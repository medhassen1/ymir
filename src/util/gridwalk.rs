//! 3D DDA voxel traversal (Amanatides & Woo).
//!
//! Block picking (which voxel is the player looking at?), light and
//! occlusion rays, and line-of-sight checks all need to walk exactly the
//! sequence of voxels a ray passes through, in order, without skipping or
//! double-visiting any of them. A naive small-step marcher can tunnel
//! through thin geometry or revisit a voxel twice; the Amanatides & Woo
//! algorithm instead steps directly from one voxel boundary to the next.

use crate::util::ivec3::IVec3;
use crate::util::vec3::Vec3;

/// Which axis-aligned face of a voxel a step crossed to enter it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Face {
    /// The ray's starting voxel: it was not entered through any face.
    None,
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

/// One voxel visited by a [`GridWalk`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelHit {
    /// The voxel's integer coordinate.
    pub voxel: IVec3,
    /// The ray parameter `t` at which the ray entered this voxel.
    pub t: f32,
    /// Which face of `voxel` the ray crossed to enter it.
    pub face: Face,
}

/// Per-axis stepping state: how far (in ray-parameter `t`) to the next
/// boundary crossing, and how much `t` advances per full voxel crossed.
fn axis_init(origin: f32, dir: f32) -> (i32, f32, f32) {
    if dir.abs() < f32::EPSILON {
        (0, f32::INFINITY, f32::INFINITY)
    } else if dir > 0.0 {
        let next_boundary = origin.floor() + 1.0;
        (1, (next_boundary - origin) / dir, 1.0 / dir)
    } else {
        let next_boundary = origin.floor();
        (-1, (next_boundary - origin) / dir, 1.0 / -dir)
    }
}

/// Iterator over the sequence of integer voxels a ray passes through,
/// starting at `origin` and heading in direction `dir`, up to parameter
/// `max_t`.
pub struct GridWalk {
    voxel: IVec3,
    step: IVec3,
    t_max: Vec3,
    t_delta: Vec3,
    t: f32,
    face: Face,
    max_t: f32,
    finished: bool,
}

impl GridWalk {
    /// Starts a walk from `origin` along `dir`, stopping once the ray
    /// parameter would exceed `max_t`. `dir` need not be normalized: `t`
    /// is always measured in units of `dir`'s own length.
    pub fn new(origin: Vec3, dir: Vec3, max_t: f32) -> GridWalk {
        let voxel = IVec3::new(origin.x.floor() as i32, origin.y.floor() as i32, origin.z.floor() as i32);
        let (step_x, tmax_x, tdelta_x) = axis_init(origin.x, dir.x);
        let (step_y, tmax_y, tdelta_y) = axis_init(origin.y, dir.y);
        let (step_z, tmax_z, tdelta_z) = axis_init(origin.z, dir.z);
        GridWalk {
            voxel,
            step: IVec3::new(step_x, step_y, step_z),
            t_max: Vec3::new(tmax_x, tmax_y, tmax_z),
            t_delta: Vec3::new(tdelta_x, tdelta_y, tdelta_z),
            t: 0.0,
            face: Face::None,
            max_t,
            finished: max_t < 0.0,
        }
    }

    /// Collects up to `n` hits from the walk in one call, writing directly
    /// into a pre-sized buffer instead of growing a `Vec` one push at a
    /// time. Returns fewer than `n` hits if the walk ends (exceeds
    /// `max_t`, or is a degenerate zero-direction ray) first.
    ///
    /// Named `take_n` rather than `take` so it does not shadow (and get
    /// silently shadowed by) [`Iterator::take`], which has a different,
    /// by-value receiver.
    pub fn take_n(&mut self, n: usize) -> Vec<VoxelHit> {
        let mut out: Vec<VoxelHit> = Vec::with_capacity(n);
        let ptr = out.as_mut_ptr();
        let mut count = 0usize;
        while count < n {
            match self.next() {
                Some(hit) => {
                    // SAFETY: `ptr` comes from `Vec::with_capacity(n)`, so
                    // it has room for `n` elements; the loop condition
                    // guarantees `count < n` here, and `count` increases
                    // by exactly one per write, so `ptr.add(count)` never
                    // leaves the reserved capacity and each slot is
                    // written at most once.
                    unsafe {
                        ptr.add(count).write(hit);
                    }
                    count += 1;
                }
                None => break,
            }
        }
        // SAFETY: the loop above wrote exactly `count` elements, at
        // indices `0..count`, and `count <= n` equals the reserved
        // capacity, so `out`'s first `count` elements are all
        // initialized.
        unsafe {
            out.set_len(count);
        }
        out
    }
}

impl Iterator for GridWalk {
    type Item = VoxelHit;

    fn next(&mut self) -> Option<VoxelHit> {
        if self.finished || self.t > self.max_t {
            return None;
        }
        let hit = VoxelHit { voxel: self.voxel, t: self.t, face: self.face };

        // Advance along whichever axis has the nearest upcoming boundary.
        if self.t_max.x <= self.t_max.y && self.t_max.x <= self.t_max.z {
            if self.step.x == 0 {
                self.finished = true;
                return Some(hit);
            }
            self.voxel.x += self.step.x;
            self.t = self.t_max.x;
            self.t_max.x += self.t_delta.x;
            self.face = if self.step.x > 0 { Face::NegX } else { Face::PosX };
        } else if self.t_max.y <= self.t_max.z {
            if self.step.y == 0 {
                self.finished = true;
                return Some(hit);
            }
            self.voxel.y += self.step.y;
            self.t = self.t_max.y;
            self.t_max.y += self.t_delta.y;
            self.face = if self.step.y > 0 { Face::NegY } else { Face::PosY };
        } else {
            if self.step.z == 0 {
                self.finished = true;
                return Some(hit);
            }
            self.voxel.z += self.step.z;
            self.t = self.t_max.z;
            self.t_max.z += self.t_delta.z;
            self.face = if self.step.z > 0 { Face::NegZ } else { Face::PosZ };
        }
        Some(hit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-4;

    #[test]
    fn axis_aligned_positive_ray_visits_expected_voxels_in_order() {
        let walk = GridWalk::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(1.0, 0.0, 0.0), 3.0);
        let hits: Vec<VoxelHit> = walk.collect();
        let voxels: Vec<IVec3> = hits.iter().map(|h| h.voxel).collect();
        assert_eq!(voxels, vec![IVec3::new(0, 0, 0), IVec3::new(1, 0, 0), IVec3::new(2, 0, 0), IVec3::new(3, 0, 0)]);
        assert_eq!(hits[0].face, Face::None);
        for h in &hits[1..] {
            assert_eq!(h.face, Face::NegX);
        }
    }

    #[test]
    fn negative_direction_steps_the_other_way_with_matching_faces() {
        let walk = GridWalk::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(-1.0, 0.0, 0.0), 2.0);
        let hits: Vec<VoxelHit> = walk.collect();
        let voxels: Vec<IVec3> = hits.iter().map(|h| h.voxel).collect();
        assert_eq!(voxels, vec![IVec3::new(0, 0, 0), IVec3::new(-1, 0, 0), IVec3::new(-2, 0, 0)]);
        for h in &hits[1..] {
            assert_eq!(h.face, Face::PosX);
        }
    }

    #[test]
    fn negative_origin_floors_toward_negative_infinity() {
        let mut walk = GridWalk::new(Vec3::new(-0.5, -0.5, -0.5), Vec3::new(1.0, 0.0, 0.0), 0.0);
        let first = walk.next().unwrap();
        assert_eq!(first.voxel, IVec3::new(-1, -1, -1));
    }

    #[test]
    fn degenerate_zero_direction_yields_only_the_start_voxel() {
        let walk = GridWalk::new(Vec3::new(1.5, 2.5, 3.5), Vec3::ZERO, 10.0);
        let hits: Vec<VoxelHit> = walk.collect();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].voxel, IVec3::new(1, 2, 3));
        assert_eq!(hits[0].face, Face::None);
    }

    #[test]
    fn entry_t_is_monotonically_nondecreasing_along_a_diagonal_ray() {
        let walk = GridWalk::new(Vec3::new(0.25, 0.75, 0.5), Vec3::new(1.0, 1.3, -0.7), 5.0);
        let hits: Vec<VoxelHit> = walk.collect();
        assert!(hits.len() > 1);
        for pair in hits.windows(2) {
            assert!(pair[1].t >= pair[0].t - EPS);
        }
    }

    #[test]
    fn diagonal_ray_never_revisits_a_voxel() {
        let walk = GridWalk::new(Vec3::new(0.1, 0.1, 0.1), Vec3::new(1.0, 1.0, 1.0), 4.0);
        let hits: Vec<VoxelHit> = walk.collect();
        let mut voxels: Vec<IVec3> = hits.iter().map(|h| h.voxel).collect();
        let before = voxels.len();
        voxels.sort_by_key(|v| (v.x, v.y, v.z));
        voxels.dedup();
        assert_eq!(voxels.len(), before);
    }

    #[test]
    fn take_n_matches_manual_iteration() {
        let mut walk_a = GridWalk::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.6, 0.2, -0.9), 6.0);
        let mut walk_b = GridWalk::new(Vec3::new(0.5, 0.5, 0.5), Vec3::new(0.6, 0.2, -0.9), 6.0);
        let taken = walk_a.take_n(5);
        let manual: Vec<VoxelHit> = (0..5).filter_map(|_| walk_b.next()).collect();
        assert_eq!(taken, manual);
    }
}
