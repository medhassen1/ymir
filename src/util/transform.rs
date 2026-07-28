//! Translation-rotation-scale (TRS) transforms.
//!
//! Placing a structure template (a shipwreck, a ruin, a scanned build) into
//! the world means turning the template's local block coordinates into
//! world coordinates: rotate it to face the right way, optionally scale
//! it, then translate it to the target position. `Transform` packages that
//! as one composable value instead of three loose fields threaded through
//! every call site.

use crate::util::vec3::Vec3;
use std::ops::Mul;

/// A minimal unit quaternion, defined locally so [`Transform`] does not
/// pull in the crate's shared `quat` module -- this file is a
/// self-contained TRS transform, not a general rotation library.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rotation {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Rotation {
    /// The identity rotation (no rotation at all).
    pub const IDENTITY: Rotation = Rotation { x: 0.0, y: 0.0, z: 0.0, w: 1.0 };

    /// Builds a unit quaternion representing a rotation of `angle_radians`
    /// around `axis`. Returns [`Rotation::IDENTITY`] if `axis` is the zero
    /// vector, since a rotation "around nothing" is not well-defined.
    pub fn from_axis_angle(axis: Vec3, angle_radians: f32) -> Rotation {
        let axis = axis.normalize();
        if axis == Vec3::ZERO {
            return Rotation::IDENTITY;
        }
        let half = angle_radians * 0.5;
        let s = half.sin();
        Rotation { x: axis.x * s, y: axis.y * s, z: axis.z * s, w: half.cos() }
    }

    /// The conjugate (for a unit quaternion, the inverse rotation).
    pub fn conjugate(self) -> Rotation {
        Rotation { x: -self.x, y: -self.y, z: -self.z, w: self.w }
    }

    /// Rotates `v` by this quaternion (assumed to be unit-length, which
    /// every `Rotation` built via [`Rotation::from_axis_angle`] or
    /// [`Rotation::IDENTITY`] is).
    pub fn rotate(self, v: Vec3) -> Vec3 {
        let as_quat = Rotation { x: v.x, y: v.y, z: v.z, w: 0.0 };
        let r = self * as_quat * self.conjugate();
        Vec3::new(r.x, r.y, r.z)
    }
}

impl Mul for Rotation {
    type Output = Rotation;

    /// The Hamilton product `self * other`: applying the result rotates by
    /// `other` first, then by `self`.
    fn mul(self, other: Rotation) -> Rotation {
        Rotation {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }
}

/// A translation-rotation-scale transform, applied to a point as
/// `rotation * (scale * point) + translation` (scale in local axes first,
/// then rotate, then translate).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Rotation,
    pub scale: Vec3,
}

impl Transform {
    /// The identity transform: no translation, no rotation, unit scale.
    pub const IDENTITY: Transform = Transform { translation: Vec3::ZERO, rotation: Rotation::IDENTITY, scale: Vec3::ONE };

    /// Builds a transform from its three components directly.
    pub fn new(translation: Vec3, rotation: Rotation, scale: Vec3) -> Transform {
        Transform { translation, rotation, scale }
    }

    /// A transform that only translates.
    pub fn from_translation(t: Vec3) -> Transform {
        Transform { translation: t, rotation: Rotation::IDENTITY, scale: Vec3::ONE }
    }

    /// Transforms a point: scales it, rotates it, then translates it.
    pub fn transform_point(&self, p: Vec3) -> Vec3 {
        let scaled = Vec3::new(p.x * self.scale.x, p.y * self.scale.y, p.z * self.scale.z);
        self.rotation.rotate(scaled) + self.translation
    }

    /// Transforms a direction vector: scales and rotates it, but does not
    /// translate (a direction has no position to move).
    pub fn transform_direction(&self, d: Vec3) -> Vec3 {
        let scaled = Vec3::new(d.x * self.scale.x, d.y * self.scale.y, d.z * self.scale.z);
        self.rotation.rotate(scaled)
    }

    /// Composes two transforms so that `self.compose(other).transform_point(p)`
    /// equals `other.transform_point(self.transform_point(p))`: applying
    /// `self` first, then `other`.
    pub fn compose(&self, other: &Transform) -> Transform {
        Transform {
            translation: other.transform_point(self.translation),
            rotation: other.rotation * self.rotation,
            scale: Vec3::new(self.scale.x * other.scale.x, self.scale.y * other.scale.y, self.scale.z * other.scale.z),
        }
    }

    /// The inverse transform, such that composing a transform with its
    /// inverse yields the identity. Exact whenever `scale` is uniform (or
    /// the rotation only permutes axes, e.g. a multiple of 90 degrees) --
    /// `ymir` places structure templates with uniform scale, which is the
    /// case this is built for. For an arbitrary non-uniform scale
    /// combined with an arbitrary rotation there is, in general, no exact
    /// inverse expressible in this same translation/rotation/scale form,
    /// because rotation and non-uniform scaling do not commute; this
    /// still returns the closest such transform (rotate by the
    /// conjugate, scale by the reciprocal, then cancel the translation).
    pub fn inverse(&self) -> Transform {
        let inv_scale = Vec3::new(1.0 / self.scale.x, 1.0 / self.scale.y, 1.0 / self.scale.z);
        let inv_rotation = self.rotation.conjugate();
        let unrotated = inv_rotation.rotate(-self.translation);
        let inv_translation =
            Vec3::new(unrotated.x * inv_scale.x, unrotated.y * inv_scale.y, unrotated.z * inv_scale.z);
        Transform { translation: inv_translation, rotation: inv_rotation, scale: inv_scale }
    }

    /// Applies [`transform_point`](Transform::transform_point) to many
    /// points at once, writing directly into a pre-sized buffer instead of
    /// growing a `Vec` one push at a time. Used to place every block of a
    /// structure template's point list in one pass.
    pub fn transform_points_batch(&self, points: &[Vec3]) -> Vec<Vec3> {
        let mut out: Vec<Vec3> = Vec::with_capacity(points.len());
        let ptr = out.as_mut_ptr();
        for (i, &p) in points.iter().enumerate() {
            let tp = self.transform_point(p);
            // SAFETY: `ptr` comes from `Vec::with_capacity(points.len())`,
            // so it has room for `points.len()` elements; `i` ranges over
            // `0..points.len()` (the enumeration of `points`), so
            // `ptr.add(i)` stays within that reserved capacity, and each
            // index is written exactly once before `set_len` runs below.
            unsafe {
                ptr.add(i).write(tp);
            }
        }
        // SAFETY: the loop above wrote every index `0..points.len()`
        // exactly once, so `out`'s first `points.len()` elements are all
        // initialized, and that length does not exceed the reserved
        // capacity.
        unsafe {
            out.set_len(points.len());
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPS: f32 = 1e-4;

    fn approx_eq(a: Vec3, b: Vec3) -> bool {
        (a.x - b.x).abs() < EPS && (a.y - b.y).abs() < EPS && (a.z - b.z).abs() < EPS
    }

    #[test]
    fn identity_transform_leaves_points_and_directions_unchanged() {
        let p = Vec3::new(3.0, -2.0, 7.0);
        assert!(approx_eq(Transform::IDENTITY.transform_point(p), p));
        assert!(approx_eq(Transform::IDENTITY.transform_direction(p), p));
    }

    #[test]
    fn translation_only_shifts_points_but_not_directions() {
        let t = Transform::from_translation(Vec3::new(1.0, 2.0, 3.0));
        let p = Vec3::new(0.0, 0.0, 0.0);
        assert!(approx_eq(t.transform_point(p), Vec3::new(1.0, 2.0, 3.0)));
        assert!(approx_eq(t.transform_direction(Vec3::new(5.0, 0.0, 0.0)), Vec3::new(5.0, 0.0, 0.0)));
    }

    #[test]
    fn rotation_by_90_degrees_around_y_maps_x_axis_to_negative_z() {
        let rot = Rotation::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), PI / 2.0);
        let rotated = rot.rotate(Vec3::new(1.0, 0.0, 0.0));
        assert!(approx_eq(rotated, Vec3::new(0.0, 0.0, -1.0)));
    }

    #[test]
    fn scale_is_applied_before_rotation() {
        let transform = Transform::new(Vec3::ZERO, Rotation::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), PI / 2.0), Vec3::new(2.0, 1.0, 1.0));
        // (1,0,0) scales to (2,0,0), then rotates 90 degrees around Y to (0,0,-2).
        let result = transform.transform_point(Vec3::new(1.0, 0.0, 0.0));
        assert!(approx_eq(result, Vec3::new(0.0, 0.0, -2.0)));
    }

    #[test]
    fn inverse_undoes_a_uniform_scale_rotation_and_translation() {
        let transform = Transform::new(
            Vec3::new(5.0, -3.0, 2.0),
            Rotation::from_axis_angle(Vec3::new(0.3, 1.0, -0.2), 1.1),
            Vec3::splat(2.5),
        );
        let inv = transform.inverse();
        for p in [Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO, Vec3::new(-4.0, 0.5, 9.0)] {
            let round_tripped = inv.transform_point(transform.transform_point(p));
            assert!(approx_eq(round_tripped, p));
        }
    }

    #[test]
    fn compose_matches_applying_each_transform_in_sequence() {
        let a = Transform::from_translation(Vec3::new(1.0, 0.0, 0.0));
        let b = Transform::new(Vec3::new(0.0, 5.0, 0.0), Rotation::from_axis_angle(Vec3::new(0.0, 1.0, 0.0), PI), Vec3::ONE);
        let composed = a.compose(&b);
        let p = Vec3::new(2.0, 0.0, 0.0);
        let expected = b.transform_point(a.transform_point(p));
        assert!(approx_eq(composed.transform_point(p), expected));
    }

    #[test]
    fn transform_points_batch_matches_scalar_transform_point() {
        let transform = Transform::new(Vec3::new(1.0, 1.0, 1.0), Rotation::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), 0.7), Vec3::splat(1.5));
        let points = [Vec3::new(1.0, 2.0, 3.0), Vec3::ZERO, Vec3::new(-1.0, -2.0, -3.0)];
        let batch = transform.transform_points_batch(&points);
        for (i, &p) in points.iter().enumerate() {
            assert!(approx_eq(batch[i], transform.transform_point(p)));
        }
    }
}
