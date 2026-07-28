//! Sparse octree over a fixed cubic world extent.
//!
//! Entities, light-update queues, and structure-placement queries all need
//! "what's near this point in 3D" answers over a world far too large to
//! hold a dense grid of. `Octree` gives them a point index that only pays
//! for the regions actually populated, backed by a flat node pool with
//! index-based child links so the whole structure can be dropped or
//! serialized without chasing pointers.

use crate::util::ivec3::IVec3;
use crate::util::vec3::Vec3;

/// Maximum points a leaf holds before it subdivides (assuming the depth
/// limit has not been reached).
const LEAF_CAPACITY: usize = 4;

/// One node in the pool. Internal nodes have `children: Some(_)` and an
/// empty `entries`; leaves have `children: None` and hold their points
/// directly.
struct Node {
    center: Vec3,
    half: f32,
    depth: u32,
    children: Option<[u32; 8]>,
    entries: Vec<(Vec3, u32)>,
}

/// A sparse octree storing `(position, payload)` pairs over a fixed cubic
/// region. Nodes live in one `Vec` and reference each other by index
/// (never by raw pointer), so the tree is trivially relocatable and its
/// memory is one contiguous allocation per instance.
pub struct Octree {
    nodes: Vec<Node>,
    max_depth: u32,
}

impl Octree {
    /// Creates an empty octree covering the cube from `origin` to
    /// `origin + Vec3::splat(size)`, subdividing at most `max_depth`
    /// times below the root.
    pub fn new(origin: Vec3, size: f32, max_depth: u32) -> Octree {
        let half = size * 0.5;
        let root = Node {
            center: origin + Vec3::splat(half),
            half,
            depth: 0,
            children: None,
            entries: Vec::new(),
        };
        Octree { nodes: vec![root], max_depth }
    }

    /// The total number of pooled nodes (internal and leaf), for tests and
    /// capacity introspection.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The total number of points stored across every leaf.
    pub fn len(&self) -> usize {
        self.leaves().map(|leaf| leaf.len()).sum()
    }

    /// Whether the octree holds no points.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Inserts `point` with `payload`, returning `false` without modifying
    /// the tree if `point` lies outside the root's cubic extent.
    pub fn insert(&mut self, point: Vec3, payload: u32) -> bool {
        let (center, half) = {
            let root = &self.nodes[0];
            (root.center, root.half)
        };
        if !Self::cube_contains(center, half, point) {
            return false;
        }
        self.insert_at(0, point, payload);
        true
    }

    /// Convenience wrapper for inserting an integer voxel coordinate,
    /// using its center as the stored point (`coord + (0.5, 0.5, 0.5)`).
    pub fn insert_voxel(&mut self, coord: IVec3, payload: u32) -> bool {
        self.insert(coord.to_vec3() + Vec3::splat(0.5), payload)
    }

    fn insert_at(&mut self, idx: usize, point: Vec3, payload: u32) {
        let children = self.nodes[idx].children;
        if let Some(children) = children {
            let center = self.nodes[idx].center;
            let octant = Self::octant(center, point);
            // SAFETY: `Self::octant` only ever sets bits `1`, `2`, and
            // `4`, so its return value is always in `0..8`, matching the
            // fixed length-8 `children` array.
            let child_idx = unsafe { *children.get_unchecked(octant) } as usize;
            self.insert_at(child_idx, point, payload);
            return;
        }

        let depth = self.nodes[idx].depth;
        let can_subdivide = depth < self.max_depth;
        if self.nodes[idx].entries.len() < LEAF_CAPACITY || !can_subdivide {
            self.nodes[idx].entries.push((point, payload));
            return;
        }

        self.subdivide(idx);
        self.insert_at(idx, point, payload);
    }

    /// Turns leaf `idx` into an internal node with 8 fresh leaf children,
    /// redistributing its existing entries among them.
    fn subdivide(&mut self, idx: usize) {
        let (center, half, depth) = {
            let node = &self.nodes[idx];
            (node.center, node.half, node.depth)
        };
        let child_half = half * 0.5;

        let mut child_ids = [0u32; 8];
        for (octant, slot) in child_ids.iter_mut().enumerate() {
            let offset = Vec3::new(
                if octant & 1 != 0 { child_half } else { -child_half },
                if octant & 2 != 0 { child_half } else { -child_half },
                if octant & 4 != 0 { child_half } else { -child_half },
            );
            *slot = self.nodes.len() as u32;
            self.nodes.push(Node {
                center: center + offset,
                half: child_half,
                depth: depth + 1,
                children: None,
                entries: Vec::new(),
            });
        }

        let old_entries = std::mem::take(&mut self.nodes[idx].entries);
        self.nodes[idx].children = Some(child_ids);
        for (p, payload) in old_entries {
            let octant = Self::octant(center, p);
            // SAFETY: `Self::octant` only ever sets bits `1`, `2`, and
            // `4`, so its return value is always in `0..8`, matching the
            // fixed length-8 `child_ids` array, which the loop above
            // populated completely.
            let child_idx = unsafe { *child_ids.get_unchecked(octant) } as usize;
            self.insert_at(child_idx, p, payload);
        }
    }

    /// Collects every stored `(point, payload)` whose point lies within
    /// the axis-aligned box `[region_min, region_max]` (inclusive).
    pub fn query_region(&self, region_min: Vec3, region_max: Vec3) -> Vec<(Vec3, u32)> {
        let mut out = Vec::new();
        self.query_at(0, region_min, region_max, &mut out);
        out
    }

    fn query_at(&self, idx: usize, region_min: Vec3, region_max: Vec3, out: &mut Vec<(Vec3, u32)>) {
        // SAFETY: `idx` is either the root index `0` (always valid, since
        // `nodes` starts with one element and is never emptied) or a
        // child index recorded as `self.nodes.len()` immediately before
        // the matching `push` in `subdivide`. Because `nodes` only ever
        // grows, every previously recorded index stays in bounds.
        let node = unsafe { self.nodes.get_unchecked(idx) };
        let node_min = node.center - Vec3::splat(node.half);
        let node_max = node.center + Vec3::splat(node.half);
        if !Self::boxes_overlap(node_min, node_max, region_min, region_max) {
            return;
        }
        match node.children {
            Some(children) => {
                for c in children {
                    self.query_at(c as usize, region_min, region_max, out);
                }
            }
            None => {
                for &(p, payload) in &node.entries {
                    if Self::point_in_box(p, region_min, region_max) {
                        out.push((p, payload));
                    }
                }
            }
        }
    }

    /// An iterator over every leaf's point slice, internal (non-leaf)
    /// nodes skipped entirely.
    pub fn leaves(&self) -> Leaves<'_> {
        Leaves { tree: self, stack: vec![0] }
    }

    fn octant(center: Vec3, point: Vec3) -> usize {
        let mut idx = 0usize;
        if point.x >= center.x {
            idx |= 1;
        }
        if point.y >= center.y {
            idx |= 2;
        }
        if point.z >= center.z {
            idx |= 4;
        }
        idx
    }

    fn cube_contains(center: Vec3, half: f32, point: Vec3) -> bool {
        (point.x - center.x).abs() <= half
            && (point.y - center.y).abs() <= half
            && (point.z - center.z).abs() <= half
    }

    fn boxes_overlap(a_min: Vec3, a_max: Vec3, b_min: Vec3, b_max: Vec3) -> bool {
        a_min.x <= b_max.x
            && a_max.x >= b_min.x
            && a_min.y <= b_max.y
            && a_max.y >= b_min.y
            && a_min.z <= b_max.z
            && a_max.z >= b_min.z
    }

    fn point_in_box(p: Vec3, region_min: Vec3, region_max: Vec3) -> bool {
        p.x >= region_min.x
            && p.x <= region_max.x
            && p.y >= region_min.y
            && p.y <= region_max.y
            && p.z >= region_min.z
            && p.z <= region_max.z
    }
}

/// Depth-first iterator over a tree's leaf point slices. See
/// [`Octree::leaves`].
pub struct Leaves<'a> {
    tree: &'a Octree,
    stack: Vec<usize>,
}

impl<'a> Iterator for Leaves<'a> {
    type Item = &'a [(Vec3, u32)];

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(idx) = self.stack.pop() {
            // SAFETY: identical reasoning to `Octree::query_at`: every
            // index pushed onto `stack` either started as the root `0` or
            // came from a node's `children`, which only ever holds
            // indices recorded before their target was pushed into a
            // `nodes` `Vec` that never shrinks.
            let node = unsafe { self.tree.nodes.get_unchecked(idx) };
            match node.children {
                Some(children) => self.stack.extend(children.iter().map(|&c| c as usize)),
                None => return Some(&node.entries),
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_outside_root_extent_is_rejected() {
        let mut tree = Octree::new(Vec3::ZERO, 16.0, 4);
        assert!(!tree.insert(Vec3::new(100.0, 0.0, 0.0), 1));
        assert_eq!(tree.len(), 0);
    }

    #[test]
    fn insert_and_query_region_finds_the_point() {
        let mut tree = Octree::new(Vec3::ZERO, 16.0, 4);
        assert!(tree.insert(Vec3::new(2.0, 2.0, 2.0), 42));
        let hits = tree.query_region(Vec3::new(0.0, 0.0, 0.0), Vec3::new(4.0, 4.0, 4.0));
        assert_eq!(hits, vec![(Vec3::new(2.0, 2.0, 2.0), 42)]);
    }

    #[test]
    fn query_region_excludes_points_outside_it() {
        let mut tree = Octree::new(Vec3::ZERO, 16.0, 4);
        tree.insert(Vec3::new(1.0, 1.0, 1.0), 1);
        tree.insert(Vec3::new(15.0, 15.0, 15.0), 2);
        let hits = tree.query_region(Vec3::new(0.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(hits, vec![(Vec3::new(1.0, 1.0, 1.0), 1)]);
    }

    #[test]
    fn subdivision_happens_after_leaf_capacity_is_exceeded() {
        let mut tree = Octree::new(Vec3::ZERO, 16.0, 6);
        assert_eq!(tree.node_count(), 1);
        // Points scattered on both sides of the root's center (8, 8, 8),
        // so they land in more than one octant, forcing a genuine split
        // rather than repeatedly landing in the same child.
        for i in 0..(LEAF_CAPACITY as i32 + 1) {
            let s = if i % 2 == 0 { 1.0 } else { -1.0 };
            let offset = s * (i + 1) as f32 * 0.1;
            assert!(tree.insert(Vec3::splat(8.0 + offset), i as u32));
        }
        assert!(tree.node_count() > 1);
        assert_eq!(tree.len(), LEAF_CAPACITY + 1);
    }

    #[test]
    fn depth_limit_stops_infinite_subdivision_for_coincident_points() {
        let mut tree = Octree::new(Vec3::ZERO, 16.0, 3);
        for i in 0..40u32 {
            assert!(tree.insert(Vec3::new(1.0, 1.0, 1.0), i));
        }
        assert_eq!(tree.len(), 40);
        // Bounded by the geometric series of a depth-3 octree; well under
        // a runaway/unbounded node count.
        assert!(tree.node_count() < 200);
    }

    #[test]
    fn leaves_iterator_visits_every_inserted_point_exactly_once() {
        let mut tree = Octree::new(Vec3::ZERO, 16.0, 5);
        let mut expected: Vec<u32> = Vec::new();
        for i in 0..30u32 {
            let p = Vec3::new((i % 5) as f32, (i % 3) as f32, (i % 7) as f32);
            tree.insert(p, i);
            expected.push(i);
        }
        let mut seen: Vec<u32> = tree.leaves().flat_map(|leaf| leaf.iter().map(|&(_, payload)| payload)).collect();
        seen.sort_unstable();
        expected.sort_unstable();
        assert_eq!(seen, expected);
    }

    #[test]
    fn insert_voxel_matches_the_equivalent_vec3_insert() {
        let mut a = Octree::new(Vec3::ZERO, 16.0, 4);
        let mut b = Octree::new(Vec3::ZERO, 16.0, 4);
        assert!(a.insert_voxel(IVec3::new(3, 4, 5), 7));
        assert!(b.insert(Vec3::new(3.5, 4.5, 5.5), 7));
        assert_eq!(a.query_region(Vec3::ZERO, Vec3::splat(16.0)), b.query_region(Vec3::ZERO, Vec3::splat(16.0)));
    }
}
