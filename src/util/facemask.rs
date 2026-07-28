//! The 6 block faces, their geometry, culling rule, and a compact bitset.
//!
//! Face culling, meshing sweep order, and AO lookups all need the same
//! facts about the 6 axis-aligned faces: the opposite face, its normal and
//! neighbour offset, and a cheap way to track a subset of them.

/// One of the 6 axis-aligned faces of a block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Face {
    /// Facing toward `+X`.
    PosX = 0,
    /// Facing toward `-X`.
    NegX = 1,
    /// Facing toward `+Y`.
    PosY = 2,
    /// Facing toward `-Y`.
    NegY = 3,
    /// Facing toward `+Z`.
    PosZ = 4,
    /// Facing toward `-Z`.
    NegZ = 5,
}

const NORMALS: [[i32; 3]; 6] = [
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
];

const OPPOSITES: [Face; 6] = [
    Face::NegX,
    Face::PosX,
    Face::NegY,
    Face::PosY,
    Face::NegZ,
    Face::PosZ,
];

const ALL_FACES: [Face; 6] = [
    Face::PosX,
    Face::NegX,
    Face::PosY,
    Face::NegY,
    Face::PosZ,
    Face::NegZ,
];

impl Face {
    /// All 6 faces, in a fixed, stable order.
    pub const ALL: [Face; 6] = ALL_FACES;

    /// This face's index in `0..6`, matching its `#[repr(u8)]` discriminant.
    #[inline]
    pub fn index(self) -> usize {
        self as u8 as usize
    }

    /// Recover a face from an index in `0..6`, or `None` if out of range.
    pub fn from_index(idx: usize) -> Option<Face> {
        ALL_FACES.iter().copied().find(|f| f.index() == idx)
    }

    /// The face pointing the opposite direction (e.g. `PosX` <-> `NegX`).
    #[inline]
    pub fn opposite(self) -> Face {
        let idx = self.index();
        // SAFETY: `Face::index()` always returns a value in `0..6` (it is
        // the `#[repr(u8)]` discriminant of a 6-variant enum), and
        // `OPPOSITES` has exactly 6 entries.
        unsafe { *OPPOSITES.get_unchecked(idx) }
    }

    /// This face's outward unit normal, as integer components.
    #[inline]
    pub fn normal(self) -> [i32; 3] {
        let idx = self.index();
        // SAFETY: `idx` is in `0..6` for the same reason as in `opposite`,
        // and `NORMALS` has exactly 6 entries.
        unsafe { *NORMALS.get_unchecked(idx) }
    }

    /// The block-coordinate offset of the neighbour this face touches.
    /// Identical to [`Face::normal`] — the neighbour in a given direction is
    /// always exactly one block along that direction's unit normal.
    #[inline]
    pub fn offset(self) -> [i32; 3] {
        self.normal()
    }
}

/// Decide whether each of two blocks sharing a boundary along one axis
/// should render the face pointing at the other. A face is only worth
/// drawing if its own block is solid and the neighbour it looks into is
/// not — two solid blocks cull both faces (nothing sees the seam between
/// them) and two empty "blocks" have no faces to begin with.
#[inline]
pub fn cull_adjacent(a_solid: bool, b_solid: bool) -> (bool, bool) {
    (a_solid && !b_solid, b_solid && !a_solid)
}

/// A compact bitset of [`Face`] values, one bit per face, backed by a
/// single byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FaceSet(u8);

impl FaceSet {
    /// The empty set.
    pub const EMPTY: FaceSet = FaceSet(0);
    /// The set containing all 6 faces.
    pub const ALL: FaceSet = FaceSet(0b0011_1111);

    /// Add `face` to the set.
    #[inline]
    pub fn insert(&mut self, face: Face) {
        self.0 |= 1 << face.index();
    }

    /// Remove `face` from the set.
    #[inline]
    pub fn remove(&mut self, face: Face) {
        self.0 &= !(1 << face.index());
    }

    /// Whether `face` is present in the set.
    #[inline]
    pub fn contains(&self, face: Face) -> bool {
        self.0 & (1 << face.index()) != 0
    }

    /// Number of faces currently in the set.
    #[inline]
    pub fn len(&self) -> u32 {
        self.0.count_ones()
    }

    /// Whether the set has no faces.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0 == 0
    }

    /// Iterate the faces currently in the set, lowest index first.
    pub fn iter(&self) -> FaceSetIter {
        FaceSetIter { bits: self.0 }
    }
}

/// Iterator over the faces present in a [`FaceSet`], produced by
/// [`FaceSet::iter`].
pub struct FaceSetIter {
    bits: u8,
}

impl Iterator for FaceSetIter {
    type Item = Face;

    fn next(&mut self) -> Option<Face> {
        if self.bits == 0 {
            return None;
        }
        let idx = self.bits.trailing_zeros() as usize;
        self.bits &= self.bits - 1; // clear the lowest set bit
        // SAFETY: `FaceSet` only ever sets bits produced by `Face::index()`
        // (via `insert`), which are always in `0..6`, and `FaceSet::ALL`
        // (the widest possible value) masks exactly those 6 bits. So any
        // bit set in `self.bits` has `trailing_zeros() < 6`, matching
        // `ALL_FACES`'s length.
        Some(unsafe { *ALL_FACES.get_unchecked(idx) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opposite_is_an_involution_for_every_face() {
        for face in Face::ALL {
            assert_eq!(face.opposite().opposite(), face);
            assert_ne!(face.opposite(), face);
        }
    }

    #[test]
    fn normals_are_unit_axis_vectors() {
        assert_eq!(Face::PosX.normal(), [1, 0, 0]);
        assert_eq!(Face::NegX.normal(), [-1, 0, 0]);
        assert_eq!(Face::PosY.normal(), [0, 1, 0]);
        assert_eq!(Face::NegY.normal(), [0, -1, 0]);
        assert_eq!(Face::PosZ.normal(), [0, 0, 1]);
        assert_eq!(Face::NegZ.normal(), [0, 0, -1]);
    }

    #[test]
    fn offset_matches_normal_for_every_face() {
        for face in Face::ALL {
            assert_eq!(face.offset(), face.normal());
        }
    }

    #[test]
    fn from_index_round_trips_with_index() {
        for face in Face::ALL {
            assert_eq!(Face::from_index(face.index()), Some(face));
        }
        assert_eq!(Face::from_index(6), None);
    }

    #[test]
    fn cull_adjacent_only_renders_the_solid_side_facing_empty_space() {
        assert_eq!(cull_adjacent(true, false), (true, false));
        assert_eq!(cull_adjacent(false, true), (false, true));
        assert_eq!(cull_adjacent(true, true), (false, false));
        assert_eq!(cull_adjacent(false, false), (false, false));
    }

    #[test]
    fn face_set_insert_remove_and_contains() {
        let mut set = FaceSet::EMPTY;
        assert!(set.is_empty());
        set.insert(Face::PosY);
        set.insert(Face::NegZ);
        assert!(set.contains(Face::PosY));
        assert!(set.contains(Face::NegZ));
        assert!(!set.contains(Face::PosX));
        assert_eq!(set.len(), 2);
        set.remove(Face::PosY);
        assert!(!set.contains(Face::PosY));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn face_set_iterates_every_inserted_face_exactly_once() {
        let mut set = FaceSet::EMPTY;
        for face in [Face::PosX, Face::NegY, Face::PosZ] {
            set.insert(face);
        }
        let collected: Vec<Face> = set.iter().collect();
        assert_eq!(collected.len(), 3);
        for face in [Face::PosX, Face::NegY, Face::PosZ] {
            assert!(collected.contains(&face));
        }
        assert_eq!(FaceSet::ALL.iter().count(), 6);
        assert_eq!(FaceSet::EMPTY.iter().count(), 0);
    }
}
