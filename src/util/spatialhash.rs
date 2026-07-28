//! Hashing 3D integer coordinates into buckets, plus a small
//! coordinate-keyed map built on top of it.
//!
//! A loaded voxel world routinely needs to answer "is there a chunk (or
//! entity, or light-update record) at this coordinate" for coordinates
//! spanning the full range of `i32`. Rather than reach for a general
//! hash-map keyed on a tuple (which usually means a SipHash-quality,
//! DoS-resistant but comparatively slow hasher), `ymir` mixes coordinates
//! directly with a cheap, well-distributed finalizer and uses that to
//! drive its own small open-addressed table.

/// The 64-bit finalizer from MurmurHash3's 128-bit variant, used here
/// purely as a strong, cheap integer avalanche mix.
#[inline]
fn mix64(mut x: u64) -> u64 {
    x ^= x >> 33;
    x = x.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    x ^= x >> 33;
    x = x.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    x ^= x >> 33;
    x
}

/// Hashes a chunk column coordinate `(x, z)` into a bucket index. The two
/// 32-bit coordinates are packed into one 64-bit word (as their unsigned
/// bit patterns, so negative coordinates round-trip through the packing
/// without sign-extension surprises) before mixing.
pub fn hash_chunk(x: i32, z: i32) -> u64 {
    let packed = ((x as u32 as u64) << 32) | (z as u32 as u64);
    mix64(packed)
}

/// Hashes a full block coordinate `(x, y, z)` into a bucket index, by
/// folding each axis through the mixer in turn with a distinct odd
/// multiplier so permuted coordinates (e.g. swapping `x` and `z`) do not
/// collide.
pub fn hash_block(x: i32, y: i32, z: i32) -> u64 {
    let h = mix64((x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
    let h = mix64(h ^ (y as u32 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F));
    mix64(h ^ (z as u32 as u64).wrapping_mul(0x1656_67B1_9E37_79F9))
}

/// One slot in a [`CoordMap`]'s backing table.
#[derive(Debug, Clone, Copy)]
struct Slot {
    key: u64,
    value: u32,
    occupied: bool,
}

impl Slot {
    const EMPTY: Slot = Slot { key: 0, value: 0, occupied: false };
}

/// A small open-addressed map from a 3D block coordinate to a `u32`
/// (e.g. a palette index, a light level, or a packed state ID). Capacity
/// is always a power of two, so a slot's home bucket is `key & (cap - 1)`
/// with no division; collisions are resolved by linear probing.
#[derive(Debug, Clone)]
pub struct CoordMap {
    slots: Vec<Slot>,
    len: usize,
}

impl Default for CoordMap {
    fn default() -> Self {
        Self::new()
    }
}

impl CoordMap {
    /// Creates an empty map with a small default capacity.
    pub fn new() -> Self {
        Self::with_capacity(8)
    }

    /// Creates an empty map with room for at least `capacity` entries
    /// before its first internal resize.
    pub fn with_capacity(capacity: usize) -> Self {
        let cap = capacity.next_power_of_two().max(4);
        CoordMap { slots: vec![Slot::EMPTY; cap], len: 0 }
    }

    /// The number of entries currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the map holds no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// Inserts `value` at `(x, y, z)`, returning the previous value if
    /// the coordinate was already present.
    pub fn insert(&mut self, x: i32, y: i32, z: i32, value: u32) -> Option<u32> {
        if (self.len + 1) * 4 >= self.slots.len() * 3 {
            self.grow();
        }
        let key = hash_block(x, y, z);
        self.insert_by_key(key, value)
    }

    /// Looks up the value stored at `(x, y, z)`, if any.
    pub fn get(&self, x: i32, y: i32, z: i32) -> Option<u32> {
        let key = hash_block(x, y, z);
        let mask = self.mask();
        let mut idx = (key as usize) & mask;
        loop {
            // SAFETY: `idx` starts as `key as usize & mask` and is only
            // ever advanced via `(idx + 1) & mask`, so it is always in
            // `0..slots.len()` (`mask == slots.len() - 1` and
            // `slots.len()` is a power of two).
            let slot = unsafe { self.slots.get_unchecked(idx) };
            if !slot.occupied {
                return None;
            }
            if slot.key == key {
                return Some(slot.value);
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Shared insert path used both by `insert` (which must compute the
    /// key from coordinates) and `grow` (which already has the key from
    /// an existing slot and must not re-hash coordinates it no longer has).
    fn insert_by_key(&mut self, key: u64, value: u32) -> Option<u32> {
        let mask = self.mask();
        let mut idx = (key as usize) & mask;
        loop {
            // SAFETY: identical reasoning to `get`: `idx` is always
            // `(key as usize) & mask` advanced by steps of `(idx + 1) &
            // mask`, so it never leaves `0..slots.len()`.
            let slot = unsafe { self.slots.get_unchecked_mut(idx) };
            if !slot.occupied {
                *slot = Slot { key, value, occupied: true };
                self.len += 1;
                return None;
            }
            if slot.key == key {
                let old = slot.value;
                slot.value = value;
                return Some(old);
            }
            idx = (idx + 1) & mask;
        }
    }

    /// Doubles capacity and reinserts every occupied slot. Because
    /// capacity always at least doubles and insertion keeps the load
    /// factor under 75%, linear probing is guaranteed to find an empty
    /// slot for every reinserted entry without looping forever.
    fn grow(&mut self) {
        let new_cap = self.slots.len() * 2;
        let old = std::mem::replace(&mut self.slots, vec![Slot::EMPTY; new_cap]);
        self.len = 0;
        for slot in old.into_iter().filter(|s| s.occupied) {
            self.insert_by_key(slot.key, slot.value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_functions_are_deterministic() {
        assert_eq!(hash_chunk(3, -7), hash_chunk(3, -7));
        assert_eq!(hash_block(1, 2, 3), hash_block(1, 2, 3));
    }

    #[test]
    fn nearby_coordinates_scatter_to_different_hashes() {
        assert_ne!(hash_chunk(0, 0), hash_chunk(1, 0));
        assert_ne!(hash_block(0, 0, 0), hash_block(0, 0, 1));
        assert_ne!(hash_block(1, 2, 3), hash_block(3, 2, 1));
    }

    #[test]
    fn insert_and_get_round_trip_over_a_coordinate_grid() {
        let mut map = CoordMap::new();
        let mut expected = 0u32;
        for x in -4..4 {
            for y in 0..6 {
                for z in -4..4 {
                    map.insert(x, y, z, expected);
                    expected += 1;
                }
            }
        }
        assert_eq!(map.len(), expected as usize);

        let mut check = 0u32;
        for x in -4..4 {
            for y in 0..6 {
                for z in -4..4 {
                    assert_eq!(map.get(x, y, z), Some(check));
                    check += 1;
                }
            }
        }
    }

    #[test]
    fn overwriting_a_coordinate_returns_the_previous_value() {
        let mut map = CoordMap::new();
        assert_eq!(map.insert(1, 1, 1, 10), None);
        assert_eq!(map.insert(1, 1, 1, 20), Some(10));
        assert_eq!(map.get(1, 1, 1), Some(20));
        assert_eq!(map.len(), 1);
    }

    #[test]
    fn missing_coordinate_returns_none() {
        let map = CoordMap::new();
        assert_eq!(map.get(99, 99, 99), None);
    }

    #[test]
    fn survives_growth_with_all_entries_intact() {
        let mut map = CoordMap::with_capacity(4);
        for i in 0..500i32 {
            map.insert(i, -i, i * 2, i as u32);
        }
        assert_eq!(map.len(), 500);
        for i in 0..500i32 {
            assert_eq!(map.get(i, -i, i * 2), Some(i as u32));
        }
        assert_eq!(map.get(500, -500, 1000), None);
    }
}
