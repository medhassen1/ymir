//! World <-> chunk <-> section <-> region coordinate conversions.
//!
//! Every store access must land on the right chunk, section, linear index,
//! and region file; getting negative coordinates wrong is the most common
//! bug here, so every conversion is built on `div_euclid`/`rem_euclid`.

/// Blocks per chunk-section edge.
pub const SECTION_EDGE: usize = 16;

/// Chunks per region edge (a region is a `REGION_EDGE x REGION_EDGE` grid of
/// chunk columns, mirroring the on-disk container this crate persists to).
pub const REGION_EDGE: i32 = 32;

/// Split a single world block coordinate into its chunk index and local
/// in-chunk offset (`0..SECTION_EDGE`). Correct for negative coordinates:
/// block `-1` is local offset `15` of chunk `-1`, not a negative offset.
#[inline]
pub fn world_to_chunk_component(block: i32) -> (i32, u8) {
    let chunk = block.div_euclid(SECTION_EDGE as i32);
    let local = block.rem_euclid(SECTION_EDGE as i32) as u8;
    (chunk, local)
}

/// The world block coordinate of the `-x/-y/-z` corner of a chunk index.
#[inline]
pub fn chunk_base_block(chunk: i32) -> i32 {
    chunk * SECTION_EDGE as i32
}

/// The vertical section index containing world Y coordinate `y` (sections
/// are `SECTION_EDGE` blocks tall and stack through negative Y as well).
#[inline]
pub fn section_index(y: i32) -> i32 {
    y.div_euclid(SECTION_EDGE as i32)
}

/// The local Y offset (`0..SECTION_EDGE`) of world Y coordinate `y` within
/// its section.
#[inline]
pub fn local_y(y: i32) -> u8 {
    y.rem_euclid(SECTION_EDGE as i32) as u8
}

/// The linear index of block `(x, y, z)` within one section's flat block
/// array, using the layout `y * 256 + z * 16 + x`. Panics if any coordinate
/// is `>= SECTION_EDGE`.
#[inline]
pub fn linear_index(x: usize, y: usize, z: usize) -> usize {
    assert!(x < SECTION_EDGE && y < SECTION_EDGE && z < SECTION_EDGE, "coordinate out of range");
    y * SECTION_EDGE * SECTION_EDGE + z * SECTION_EDGE + x
}

/// Inverse of [`linear_index`]: recover `(x, y, z)` from a section-local
/// linear index. Panics if `idx >= SECTION_EDGE^3`.
#[inline]
pub fn index_to_xyz(idx: usize) -> (usize, usize, usize) {
    let volume = SECTION_EDGE * SECTION_EDGE * SECTION_EDGE;
    assert!(idx < volume, "index out of range");
    let y = idx / (SECTION_EDGE * SECTION_EDGE);
    let rem = idx % (SECTION_EDGE * SECTION_EDGE);
    let z = rem / SECTION_EDGE;
    let x = rem % SECTION_EDGE;
    (x, y, z)
}

/// Convert many `(x, y, z)` triples to linear indices in one pass, writing
/// straight into a pre-sized buffer instead of growing a `Vec` one `push`
/// at a time.
pub fn linear_index_batch(coords: &[(u8, u8, u8)]) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(coords.len());
    let ptr = out.as_mut_ptr();
    for (i, &(x, y, z)) in coords.iter().enumerate() {
        let idx = linear_index(x as usize, y as usize, z as usize);
        debug_assert!(idx <= u16::MAX as usize);
        // SAFETY: `ptr` comes from `Vec::with_capacity(coords.len())`, so it
        // has room for `coords.len()` elements; `i` ranges over
        // `0..coords.len()` (the enumeration of `coords`), so `ptr.add(i)`
        // stays within that reserved capacity, and this loop writes each
        // index in `0..coords.len()` exactly once before `set_len` runs.
        unsafe {
            ptr.add(i).write(idx as u16);
        }
    }
    // SAFETY: the loop above wrote every index `0..coords.len()` exactly
    // once, so `out`'s first `coords.len()` elements are all initialized,
    // and that length does not exceed the capacity reserved above.
    unsafe {
        out.set_len(coords.len());
    }
    out
}

/// The region coordinate containing chunk `(cx, cz)`.
#[inline]
pub fn chunk_to_region(cx: i32, cz: i32) -> (i32, i32) {
    (cx.div_euclid(REGION_EDGE), cz.div_euclid(REGION_EDGE))
}

/// Pack a chunk coordinate into a single `u64`, suitable as a hash-map key
/// without hashing a tuple. The two `i32` halves occupy the low and high 32
/// bits respectively (native-endianness-dependent, but consistent with
/// [`unchunk_key`]).
#[inline]
pub fn chunk_key(x: i32, z: i32) -> u64 {
    // SAFETY: `[i32; 2]` and `u64` both have size 8 (and `transmute`
    // ignores the source/destination alignment of the *values* being
    // converted, only their size). Every 8-byte pattern is a valid `u64`,
    // so this reinterpretation cannot produce an invalid value; it carries
    // no meaning beyond "the bits of `[x, z]` read back as one integer".
    unsafe { std::mem::transmute::<[i32; 2], u64>([x, z]) }
}

/// Inverse of [`chunk_key`].
#[inline]
pub fn unchunk_key(key: u64) -> (i32, i32) {
    // SAFETY: `u64` and `[i32; 2]` both have size 8, and every bit pattern
    // is a valid `i32`, so reinterpreting the integer's bits as two `i32`s
    // cannot produce an invalid value. This is the exact inverse of the
    // transmute in `chunk_key`.
    let parts: [i32; 2] = unsafe { std::mem::transmute::<u64, [i32; 2]>(key) };
    (parts[0], parts[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_to_chunk_handles_negative_coordinates_correctly() {
        assert_eq!(world_to_chunk_component(0), (0, 0));
        assert_eq!(world_to_chunk_component(15), (0, 15));
        assert_eq!(world_to_chunk_component(16), (1, 0));
        assert_eq!(world_to_chunk_component(-1), (-1, 15));
        assert_eq!(world_to_chunk_component(-16), (-1, 0));
        assert_eq!(world_to_chunk_component(-17), (-2, 15));
    }

    #[test]
    fn chunk_base_block_reconstructs_original_with_local_offset() {
        for block in [-33, -17, -1, 0, 1, 15, 16, 1000] {
            let (chunk, local) = world_to_chunk_component(block);
            assert_eq!(chunk_base_block(chunk) + local as i32, block);
        }
    }

    #[test]
    fn section_index_and_local_y_handle_negative_y() {
        assert_eq!(section_index(-1), -1);
        assert_eq!(local_y(-1), 15);
        assert_eq!(section_index(-16), -1);
        assert_eq!(local_y(-16), 0);
        assert_eq!(section_index(16), 1);
        assert_eq!(local_y(16), 0);
    }

    #[test]
    fn linear_index_matches_documented_layout_and_round_trips() {
        // y * 256 + z * 16 + x
        assert_eq!(linear_index(1, 2, 3), 2 * 256 + 3 * 16 + 1);
        assert_eq!(linear_index(0, 0, 0), 0);
        assert_eq!(linear_index(15, 15, 15), 4095);
        for &(x, y, z) in &[(0, 0, 0), (15, 0, 0), (0, 15, 0), (0, 0, 15), (7, 8, 9)] {
            let idx = linear_index(x, y, z);
            assert_eq!(index_to_xyz(idx), (x, y, z));
        }
    }

    #[test]
    fn linear_index_batch_matches_scalar_version() {
        let coords: Vec<(u8, u8, u8)> = (0..16u8)
            .map(|i| (i, (i * 3) % 16, (i * 7) % 16))
            .collect();
        let batch = linear_index_batch(&coords);
        for (i, &(x, y, z)) in coords.iter().enumerate() {
            assert_eq!(batch[i] as usize, linear_index(x as usize, y as usize, z as usize));
        }
    }

    #[test]
    fn chunk_to_region_handles_negative_chunk_coordinates() {
        assert_eq!(chunk_to_region(0, 0), (0, 0));
        assert_eq!(chunk_to_region(31, 31), (0, 0));
        assert_eq!(chunk_to_region(32, 32), (1, 1));
        assert_eq!(chunk_to_region(-1, -1), (-1, -1));
        assert_eq!(chunk_to_region(-32, -32), (-1, -1));
        assert_eq!(chunk_to_region(-33, 0), (-2, 0));
    }

    #[test]
    fn chunk_key_round_trips_positive_and_negative_pairs() {
        for &(x, z) in &[(0, 0), (1, -1), (-1000, 2000), (i32::MIN, i32::MAX)] {
            let key = chunk_key(x, z);
            assert_eq!(unchunk_key(key), (x, z));
        }
    }
}
