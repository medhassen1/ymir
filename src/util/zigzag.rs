//! Zigzag mapping between signed and unsigned integers. Voxel deltas (light
//! changes, coordinate offsets) are signed but usually small; zigzag folds
//! negatives next to their positive counterparts so a downstream varint or
//! bit-packed encoder sees small magnitudes instead of numbers near the
//! unsigned type's maximum.

use std::slice;

/// Maps a signed `i16` to its zigzag-encoded `u16` representation.
#[inline]
pub fn encode_i16(v: i16) -> u16 {
    ((v << 1) ^ (v >> 15)) as u16
}

/// Inverse of [`encode_i16`].
#[inline]
pub fn decode_i16(z: u16) -> i16 {
    ((z >> 1) as i16) ^ -((z & 1) as i16)
}

/// Maps a signed `i32` to its zigzag-encoded `u32` representation.
#[inline]
pub fn encode_i32(v: i32) -> u32 {
    ((v << 1) ^ (v >> 31)) as u32
}

/// Inverse of [`encode_i32`].
#[inline]
pub fn decode_i32(z: u32) -> i32 {
    ((z >> 1) as i32) ^ -((z & 1) as i32)
}

/// Maps a signed `i64` to its zigzag-encoded `u64` representation.
#[inline]
pub fn encode_i64(v: i64) -> u64 {
    ((v << 1) ^ (v >> 63)) as u64
}

/// Inverse of [`encode_i64`].
#[inline]
pub fn decode_i64(z: u64) -> i64 {
    ((z >> 1) as i64) ^ -((z & 1) as i64)
}

/// Zigzag-encodes every element of `src` into `dst`, which must have the
/// same length as `src`.
///
/// # Panics
/// Panics if `src.len() != dst.len()`.
pub fn encode_slice_i32(src: &[i32], dst: &mut [u32]) {
    assert_eq!(src.len(), dst.len(), "zigzag: length mismatch");
    let len = src.len();
    for i in 0..len {
        // SAFETY: `i` ranges over `0..len`, and both `src` and `dst` were
        // just asserted to have length `len`, so both indexed accesses are
        // in bounds for the entire loop.
        unsafe {
            *dst.get_unchecked_mut(i) = encode_i32(*src.get_unchecked(i));
        }
    }
}

/// Zigzag-decodes every element of `src` into `dst`, which must have the
/// same length as `src`.
///
/// # Panics
/// Panics if `src.len() != dst.len()`.
pub fn decode_slice_i32(src: &[u32], dst: &mut [i32]) {
    assert_eq!(src.len(), dst.len(), "zigzag: length mismatch");
    let len = src.len();
    for i in 0..len {
        // SAFETY: identical reasoning to `encode_slice_i32`: `i < len` and
        // both slices have length `len`, confirmed by the assertion above.
        unsafe {
            *dst.get_unchecked_mut(i) = decode_i32(*src.get_unchecked(i));
        }
    }
}

/// Returns a zero-copy view of a zigzag-encoded `u64` buffer reinterpreted
/// as raw `i64` bit patterns (i.e. *not* zigzag-decoded, just the same bits
/// viewed through the other type). Useful when a lower layer wants to treat
/// an already-encoded buffer as opaque signed storage without copying.
pub fn view_as_i64_bits(buf: &[u64]) -> &[i64] {
    // SAFETY: `u64` and `i64` have identical size and alignment, and every
    // 64-bit pattern is a valid value of both types, so reinterpreting the
    // same memory region with the same length and lifetime as `buf` cannot
    // produce an invalid value or read past the original allocation.
    unsafe { slice::from_raw_parts(buf.as_ptr() as *const i64, buf.len()) }
}

/// Zigzag-encodes a whole slice into a freshly allocated vector.
pub fn encode_vec_i64(src: &[i64]) -> Vec<u64> {
    src.iter().map(|&v| encode_i64(v)).collect()
}

/// Zigzag-decodes a whole slice into a freshly allocated vector.
pub fn decode_vec_i64(src: &[u64]) -> Vec<i64> {
    src.iter().map(|&z| decode_i64(z)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_i16_boundaries() {
        for v in [0i16, 1, -1, i16::MIN, i16::MAX, 100, -100] {
            assert_eq!(decode_i16(encode_i16(v)), v);
        }
    }

    #[test]
    fn round_trip_i32_boundaries() {
        for v in [0i32, 1, -1, i32::MIN, i32::MAX, 12345, -12345] {
            assert_eq!(decode_i32(encode_i32(v)), v);
        }
    }

    #[test]
    fn round_trip_i64_boundaries() {
        for v in [0i64, 1, -1, i64::MIN, i64::MAX, 1 << 40, -(1 << 40)] {
            assert_eq!(decode_i64(encode_i64(v)), v);
        }
    }

    #[test]
    fn zigzag_ordering_is_small_magnitude_first() {
        // Zigzag should interleave: 0,-1,1,-2,2,-3,3,...
        assert_eq!(encode_i32(0), 0);
        assert_eq!(encode_i32(-1), 1);
        assert_eq!(encode_i32(1), 2);
        assert_eq!(encode_i32(-2), 3);
        assert_eq!(encode_i32(2), 4);
    }

    #[test]
    fn slice_helpers_round_trip() {
        let src: Vec<i32> = (-50..50).collect();
        let mut encoded = vec![0u32; src.len()];
        encode_slice_i32(&src, &mut encoded);
        let mut decoded = vec![0i32; src.len()];
        decode_slice_i32(&encoded, &mut decoded);
        assert_eq!(decoded, src);
    }

    #[test]
    fn vec_helpers_round_trip_empty_and_nonempty() {
        assert!(encode_vec_i64(&[]).is_empty());
        let src = vec![0i64, -1, i64::MIN, i64::MAX, 42];
        let encoded = encode_vec_i64(&src);
        let decoded = decode_vec_i64(&encoded);
        assert_eq!(decoded, src);
    }

    #[test]
    fn view_as_i64_bits_preserves_bit_pattern() {
        let buf: Vec<u64> = vec![0, 1, u64::MAX, 1 << 63];
        let view = view_as_i64_bits(&buf);
        assert_eq!(view.len(), buf.len());
        for (a, b) in buf.iter().zip(view.iter()) {
            assert_eq!(*a, *b as u64);
        }
    }
}
