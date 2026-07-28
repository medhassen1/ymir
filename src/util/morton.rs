//! 3D and 2D Morton (Z-order) codes: bit-interleaved space-filling-curve
//! indices that keep spatially nearby voxels numerically close together.
//!
//! `ymir` uses Morton order to key sparse chunk/section maps and to walk
//! voxel neighborhoods with good cache locality: two coordinates that are
//! close in space almost always produce codes that are close in value,
//! a guarantee a plain row-major index cannot give across every axis at
//! once.

use crate::util::ivec3::IVec3;

/// Bias added to a signed axis before encoding it into the unsigned,
/// 21-bit-per-axis Morton space, and subtracted back out on decode. Centers
/// the representable range on the origin: `-BIAS_21 ..= BIAS_21 - 1`.
pub const BIAS_21: i32 = 1 << 20;

/// Mask keeping only the low 21 bits of a coordinate, the per-axis budget
/// for [`encode3_64`]/[`decode3_64`].
const LOW21: u64 = (1 << 21) - 1;

/// Mask keeping only the low 10 bits of a coordinate, the per-axis budget
/// for [`encode3_32`]/[`decode3_32`].
const LOW10: u32 = (1 << 10) - 1;

/// Spreads a 21-bit value so each original bit lands 3 apart (bit `i` moves
/// to bit `3*i`), leaving room for two more interleaved axes. Classic
/// "magic number" bit-spreading: a fixed sequence of shift-or-mask steps,
/// no branches or loops.
fn split_by_3_bits_64(a: u32) -> u64 {
    let mut x = a as u64 & LOW21;
    x = (x | (x << 32)) & 0x1f00000000ffff;
    x = (x | (x << 16)) & 0x1f0000ff0000ff;
    x = (x | (x << 8)) & 0x100f00f00f00f00f;
    x = (x | (x << 4)) & 0x10c30c30c30c30c3;
    x = (x | (x << 2)) & 0x1249249249249249;
    x
}

/// Inverse of [`split_by_3_bits_64`]: gathers every third bit back into a
/// contiguous 21-bit value.
fn compact_by_3_bits_64(x: u64) -> u32 {
    let mut x = x & 0x1249249249249249;
    x = (x | (x >> 2)) & 0x10c30c30c30c30c3;
    x = (x | (x >> 4)) & 0x100f00f00f00f00f;
    x = (x | (x >> 8)) & 0x1f0000ff0000ff;
    x = (x | (x >> 16)) & 0x1f00000000ffff;
    x = (x | (x >> 32)) & LOW21;
    x as u32
}

/// Encodes three axis values into a 63-bit 3D Morton code, using the low 21
/// bits of each (higher bits are silently discarded).
pub fn encode3_64(x: u32, y: u32, z: u32) -> u64 {
    split_by_3_bits_64(x) | (split_by_3_bits_64(y) << 1) | (split_by_3_bits_64(z) << 2)
}

/// Decodes a 63-bit 3D Morton code back into its three 21-bit axis values.
pub fn decode3_64(code: u64) -> (u32, u32, u32) {
    (
        compact_by_3_bits_64(code),
        compact_by_3_bits_64(code >> 1),
        compact_by_3_bits_64(code >> 2),
    )
}

/// Spreads a 10-bit value so each original bit lands 3 apart, the 32-bit
/// analogue of [`split_by_3_bits_64`] for callers that only need a small
/// region's worth of range and want to stay in a `u32`.
fn split_by_3_bits_32(a: u32) -> u32 {
    let mut x = a & LOW10;
    x = (x | (x << 16)) & 0xff0000ff;
    x = (x | (x << 8)) & 0x0300f00f;
    x = (x | (x << 4)) & 0x030c30c3;
    x = (x | (x << 2)) & 0x09249249;
    x
}

/// Inverse of [`split_by_3_bits_32`].
fn compact_by_3_bits_32(x: u32) -> u32 {
    let mut x = x & 0x09249249;
    x = (x | (x >> 2)) & 0x030c30c3;
    x = (x | (x >> 4)) & 0x0300f00f;
    x = (x | (x >> 8)) & 0xff0000ff;
    x = (x | (x >> 16)) & LOW10;
    x
}

/// Encodes three axis values into a 30-bit 3D Morton code, using the low 10
/// bits of each.
pub fn encode3_32(x: u32, y: u32, z: u32) -> u32 {
    split_by_3_bits_32(x) | (split_by_3_bits_32(y) << 1) | (split_by_3_bits_32(z) << 2)
}

/// Decodes a 30-bit 3D Morton code back into its three 10-bit axis values.
pub fn decode3_32(code: u32) -> (u32, u32, u32) {
    (
        compact_by_3_bits_32(code),
        compact_by_3_bits_32(code >> 1),
        compact_by_3_bits_32(code >> 2),
    )
}

/// Spreads a full 32-bit value one bit apart, the 2D analogue used by
/// [`encode2_64`].
fn split_by_1_bit_64(a: u32) -> u64 {
    let mut x = a as u64;
    x = (x | (x << 16)) & 0x0000ffff0000ffff;
    x = (x | (x << 8)) & 0x00ff00ff00ff00ff;
    x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0f;
    x = (x | (x << 2)) & 0x3333333333333333;
    x = (x | (x << 1)) & 0x5555555555555555;
    x
}

/// Inverse of [`split_by_1_bit_64`].
fn compact_by_1_bit_64(x: u64) -> u32 {
    let mut x = x & 0x5555555555555555;
    x = (x | (x >> 1)) & 0x3333333333333333;
    x = (x | (x >> 2)) & 0x0f0f0f0f0f0f0f0f;
    x = (x | (x >> 4)) & 0x00ff00ff00ff00ff;
    x = (x | (x >> 8)) & 0x0000ffff0000ffff;
    x = (x | (x >> 16)) & 0x00000000ffffffff;
    x as u32
}

/// Encodes two full 32-bit axis values into a 64-bit 2D Morton code, for
/// region-grid or heightmap indexing.
pub fn encode2_64(x: u32, y: u32) -> u64 {
    split_by_1_bit_64(x) | (split_by_1_bit_64(y) << 1)
}

/// Decodes a 64-bit 2D Morton code back into its two axis values.
pub fn decode2_64(code: u64) -> (u32, u32) {
    (compact_by_1_bit_64(code), compact_by_1_bit_64(code >> 1))
}

/// Spreads a `u16` axis value one bit apart via magic-number shifts (the
/// reference implementation [`encode2_32_lut`] is checked against).
fn split_by_1_bit_32(a: u16) -> u32 {
    let mut x = a as u32;
    x = (x | (x << 8)) & 0x00ff00ff;
    x = (x | (x << 4)) & 0x0f0f0f0f;
    x = (x | (x << 2)) & 0x33333333;
    x = (x | (x << 1)) & 0x55555555;
    x
}

/// Inverse of [`split_by_1_bit_32`].
fn compact_by_1_bit_32(x: u32) -> u16 {
    let mut x = x & 0x55555555;
    x = (x | (x >> 1)) & 0x33333333;
    x = (x | (x >> 2)) & 0x0f0f0f0f;
    x = (x | (x >> 4)) & 0x00ff00ff;
    x = (x | (x >> 8)) & 0x0000ffff;
    x as u16
}

/// Encodes two 16-bit axis values into a 32-bit 2D Morton code, the small
/// range a region's local chunk grid actually needs.
pub fn encode2_32(x: u16, y: u16) -> u32 {
    split_by_1_bit_32(x) | (split_by_1_bit_32(y) << 1)
}

/// Decodes a 32-bit 2D Morton code back into its two 16-bit axis values.
pub fn decode2_32(code: u32) -> (u16, u16) {
    (compact_by_1_bit_32(code), compact_by_1_bit_32(code >> 1))
}

/// Precomputed at compile time: `SPREAD_LUT[b]` spreads the 8 bits of `b`
/// one bit apart (bit `i` of `b` becomes bit `2*i` of the result). Table
/// lookup alternative to [`split_by_1_bit_32`]'s shift chain, used by
/// [`encode2_32_lut`].
const fn build_spread_lut() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut b = 0usize;
    while b < 256 {
        let mut result = 0u32;
        let mut bit = 0;
        while bit < 8 {
            if (b >> bit) & 1 != 0 {
                result |= 1 << (bit * 2);
            }
            bit += 1;
        }
        table[b] = result;
        b += 1;
    }
    table
}

/// The 256-entry byte bit-spreading table used by [`encode2_32_lut`].
const SPREAD_LUT: [u32; 256] = build_spread_lut();

/// Spreads a `u16` axis value into a 32-bit value with one zero bit between
/// each original bit, via two table lookups instead of five shift/mask
/// pairs.
fn spread16_lut(v: u16) -> u32 {
    let lo = (v as usize) & 0xff;
    let hi = (v as usize >> 8) & 0xff;
    // SAFETY: `lo` is masked with `0xff`, so it is always in `0..256`,
    // matching `SPREAD_LUT`'s fixed length of 256.
    let lo_spread = unsafe { *SPREAD_LUT.get_unchecked(lo) };
    // SAFETY: `hi` is masked with `0xff`, so it is always in `0..256`,
    // matching `SPREAD_LUT`'s fixed length of 256.
    let hi_spread = unsafe { *SPREAD_LUT.get_unchecked(hi) };
    lo_spread | (hi_spread << 16)
}

/// Table-driven equivalent of [`encode2_32`]. Produces identical output;
/// provided for callers whose hot loop would rather pay one cache-resident
/// lookup per byte than a chain of shifts.
pub fn encode2_32_lut(x: u16, y: u16) -> u32 {
    spread16_lut(x) | (spread16_lut(y) << 1)
}

/// Encodes an [`IVec3`] world coordinate into a 63-bit Morton code, biasing
/// each axis by [`BIAS_21`] so both negative and positive coordinates in
/// `-BIAS_21 ..= BIAS_21 - 1` map to distinct, non-negative Morton input.
pub fn encode_ivec3(v: IVec3) -> u64 {
    encode3_64((v.x + BIAS_21) as u32, (v.y + BIAS_21) as u32, (v.z + BIAS_21) as u32)
}

/// Inverse of [`encode_ivec3`].
pub fn decode_ivec3(code: u64) -> IVec3 {
    let (x, y, z) = decode3_64(code);
    IVec3::new(x as i32 - BIAS_21, y as i32 - BIAS_21, z as i32 - BIAS_21)
}

/// Encodes many `(x, y, z)` coordinate triples into 64-bit 3D Morton codes
/// in one pass, writing directly into a pre-sized buffer instead of
/// growing a `Vec` one push at a time.
pub fn encode3_64_batch(coords: &[(u32, u32, u32)]) -> Vec<u64> {
    let mut out: Vec<u64> = Vec::with_capacity(coords.len());
    let ptr = out.as_mut_ptr();
    for (i, &(x, y, z)) in coords.iter().enumerate() {
        let code = encode3_64(x, y, z);
        // SAFETY: `ptr` comes from `Vec::with_capacity(coords.len())`, so
        // it has room for `coords.len()` elements; `i` ranges over
        // `0..coords.len()` (the enumeration of `coords`), so `ptr.add(i)`
        // stays within that reserved capacity, and each index is written
        // exactly once before `set_len` runs below.
        unsafe {
            ptr.add(i).write(code);
        }
    }
    // SAFETY: the loop above wrote every index `0..coords.len()` exactly
    // once, so `out`'s first `coords.len()` elements are all initialized,
    // and that length does not exceed the reserved capacity.
    unsafe {
        out.set_len(coords.len());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode3_64_round_trips_across_the_full_single_axis_range() {
        for x in 0..(1u32 << 21) {
            let code = encode3_64(x, 0, 0);
            assert_eq!(decode3_64(code), (x, 0, 0));
        }
        for y in (0..(1u32 << 21)).step_by(7) {
            let code = encode3_64(0, y, 0);
            assert_eq!(decode3_64(code), (0, y, 0));
        }
        for z in (0..(1u32 << 21)).step_by(7) {
            let code = encode3_64(0, 0, z);
            assert_eq!(decode3_64(code), (0, 0, z));
        }
    }

    #[test]
    fn encode3_32_round_trips_across_the_full_10_bit_range() {
        for x in 0..(1u32 << 10) {
            for y in 0..(1u32 << 10) {
                let code = encode3_32(x, y, 5);
                assert_eq!(decode3_32(code), (x, y, 5));
            }
        }
    }

    #[test]
    fn encode3_64_produces_the_expected_low_bit_pattern() {
        assert_eq!(encode3_64(1, 0, 0), 1);
        assert_eq!(encode3_64(0, 1, 0), 2);
        assert_eq!(encode3_64(0, 0, 1), 4);
        assert_eq!(encode3_64(1, 1, 1), 7);
        assert_eq!(encode3_64(2, 0, 0), 8);
    }

    #[test]
    fn encode2_64_round_trips_a_sample_of_the_full_32_bit_range() {
        let samples = [0u32, 1, 2, 12345, u32::MAX / 3, u32::MAX - 1, u32::MAX];
        for &x in &samples {
            for &y in &samples {
                let code = encode2_64(x, y);
                assert_eq!(decode2_64(code), (x, y));
            }
        }
    }

    #[test]
    fn lut_variant_matches_the_magic_number_variant_and_round_trips() {
        for x in (0..u16::MAX).step_by(41) {
            for y in (0..u16::MAX).step_by(97) {
                let magic = encode2_32(x, y);
                let lut = encode2_32_lut(x, y);
                assert_eq!(magic, lut);
                assert_eq!(decode2_32(lut), (x, y));
            }
        }
    }

    #[test]
    fn ivec3_convenience_round_trips_negative_and_positive_coordinates() {
        for &v in &[
            IVec3::new(0, 0, 0),
            IVec3::new(-1, -1, -1),
            IVec3::new(1000, -2000, 3000),
            IVec3::new(-BIAS_21, -BIAS_21, -BIAS_21),
            IVec3::new(BIAS_21 - 1, BIAS_21 - 1, BIAS_21 - 1),
        ] {
            let code = encode_ivec3(v);
            assert_eq!(decode_ivec3(code), v);
        }
    }

    #[test]
    fn batch_encoding_matches_scalar_encoding() {
        let coords: Vec<(u32, u32, u32)> =
            (0..500u32).map(|i| (i, i.wrapping_mul(3) % 2000, i.wrapping_mul(7) % 2000)).collect();
        let batch = encode3_64_batch(&coords);
        for (i, &(x, y, z)) in coords.iter().enumerate() {
            assert_eq!(batch[i], encode3_64(x, y, z));
        }
    }
}
