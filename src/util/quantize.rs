//! Vertex attribute quantization for the mesher's GPU-bound output.
//!
//! Packing positions, UVs, and unit floats into small integer codes shrinks
//! vertex buffers several-fold versus plain `f32`: unorm16 positions, unorm
//! UVs, and snorm normals/weights, each with its dequantize inverse.

use std::mem::MaybeUninit;

/// Blocks per chunk-section edge; the default position quantization extent.
pub const SECTION_EDGE: usize = 16;

/// Quantize `x` (expected in `[0, extent]`) to an unsigned 16-bit fixed-point
/// code spanning the full `u16` range. Values outside the range are clamped.
#[inline]
pub fn quantize_unorm16(x: f32, extent: f32) -> u16 {
    debug_assert!(extent > 0.0);
    let t = (x / extent).clamp(0.0, 1.0);
    (t * u16::MAX as f32).round() as u16
}

/// Inverse of [`quantize_unorm16`]: recover an approximate `x` in `[0, extent]`.
#[inline]
pub fn dequantize_unorm16(q: u16, extent: f32) -> f32 {
    (q as f32 / u16::MAX as f32) * extent
}

/// Quantize `x` in `[0, 1]` (a texture coordinate) to an 8-bit unorm code.
#[inline]
pub fn quantize_unorm8(x: f32) -> u8 {
    let t = x.clamp(0.0, 1.0);
    (t * u8::MAX as f32).round() as u8
}

/// Inverse of [`quantize_unorm8`].
#[inline]
pub fn dequantize_unorm8(q: u8) -> f32 {
    q as f32 / u8::MAX as f32
}

/// Quantize `x` in `[0, 1]` to a 16-bit unorm code (higher-precision UVs).
#[inline]
pub fn quantize_unorm16_uv(x: f32) -> u16 {
    quantize_unorm16(x, 1.0)
}

/// Inverse of [`quantize_unorm16_uv`].
#[inline]
pub fn dequantize_unorm16_uv(q: u16) -> f32 {
    dequantize_unorm16(q, 1.0)
}

/// Quantize `x` in `[-1, 1]` (normals, tangent components) to an 8-bit
/// signed-normalized code.
#[inline]
pub fn quantize_snorm8(x: f32) -> i8 {
    let t = x.clamp(-1.0, 1.0);
    (t * i8::MAX as f32).round() as i8
}

/// Inverse of [`quantize_snorm8`].
#[inline]
pub fn dequantize_snorm8(q: i8) -> f32 {
    (q as f32 / i8::MAX as f32).clamp(-1.0, 1.0)
}

/// Quantize `x` in `[-1, 1]` to a 16-bit signed-normalized code.
#[inline]
pub fn quantize_snorm16(x: f32) -> i16 {
    let t = x.clamp(-1.0, 1.0);
    (t * i16::MAX as f32).round() as i16
}

/// Inverse of [`quantize_snorm16`].
#[inline]
pub fn dequantize_snorm16(q: i16) -> f32 {
    (q as f32 / i16::MAX as f32).clamp(-1.0, 1.0)
}

/// A chunk-local vertex position packed into three unorm16 lanes. Laid out
/// with no padding so a batch of these can be viewed directly as a byte
/// buffer for upload.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizedPosition {
    /// Quantized X coordinate.
    pub x: u16,
    /// Quantized Y coordinate.
    pub y: u16,
    /// Quantized Z coordinate.
    pub z: u16,
}

impl QuantizedPosition {
    /// Quantize a world-space-local position over `[0, extent]` per axis.
    pub fn quantize(pos: [f32; 3], extent: f32) -> Self {
        QuantizedPosition {
            x: quantize_unorm16(pos[0], extent),
            y: quantize_unorm16(pos[1], extent),
            z: quantize_unorm16(pos[2], extent),
        }
    }

    /// Recover the approximate original position.
    pub fn dequantize(self, extent: f32) -> [f32; 3] {
        [
            dequantize_unorm16(self.x, extent),
            dequantize_unorm16(self.y, extent),
            dequantize_unorm16(self.z, extent),
        ]
    }
}

/// Quantize a batch of positions, one call per element (a plain, safe loop —
/// see [`quantize_triangle`] for the fixed-arity fast path used per-triangle).
pub fn quantize_batch(positions: &[[f32; 3]], extent: f32) -> Vec<QuantizedPosition> {
    positions
        .iter()
        .map(|p| QuantizedPosition::quantize(*p, extent))
        .collect()
}

/// Quantize the three corner positions of one mesher triangle at once. This
/// is the hot path: greedy meshing emits geometry triangle-by-triangle, and
/// building the fixed-size result in place avoids a default value or a
/// throwaway heap allocation.
pub fn quantize_triangle(tri: [[f32; 3]; 3], extent: f32) -> [QuantizedPosition; 3] {
    let mut out: MaybeUninit<[QuantizedPosition; 3]> = MaybeUninit::uninit();
    let base = out.as_mut_ptr() as *mut QuantizedPosition;
    for (i, p) in tri.iter().enumerate() {
        // SAFETY: `base` points at the start of an array of 3
        // `QuantizedPosition` slots carved out of `out`'s own storage, and
        // `i` ranges over `0..3` (the length of `tri`), so `base.add(i)`
        // stays within that array. Each slot is written exactly once before
        // `assume_init` is called below, and none is read before its write.
        unsafe { base.add(i).write(QuantizedPosition::quantize(*p, extent)) };
    }
    // SAFETY: the loop above wrote all 3 elements (indices 0, 1, 2) of the
    // array through `base`, which points into `out`'s storage, so every byte
    // of `out` is now initialized and `assume_init` is valid.
    unsafe { out.assume_init() }
}

/// View a slice of packed positions as raw little/native-endian bytes,
/// suitable for a vertex buffer upload without a copy.
pub fn positions_as_bytes(positions: &[QuantizedPosition]) -> &[u8] {
    let len_bytes = std::mem::size_of_val(positions);
    // SAFETY: `QuantizedPosition` is `repr(C)` with three `u16` fields and no
    // padding (size 6, align 2), so every byte of every element is a valid,
    // initialized `u8`. The source slice is a valid, live borrow of
    // `positions`, so the pointer is non-null and the `len_bytes`-byte range
    // stays within the allocation for the lifetime of the returned slice,
    // which borrows from `positions`. `u8` has alignment 1, so no alignment
    // requirement is violated.
    unsafe { std::slice::from_raw_parts(positions.as_ptr() as *const u8, len_bytes) }
}

/// Pack three snorm8 components into a byte triple. `i8` and `u8` share size
/// and alignment and every bit pattern of one is a valid instance of the
/// other, so this is a pure reinterpretation with no value transformation
/// beyond the two's-complement bit pattern itself.
pub fn snorm8_triple_to_bytes(v: [i8; 3]) -> [u8; 3] {
    // SAFETY: `[i8; 3]` and `[u8; 3]` have identical size (3) and alignment
    // (1), and both types accept every possible bit pattern as a valid
    // value, so reinterpreting one array as the other cannot produce an
    // invalid bit pattern for the target type.
    unsafe { std::mem::transmute::<[i8; 3], [u8; 3]>(v) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unorm16_round_trip_within_error_bound() {
        let extent = SECTION_EDGE as f32;
        let max_err = extent / u16::MAX as f32;
        for i in 0..=20 {
            let x = extent * (i as f32 / 20.0);
            let q = quantize_unorm16(x, extent);
            let back = dequantize_unorm16(q, extent);
            assert!((back - x).abs() <= max_err + 1e-4, "x={x} back={back}");
        }
    }

    #[test]
    fn unorm16_clamps_out_of_range() {
        assert_eq!(quantize_unorm16(-5.0, 16.0), 0);
        assert_eq!(quantize_unorm16(100.0, 16.0), u16::MAX);
    }

    #[test]
    fn unorm8_uv_round_trip_within_error_bound() {
        let max_err = 1.0 / u8::MAX as f32;
        for i in 0..=10 {
            let u = i as f32 / 10.0;
            let q = quantize_unorm8(u);
            let back = dequantize_unorm8(q);
            assert!((back - u).abs() <= max_err + 1e-6);
        }
    }

    #[test]
    fn snorm8_endpoints_are_exact_sign() {
        assert_eq!(quantize_snorm8(1.0), i8::MAX);
        assert_eq!(quantize_snorm8(-1.0), -i8::MAX);
        assert!(dequantize_snorm8(i8::MAX) > 0.99);
        assert!(dequantize_snorm8(-i8::MAX) < -0.99);
    }

    #[test]
    fn snorm16_round_trip_within_error_bound() {
        let max_err = 1.0 / i16::MAX as f32;
        for i in -10..=10 {
            let x = i as f32 / 10.0;
            let q = quantize_snorm16(x);
            let back = dequantize_snorm16(q);
            assert!((back - x).abs() <= max_err + 1e-4);
        }
    }

    #[test]
    fn quantized_position_round_trip_batch_and_triangle_agree() {
        let extent = SECTION_EDGE as f32;
        let pts = [[0.0, 0.0, 0.0], [8.0, 15.999, 3.5], [16.0, 16.0, 16.0]];
        let batch = quantize_batch(&pts, extent);
        let tri = quantize_triangle(pts, extent);
        for i in 0..3 {
            let scalar = QuantizedPosition::quantize(pts[i], extent);
            assert_eq!(batch[i], scalar);
            assert_eq!(tri[i], scalar);
            let back = batch[i].dequantize(extent);
            for k in 0..3 {
                assert!((back[k] - pts[i][k]).abs() < 0.01);
            }
        }
    }

    #[test]
    fn byte_reinterpretation_is_exact() {
        let pts = [QuantizedPosition { x: 1, y: 2, z: 3 }];
        let bytes = positions_as_bytes(&pts);
        assert_eq!(bytes.len(), 6);
        assert_eq!(&bytes[0..2], &1u16.to_ne_bytes());
        assert_eq!(&bytes[2..4], &2u16.to_ne_bytes());
        assert_eq!(&bytes[4..6], &3u16.to_ne_bytes());

        let signed = [-1i8, 0, 127];
        assert_eq!(snorm8_triple_to_bytes(signed), [0xFFu8, 0x00, 0x7F]);
    }
}
