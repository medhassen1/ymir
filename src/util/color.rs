//! Color packing, gamma conversion, HSV, and biome tinting.
//!
//! Textures are authored in sRGB but blending is only correct in linear
//! light, so this module provides sRGB<->linear conversion, RGB<->HSV, a
//! packed `u32` vertex color, and the tint blend used to recolor by biome.

use std::sync::OnceLock;

/// Pack four 8-bit channels into a single `u32`, byte order `[r, g, b, a]`.
pub fn pack_rgba8(r: u8, g: u8, b: u8, a: u8) -> u32 {
    u32::from_ne_bytes([r, g, b, a])
}

/// Inverse of [`pack_rgba8`]: split a packed color back into its channels.
pub fn unpack_rgba8(packed: u32) -> (u8, u8, u8, u8) {
    let bytes = packed.to_ne_bytes();
    (bytes[0], bytes[1], bytes[2], bytes[3])
}

/// View a slice of already-packed RGBA8 colors as raw bytes (4 per color,
/// native-endian), for handing a vertex-color buffer to an upload routine
/// without a copy.
pub fn rgba_slice_as_bytes(colors: &[u32]) -> &[u8] {
    let len_bytes = std::mem::size_of_val(colors);
    // SAFETY: `colors` is a valid, live slice of `u32`, so `colors.as_ptr()`
    // is non-null and points at `len_bytes` initialized bytes (every bit
    // pattern of a `u32` is four valid, initialized `u8`s), and the
    // returned slice borrows from `colors` so it cannot outlive the
    // allocation it points into. `u8` has alignment 1, so the pointer
    // (already aligned for `u32`, a stricter requirement) needs no
    // realignment.
    unsafe { std::slice::from_raw_parts(colors.as_ptr() as *const u8, len_bytes) }
}

fn srgb_channel_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Lazily built 256-entry sRGB-encoded-byte -> linear-float lookup table.
/// Building it once and indexing it avoids a `powf` call per channel per
/// vertex during chunk mesh upload.
static SRGB_TO_LINEAR_TABLE: OnceLock<[f32; 256]> = OnceLock::new();

fn srgb_table() -> &'static [f32; 256] {
    SRGB_TO_LINEAR_TABLE.get_or_init(|| std::array::from_fn(|i| srgb_channel_to_linear(i as f32 / 255.0)))
}

/// Convert one sRGB-encoded byte channel to a linear-light float in `[0, 1]`.
pub fn srgb8_to_linear(v: u8) -> f32 {
    let table = srgb_table();
    // SAFETY: `v` is a `u8`, so `v as usize` is always in `0..=255`, and
    // `table` has exactly 256 elements — the index is always in bounds.
    unsafe { *table.get_unchecked(v as usize) }
}

/// Convert one linear-light float in `[0, 1]` to an sRGB-encoded byte.
pub fn linear_to_srgb8(c: f32) -> u8 {
    let c = c.clamp(0.0, 1.0);
    let encoded = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

/// Convert a linear RGB triple to HSV (`h` in `[0, 360)`, `s`/`v` in `[0,1]`).
pub fn rgb_to_hsv(rgb: [f32; 3]) -> [f32; 3] {
    let (r, g, b) = (rgb[0], rgb[1], rgb[2]);
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let h = if delta.abs() < 1e-8 {
        0.0
    } else if max == r {
        60.0 * (((g - b) / delta).rem_euclid(6.0))
    } else if max == g {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };
    let s = if max <= 1e-8 { 0.0 } else { delta / max };
    [h, s, max]
}

/// Convert an HSV triple back to linear RGB.
pub fn hsv_to_rgb(hsv: [f32; 3]) -> [f32; 3] {
    let (h, s, v) = (hsv[0].rem_euclid(360.0), hsv[1].clamp(0.0, 1.0), hsv[2]);
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
    let m = v - c;
    let (r1, g1, b1) = match (h / 60.0) as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    [r1 + m, g1 + m, b1 + m]
}

/// Linearly interpolate two linear-light colors. `t` is not clamped, so
/// callers doing extrapolation (e.g. contrast boosts) can pass values
/// outside `[0, 1]` deliberately.
pub fn lerp_linear(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

/// Blend a base linear color with a biome tint color at the given strength.
/// At `strength = 0` the base is unchanged; at `strength = 1` the base is
/// fully multiplied by the tint (the standard grass/leaves/water recolor
/// used by voxel renderers, where the tint comes from a biome color map).
pub fn tint(base: [f32; 3], tint_color: [f32; 3], strength: f32) -> [f32; 3] {
    let s = strength.clamp(0.0, 1.0);
    let tinted = [base[0] * tint_color[0], base[1] * tint_color[1], base[2] * tint_color[2]];
    lerp_linear(base, tinted, s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rgba_pack_unpack_round_trip() {
        let packed = pack_rgba8(10, 20, 30, 255);
        assert_eq!(unpack_rgba8(packed), (10, 20, 30, 255));
        let packed2 = pack_rgba8(0, 0, 0, 0);
        assert_eq!(unpack_rgba8(packed2), (0, 0, 0, 0));

        let colors = [packed, packed2];
        let bytes = rgba_slice_as_bytes(&colors);
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[0..4], &packed.to_ne_bytes());
        assert_eq!(&bytes[4..8], &packed2.to_ne_bytes());
    }

    #[test]
    fn srgb_table_endpoints_and_monotonic() {
        assert_eq!(srgb8_to_linear(0), 0.0);
        assert!((srgb8_to_linear(255) - 1.0).abs() < 1e-5);
        let mut prev = -1.0;
        for v in 0..=255u8 {
            let cur = srgb8_to_linear(v);
            assert!(cur >= prev, "table not monotonic at {v}");
            prev = cur;
        }
    }

    #[test]
    fn srgb_round_trip_within_one_byte() {
        for v in [0u8, 1, 17, 64, 128, 200, 254, 255] {
            let lin = srgb8_to_linear(v);
            let back = linear_to_srgb8(lin);
            assert!((back as i32 - v as i32).abs() <= 1, "v={v} back={back}");
        }
    }

    #[test]
    fn hsv_round_trip_on_primaries() {
        let colors = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.2, 0.4, 0.6],
        ];
        for c in colors {
            let hsv = rgb_to_hsv(c);
            let back = hsv_to_rgb(hsv);
            for i in 0..3 {
                assert!((back[i] - c[i]).abs() < 1e-4, "c={c:?} back={back:?}");
            }
        }
    }

    #[test]
    fn lerp_linear_hits_endpoints_and_midpoint() {
        let a = [0.0, 0.0, 0.0];
        let b = [1.0, 1.0, 1.0];
        assert_eq!(lerp_linear(a, b, 0.0), a);
        assert_eq!(lerp_linear(a, b, 1.0), b);
        assert_eq!(lerp_linear(a, b, 0.5), [0.5, 0.5, 0.5]);
    }

    #[test]
    fn tint_at_zero_is_identity_at_one_is_multiplicative() {
        let base = [0.8, 0.6, 0.4];
        let tint_color = [0.5, 1.0, 0.0];
        assert_eq!(tint(base, tint_color, 0.0), base);
        let full = tint(base, tint_color, 1.0);
        assert!((full[0] - 0.4).abs() < 1e-6);
        assert!((full[1] - 0.6).abs() < 1e-6);
        assert!((full[2] - 0.0).abs() < 1e-6);
    }
}
