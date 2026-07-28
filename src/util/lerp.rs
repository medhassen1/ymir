//! Scalar interpolation and easing.
//!
//! Terrain density fields, light falloff, biome blending, and camera or
//! structure placement all need to turn a handful of sample points into a
//! smooth value in between. Every voxel engine ends up rewriting `lerp`,
//! `smoothstep`, and trilinear sampling; `ymir` keeps one tested copy here
//! so density-field and lighting code never has to.

/// Linearly interpolates between `a` and `b`. `t = 0` yields `a`, `t = 1`
/// yields `b`; `t` outside `[0, 1]` extrapolates rather than clamping.
pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Inverse of [`lerp`]: given a value `v` and the same `a`/`b` endpoints,
/// returns the `t` that would reproduce it. Returns `0.0` when `a` and `b`
/// are equal (a degenerate, zero-width range) rather than dividing by
/// zero.
pub fn inverse_lerp(a: f32, b: f32, v: f32) -> f32 {
    let span = b - a;
    if span.abs() < f32::EPSILON {
        0.0
    } else {
        (v - a) / span
    }
}

/// Maps `v` from the range `[in_min, in_max]` to `[out_min, out_max]`,
/// preserving its relative position between the endpoints.
pub fn remap(v: f32, in_min: f32, in_max: f32, out_min: f32, out_max: f32) -> f32 {
    lerp(out_min, out_max, inverse_lerp(in_min, in_max, v))
}

/// Clamps `v` into `[lo, hi]`.
pub fn clamp(v: f32, lo: f32, hi: f32) -> f32 {
    v.clamp(lo, hi)
}

/// Clamps `v` into `[0, 1]`.
pub fn clamp01(v: f32) -> f32 {
    v.clamp(0.0, 1.0)
}

/// Hermite smoothstep: `0` at or before `edge0`, `1` at or after `edge1`,
/// an S-curve with zero first derivative at both ends in between. Cheaper
/// than [`smootherstep`] but visibly kinks in its second derivative, which
/// shows up as a seam when used to blend biome weights across a wide area.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp01(inverse_lerp(edge0, edge1, x));
    t * t * (3.0 - 2.0 * t)
}

/// Ken Perlin's improved smoothstep: same endpoint behavior as
/// [`smoothstep`], but with zero first *and* second derivative at both
/// ends, so chained density-field samples don't show a curvature seam.
pub fn smootherstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = clamp01(inverse_lerp(edge0, edge1, x));
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

/// Cubic Hermite interpolation between `p0` (at `t = 0`) and `p1` (at
/// `t = 1`), with explicit tangents `m0`/`m1` at each end. The building
/// block [`catmull_rom`] is defined in terms of.
pub fn cubic_hermite(p0: f32, m0: f32, p1: f32, m1: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    let h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    let h10 = t3 - 2.0 * t2 + t;
    let h01 = -2.0 * t3 + 3.0 * t2;
    let h11 = t3 - t2;
    h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1
}

/// Catmull-Rom spline through four consecutive samples, interpolating
/// between `p1` (at `t = 0`) and `p2` (at `t = 1`) using `p0`/`p3` only to
/// shape the tangents. Used to smooth a coarse noise lattice (e.g. biome
/// temperature samples) into a continuous field without the ringing a
/// higher-degree fit would introduce.
pub fn catmull_rom(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

/// Bilinear interpolation across a unit square. `corners` is ordered
/// `[v(x0,y0), v(x1,y0), v(x0,y1), v(x1,y1)]`; `tx`/`ty` are the fractional
/// position within the square along each axis.
pub fn bilinear(corners: [f32; 4], tx: f32, ty: f32) -> f32 {
    let bottom = lerp(corners[0], corners[1], tx);
    let top = lerp(corners[2], corners[3], tx);
    lerp(bottom, top, ty)
}

/// Trilinear interpolation across a unit cube. `corners` is ordered so bit
/// `0`/`1`/`2` of the index selects the `x`/`y`/`z` corner, i.e.
/// `[v000, v100, v010, v110, v001, v101, v011, v111]`; `tx`/`ty`/`tz` are
/// the fractional position within the cube along each axis. This is the
/// core operation for sampling a chunk's density field at an arbitrary
/// point instead of only at integer voxel corners.
pub fn trilinear(corners: [f32; 8], tx: f32, ty: f32, tz: f32) -> f32 {
    let c00 = lerp(corners[0], corners[1], tx);
    let c10 = lerp(corners[2], corners[3], tx);
    let c01 = lerp(corners[4], corners[5], tx);
    let c11 = lerp(corners[6], corners[7], tx);
    let c0 = lerp(c00, c10, ty);
    let c1 = lerp(c01, c11, ty);
    lerp(c0, c1, tz)
}

/// Applies [`lerp`] with a fixed `a`/`b` across many interpolation
/// parameters at once, writing directly into a pre-sized buffer instead of
/// growing a `Vec` one push at a time. Handy for resampling a whole row of
/// a density field against two known endpoint samples.
pub fn lerp_batch(a: f32, b: f32, ts: &[f32]) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::with_capacity(ts.len());
    let ptr = out.as_mut_ptr();
    for (i, &t) in ts.iter().enumerate() {
        let v = lerp(a, b, t);
        // SAFETY: `ptr` comes from `Vec::with_capacity(ts.len())`, so it
        // has room for `ts.len()` elements; `i` ranges over `0..ts.len()`
        // (the enumeration of `ts`), so `ptr.add(i)` stays within that
        // reserved capacity, and each index is written exactly once
        // before `set_len` runs below.
        unsafe {
            ptr.add(i).write(v);
        }
    }
    // SAFETY: the loop above wrote every index `0..ts.len()` exactly
    // once, so `out`'s first `ts.len()` elements are all initialized, and
    // that length does not exceed the reserved capacity.
    unsafe {
        out.set_len(ts.len());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    #[test]
    fn lerp_hits_endpoints_and_midpoint() {
        assert!((lerp(0.0, 10.0, 0.0) - 0.0).abs() < EPS);
        assert!((lerp(0.0, 10.0, 1.0) - 10.0).abs() < EPS);
        assert!((lerp(0.0, 10.0, 0.5) - 5.0).abs() < EPS);
        assert!((lerp(2.0, 2.0, 0.7) - 2.0).abs() < EPS);
    }

    #[test]
    fn inverse_lerp_and_remap_round_trip() {
        let t = inverse_lerp(10.0, 20.0, 15.0);
        assert!((t - 0.5).abs() < EPS);
        assert!((inverse_lerp(5.0, 5.0, 5.0) - 0.0).abs() < EPS);

        let remapped = remap(15.0, 10.0, 20.0, 0.0, 100.0);
        assert!((remapped - 50.0).abs() < EPS);
    }

    #[test]
    fn clamp_functions_bound_their_range() {
        assert!((clamp(-5.0, 0.0, 1.0) - 0.0).abs() < EPS);
        assert!((clamp(5.0, 0.0, 1.0) - 1.0).abs() < EPS);
        assert!((clamp01(0.3) - 0.3).abs() < EPS);
        assert!((clamp01(-1.0) - 0.0).abs() < EPS);
    }

    #[test]
    fn smoothstep_and_smootherstep_match_at_endpoints_and_midpoint() {
        assert!((smoothstep(0.0, 1.0, 0.0) - 0.0).abs() < EPS);
        assert!((smoothstep(0.0, 1.0, 1.0) - 1.0).abs() < EPS);
        assert!((smoothstep(0.0, 1.0, 0.5) - 0.5).abs() < EPS);

        assert!((smootherstep(0.0, 1.0, 0.0) - 0.0).abs() < EPS);
        assert!((smootherstep(0.0, 1.0, 1.0) - 1.0).abs() < EPS);
        assert!((smootherstep(0.0, 1.0, 0.5) - 0.5).abs() < EPS);

        // Below the lower edge is clamped flat, above the upper edge too.
        assert!((smoothstep(0.0, 1.0, -5.0) - 0.0).abs() < EPS);
        assert!((smootherstep(0.0, 1.0, 5.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn catmull_rom_and_hermite_pass_through_their_control_points() {
        assert!((catmull_rom(0.0, 3.0, 7.0, 10.0, 0.0) - 3.0).abs() < EPS);
        assert!((catmull_rom(0.0, 3.0, 7.0, 10.0, 1.0) - 7.0).abs() < EPS);

        assert!((cubic_hermite(2.0, 0.0, 9.0, 0.0, 0.0) - 2.0).abs() < EPS);
        assert!((cubic_hermite(2.0, 0.0, 9.0, 0.0, 1.0) - 9.0).abs() < EPS);
    }

    #[test]
    fn bilinear_and_trilinear_reduce_to_corner_values() {
        let square = [1.0, 2.0, 3.0, 4.0];
        assert!((bilinear(square, 0.0, 0.0) - 1.0).abs() < EPS);
        assert!((bilinear(square, 1.0, 0.0) - 2.0).abs() < EPS);
        assert!((bilinear(square, 0.0, 1.0) - 3.0).abs() < EPS);
        assert!((bilinear(square, 1.0, 1.0) - 4.0).abs() < EPS);

        let cube = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0];
        let corners_t = [
            (0.0, 0.0, 0.0, 10.0),
            (1.0, 0.0, 0.0, 20.0),
            (0.0, 1.0, 0.0, 30.0),
            (1.0, 1.0, 0.0, 40.0),
            (0.0, 0.0, 1.0, 50.0),
            (1.0, 0.0, 1.0, 60.0),
            (0.0, 1.0, 1.0, 70.0),
            (1.0, 1.0, 1.0, 80.0),
        ];
        for (tx, ty, tz, expected) in corners_t {
            assert!((trilinear(cube, tx, ty, tz) - expected).abs() < EPS);
        }
        // Center of the cube is the plain average of all eight corners.
        let center = trilinear(cube, 0.5, 0.5, 0.5);
        let mean: f32 = cube.iter().sum::<f32>() / 8.0;
        assert!((center - mean).abs() < EPS);
    }

    #[test]
    fn lerp_batch_matches_scalar_lerp() {
        let ts: Vec<f32> = (0..=20).map(|i| i as f32 / 20.0).collect();
        let batch = lerp_batch(-3.0, 8.0, &ts);
        for (i, &t) in ts.iter().enumerate() {
            assert!((batch[i] - lerp(-3.0, 8.0, t)).abs() < EPS);
        }
    }
}
