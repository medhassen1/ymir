//! Fractal Brownian motion (fBm) and its ridged/billow variants.
//!
//! A single octave of Perlin/simplex/value noise looks too regular to
//! pass as real terrain. `ymir` layers several octaves of a caller-chosen
//! noise function at increasing frequency and decreasing amplitude to get
//! natural-looking height fields; the ridged and billow variants reshape
//! each octave before accumulating it, which is what produces sharp
//! mountain ridges or rolling, cloud-like billows instead of plain hills.
//! These are free functions rather than a struct because they carry no
//! state of their own — all state lives in the noise closure the caller
//! supplies (typically a [`crate::util::perlin::PerlinNoise`] or
//! [`crate::util::simplex::SimplexNoise`] sampler).

use std::mem::MaybeUninit;

/// The largest octave count these functions will precompute weights for;
/// requests above this are clamped rather than allocating.
const MAX_OCTAVES: usize = 32;

/// Precomputes the per-octave amplitude weights `[1, gain, gain^2, ...]`
/// for `octaves` (clamped to [`MAX_OCTAVES`]) and returns them alongside
/// their sum, which is used to normalize the accumulated result back into
/// roughly `[-1, 1]`.
fn octave_weights(octaves: u32, gain: f64) -> ([f64; MAX_OCTAVES], usize, f64) {
    let n = (octaves as usize).min(MAX_OCTAVES);
    // SAFETY: an array of `MaybeUninit<f64>` has no validity requirement
    // on its bytes, so leaving it momentarily uninitialized here (before
    // the loop below writes every slot) is defined behavior.
    let mut weights: [MaybeUninit<f64>; MAX_OCTAVES] = unsafe { MaybeUninit::uninit().assume_init() };
    let mut amp = 1.0;
    let mut total = 0.0;
    for (i, slot) in weights.iter_mut().enumerate() {
        let w = if i < n { amp } else { 0.0 };
        *slot = MaybeUninit::new(w);
        if i < n {
            total += amp;
            amp *= gain;
        }
    }
    // SAFETY: the loop above wrote every one of the `MAX_OCTAVES` slots
    // via `MaybeUninit::new`, so the whole array is initialized;
    // `[MaybeUninit<f64>; MAX_OCTAVES]` and `[f64; MAX_OCTAVES]` share
    // size and alignment, so this transmute is defined behavior.
    let weights =
        unsafe { std::mem::transmute::<[MaybeUninit<f64>; MAX_OCTAVES], [f64; MAX_OCTAVES]>(weights) };
    (weights, n, total)
}

/// Reads the `i`-th precomputed octave weight. Every call site below only
/// ever passes an `i` taken from a `0..n` loop, where `n <= MAX_OCTAVES`
/// is the clamp `octave_weights` already applied, so `i` is always a
/// valid index into `weights`.
#[inline]
fn weight_at(weights: &[f64; MAX_OCTAVES], i: usize) -> f64 {
    // SAFETY: see the doc comment above — `i < n <= MAX_OCTAVES ==
    // weights.len()` at every call site.
    unsafe { *weights.get_unchecked(i) }
}

/// Accumulates `octaves` layers of `noise(x * freq, y * freq)`, doubling
/// (or `lacunarity`-multiplying) the frequency and scaling the amplitude
/// by `gain` each octave, then normalizes by the total amplitude so a
/// noise function bounded in `[-1, 1]` stays (nominally) in `[-1, 1]`.
pub fn fbm2<F: Fn(f64, f64) -> f64>(noise: F, x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let (weights, n, total) = octave_weights(octaves, gain);
    if total == 0.0 {
        return 0.0;
    }
    let mut freq = 1.0;
    let mut sum = 0.0;
    for i in 0..n {
        sum += weight_at(&weights, i) * noise(x * freq, y * freq);
        freq *= lacunarity;
    }
    sum / total
}

/// The 3D counterpart of [`fbm2`].
pub fn fbm3<F: Fn(f64, f64, f64) -> f64>(
    noise: F,
    x: f64,
    y: f64,
    z: f64,
    octaves: u32,
    lacunarity: f64,
    gain: f64,
) -> f64 {
    let (weights, n, total) = octave_weights(octaves, gain);
    if total == 0.0 {
        return 0.0;
    }
    let mut freq = 1.0;
    let mut sum = 0.0;
    for i in 0..n {
        sum += weight_at(&weights, i) * noise(x * freq, y * freq, z * freq);
        freq *= lacunarity;
    }
    sum / total
}

/// Billowy fBm: each octave contributes `2 * |noise| - 1` instead of the
/// raw noise value, which folds troughs upward into puffy, cloud-like
/// mounds rather than smooth valleys.
pub fn billow2<F: Fn(f64, f64) -> f64>(noise: F, x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let (weights, n, total) = octave_weights(octaves, gain);
    if total == 0.0 {
        return 0.0;
    }
    let mut freq = 1.0;
    let mut sum = 0.0;
    for i in 0..n {
        let billowed = 2.0 * noise(x * freq, y * freq).abs() - 1.0;
        sum += weight_at(&weights, i) * billowed;
        freq *= lacunarity;
    }
    sum / total
}

/// Ridged multifractal noise: each octave contributes `(1 - |noise|)^2`,
/// weighted by the previous octave's contribution, which sharpens ridges
/// and lets high-frequency detail show through only near existing ridges.
/// Unlike [`fbm2`]/[`billow2`], the result is not amplitude-normalized to
/// `[-1, 1]` (ridged multifractal is conventionally left unnormalized,
/// typically landing in `[0, ~1]`); scale it to taste at the call site.
pub fn ridged2<F: Fn(f64, f64) -> f64>(noise: F, x: f64, y: f64, octaves: u32, lacunarity: f64, gain: f64) -> f64 {
    let mut freq = 1.0;
    let mut amp = 0.5;
    let mut prev = 1.0;
    let mut sum = 0.0;
    for _ in 0..octaves.min(MAX_OCTAVES as u32) {
        let n = noise(x * freq, y * freq);
        let signal = 1.0 - n.abs();
        let signal = signal * signal * prev;
        prev = signal;
        sum += signal * amp;
        freq *= lacunarity;
        amp *= gain;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbm2_of_a_constant_function_equals_that_constant() {
        // Every octave contributes `weight * 1.0`; dividing by the total
        // weight recovers exactly 1.0, independent of octave count.
        let v = fbm2(|_, _| 1.0, 3.0, -4.0, 5, 2.0, 0.5);
        assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {v}");
    }

    #[test]
    fn fbm3_of_zero_is_zero() {
        let v = fbm3(|_, _, _| 0.0, 1.0, 2.0, 3.0, 4, 2.0, 0.5);
        assert_eq!(v, 0.0);
    }

    #[test]
    fn zero_octaves_yields_zero() {
        assert_eq!(fbm2(|_, _| 1.0, 0.0, 0.0, 0, 2.0, 0.5), 0.0);
        assert_eq!(billow2(|_, _| 1.0, 0.0, 0.0, 0, 2.0, 0.5), 0.0);
    }

    #[test]
    fn billow_of_a_constant_function_equals_that_constant() {
        // 2*|1.0| - 1 == 1.0 for every octave, so the normalized sum is 1.0.
        let v = billow2(|_, _| 1.0, 0.0, 0.0, 6, 2.0, 0.5);
        assert!((v - 1.0).abs() < 1e-12, "expected 1.0, got {v}");
    }

    #[test]
    fn ridged_single_octave_matches_hand_computed_value() {
        // octaves=1: signal = (1 - |0|)^2 * prev(1.0) = 1.0, amp starts at
        // 0.5, so sum = 1.0 * 0.5 = 0.5 exactly.
        let v = ridged2(|_, _| 0.0, 0.0, 0.0, 1, 2.0, 0.5);
        assert!((v - 0.5).abs() < 1e-12, "expected 0.5, got {v}");
    }

    #[test]
    fn is_deterministic_for_the_same_closure_and_inputs() {
        let f = |x: f64, y: f64| (x * 12.9898 + y * 78.233).sin();
        let a = fbm2(f, 1.23, -4.56, 6, 2.1, 0.45);
        let b = fbm2(f, 1.23, -4.56, 6, 2.1, 0.45);
        assert_eq!(a, b);
    }

    #[test]
    fn more_octaves_still_normalizes_to_a_constant_input() {
        for octaves in [1u32, 8, 32, 40] {
            let v = fbm2(|_, _| 1.0, 5.0, 5.0, octaves, 2.0, 0.5);
            assert!((v - 1.0).abs() < 1e-9, "octaves={octaves} gave {v}");
        }
    }
}
