//! Biome classification and neighbourhood tint blending.
//!
//! A hard biome boundary produces a visible seam in grass/foliage color, so
//! this module holds the biome registry (temperature, humidity, tint),
//! nearest-biome classification, and a 3x3 distance-weighted tint blend.

/// Biome cells per chunk edge (biomes are `BIOME_EDGE^3` blocks wide).
pub const BIOME_EDGE: usize = 4;

/// A single biome's classification point and base render tint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiomeRecord {
    /// Stable name, used by world-gen configuration and debug tooling.
    pub name: &'static str,
    /// Mean annual temperature, arbitrary units, roughly `[-1, 1]`.
    pub temperature: f32,
    /// Humidity, roughly `[0, 1]`.
    pub humidity: f32,
    /// Linear-light tint applied to grass/foliage/water in this biome.
    pub tint: [f32; 3],
}

/// The built-in biome registry. Real world-gen would load this from
/// configuration; a fixed table is enough for classification and blending.
pub const BIOMES: [BiomeRecord; 6] = [
    BiomeRecord { name: "tundra", temperature: -0.8, humidity: 0.2, tint: [0.6, 0.7, 0.6] },
    BiomeRecord { name: "taiga", temperature: -0.3, humidity: 0.5, tint: [0.4, 0.6, 0.5] },
    BiomeRecord { name: "plains", temperature: 0.3, humidity: 0.4, tint: [0.5, 0.8, 0.3] },
    BiomeRecord { name: "forest", temperature: 0.2, humidity: 0.7, tint: [0.3, 0.6, 0.25] },
    BiomeRecord { name: "desert", temperature: 0.9, humidity: 0.1, tint: [0.9, 0.75, 0.4] },
    BiomeRecord { name: "swamp", temperature: 0.4, humidity: 0.9, tint: [0.35, 0.45, 0.3] },
];

fn squared_distance(temperature: f32, humidity: f32, biome: &BiomeRecord) -> f32 {
    let dt = temperature - biome.temperature;
    let dh = humidity - biome.humidity;
    dt * dt + dh * dh
}

/// Classify a `(temperature, humidity)` point to the index of its nearest
/// biome in [`BIOMES`] (squared Euclidean distance in temperature/humidity
/// space). [`BIOMES`] is non-empty, so this always returns a valid index.
pub fn classify(temperature: f32, humidity: f32) -> usize {
    let mut best_idx = 0usize;
    let mut best_dist = f32::INFINITY;
    for (i, biome) in BIOMES.iter().enumerate() {
        let d = squared_distance(temperature, humidity, biome);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx
}

/// Look up the base tint of biome `idx`. Panics if `idx >= BIOMES.len()`.
pub fn tint_of(idx: usize) -> [f32; 3] {
    assert!(idx < BIOMES.len(), "biome index out of range");
    // SAFETY: the assert above guarantees `idx < BIOMES.len()`.
    unsafe { BIOMES.get_unchecked(idx).tint }
}

/// Classify a point directly to its nearest biome's tint, combining
/// [`classify`] and [`tint_of`].
pub fn classify_tint(temperature: f32, humidity: f32) -> [f32; 3] {
    tint_of(classify(temperature, humidity))
}

/// Distance-based weights for a 3x3 neighbourhood centered on the target
/// cell (index 4, row-major `dz in -1..=1, dx in -1..=1`), using an inverse
/// linear falloff: the center cell gets weight 1, orthogonal neighbours
/// `1 / (1 + 1)`, and diagonal neighbours `1 / (1 + sqrt(2))`.
pub fn default_weights_3x3() -> [f32; 9] {
    let mut weights = [0.0f32; 9];
    let mut i = 0;
    for dz in -1..=1i32 {
        for dx in -1..=1i32 {
            let dist = ((dx * dx + dz * dz) as f32).sqrt();
            weights[i] = 1.0 / (1.0 + dist);
            i += 1;
        }
    }
    weights
}

/// Blend 9 neighbourhood tints (row-major, center at index 4) using 9
/// matching weights, normalizing by the weight sum so the result stays a
/// valid color even if the caller's weights don't already sum to 1.
pub fn blend_tint_3x3(neighbors: &[[f32; 3]; 9], weights: &[f32; 9]) -> [f32; 3] {
    let mut sum = [0.0f32; 3];
    let mut weight_total = 0.0f32;
    for i in 0..9 {
        // SAFETY: `i` is a loop counter ranging over the literal `0..9`,
        // and both `neighbors` and `weights` are arrays of exactly 9
        // elements, so `i` is always a valid index into either.
        let (tint, w) = unsafe { (*neighbors.get_unchecked(i), *weights.get_unchecked(i)) };
        sum[0] += tint[0] * w;
        sum[1] += tint[1] * w;
        sum[2] += tint[2] * w;
        weight_total += w;
    }
    if weight_total <= 0.0 {
        return neighbors[4];
    }
    [sum[0] / weight_total, sum[1] / weight_total, sum[2] / weight_total]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_returns_exact_match_for_each_biomes_own_point() {
        for (i, biome) in BIOMES.iter().enumerate() {
            assert_eq!(classify(biome.temperature, biome.humidity), i);
        }
    }

    #[test]
    fn classify_picks_nearer_of_two_close_biomes() {
        // Plains (0.3, 0.4) is closer than forest (0.2, 0.7) to this point.
        let idx = classify(0.32, 0.42);
        assert_eq!(BIOMES[idx].name, "plains");
    }

    #[test]
    fn tint_of_matches_registry_and_panics_out_of_range() {
        assert_eq!(tint_of(0), BIOMES[0].tint);
        assert_eq!(tint_of(BIOMES.len() - 1), BIOMES[BIOMES.len() - 1].tint);
        let result = std::panic::catch_unwind(|| tint_of(BIOMES.len()));
        assert!(result.is_err());
    }

    #[test]
    fn classify_tint_matches_classify_then_tint_of() {
        let t = classify_tint(-0.8, 0.2);
        assert_eq!(t, BIOMES[0].tint);
    }

    #[test]
    fn default_weights_center_is_heaviest_and_symmetric() {
        let w = default_weights_3x3();
        assert_eq!(w[4], 1.0);
        // Orthogonal neighbours (indices 1, 3, 5, 7) all equal.
        assert!((w[1] - w[3]).abs() < 1e-6);
        assert!((w[3] - w[5]).abs() < 1e-6);
        assert!((w[5] - w[7]).abs() < 1e-6);
        // Corners (0, 2, 6, 8) all equal and lighter than the orthogonal ring.
        assert!((w[0] - w[2]).abs() < 1e-6);
        assert!(w[0] < w[1]);
    }

    #[test]
    fn uniform_tint_neighbourhood_blends_to_itself_regardless_of_weights() {
        let neighbors = [[0.4, 0.5, 0.6]; 9];
        let weights = default_weights_3x3();
        let blended = blend_tint_3x3(&neighbors, &weights);
        for i in 0..3 {
            assert!((blended[i] - neighbors[0][i]).abs() < 1e-5);
        }
    }

    #[test]
    fn equal_weights_average_two_distinct_tints_evenly() {
        let mut neighbors = [[0.0f32, 0.0, 0.0]; 9];
        neighbors[0] = [1.0, 0.0, 0.0];
        neighbors[1] = [0.0, 1.0, 0.0];
        let weights = [1.0f32; 9];
        let blended = blend_tint_3x3(&neighbors, &weights);
        assert!((blended[0] - 1.0 / 9.0).abs() < 1e-5);
        assert!((blended[1] - 1.0 / 9.0).abs() < 1e-5);
    }
}
