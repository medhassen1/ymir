//! PCG32 (XSH-RR): a 64-bit-state, 32-bit-output permuted congruential
//! generator.
//!
//! `ymir` reaches for PCG32 wherever a subsystem needs many *independent,
//! named* random streams from the same 64-bit seed instead of a single
//! shared generator — for example, terrain height, cave carving, ore
//! placement, and loot rolls all want reproducible-but-uncorrelated
//! sequences from one world seed. PCG's "stream selection" mechanism
//! (a per-generator odd increment) makes that free: two generators with
//! the same seed but different streams never share a state trajectory.

/// The 64-bit LCG multiplier used by all PCG32 variants (a constant
/// chosen for its spectral properties, from O'Neill's reference PCG-C
/// implementation).
const MULTIPLIER: u64 = 6364136223846793005;

/// A small table of pre-mixed odd increments for `ymir`'s conventional
/// subsystem streams, selected via [`Pcg32::from_named_stream`]. Values
/// are arbitrary but fixed, so the same name always yields the same
/// stream across runs.
const NAMED_STREAM_SEQUENCES: [u64; 4] = [
    0x9E37_79B9_7F4A_7C15,
    0xBF58_476D_1CE4_E5B9,
    0x94D0_49BB_1331_11EB,
    0xD6E8_FEB8_6659_FD93,
];

/// Selects the terrain-height stream in [`Pcg32::from_named_stream`].
pub const STREAM_TERRAIN: u32 = 0;
/// Selects the cave/carving stream in [`Pcg32::from_named_stream`].
pub const STREAM_CAVES: u32 = 1;
/// Selects the loot-table stream in [`Pcg32::from_named_stream`].
pub const STREAM_LOOT: u32 = 2;
/// Selects the mob-spawning stream in [`Pcg32::from_named_stream`].
pub const STREAM_MOBS: u32 = 3;

/// A PCG32 generator using the XSH-RR (xorshift, random rotation) output
/// permutation: 64 bits of LCG state, 32 bits of output per step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    /// Creates a generator from a `seed` and a `stream` selector. Two
    /// generators built with the same seed but different streams produce
    /// completely different (and never-colliding) output sequences; two
    /// built with the same seed *and* stream are identical.
    pub fn new(seed: u64, stream: u64) -> Self {
        let mut rng = Pcg32 {
            state: 0,
            inc: (stream << 1) | 1,
        };
        rng.step();
        rng.state = rng.state.wrapping_add(seed);
        rng.step();
        rng
    }

    /// Creates a generator from a seed and one of `ymir`'s named streams
    /// (see the `STREAM_*` constants), so callers don't have to invent
    /// their own stream constants for common subsystems.
    pub fn from_named_stream(seed: u64, name: u32) -> Self {
        let idx = (name & 0b11) as usize;
        // SAFETY: `idx` is masked with `0b11` (3), so it is always in
        // `0..4`, matching `NAMED_STREAM_SEQUENCES`'s fixed length of 4.
        let stream = unsafe { *NAMED_STREAM_SEQUENCES.get_unchecked(idx) };
        Pcg32::new(seed, stream)
    }

    /// Advances the LCG state by one step and applies the XSH-RR output
    /// permutation to the *pre-advance* state, per the PCG reference
    /// algorithm.
    fn step(&mut self) -> u32 {
        let old_state = self.state;
        self.state = old_state.wrapping_mul(MULTIPLIER).wrapping_add(self.inc);
        let xorshifted = (((old_state >> 18) ^ old_state) >> 27) as u32;
        let rot = (old_state >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Draws the next 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        self.step()
    }

    /// Draws a uniformly distributed integer in `[0, bound)` using PCG's
    /// threshold-rejection method, which avoids modulo bias.
    pub fn next_bounded(&mut self, bound: u32) -> u32 {
        assert!(bound > 0, "bound must be positive");
        let threshold = bound.wrapping_neg() % bound;
        loop {
            let r = self.next_u32();
            if r >= threshold {
                return r % bound;
            }
        }
    }

    /// Draws a uniformly distributed `f64` in `[0, 1)` by combining two
    /// 32-bit draws into a 53-bit mantissa.
    pub fn next_f64(&mut self) -> f64 {
        let hi = self.next_u32() as u64;
        let lo = self.next_u32() as u64;
        let bits = ((hi << 21) ^ (lo >> 11)) & ((1u64 << 53) - 1);
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Fills `dst` with random bytes, four at a time from each 32-bit draw.
    pub fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut chunks = dst.chunks_exact_mut(4);
        for chunk in &mut chunks {
            let word = self.next_u32().to_le_bytes();
            // SAFETY: `chunk` has exactly 4 bytes (from `chunks_exact_mut(4)`)
            // and `word` is a local `[u8; 4]`; both are valid, properly
            // aligned (`u8` has alignment 1) regions of 4 bytes, and `word`
            // is a distinct stack allocation from `dst`'s backing storage,
            // so the copy cannot overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(word.as_ptr(), chunk.as_mut_ptr(), 4);
            }
        }
        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let word = self.next_u32().to_le_bytes();
            rem.copy_from_slice(&word[..rem.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_seed_42_stream_54() {
        // Matches the canonical `pcg32_demo` output for seed=42, seq=54.
        let mut rng = Pcg32::new(42, 54);
        let expected: [u32; 6] = [
            0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e,
        ];
        for e in expected {
            assert_eq!(rng.next_u32(), e);
        }
    }

    #[test]
    fn same_seed_different_stream_diverges() {
        let mut a = Pcg32::new(1, 1);
        let mut b = Pcg32::new(1, 2);
        let mut collisions = 0;
        for _ in 0..32 {
            if a.next_u32() == b.next_u32() {
                collisions += 1;
            }
        }
        assert_eq!(collisions, 0);
    }

    #[test]
    fn named_streams_are_pairwise_distinct() {
        let names = [STREAM_TERRAIN, STREAM_CAVES, STREAM_LOOT, STREAM_MOBS];
        let mut firsts = Vec::new();
        for &n in &names {
            let mut rng = Pcg32::from_named_stream(0xF00D, n);
            firsts.push(rng.next_u32());
        }
        for i in 0..firsts.len() {
            for j in (i + 1)..firsts.len() {
                assert_ne!(firsts[i], firsts[j], "streams {i} and {j} collided");
            }
        }
    }

    #[test]
    fn deterministic_replay() {
        let mut a = Pcg32::new(999, 11);
        let mut b = Pcg32::new(999, 11);
        for _ in 0..50 {
            assert_eq!(a.next_u32(), b.next_u32());
        }
    }

    #[test]
    fn next_bounded_stays_in_range() {
        let mut rng = Pcg32::new(3, 7);
        for _ in 0..5000 {
            let v = rng.next_bounded(13);
            assert!(v < 13);
        }
    }

    #[test]
    fn next_f64_is_in_unit_interval() {
        let mut rng = Pcg32::new(8, 8);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "value out of range: {x}");
        }
    }

    #[test]
    fn fill_bytes_matches_streamed_words() {
        let mut a = Pcg32::new(2024, 1);
        let mut b = Pcg32::new(2024, 1);
        let mut buf = [0u8; 10];
        a.fill_bytes(&mut buf);

        let mut expected = Vec::new();
        for _ in 0..2 {
            expected.extend_from_slice(&b.next_u32().to_le_bytes());
        }
        expected.extend_from_slice(&b.next_u32().to_le_bytes()[..2]);
        assert_eq!(&buf[..], &expected[..]);
    }
}
