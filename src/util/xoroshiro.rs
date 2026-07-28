//! xoroshiro128++: a small, fast, statistically strong 64-bit generator.
//!
//! `ymir` uses this as the default per-world (or per-region) generator:
//! it has a longer period and better statistical quality than SplitMix64
//! and, unlike SplitMix64, supports [`jump`](Xoroshiro128PlusPlus::jump)
//! and [`long_jump`](Xoroshiro128PlusPlus::long_jump), which advance the
//! stream by 2^64 or 2^96 steps respectively. That makes it possible to
//! carve out non-overlapping sub-streams for parallel chunk generation
//! from a single seeded generator, rather than hashing coordinates for
//! every draw.

/// Rotates `x` left by `k` bits (`k` in `1..64`).
#[inline]
fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

/// One step of SplitMix64, used only to expand a single seed into the two
/// 64-bit words of xoroshiro's state. Kept private and self-contained
/// rather than depending on [`crate::util::splitmix64`], since each
/// `ymir::util` module should stand on its own.
fn splitmix64_step(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// The jump polynomial equivalent to 2^64 calls to [`Xoroshiro128PlusPlus::next_u64`].
const JUMP: [u64; 2] = [0x2bd7_a6a6_e99c_2ddc, 0x0992_ccaf_6a6f_ca05];

/// The jump polynomial equivalent to 2^96 calls to [`Xoroshiro128PlusPlus::next_u64`].
const LONG_JUMP: [u64; 2] = [0x360f_d5f2_cf8d_5d99, 0x9c6e_6877_736c_46e3];

/// A xoroshiro128++ generator: 128 bits of state, period 2^128 - 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Xoroshiro128PlusPlus {
    s0: u64,
    s1: u64,
}

impl Xoroshiro128PlusPlus {
    /// Seeds a generator from a single `u64` by expanding it through two
    /// steps of SplitMix64, as recommended by the algorithm's authors. Any
    /// seed, including zero, yields a valid non-degenerate state (the
    /// all-zero state is the one state xoroshiro must never enter, and
    /// SplitMix64 expansion makes that vanishingly unlikely for any input).
    pub fn new(seed: u64) -> Self {
        let mut state = seed;
        let s0 = splitmix64_step(&mut state);
        let s1 = splitmix64_step(&mut state);
        Xoroshiro128PlusPlus { s0, s1 }
    }

    /// Builds a generator directly from a raw 128-bit state. Both words
    /// being zero is invalid (it is the fixed point of the update
    /// function); such a state is replaced with a fallback so the
    /// generator always produces output.
    pub fn from_state(s0: u64, s1: u64) -> Self {
        if s0 == 0 && s1 == 0 {
            Xoroshiro128PlusPlus { s0: 1, s1: 0 }
        } else {
            Xoroshiro128PlusPlus { s0, s1 }
        }
    }

    /// Draws the next 64-bit output and advances the state.
    pub fn next_u64(&mut self) -> u64 {
        let s0 = self.s0;
        let mut s1 = self.s1;
        let result = rotl(s0.wrapping_add(s1), 17).wrapping_add(s0);

        s1 ^= s0;
        self.s0 = rotl(s0, 49) ^ s1 ^ (s1 << 21);
        self.s1 = rotl(s1, 28);

        result
    }

    /// Draws the next 32-bit output from the high half of a 64-bit draw.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Draws a uniformly distributed `f64` in `[0, 1)`.
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Draws a uniformly distributed integer in `[0, bound)` using
    /// Lemire's rejection method (no modulo bias).
    pub fn next_bounded(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "bound must be positive");
        let mut product = (self.next_u64() as u128) * (bound as u128);
        let mut low = product as u64;
        if low < bound {
            let threshold = bound.wrapping_neg() % bound;
            while low < threshold {
                product = (self.next_u64() as u128) * (bound as u128);
                low = product as u64;
            }
        }
        (product >> 64) as u64
    }

    /// Advances the state as if [`next_u64`](Self::next_u64) had been
    /// called 2^64 times, without materializing every intermediate draw.
    /// Used to derive an independent, non-overlapping stream for another
    /// subsystem from the same base generator.
    pub fn jump(&mut self) {
        self.apply_jump_polynomial(JUMP);
    }

    /// Advances the state as if `next_u64` had been called 2^96 times.
    /// Combined with `jump`, this lets a world seed be partitioned into a
    /// grid of far-apart streams (e.g. one long-jump per region, one jump
    /// per chunk within a region).
    pub fn long_jump(&mut self) {
        self.apply_jump_polynomial(LONG_JUMP);
    }

    /// Shared implementation for `jump` and `long_jump`: walks the bits of
    /// a jump polynomial, XORing the pre-jump state into an accumulator
    /// wherever a bit is set, while stepping the generator once per bit.
    fn apply_jump_polynomial(&mut self, table: [u64; 2]) {
        let mut acc0 = 0u64;
        let mut acc1 = 0u64;
        for i in 0..table.len() {
            // SAFETY: `i` is produced by `0..table.len()`, so it is always
            // a valid index into `table` by construction.
            let word = unsafe { *table.get_unchecked(i) };
            for b in 0..64u32 {
                if word & (1u64 << b) != 0 {
                    acc0 ^= self.s0;
                    acc1 ^= self.s1;
                }
                self.next_u64();
            }
        }
        self.s0 = acc0;
        self.s1 = acc1;
    }

    /// Fills `dst` with random bytes, eight at a time from each 64-bit
    /// draw, avoiding a separate byte-by-byte loop for buffer seeding.
    pub fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut chunks = dst.chunks_exact_mut(8);
        for chunk in &mut chunks {
            let word = self.next_u64().to_le_bytes();
            // SAFETY: `chunk` has exactly 8 bytes (from `chunks_exact_mut(8)`)
            // and `word` is a local `[u8; 8]`; both are valid for 8 bytes of
            // `u8` (alignment 1), and since `word` is a distinct stack
            // allocation from `dst`'s backing storage, the two never overlap.
            unsafe {
                std::ptr::copy_nonoverlapping(word.as_ptr(), chunk.as_mut_ptr(), 8);
            }
        }
        let rem = chunks.into_remainder();
        if !rem.is_empty() {
            let word = self.next_u64().to_le_bytes();
            rem.copy_from_slice(&word[..rem.len()]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_seed_12345() {
        let mut rng = Xoroshiro128PlusPlus::new(12345);
        let expected: [u64; 5] = [
            0xe08ec422beebbea0,
            0xc5454d3ad5892bf0,
            0x5223964c36832da0,
            0x8ea7792a1152a13a,
            0x2a085815e39fccff,
        ];
        for e in expected {
            assert_eq!(rng.next_u64(), e);
        }
    }

    #[test]
    fn jump_matches_known_state_and_outputs() {
        let mut rng = Xoroshiro128PlusPlus::new(12345);
        rng.jump();
        assert_eq!(rng.s0, 0x91b1bb7b28cee9ab);
        assert_eq!(rng.s1, 0x1e58033e93f41b84);
        let expected: [u64; 3] = [0x0f253501332e49be, 0xab40fecf069a032d, 0x2cb4c0cc2a2cbdab];
        for e in expected {
            assert_eq!(rng.next_u64(), e);
        }
    }

    #[test]
    fn long_jump_matches_known_state_and_outputs() {
        let mut rng = Xoroshiro128PlusPlus::new(12345);
        rng.long_jump();
        assert_eq!(rng.s0, 0x95a2c696353d2df6);
        assert_eq!(rng.s1, 0xfcdc437e56ccf59b);
        let expected: [u64; 3] = [0xa9cbdeaa7c6052f4, 0x6bdb44c21867ddd4, 0xf99c4c6830ef4282];
        for e in expected {
            assert_eq!(rng.next_u64(), e);
        }
    }

    #[test]
    fn jump_produces_a_different_stream_than_no_jump() {
        let base = Xoroshiro128PlusPlus::new(9001);
        let mut unjumped = base.clone();
        let mut jumped = base.clone();
        jumped.jump();
        let mut collisions = 0;
        for _ in 0..16 {
            if unjumped.next_u64() == jumped.next_u64() {
                collisions += 1;
            }
        }
        assert!(collisions == 0, "jumped and un-jumped streams collided");
    }

    #[test]
    fn all_zero_seed_state_is_rejected() {
        let mut rng = Xoroshiro128PlusPlus::from_state(0, 0);
        // Must not stay stuck at the degenerate all-zero fixed point.
        assert_ne!((rng.s0, rng.s1), (0, 0));
        let _ = rng.next_u64();
        assert_ne!((rng.s0, rng.s1), (0, 0));
    }

    #[test]
    fn next_bounded_stays_in_range() {
        let mut rng = Xoroshiro128PlusPlus::new(4242);
        for _ in 0..2000 {
            let v = rng.next_bounded(37);
            assert!(v < 37);
        }
    }

    #[test]
    fn fill_bytes_matches_streamed_words() {
        let mut a = Xoroshiro128PlusPlus::new(777);
        let mut b = Xoroshiro128PlusPlus::new(777);
        let mut buf = [0u8; 17];
        a.fill_bytes(&mut buf);

        let mut expected = Vec::new();
        for _ in 0..2 {
            expected.extend_from_slice(&b.next_u64().to_le_bytes());
        }
        expected.extend_from_slice(&b.next_u64().to_le_bytes()[..1]);
        assert_eq!(&buf[..], &expected[..]);
    }
}
