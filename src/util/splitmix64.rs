//! SplitMix64: a small, fast 64-bit pseudo-random generator.
//!
//! A voxel world needs many *independent* random streams derived from one
//! world seed: terrain noise, cave carving, structure placement, loot
//! tables, mob spawns. SplitMix64 is the standard way to expand a single
//! `u64` seed into as many well-mixed sub-seeds as needed (it is what
//! [`crate::util::xoroshiro`] uses internally to build its own state), and
//! it is also useful on its own wherever a cheap, deterministic generator
//! is enough.

use std::mem::MaybeUninit;

/// The golden-ratio increment recommended by Vigna's reference
/// implementation; any odd increment works, but this one has good
/// avalanche behavior across all seeds, including zero.
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;

/// A SplitMix64 generator. Eight bytes of state, no side tables, and no
/// jump function (it is not designed for stream splitting the way
/// xoroshiro is) — but it is the workhorse for turning one seed into many.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Creates a generator from an arbitrary seed. Every seed, including
    /// zero, produces a well-mixed output stream.
    pub fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }

    /// Returns the current raw state without advancing the generator.
    /// Useful for saving/restoring a stream (e.g. per-chunk seeds).
    pub fn state(&self) -> u64 {
        self.state
    }

    /// Draws the next 64-bit output and advances the state.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Draws the next 32-bit output. Uses the high half of a 64-bit draw,
    /// since the upper bits of SplitMix64's output mix slightly better than
    /// the lower ones.
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Draws a uniformly distributed `f64` in `[0, 1)`, using the top 53
    /// bits of a 64-bit draw (the width of an `f64` mantissa).
    pub fn next_f64(&mut self) -> f64 {
        let bits = self.next_u64() >> 11;
        bits as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Draws a uniformly distributed integer in `[0, bound)` with Lemire's
    /// rejection method, which avoids the modulo bias that a plain
    /// `next_u64() % bound` would introduce for bounds that don't divide
    /// 2^64 evenly.
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

    /// Fills `dst` with random bytes, drawing one 64-bit word per 8 bytes.
    /// This is the byte-oriented entry point used to seed hash keys or
    /// other fixed-size arrays without a separate per-byte loop.
    pub fn fill_bytes(&mut self, dst: &mut [u8]) {
        let mut chunks = dst.chunks_exact_mut(8);
        for chunk in &mut chunks {
            let word = self.next_u64().to_le_bytes();
            // SAFETY: `chunk` is exactly 8 bytes long, guaranteed by
            // `chunks_exact_mut(8)`; `word` is a local `[u8; 8]`, so both
            // the source and destination are valid for 8 bytes and, since
            // `word` lives on this function's stack frame while `chunk`
            // borrows from `dst`, the two regions cannot overlap.
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

    /// Draws `N` independent 64-bit values at once, e.g. to build a small
    /// seed table for a permutation array in one call.
    pub fn next_array<const N: usize>(&mut self) -> [u64; N] {
        // SAFETY: an array of `MaybeUninit<u64>` never needs its elements
        // to be initialized (unlike `[u64; N]`), because `MaybeUninit`
        // itself has no validity requirement on its bytes. This is the
        // standard pattern for building an array element-by-element
        // without a placeholder default value.
        let mut buf: [MaybeUninit<u64>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        for slot in buf.iter_mut() {
            *slot = MaybeUninit::new(self.next_u64());
        }
        // SAFETY: the loop above just wrote every one of the `N` slots via
        // `MaybeUninit::new`, so all `N` `u64`s are now initialized.
        // `[MaybeUninit<u64>; N]` and `[u64; N]` have identical size and
        // alignment (`MaybeUninit<T>` is layout-compatible with `T`), so
        // transmuting the fully initialized array is defined behavior.
        unsafe { std::mem::transmute_copy::<[MaybeUninit<u64>; N], [u64; N]>(&buf) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_seed_zero() {
        // Reference vectors from Vigna's public-domain splitmix64.c,
        // seeded with state = 0.
        let mut rng = SplitMix64::new(0);
        let expected: [u64; 5] = [
            0xe220a8397b1dcdaf,
            0x6e789e6aa1b965f4,
            0x06c45d188009454f,
            0xf88bb8a8724c81ec,
            0x1b39896a51a8749b,
        ];
        for e in expected {
            assert_eq!(rng.next_u64(), e);
        }
    }

    #[test]
    fn known_answer_seed_42() {
        let mut rng = SplitMix64::new(42);
        let expected: [u64; 3] = [0xbdd732262feb6e95, 0x28efe333b266f103, 0x47526757130f9f52];
        for e in expected {
            assert_eq!(rng.next_u64(), e);
        }
    }

    #[test]
    fn deterministic_replay() {
        let mut a = SplitMix64::new(0x00C0_FFEE);
        let mut b = SplitMix64::new(0x00C0_FFEE);
        for _ in 0..64 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn next_f64_is_in_unit_interval() {
        let mut rng = SplitMix64::new(7);
        for _ in 0..1000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "value out of range: {x}");
        }
    }

    #[test]
    fn next_bounded_stays_in_range_and_hits_zero_and_nonzero() {
        let mut rng = SplitMix64::new(99);
        let mut saw_zero = false;
        let mut saw_nonzero = false;
        for _ in 0..2000 {
            let v = rng.next_bounded(10);
            assert!(v < 10);
            if v == 0 {
                saw_zero = true;
            } else {
                saw_nonzero = true;
            }
        }
        assert!(saw_zero && saw_nonzero);
    }

    #[test]
    fn fill_bytes_matches_streamed_words() {
        let mut a = SplitMix64::new(1234);
        let mut b = SplitMix64::new(1234);
        let mut buf = [0u8; 20];
        a.fill_bytes(&mut buf);

        let mut expected = Vec::new();
        for _ in 0..2 {
            expected.extend_from_slice(&b.next_u64().to_le_bytes());
        }
        expected.extend_from_slice(&b.next_u64().to_le_bytes()[..4]);
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn next_array_matches_sequential_draws() {
        let mut a = SplitMix64::new(55);
        let mut b = SplitMix64::new(55);
        let arr: [u64; 4] = a.next_array();
        for v in arr {
            assert_eq!(v, b.next_u64());
        }
    }
}
