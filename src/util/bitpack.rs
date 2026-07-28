//! Fixed-width bit packing into `u64` words. A chunk section's block
//! palette rarely needs a full byte per voxel: with 16 distinct block
//! types, 4 bits per index suffices. This packs and unpacks arbitrary-width
//! (1..=32 bit) unsigned fields into a dense `u64` word array.

use std::mem::MaybeUninit;

const WORD_BITS: u32 = 64;

/// Returns the number of `u64` words needed to hold `count` fields of
/// `bits` width each (1..=32).
///
/// # Panics
/// Panics if `bits` is 0 or greater than 32.
pub fn packed_len(count: usize, bits: u32) -> usize {
    assert!((1..=32).contains(&bits), "bitpack: bits out of range");
    let total_bits = count as u64 * bits as u64;
    total_bits.div_ceil(WORD_BITS as u64) as usize
}

/// Appends fixed-width fields to a growable `u64` word buffer, one field at
/// a time, packing them tightly across word boundaries.
pub struct BitWriter {
    words: Vec<u64>,
    bit_pos: u64,
}

impl BitWriter {
    /// Creates an empty writer.
    pub fn new() -> Self {
        BitWriter { words: Vec::new(), bit_pos: 0 }
    }

    /// Writes the low `bits` bits of `value` (higher bits are ignored) as
    /// the next field.
    ///
    /// # Panics
    /// Panics if `bits` is 0 or greater than 32.
    pub fn write_bits(&mut self, value: u32, bits: u32) {
        assert!((1..=32).contains(&bits), "bitpack: bits out of range");
        let mask = if bits == 32 { u32::MAX as u64 } else { (1u64 << bits) - 1 };
        let mut v = (value as u64) & mask;
        let mut remaining = bits;
        while remaining > 0 {
            let word_idx = (self.bit_pos / WORD_BITS as u64) as usize;
            let bit_off = (self.bit_pos % WORD_BITS as u64) as u32;
            if word_idx >= self.words.len() {
                self.words.push(0);
            }
            let space = WORD_BITS - bit_off;
            let take = remaining.min(space);
            let chunk = v & ((1u64 << take) - 1);
            // SAFETY: `word_idx` was just checked against `self.words.len()`
            // and a new word pushed if it was out of range, so `word_idx`
            // is strictly less than `self.words.len()` at this point.
            unsafe {
                *self.words.get_unchecked_mut(word_idx) |= chunk << bit_off;
            }
            v >>= take;
            remaining -= take;
            self.bit_pos += take as u64;
        }
    }

    /// Consumes the writer and returns the packed word array.
    pub fn into_words(self) -> Vec<u64> {
        self.words
    }

    /// Number of fields' worth of bits written so far.
    pub fn bit_len(&self) -> u64 {
        self.bit_pos
    }
}

impl Default for BitWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Reads fixed-width fields back out of a `u64` word slice written by
/// [`BitWriter`] (or any buffer with the same bit layout).
pub struct BitReader<'a> {
    words: &'a [u64],
    bit_pos: u64,
}

impl<'a> BitReader<'a> {
    /// Creates a reader positioned at the start of `words`.
    pub fn new(words: &'a [u64]) -> Self {
        BitReader { words, bit_pos: 0 }
    }

    /// Reads the next `bits`-wide field, or `None` if fewer than `bits`
    /// bits remain in the backing word slice.
    ///
    /// # Panics
    /// Panics if `bits` is 0 or greater than 32.
    pub fn read_bits(&mut self, bits: u32) -> Option<u32> {
        assert!((1..=32).contains(&bits), "bitpack: bits out of range");
        let total_bits = self.words.len() as u64 * WORD_BITS as u64;
        if self.bit_pos + bits as u64 > total_bits {
            return None;
        }
        let mut result: u64 = 0;
        let mut got = 0u32;
        while got < bits {
            let word_idx = (self.bit_pos / WORD_BITS as u64) as usize;
            let bit_off = (self.bit_pos % WORD_BITS as u64) as u32;
            // SAFETY: the range check above guarantees `self.bit_pos +
            // (bits - got) <= total_bits`, so `word_idx` computed from any
            // `self.bit_pos` visited in this loop is strictly less than
            // `self.words.len()`.
            let word = unsafe { *self.words.get_unchecked(word_idx) };
            let space = WORD_BITS - bit_off;
            let take = (bits - got).min(space);
            let chunk = (word >> bit_off) & ((1u64 << take) - 1);
            result |= chunk << got;
            got += take;
            self.bit_pos += take as u64;
        }
        Some(result as u32)
    }

    /// Reads exactly `N` fields of `bits` width into a fixed-size array,
    /// or `None` if the backing slice runs out first.
    pub fn read_array<const N: usize>(&mut self, bits: u32) -> Option<[u32; N]> {
        // SAFETY: the outer `MaybeUninit` here wraps `[MaybeUninit<u32>; N]`,
        // not `[u32; N]` directly. An array of `MaybeUninit<u32>` is a valid
        // value no matter what bytes its storage holds, because each element
        // itself carries no initialization requirement, so asserting the
        // *array-of-MaybeUninit* is initialized is always sound.
        let mut buf: [MaybeUninit<u32>; N] = unsafe { MaybeUninit::uninit().assume_init() };
        for slot in buf.iter_mut().take(N) {
            let v = self.read_bits(bits)?;
            slot.write(v);
        }
        // SAFETY: the loop above wrote every one of the `N` slots via
        // `MaybeUninit::write` (returning early with `None` before this
        // point if any `read_bits` call failed), so all `N` elements are
        // initialized and it is sound to reinterpret the array as `[u32; N]`.
        Some(unsafe { std::mem::transmute_copy::<[MaybeUninit<u32>; N], [u32; N]>(&buf) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_len_matches_bit_math() {
        assert_eq!(packed_len(0, 4), 0);
        assert_eq!(packed_len(16, 4), 1);
        assert_eq!(packed_len(17, 4), 2);
        assert_eq!(packed_len(4096, 4), 256);
        assert_eq!(packed_len(1, 32), 1);
        assert_eq!(packed_len(2, 32), 1);
    }

    #[test]
    fn round_trip_single_width_field_stream() {
        for bits in [1u32, 3, 4, 5, 9, 17, 32] {
            let mut w = BitWriter::new();
            let mask = if bits == 32 { u32::MAX } else { (1u32 << bits) - 1 };
            let values: Vec<u32> =
                (0..200u32).map(|i| i.wrapping_mul(2654435761) & mask).collect();
            for &v in &values {
                w.write_bits(v, bits);
            }
            let words = w.into_words();
            let mut r = BitReader::new(&words);
            for &expected in &values {
                assert_eq!(r.read_bits(bits), Some(expected));
            }
        }
    }

    #[test]
    fn reader_returns_none_past_end() {
        let mut w = BitWriter::new();
        w.write_bits(7, 4);
        let words = w.into_words();
        let mut r = BitReader::new(&words);
        assert_eq!(r.read_bits(4), Some(7));
        // Remaining bits in the single word are zero-padded, so more reads
        // succeed (returning 0) until the word itself is exhausted (max
        // field width is 32, so the remaining 60 bits are read in two
        // calls).
        assert_eq!(r.read_bits(32), Some(0));
        assert_eq!(r.read_bits(28), Some(0));
        assert_eq!(r.read_bits(1), None);
    }

    #[test]
    fn palette_index_pattern_4_bits() {
        // A 16-entry palette section: 4096 voxels, 4 bits each.
        let mut w = BitWriter::new();
        let indices: Vec<u32> = (0..4096u32).map(|i| i % 16).collect();
        for &idx in &indices {
            w.write_bits(idx, 4);
        }
        let words = w.into_words();
        assert_eq!(words.len(), packed_len(4096, 4));
        let mut r = BitReader::new(&words);
        for &expected in &indices {
            assert_eq!(r.read_bits(4), Some(expected));
        }
    }

    #[test]
    fn read_array_round_trip() {
        let mut w = BitWriter::new();
        let values: [u32; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        for &v in &values {
            w.write_bits(v, 6);
        }
        let words = w.into_words();
        let mut r = BitReader::new(&words);
        let out: [u32; 8] = r.read_array(6).unwrap();
        assert_eq!(out, values);
    }

    #[test]
    fn empty_writer_produces_no_words() {
        let w = BitWriter::new();
        assert_eq!(w.bit_len(), 0);
        assert!(w.into_words().is_empty());
    }
}
