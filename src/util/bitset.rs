//! A growable bitset backed by `u64` words. Voxel engines track sets like
//! "which chunks in this radius are loaded" or "which faces need
//! remeshing"; a word-packed bitset answers membership and set-algebra
//! queries far more cheaply than a `Vec<bool>` or a hash set.

const BITS_PER_WORD: usize = 64;

/// A resizable set of bit indices, packed 64 per `u64` word.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BitSet {
    words: Vec<u64>,
}

impl BitSet {
    /// Creates an empty bitset with no words allocated.
    pub fn new() -> Self {
        BitSet { words: Vec::new() }
    }

    /// Creates an empty bitset with room for at least `bits` indices
    /// without reallocating.
    pub fn with_capacity_bits(bits: usize) -> Self {
        let words = bits.div_ceil(BITS_PER_WORD);
        BitSet { words: Vec::with_capacity(words) }
    }

    /// The highest index the set can currently hold without growing.
    pub fn capacity_bits(&self) -> usize {
        self.words.len() * BITS_PER_WORD
    }

    fn ensure_capacity(&mut self, index: usize) {
        let needed = index / BITS_PER_WORD + 1;
        if needed > self.words.len() {
            self.words.resize(needed, 0);
        }
    }

    /// Sets bit `index`, growing the backing storage if needed.
    pub fn set(&mut self, index: usize) {
        self.ensure_capacity(index);
        let word_idx = index / BITS_PER_WORD;
        let bit_idx = index % BITS_PER_WORD;
        // SAFETY: `ensure_capacity` just grew `self.words` so that
        // `word_idx = index / BITS_PER_WORD` is strictly less than
        // `self.words.len()`.
        unsafe {
            *self.words.get_unchecked_mut(word_idx) |= 1u64 << bit_idx;
        }
    }

    /// Clears bit `index`. A no-op if `index` is beyond current capacity.
    pub fn clear(&mut self, index: usize) {
        let word_idx = index / BITS_PER_WORD;
        if word_idx >= self.words.len() {
            return;
        }
        let bit_idx = index % BITS_PER_WORD;
        // SAFETY: the guard above returns early unless `word_idx <
        // self.words.len()`, so this access is in bounds.
        unsafe {
            *self.words.get_unchecked_mut(word_idx) &= !(1u64 << bit_idx);
        }
    }

    /// Toggles bit `index`, growing the backing storage if needed.
    pub fn toggle(&mut self, index: usize) {
        self.ensure_capacity(index);
        let word_idx = index / BITS_PER_WORD;
        let bit_idx = index % BITS_PER_WORD;
        self.words[word_idx] ^= 1u64 << bit_idx;
    }

    /// Returns whether bit `index` is set. Indices beyond capacity are
    /// treated as unset.
    pub fn test(&self, index: usize) -> bool {
        let word_idx = index / BITS_PER_WORD;
        match self.words.get(word_idx) {
            Some(word) => word & (1u64 << (index % BITS_PER_WORD)) != 0,
            None => false,
        }
    }

    /// Total number of set bits.
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Whether no bits are set.
    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|&w| w == 0)
    }

    /// Sets every bit that is set in `other` (in place). Extends this set's
    /// storage if `other` is wider.
    pub fn union_with(&mut self, other: &BitSet) {
        if other.words.len() > self.words.len() {
            self.words.resize(other.words.len(), 0);
        }
        let shared = other.words.len();
        for i in 0..shared {
            // SAFETY: `i` ranges over `0..shared` where `shared =
            // other.words.len()`, and `self.words` was just resized to at
            // least that length, so both unchecked reads/writes are in
            // bounds.
            unsafe {
                *self.words.get_unchecked_mut(i) |= *other.words.get_unchecked(i);
            }
        }
    }

    /// Keeps only bits set in both this set and `other`.
    pub fn intersect_with(&mut self, other: &BitSet) {
        let shared = self.words.len().min(other.words.len());
        for i in 0..shared {
            // SAFETY: `i < shared <= self.words.len()` and `i < shared <=
            // other.words.len()`, so both accesses are in bounds.
            unsafe {
                *self.words.get_unchecked_mut(i) &= *other.words.get_unchecked(i);
            }
        }
        for word in self.words.iter_mut().skip(shared) {
            *word = 0;
        }
    }

    /// Clears every bit that is set in `other`.
    pub fn difference_with(&mut self, other: &BitSet) {
        for (mine, theirs) in self.words.iter_mut().zip(other.words.iter()) {
            *mine &= !*theirs;
        }
    }

    /// Returns the lowest set bit index, if any.
    pub fn first_set(&self) -> Option<usize> {
        self.next_set(0)
    }

    /// Returns the lowest set bit index that is `>= from`, if any.
    pub fn next_set(&self, from: usize) -> Option<usize> {
        let mut word_idx = from / BITS_PER_WORD;
        if word_idx >= self.words.len() {
            return None;
        }
        let bit_off = from % BITS_PER_WORD;
        let mut masked = self.words[word_idx] & (!0u64 << bit_off);
        loop {
            if masked != 0 {
                return Some(word_idx * BITS_PER_WORD + masked.trailing_zeros() as usize);
            }
            word_idx += 1;
            if word_idx >= self.words.len() {
                return None;
            }
            masked = self.words[word_idx];
        }
    }

    /// Iterates over all set bit indices in ascending order.
    pub fn iter_ones(&self) -> IterOnes<'_> {
        IterOnes { set: self, next: 0 }
    }
}

/// Ascending iterator over the set bit indices of a [`BitSet`].
pub struct IterOnes<'a> {
    set: &'a BitSet,
    next: usize,
}

impl<'a> Iterator for IterOnes<'a> {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        let found = self.set.next_set(self.next)?;
        self.next = found + 1;
        Some(found)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn set_clear_toggle_test_round_trip() {
        let mut bs = BitSet::new();
        assert!(!bs.test(5));
        bs.set(5);
        assert!(bs.test(5));
        bs.clear(5);
        assert!(!bs.test(5));
        bs.toggle(63);
        assert!(bs.test(63));
        bs.toggle(63);
        assert!(!bs.test(63));
    }

    #[test]
    fn grows_across_word_boundaries() {
        let mut bs = BitSet::new();
        bs.set(200);
        assert!(bs.capacity_bits() >= 201);
        assert!(bs.test(200));
        assert_eq!(bs.count_ones(), 1);
    }

    #[test]
    fn set_algebra_operations() {
        let mut a = BitSet::new();
        for i in [1, 2, 3, 100] {
            a.set(i);
        }
        let mut b = BitSet::new();
        for i in [2, 3, 4, 200] {
            b.set(i);
        }
        let mut union = a.clone();
        union.union_with(&b);
        assert_eq!(union.count_ones(), 6);

        let mut inter = a.clone();
        inter.intersect_with(&b);
        assert_eq!(inter.count_ones(), 2);
        assert!(inter.test(2) && inter.test(3));

        let mut diff = a.clone();
        diff.difference_with(&b);
        assert_eq!(diff.count_ones(), 2);
        assert!(diff.test(1) && diff.test(100));
    }

    #[test]
    fn iter_ones_yields_ascending_indices() {
        let mut bs = BitSet::new();
        for i in [5usize, 64, 65, 130, 0] {
            bs.set(i);
        }
        let collected: Vec<usize> = bs.iter_ones().collect();
        assert_eq!(collected, vec![0, 5, 64, 65, 130]);
    }

    #[test]
    fn empty_set_has_no_first_bit() {
        let bs = BitSet::new();
        assert_eq!(bs.first_set(), None);
        assert!(bs.is_empty());
    }

    #[test]
    fn next_set_from_middle_of_word() {
        let mut bs = BitSet::new();
        bs.set(10);
        bs.set(20);
        assert_eq!(bs.next_set(11), Some(20));
        assert_eq!(bs.next_set(21), None);
    }
}
