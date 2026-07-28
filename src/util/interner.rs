//! A string interner backed by one contiguous buffer. Block type names and
//! biome ids repeat constantly across a world; interning stores each
//! distinct string once and hands out a small `Symbol(u32)` everywhere
//! else, turning string comparison into integer comparison.

/// A deduplicated string handle. Two symbols are equal exactly when the
/// strings they were interned from are equal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Symbol(u32);

struct Span {
    start: u32,
    end: u32,
}

/// Interns strings into one contiguous buffer, deduping via a manual
/// open-addressing hash table over spans (rather than a `HashMap<String,_>`,
/// which would store every key's bytes a second time).
pub struct Interner {
    buffer: String,
    spans: Vec<Span>,
    // Each bucket holds the index into `spans` of a symbol whose hash maps
    // here, or `u32::MAX` for an empty bucket. `buckets.len()` is always a
    // power of two so `hash & (buckets.len() - 1)` is a valid index.
    buckets: Vec<u32>,
}

const EMPTY: u32 = u32::MAX;

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

impl Interner {
    /// Creates an empty interner.
    pub fn new() -> Self {
        Interner { buffer: String::new(), spans: Vec::new(), buckets: vec![EMPTY; 16] }
    }

    fn find(&self, s: &str, hash: u64) -> Option<u32> {
        if self.buckets.is_empty() {
            return None;
        }
        let mask = self.buckets.len() - 1;
        let mut idx = (hash as usize) & mask;
        for _ in 0..self.buckets.len() {
            let slot = self.buckets[idx];
            if slot == EMPTY {
                return None;
            }
            if self.span_str(slot) == s {
                return Some(slot);
            }
            idx = (idx + 1) & mask;
        }
        None
    }

    fn span_str(&self, sym: u32) -> &str {
        let span = &self.spans[sym as usize];
        &self.buffer[span.start as usize..span.end as usize]
    }

    fn grow_buckets(&mut self) {
        let mut new_buckets = vec![EMPTY; self.buckets.len() * 2];
        let mask = new_buckets.len() - 1;
        for sym in 0..self.spans.len() as u32 {
            let hash = fnv1a(self.span_str(sym).as_bytes());
            let mut idx = (hash as usize) & mask;
            while new_buckets[idx] != EMPTY {
                idx = (idx + 1) & mask;
            }
            new_buckets[idx] = sym;
        }
        self.buckets = new_buckets;
    }

    /// Interns `s`, returning its (possibly newly assigned) symbol. Interning
    /// the same string content again always returns the same symbol.
    pub fn intern(&mut self, s: &str) -> Symbol {
        let hash = fnv1a(s.as_bytes());
        if let Some(existing) = self.find(s, hash) {
            return Symbol(existing);
        }
        // Load factor 0.75 keeps probe sequences short.
        if (self.spans.len() + 1) * 4 >= self.buckets.len() * 3 {
            self.grow_buckets();
        }
        let start = self.buffer.len() as u32;
        self.buffer.push_str(s);
        let end = self.buffer.len() as u32;
        let sym = self.spans.len() as u32;
        self.spans.push(Span { start, end });

        let mask = self.buckets.len() - 1;
        let mut idx = (hash as usize) & mask;
        // SAFETY: `mask = buckets.len() - 1` with `buckets.len()` a power of
        // two, so `hash as usize & mask` (and every subsequent `(idx + 1) &
        // mask`) is always a value in `0..buckets.len()`: bit-masking with
        // `len - 1` for a power-of-two length can never exceed `len - 1`.
        while unsafe { *self.buckets.get_unchecked(idx) } != EMPTY {
            idx = (idx + 1) & mask;
        }
        self.buckets[idx] = sym;
        Symbol(sym)
    }

    /// Resolves a symbol back to its string, or `None` if it was not issued
    /// by this interner (out of range).
    pub fn resolve(&self, symbol: Symbol) -> Option<&str> {
        let idx = symbol.0 as usize;
        if idx >= self.spans.len() {
            return None;
        }
        // SAFETY: the bounds check above guarantees `idx < self.spans.len()`.
        let span = unsafe { self.spans.get_unchecked(idx) };
        let start = span.start as usize;
        let end = span.end as usize;
        // SAFETY: `start` and `end` were recorded in `intern` as byte
        // offsets immediately before and after a `push_str` call, so they
        // fall exactly on UTF-8 character boundaries within `self.buffer`
        // and `start <= end <= self.buffer.len()` holds because the buffer
        // only ever grows by appending. Re-validating UTF-8 on a slice we
        // already know came from a valid `&str` would be redundant work.
        let bytes = unsafe { self.buffer.as_bytes().get_unchecked(start..end) };
        Some(unsafe { std::str::from_utf8_unchecked(bytes) })
    }

    /// Number of distinct strings interned so far.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether no strings have been interned yet.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

impl Default for Interner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interning_same_string_returns_same_symbol() {
        let mut it = Interner::new();
        let a = it.intern("minecraft:stone");
        let b = it.intern("minecraft:stone");
        assert_eq!(a, b);
        assert_eq!(it.len(), 1);
    }

    #[test]
    fn distinct_strings_get_distinct_symbols() {
        let mut it = Interner::new();
        let a = it.intern("stone");
        let b = it.intern("dirt");
        assert_ne!(a, b);
        assert_eq!(it.resolve(a), Some("stone"));
        assert_eq!(it.resolve(b), Some("dirt"));
    }

    #[test]
    fn resolve_rejects_out_of_range_symbol() {
        let it = Interner::new();
        assert_eq!(it.resolve(Symbol(0)), None);
    }

    #[test]
    fn survives_growth_across_many_entries() {
        let mut it = Interner::new();
        let mut symbols = Vec::new();
        for i in 0..500 {
            let s = format!("block_{i}");
            symbols.push((s.clone(), it.intern(&s)));
        }
        for (s, sym) in &symbols {
            assert_eq!(it.resolve(*sym), Some(s.as_str()));
        }
        // Re-interning after growth must still dedupe correctly.
        let again = it.intern("block_250");
        assert_eq!(again, symbols[250].1);
        assert_eq!(it.len(), 500);
    }

    #[test]
    fn empty_string_is_a_valid_distinct_entry() {
        let mut it = Interner::new();
        let empty = it.intern("");
        assert_eq!(it.resolve(empty), Some(""));
        let non_empty = it.intern("x");
        assert_ne!(empty, non_empty);
    }

    #[test]
    fn empty_interner_reports_empty() {
        let it = Interner::new();
        assert!(it.is_empty());
        assert_eq!(it.len(), 0);
    }
}
