//! A fixed-capacity LRU cache over an intrusive doubly linked list. Decoded
//! chunk sections and mesh buffers are worth caching but too numerous to
//! keep forever; this evicts the least-recently-used entry at capacity,
//! threading recency order through indices in a `Vec` of slots.

use std::collections::HashMap;
use std::hash::Hash;

const NIL: u32 = u32::MAX;

struct Slot<K, V> {
    key: K,
    value: V,
    prev: u32,
    next: u32,
}

/// A capacity-bounded cache that evicts least-recently-used entries.
///
/// `map` resolves a key to its slot index; `slots` holds the entries and
/// the intrusive prev/next links; `head`/`tail` are the most- and
/// least-recently-used slot indices. Every index stored in `map`, in the
/// `head`/`tail` fields, or in any slot's `prev`/`next` always names a slot
/// that is currently linked into the list — removal always relinks
/// neighbors before a slot is reused, so this invariant never lapses.
pub struct LruCache<K, V> {
    map: HashMap<K, u32>,
    slots: Vec<Slot<K, V>>,
    head: u32,
    tail: u32,
    capacity: usize,
}

impl<K: Eq + Hash + Clone, V> LruCache<K, V> {
    /// Creates a cache holding at most `capacity` entries.
    ///
    /// # Panics
    /// Panics if `capacity` is 0.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "LruCache: capacity must be positive");
        LruCache { map: HashMap::new(), slots: Vec::new(), head: NIL, tail: NIL, capacity }
    }

    /// Number of entries currently cached.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Whether the cache holds no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Whether `key` is currently cached (does not affect recency).
    pub fn contains(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    fn unlink(&mut self, idx: u32) {
        let (prev, next) = {
            // SAFETY: every index passed to `unlink` is either `self.head`
            // or a slot's stored `prev`/`next`, all of which the struct-level
            // invariant guarantees are valid, currently-linked slot indices.
            let slot = unsafe { self.slots.get_unchecked(idx as usize) };
            (slot.prev, slot.next)
        };
        if prev != NIL {
            self.slots[prev as usize].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.slots[next as usize].prev = prev;
        } else {
            self.tail = prev;
        }
    }

    fn push_front(&mut self, idx: u32) {
        let old_head = self.head;
        // SAFETY: `idx` names a slot that was just allocated or unlinked by
        // the caller, and is always `< self.slots.len()` because it is
        // either a freshly pushed index or one already present in `map`.
        unsafe {
            let slot = self.slots.get_unchecked_mut(idx as usize);
            slot.prev = NIL;
            slot.next = old_head;
        }
        if old_head != NIL {
            self.slots[old_head as usize].prev = idx;
        } else {
            self.tail = idx;
        }
        self.head = idx;
    }

    fn touch(&mut self, idx: u32) {
        if self.head == idx {
            return;
        }
        self.unlink(idx);
        self.push_front(idx);
    }

    /// Looks up `key`, marking it most-recently-used on a hit.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let idx = *self.map.get(key)?;
        self.touch(idx);
        // SAFETY: `idx` came from `self.map`, which by the struct invariant
        // only ever stores indices of currently-linked (i.e. occupied)
        // slots, so this index is in bounds and its slot holds a live value.
        Some(unsafe { &self.slots.get_unchecked(idx as usize).value })
    }

    /// Inserts or updates `key` with `value`, marking it most-recently-used.
    /// If inserting a new key would exceed capacity, evicts the
    /// least-recently-used entry first.
    pub fn put(&mut self, key: K, value: V) {
        if let Some(&idx) = self.map.get(&key) {
            self.slots[idx as usize].value = value;
            self.touch(idx);
            return;
        }
        if self.map.len() >= self.capacity {
            let evict_idx = self.tail;
            self.unlink(evict_idx);
            let evicted_key = self.slots[evict_idx as usize].key.clone();
            self.map.remove(&evicted_key);
            self.slots[evict_idx as usize] = Slot { key: key.clone(), value, prev: NIL, next: NIL };
            self.map.insert(key, evict_idx);
            self.push_front(evict_idx);
            return;
        }
        let idx = self.slots.len() as u32;
        self.slots.push(Slot { key: key.clone(), value, prev: NIL, next: NIL });
        self.map.insert(key, idx);
        self.push_front(idx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_round_trip() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);
        assert_eq!(cache.get(&"a"), Some(&1));
        assert_eq!(cache.get(&"b"), Some(&2));
    }

    #[test]
    fn eviction_removes_least_recently_used() {
        let mut cache = LruCache::new(2);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.get(&"a"); // "a" is now most recently used, "b" is LRU
        cache.put("c", 3); // should evict "b"
        assert!(!cache.contains(&"b"));
        assert!(cache.contains(&"a"));
        assert!(cache.contains(&"c"));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn updating_existing_key_does_not_grow_len() {
        let mut cache = LruCache::new(3);
        cache.put("a", 1);
        cache.put("a", 2);
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.get(&"a"), Some(&2));
    }

    #[test]
    fn capacity_one_always_keeps_most_recent() {
        let mut cache = LruCache::new(1);
        cache.put("a", 1);
        cache.put("b", 2);
        assert!(!cache.contains(&"a"));
        assert_eq!(cache.get(&"b"), Some(&2));
    }

    #[test]
    fn get_on_missing_key_is_none_and_does_not_panic() {
        let mut cache: LruCache<&str, i32> = LruCache::new(2);
        assert_eq!(cache.get(&"missing"), None);
        assert!(cache.is_empty());
    }

    #[test]
    fn repeated_touch_preserves_all_entries_under_capacity() {
        let mut cache = LruCache::new(3);
        cache.put("a", 1);
        cache.put("b", 2);
        cache.put("c", 3);
        for _ in 0..5 {
            cache.get(&"a");
            cache.get(&"b");
            cache.get(&"c");
        }
        assert_eq!(cache.len(), 3);
        assert!(cache.contains(&"a") && cache.contains(&"b") && cache.contains(&"c"));
    }
}
