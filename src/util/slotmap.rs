//! A generational slot map keyed by index, not by pointer. Entities and
//! loaded chunk records need stable handles that survive other entries
//! being removed and their slots reused; a stale `Key` is rejected instead
//! of silently aliasing a different, newer value stored in the same slot.

/// A handle into a [`SlotMap`]. Two keys are equal only if they name the
/// same slot at the same generation, so a key from before a `remove` will
/// not match the slot after it is reused. The handle is a plain index pair
/// with no pointer into the slot storage, so slots can be freely
/// reallocated by the backing `Vec` without invalidating any live key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Key {
    index: u32,
    generation: u32,
}

enum Slot<T> {
    Occupied { value: T, generation: u32 },
    Vacant { next_free: Option<u32>, generation: u32 },
}

/// A dense, generation-checked map from [`Key`] to `T`.
pub struct SlotMap<T> {
    slots: Vec<Slot<T>>,
    free_head: Option<u32>,
    len: usize,
}

impl<T> SlotMap<T> {
    /// Creates an empty slot map.
    pub fn new() -> Self {
        SlotMap { slots: Vec::new(), free_head: None, len: 0 }
    }

    /// Number of currently occupied slots.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the map holds no values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Inserts `value`, returning a fresh key. Reuses a freed slot (bumping
    /// its generation) when one is available, otherwise appends a new slot.
    pub fn insert(&mut self, value: T) -> Key {
        self.len += 1;
        if let Some(idx) = self.free_head {
            let slot = &mut self.slots[idx as usize];
            let generation = match *slot {
                Slot::Vacant { next_free, generation } => {
                    self.free_head = next_free;
                    generation
                }
                Slot::Occupied { .. } => unreachable!("free list points at an occupied slot"),
            };
            *slot = Slot::Occupied { value, generation };
            Key { index: idx, generation }
        } else {
            let index = self.slots.len() as u32;
            self.slots.push(Slot::Occupied { value, generation: 0 });
            Key { index, generation: 0 }
        }
    }

    /// Returns a reference to the value for `key`, or `None` if the key is
    /// out of range, points at a vacant slot, or its generation is stale.
    pub fn get(&self, key: Key) -> Option<&T> {
        let idx = key.index as usize;
        if idx >= self.slots.len() {
            return None;
        }
        // SAFETY: the bounds check above guarantees `idx < self.slots.len()`.
        match unsafe { self.slots.get_unchecked(idx) } {
            Slot::Occupied { value, generation } if *generation == key.generation => Some(value),
            _ => None,
        }
    }

    /// Mutable counterpart of [`SlotMap::get`].
    pub fn get_mut(&mut self, key: Key) -> Option<&mut T> {
        let idx = key.index as usize;
        if idx >= self.slots.len() {
            return None;
        }
        // SAFETY: the bounds check above guarantees `idx < self.slots.len()`,
        // and `&mut self` ensures this is the only live borrow of `slots`.
        match unsafe { self.slots.get_unchecked_mut(idx) } {
            Slot::Occupied { value, generation } if *generation == key.generation => Some(value),
            _ => None,
        }
    }

    /// Removes and returns the value for `key`, pushing its slot onto the
    /// free list with a bumped generation so old keys can never match again.
    pub fn remove(&mut self, key: Key) -> Option<T> {
        let idx = key.index as usize;
        if idx >= self.slots.len() {
            return None;
        }
        // SAFETY: bounds-checked above; `idx < self.slots.len()`.
        let slot = unsafe { self.slots.get_unchecked_mut(idx) };
        match slot {
            Slot::Occupied { generation, .. } if *generation == key.generation => {
                let next_gen = generation.wrapping_add(1);
                let old = std::mem::replace(
                    slot,
                    Slot::Vacant { next_free: self.free_head, generation: next_gen },
                );
                self.free_head = Some(idx as u32);
                self.len -= 1;
                match old {
                    Slot::Occupied { value, .. } => Some(value),
                    Slot::Vacant { .. } => None,
                }
            }
            _ => None,
        }
    }

    /// Whether `key` currently names a live value.
    pub fn contains(&self, key: Key) -> bool {
        self.get(key).is_some()
    }

    /// Iterates over all currently occupied `(Key, &T)` pairs in slot
    /// order.
    pub fn iter(&self) -> impl Iterator<Item = (Key, &T)> {
        self.slots.iter().enumerate().filter_map(|(i, slot)| match slot {
            Slot::Occupied { value, generation } => {
                Some((Key { index: i as u32, generation: *generation }, value))
            }
            Slot::Vacant { .. } => None,
        })
    }
}

impl<T> Default for SlotMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_get_round_trip() {
        let mut map = SlotMap::new();
        let a = map.insert("a");
        let b = map.insert("b");
        assert_eq!(map.get(a), Some(&"a"));
        assert_eq!(map.get(b), Some(&"b"));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn stale_key_rejected_after_removal() {
        let mut map = SlotMap::new();
        let a = map.insert(10);
        assert_eq!(map.remove(a), Some(10));
        assert_eq!(map.get(a), None);
        assert!(!map.contains(a));
        assert_eq!(map.len(), 0);
    }

    #[test]
    fn free_slot_is_reused_with_new_generation() {
        let mut map = SlotMap::new();
        let a = map.insert(1);
        map.remove(a).unwrap();
        let b = map.insert(2);
        // The freed slot should be reused: same index, later generation.
        assert_eq!(b.index, a.index);
        assert_ne!(b.generation, a.generation);
        assert_eq!(map.get(a), None);
        assert_eq!(map.get(b), Some(&2));
    }

    #[test]
    fn get_mut_allows_in_place_update() {
        let mut map = SlotMap::new();
        let k = map.insert(5);
        *map.get_mut(k).unwrap() += 100;
        assert_eq!(map.get(k), Some(&105));
    }

    #[test]
    fn out_of_range_key_is_none() {
        let map: SlotMap<i32> = SlotMap::new();
        let bogus = Key { index: 999, generation: 0 };
        assert_eq!(map.get(bogus), None);
    }

    #[test]
    fn iter_visits_only_occupied_slots() {
        let mut map = SlotMap::new();
        let a = map.insert("x");
        let _b = map.insert("y");
        map.remove(a);
        let items: Vec<_> = map.iter().map(|(_, v)| *v).collect();
        assert_eq!(items, vec!["y"]);
    }
}
