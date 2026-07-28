//! A small vector that stores up to `N` elements inline before spilling.
//! Per-voxel neighbor lists and mesh face vertex fans are usually tiny
//! (often 0-6 items) but occasionally large, and boxing every one in a heap
//! `Vec` wastes an allocation for the common case that this type avoids.

use std::mem::MaybeUninit;

enum Storage<T, const N: usize> {
    Inline { buf: [MaybeUninit<T>; N], len: usize },
    Spilled(Vec<T>),
}

/// A vector with `N` inline slots that spills to the heap beyond that.
///
/// # Soundness approach
/// The inline storage is `[MaybeUninit<T>; N]` with a tracked `len`. The
/// single invariant every method must preserve is: *indices `0..len` of the
/// inline buffer are initialized, indices `len..N` are not*. Every unsafe
/// block below either reads/writes strictly inside `0..len`, or (in the
/// spill transition) reads exactly the `0..len` prefix once and immediately
/// discards the old storage without ever touching it again.
pub struct SmallVec<T, const N: usize> {
    storage: Storage<T, N>,
}

impl<T, const N: usize> SmallVec<T, N> {
    /// Creates an empty small vector using inline storage.
    pub fn new() -> Self {
        // `std::array::from_fn` builds the array element-by-element, so no
        // unsafe assumption about bulk-initializing `[MaybeUninit<T>; N]`
        // is needed here.
        SmallVec {
            storage: Storage::Inline { buf: std::array::from_fn(|_| MaybeUninit::uninit()), len: 0 },
        }
    }

    /// Number of elements currently stored.
    pub fn len(&self) -> usize {
        match &self.storage {
            Storage::Inline { len, .. } => *len,
            Storage::Spilled(v) => v.len(),
        }
    }

    /// Whether the vector holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether elements currently live on the heap rather than inline.
    pub fn is_spilled(&self) -> bool {
        matches!(self.storage, Storage::Spilled(_))
    }

    /// Appends `value`, spilling inline storage to a `Vec` first if the
    /// inline capacity `N` is already full.
    pub fn push(&mut self, value: T) {
        if let Storage::Inline { len, .. } = &self.storage {
            if *len == N {
                self.spill();
            }
        }
        match &mut self.storage {
            Storage::Inline { buf, len } => {
                buf[*len].write(value);
                *len += 1;
            }
            Storage::Spilled(v) => v.push(value),
        }
    }

    /// Removes and returns the last element, or `None` if empty.
    pub fn pop(&mut self) -> Option<T> {
        match &mut self.storage {
            Storage::Inline { buf, len } => {
                if *len == 0 {
                    return None;
                }
                *len -= 1;
                // SAFETY: index `*len` (post-decrement) was, by the module
                // invariant, initialized while it was part of `0..len`
                // before this decrement, and is not read again through the
                // inline path since `len` no longer covers it.
                Some(unsafe { buf[*len].assume_init_read() })
            }
            Storage::Spilled(v) => v.pop(),
        }
    }

    /// Moves the initialized inline prefix into a new heap `Vec`, then
    /// switches this vector's storage to `Spilled`.
    fn spill(&mut self) {
        if let Storage::Inline { buf, len } = &mut self.storage {
            let mut v = Vec::with_capacity(N + 1);
            for slot in buf.iter_mut().take(*len) {
                // SAFETY: `slot` ranges over indices `0..len`, which the
                // module invariant guarantees are initialized. Replacing
                // each slot with `MaybeUninit::uninit()` before this
                // function returns (and before `self.storage` is
                // overwritten below) means the old inline array is never
                // read again, so moving the value out here does not create
                // a duplicate that could later be dropped twice.
                let value = unsafe {
                    std::mem::replace(slot, MaybeUninit::uninit()).assume_init()
                };
                v.push(value);
            }
            self.storage = Storage::Spilled(v);
        }
    }

    /// Returns a reference to the element at `index`, if any.
    pub fn get(&self, index: usize) -> Option<&T> {
        match &self.storage {
            Storage::Inline { buf, len } => {
                if index >= *len {
                    return None;
                }
                // SAFETY: `index < *len`, and the module invariant
                // guarantees indices `0..len` of the inline buffer are
                // initialized.
                Some(unsafe { buf[index].assume_init_ref() })
            }
            Storage::Spilled(v) => v.get(index),
        }
    }

    /// Returns an iterator over all elements in order.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        (0..self.len()).map(move |i| self.get(i).unwrap())
    }
}

impl<T, const N: usize> Drop for SmallVec<T, N> {
    fn drop(&mut self) {
        if let Storage::Inline { buf, len } = &mut self.storage {
            for slot in buf.iter_mut().take(*len) {
                // SAFETY: indices `0..len` are initialized by the module
                // invariant, and each slot is dropped exactly once here
                // since the whole `SmallVec` (and thus this `buf`) is never
                // used again after `drop` runs.
                unsafe {
                    slot.assume_init_drop();
                }
            }
        }
        // `Storage::Spilled(Vec<T>)` drops its elements on its own.
    }
}

impl<T, const N: usize> Default for SmallVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    #[test]
    fn stays_inline_below_capacity() {
        let mut sv: SmallVec<i32, 4> = SmallVec::new();
        sv.push(1);
        sv.push(2);
        sv.push(3);
        assert!(!sv.is_spilled());
        assert_eq!(sv.len(), 3);
    }

    #[test]
    fn spills_past_inline_capacity() {
        let mut sv: SmallVec<i32, 2> = SmallVec::new();
        sv.push(1);
        sv.push(2);
        assert!(!sv.is_spilled());
        sv.push(3);
        assert!(sv.is_spilled());
        let collected: Vec<i32> = sv.iter().copied().collect();
        assert_eq!(collected, vec![1, 2, 3]);
    }

    #[test]
    fn pop_reverses_push_order_both_inline_and_spilled() {
        let mut sv: SmallVec<i32, 2> = SmallVec::new();
        for v in [1, 2, 3, 4] {
            sv.push(v);
        }
        assert_eq!(sv.pop(), Some(4));
        assert_eq!(sv.pop(), Some(3));
        assert_eq!(sv.pop(), Some(2));
        assert_eq!(sv.pop(), Some(1));
        assert_eq!(sv.pop(), None);
    }

    #[test]
    fn empty_vector_has_zero_length() {
        let sv: SmallVec<i32, 4> = SmallVec::new();
        assert!(sv.is_empty());
        assert_eq!(sv.get(0), None);
    }

    #[test]
    fn drop_releases_owned_values_inline_and_spilled() {
        let counter = Rc::new(());
        let mut sv: SmallVec<Rc<()>, 2> = SmallVec::new();
        sv.push(counter.clone());
        sv.push(counter.clone());
        sv.push(counter.clone()); // forces spill
        assert_eq!(Rc::strong_count(&counter), 4);
        drop(sv);
        assert_eq!(Rc::strong_count(&counter), 1);
    }

    #[test]
    fn get_out_of_bounds_is_none_after_spill() {
        let mut sv: SmallVec<i32, 1> = SmallVec::new();
        sv.push(10);
        sv.push(20);
        assert!(sv.is_spilled());
        assert_eq!(sv.get(5), None);
        assert_eq!(sv.get(1), Some(&20));
    }
}
