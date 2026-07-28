//! A fixed-capacity ring buffer. Tick schedulers and chunk-load queues want
//! a bounded FIFO that never allocates once created; this stores `T` in a
//! boxed slice of `MaybeUninit<T>` with a head index and length, so
//! push/pop are O(1) wraparound arithmetic instead of shifting elements.

use std::mem::MaybeUninit;

/// A fixed-capacity, heap-allocated ring buffer over `T`.
pub struct RingBuf<T> {
    buf: Box<[MaybeUninit<T>]>,
    head: usize,
    len: usize,
}

impl<T> RingBuf<T> {
    /// Creates an empty ring buffer that can hold up to `capacity` items.
    pub fn new(capacity: usize) -> Self {
        let mut v = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            v.push(MaybeUninit::uninit());
        }
        RingBuf { buf: v.into_boxed_slice(), head: 0, len: 0 }
    }

    /// Maximum number of elements this buffer can hold.
    pub fn capacity(&self) -> usize {
        self.buf.len()
    }

    /// Number of elements currently stored.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether the buffer is at capacity.
    pub fn is_full(&self) -> bool {
        self.len == self.buf.len()
    }

    fn slot_index(&self, offset: usize) -> usize {
        let cap = self.buf.len();
        if cap == 0 {
            0
        } else {
            (self.head + offset) % cap
        }
    }

    /// Pushes `value` at the back. Returns `Err(value)` without modifying
    /// the buffer if it is already full.
    pub fn push_back(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            return Err(value);
        }
        let idx = self.slot_index(self.len);
        // SAFETY: `idx = (head + len) % cap` is always in `0..cap` for
        // `cap > 0` (guaranteed here since `is_full` returned false only
        // when `cap > 0`), so this is an in-bounds write into a slot that
        // the invariant "elements live in `[head, head+len) mod cap`"
        // classifies as not-yet-live, meaning nothing there needs dropping.
        unsafe {
            self.buf.get_unchecked_mut(idx).write(value);
        }
        self.len += 1;
        Ok(())
    }

    /// Pushes `value` at the back, and if the buffer is full, evicts and
    /// returns the oldest element to make room.
    pub fn push_overwrite(&mut self, value: T) -> Option<T> {
        if self.is_full() {
            let evicted = self.pop_front();
            self.push_back(value).ok();
            evicted
        } else {
            self.push_back(value).ok();
            None
        }
    }

    /// Removes and returns the front (oldest) element, or `None` if empty.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let idx = self.head;
        // SAFETY: `idx = self.head` names the oldest live slot, which by
        // the buffer invariant is initialized whenever `self.len > 0`
        // (checked just above). Reading it out with `assume_init_read`
        // moves the value; the slot is then logically dead because `head`
        // advances and `len` shrinks below, so it will never be read again
        // without an intervening `write` from a future push.
        let value = unsafe { self.buf.get_unchecked(idx).assume_init_read() };
        self.head = self.slot_index(1);
        self.len -= 1;
        Some(value)
    }

    /// Returns a reference to the front element without removing it.
    pub fn front(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        // SAFETY: identical invariant as `pop_front`: slot `head` is live
        // whenever `len > 0`.
        Some(unsafe { self.buf.get_unchecked(self.head).assume_init_ref() })
    }

    /// Returns the live elements as (at most) two contiguous slices, in
    /// order: the first covers from `head` to either the end of the buffer
    /// or the end of the live run, the second (if non-empty) wraps around
    /// to the start.
    pub fn as_slices(&self) -> (&[T], &[T]) {
        if self.len == 0 {
            return (&[], &[]);
        }
        let cap = self.buf.len();
        let first_len = self.len.min(cap - self.head);
        let second_len = self.len - first_len;
        // SAFETY: `self.buf[head .. head + first_len]` is exactly the
        // (non-wrapping) prefix of the live run, so every element in it was
        // written by a `push_back`/`push_overwrite` and not yet popped.
        // When `second_len > 0` the live run wrapped past the end of the
        // buffer; those elements start at index 0, were written before
        // `head` advanced past them on an earlier wraparound push, and are
        // still live because they fall within `[head, head+len) mod cap`.
        // Both slices borrow `self.buf` immutably, matching `&self`.
        unsafe {
            let first =
                std::slice::from_raw_parts(self.buf.as_ptr().add(self.head) as *const T, first_len);
            let second = if second_len == 0 {
                &[]
            } else {
                std::slice::from_raw_parts(self.buf.as_ptr() as *const T, second_len)
            };
            (first, second)
        }
    }

    /// Iterates over live elements from front to back.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        let (a, b) = self.as_slices();
        a.iter().chain(b.iter())
    }
}

impl<T> Drop for RingBuf<T> {
    fn drop(&mut self) {
        // Drop only the live elements; slots outside `[head, head+len) mod
        // cap` were never initialized (or were already moved out by
        // `pop_front`) and must not be dropped.
        while self.pop_front().is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_preserves_fifo_order() {
        let mut rb: RingBuf<i32> = RingBuf::new(4);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.push_back(3).unwrap();
        assert_eq!(rb.pop_front(), Some(1));
        assert_eq!(rb.pop_front(), Some(2));
        assert_eq!(rb.len(), 1);
    }

    #[test]
    fn push_back_rejects_when_full() {
        let mut rb: RingBuf<i32> = RingBuf::new(2);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        assert!(rb.is_full());
        assert_eq!(rb.push_back(3), Err(3));
    }

    #[test]
    fn overwrite_mode_evicts_oldest() {
        let mut rb: RingBuf<i32> = RingBuf::new(3);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.push_back(3).unwrap();
        assert_eq!(rb.push_overwrite(4), Some(1));
        let collected: Vec<i32> = rb.iter().copied().collect();
        assert_eq!(collected, vec![2, 3, 4]);
    }

    #[test]
    fn wraparound_and_contiguous_slices() {
        let mut rb: RingBuf<i32> = RingBuf::new(3);
        rb.push_back(1).unwrap();
        rb.push_back(2).unwrap();
        rb.pop_front();
        rb.push_back(3).unwrap();
        rb.push_back(4).unwrap();
        // Buffer now wraps: logical order is [2, 3, 4].
        let (a, b) = rb.as_slices();
        let mut combined = a.to_vec();
        combined.extend_from_slice(b);
        assert_eq!(combined, vec![2, 3, 4]);
    }

    #[test]
    fn zero_capacity_buffer_rejects_everything() {
        let mut rb: RingBuf<i32> = RingBuf::new(0);
        assert!(rb.is_full());
        assert_eq!(rb.push_back(1), Err(1));
        assert_eq!(rb.pop_front(), None);
    }

    #[test]
    fn drop_releases_owned_values_without_leaking() {
        use std::rc::Rc;
        let counter = Rc::new(());
        let mut rb: RingBuf<Rc<()>> = RingBuf::new(4);
        for _ in 0..3 {
            rb.push_back(counter.clone()).unwrap();
        }
        assert_eq!(Rc::strong_count(&counter), 4);
        drop(rb);
        assert_eq!(Rc::strong_count(&counter), 1);
    }

    #[test]
    fn front_reflects_oldest_without_removing() {
        let mut rb: RingBuf<i32> = RingBuf::new(2);
        rb.push_back(9).unwrap();
        assert_eq!(rb.front(), Some(&9));
        assert_eq!(rb.len(), 1);
    }
}
