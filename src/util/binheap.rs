//! A binary min-heap ordered by an explicit comparator key. Tick scheduling
//! and deferred mesh rebuilds want a priority queue that pops the
//! earliest-due item first; rather than requiring the stored value itself
//! to implement `Ord`, this heap takes a key-extraction function instead.

/// A binary min-heap over `T`, ordered by `key(&T)` ascending.
pub struct MinHeap<T, K, F>
where
    K: Ord,
    F: Fn(&T) -> K,
{
    data: Vec<T>,
    key: F,
}

impl<T, K, F> MinHeap<T, K, F>
where
    K: Ord,
    F: Fn(&T) -> K,
{
    /// Creates an empty heap using `key` to order elements.
    pub fn new(key: F) -> Self {
        MinHeap { data: Vec::new(), key }
    }

    /// Builds a heap from an existing vector in O(n) via bottom-up sift.
    pub fn from_vec(data: Vec<T>, key: F) -> Self {
        let mut heap = MinHeap { data, key };
        let n = heap.data.len();
        for i in (0..n / 2).rev() {
            heap.sift_down(i);
        }
        heap
    }

    /// Number of elements in the heap.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the heap holds no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns a reference to the minimum element without removing it.
    pub fn peek(&self) -> Option<&T> {
        self.data.first()
    }

    /// Inserts `value`, restoring the heap property by sifting up.
    pub fn push(&mut self, value: T) {
        self.data.push(value);
        let mut i = self.data.len() - 1;
        while i > 0 {
            let parent = (i - 1) / 2;
            // SAFETY: `i < self.data.len()` (it was just pushed) and
            // `parent < i < self.data.len()` since `parent = (i-1)/2 < i`
            // whenever `i > 0`, so both reads are in bounds.
            let should_swap = unsafe {
                (self.key)(self.data.get_unchecked(i)) < (self.key)(self.data.get_unchecked(parent))
            };
            if !should_swap {
                break;
            }
            self.data.swap(i, parent);
            i = parent;
        }
    }

    /// Removes and returns the minimum element, restoring the heap property
    /// by moving the last element to the root and sifting down.
    pub fn pop(&mut self) -> Option<T> {
        if self.data.is_empty() {
            return None;
        }
        let last = self.data.pop().unwrap();
        if self.data.is_empty() {
            return Some(last);
        }
        let root = std::mem::replace(&mut self.data[0], last);
        self.sift_down(0);
        Some(root)
    }

    fn sift_down(&mut self, mut i: usize) {
        let len = self.data.len();
        loop {
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            let mut smallest = i;
            if left < len {
                // SAFETY: `left < len = self.data.len()` and `smallest < len`
                // (it starts as `i < len` and only ever gets reassigned to
                // valid in-bounds indices `left`/`right`), so both reads are
                // in bounds.
                let smaller = unsafe {
                    (self.key)(self.data.get_unchecked(left)) < (self.key)(self.data.get_unchecked(smallest))
                };
                if smaller {
                    smallest = left;
                }
            }
            if right < len {
                let smaller = unsafe {
                    (self.key)(self.data.get_unchecked(right)) < (self.key)(self.data.get_unchecked(smallest))
                };
                if smaller {
                    smallest = right;
                }
            }
            if smallest == i {
                break;
            }
            self.data.swap(i, smallest);
            i = smallest;
        }
    }

    /// Consumes the heap, returning all elements sorted ascending by key.
    /// Equivalent to popping until empty, but stated as a single operation.
    pub fn pop_all_sorted(mut self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.data.len());
        while let Some(v) = self.pop() {
            out.push(v);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_pop_yields_ascending_order() {
        let mut heap = MinHeap::new(|&x: &i32| x);
        for v in [5, 1, 9, 3, 7, 2] {
            heap.push(v);
        }
        let mut popped = Vec::new();
        while let Some(v) = heap.pop() {
            popped.push(v);
        }
        assert_eq!(popped, vec![1, 2, 3, 5, 7, 9]);
    }

    #[test]
    fn peek_matches_first_pop_without_removing() {
        let mut heap = MinHeap::new(|&x: &i32| x);
        heap.push(10);
        heap.push(4);
        heap.push(8);
        assert_eq!(heap.peek(), Some(&4));
        assert_eq!(heap.len(), 3);
        assert_eq!(heap.pop(), Some(4));
    }

    #[test]
    fn from_vec_heapifies_correctly() {
        let heap = MinHeap::from_vec(vec![9, 4, 7, 1, 8, 2, 5], |&x: &i32| x);
        let sorted = heap.pop_all_sorted();
        assert_eq!(sorted, vec![1, 2, 4, 5, 7, 8, 9]);
    }

    #[test]
    fn custom_key_orders_scheduled_ticks() {
        #[derive(Debug, PartialEq)]
        struct Job {
            tick: u32,
            name: &'static str,
        }
        let mut heap = MinHeap::new(|j: &Job| j.tick);
        heap.push(Job { tick: 100, name: "relight" });
        heap.push(Job { tick: 10, name: "remesh" });
        heap.push(Job { tick: 50, name: "save" });
        assert_eq!(heap.pop().unwrap().name, "remesh");
        assert_eq!(heap.pop().unwrap().name, "save");
        assert_eq!(heap.pop().unwrap().name, "relight");
    }

    #[test]
    fn empty_heap_pop_and_peek_are_none() {
        let mut heap: MinHeap<i32, i32, _> = MinHeap::new(|&x| x);
        assert_eq!(heap.peek(), None);
        assert_eq!(heap.pop(), None);
        assert!(heap.is_empty());
    }

    #[test]
    fn duplicate_keys_are_all_preserved() {
        let mut heap = MinHeap::new(|&x: &i32| x);
        for _ in 0..5 {
            heap.push(3);
        }
        let sorted = heap.pop_all_sorted();
        assert_eq!(sorted, vec![3, 3, 3, 3, 3]);
    }
}
