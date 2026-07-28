//! A bump arena for `Copy` POD data, chunked into fixed-size byte blocks.
//! Meshing and light propagation generate short-lived scratch data (vertex
//! buffers, visited-node arrays) that all die together at once; this
//! allocates by advancing an offset into a block and frees everything via
//! [`Arena::reset`] instead of tracking individual lifetimes.

const BLOCK_SIZE: usize = 64 * 1024;

/// A checked handle to a slice of `T` previously allocated by an [`Arena`].
///
/// # Why an index, not a reference
/// `alloc_slice` hands back a block index, byte offset, and length rather
/// than a `&[T]` or raw pointer. The arena's block list is a `Vec<Vec<u8>>`:
/// pushing a new block can reallocate that *outer* vector, and
/// [`Arena::reset`] can drop blocks entirely. A stored reference or pointer
/// would silently dangle across either event with no compiler diagnostic.
/// A `Handle` instead carries no pointer at all; it is checked against the
/// arena's current generation on every access, so a handle from before a
/// `reset` is rejected rather than read as garbage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Handle<T> {
    block: u32,
    offset: u32,
    len: u32,
    generation: u32,
    _marker: std::marker::PhantomData<T>,
}

/// A chunked bump allocator for `Copy` types.
pub struct Arena {
    blocks: Vec<Vec<u8>>,
    cursor: usize,
    generation: u32,
}

impl Arena {
    /// Creates an empty arena with no blocks allocated yet.
    pub fn new() -> Self {
        Arena { blocks: Vec::new(), cursor: 0, generation: 0 }
    }

    /// Allocates room for `count` values of `T`, copies them in, and
    /// returns a handle to the stored slice. `T` must be `Copy` so the
    /// arena never needs to run destructors on `reset`.
    pub fn alloc_slice<T: Copy>(&mut self, values: &[T]) -> Handle<T> {
        let count = values.len();
        let align = std::mem::align_of::<T>();
        let needed = std::mem::size_of_val(values);

        // Ensure the current block (if any) has enough aligned room;
        // otherwise start a fresh block sized to comfortably fit this
        // allocation plus up to `align - 1` bytes of padding.
        loop {
            if let Some(block) = self.blocks.last() {
                let base = block.as_ptr() as usize + self.cursor;
                let padding = (align - (base % align)) % align;
                if self.cursor + padding + needed <= block.len() {
                    break;
                }
            }
            let block_len = BLOCK_SIZE.max(needed + align);
            self.blocks.push(vec![0u8; block_len]);
            self.cursor = 0;
        }

        let block_idx = self.blocks.len() - 1;
        let block = self.blocks.last_mut().unwrap();
        let base = block.as_ptr() as usize + self.cursor;
        let padding = (align - (base % align)) % align;
        let start = self.cursor + padding;

        if count > 0 {
            // SAFETY: `start` is `self.cursor` rounded up to `align_of::<T>()`,
            // and the loop above only exits once `self.cursor + padding +
            // needed <= block.len()` holds for this exact `block`, so
            // `start + needed <= block.len()`. The resulting `&mut [T]` is
            // properly aligned (by construction of `start`) and points at
            // `needed = count * size_of::<T>()` bytes fully inside `block`,
            // so writing `count` `T` values into it stays in bounds. No
            // other live reference to this byte range exists: it was never
            // handed out before this call.
            unsafe {
                let dst = block.as_mut_ptr().add(start) as *mut T;
                let dst_slice = std::slice::from_raw_parts_mut(dst, count);
                dst_slice.copy_from_slice(values);
            }
        }
        self.cursor = start + needed;

        Handle {
            block: block_idx as u32,
            offset: start as u32,
            len: count as u32,
            generation: self.generation,
            _marker: std::marker::PhantomData,
        }
    }

    /// Resolves a handle back into a slice, or `None` if it was issued
    /// before the most recent [`Arena::reset`].
    pub fn get<T: Copy>(&self, handle: Handle<T>) -> Option<&[T]> {
        if handle.generation != self.generation {
            return None;
        }
        let block = self.blocks.get(handle.block as usize)?;
        let start = handle.offset as usize;
        let len = handle.len as usize;
        let byte_len = len * std::mem::size_of::<T>();
        if start + byte_len > block.len() {
            return None;
        }
        // SAFETY: the handle's generation matches, so `block`/`start`/`len`
        // were produced by `alloc_slice` on this same arena generation at
        // the byte range `[start, start + byte_len)`, which the check above
        // confirms fits within `block`. That range was written with exactly
        // `len` valid `T` values (copied from a `&[T]` in `alloc_slice`) and
        // is never mutated afterwards (the arena only appends new blocks or
        // resets entirely), so reinterpreting it as `&[T]` for the
        // lifetime of this `&self` borrow is sound.
        Some(unsafe { std::slice::from_raw_parts(block.as_ptr().add(start) as *const T, len) })
    }

    /// Drops all allocated blocks and invalidates every previously issued
    /// handle by bumping the generation counter.
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.cursor = 0;
        self.generation = self.generation.wrapping_add(1);
    }

    /// Total bytes considered committed: every byte of every block that has
    /// been fully abandoned (moved past), plus the cursor position within
    /// the current block. Any unused tail of an abandoned block counts as
    /// used, since a bump arena never reclaims it before `reset`.
    pub fn bytes_used(&self) -> usize {
        if self.blocks.is_empty() {
            return 0;
        }
        let earlier: usize = self.blocks[..self.blocks.len() - 1].iter().map(|b| b.len()).sum();
        earlier + self.cursor
    }
}

impl Default for Arena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_and_get_round_trip() {
        let mut arena = Arena::new();
        let handle = arena.alloc_slice(&[1u32, 2, 3, 4]);
        assert_eq!(arena.get(handle), Some(&[1u32, 2, 3, 4][..]));
    }

    #[test]
    fn multiple_allocations_are_independent() {
        let mut arena = Arena::new();
        let a = arena.alloc_slice(&[1i32, 2, 3]);
        let b = arena.alloc_slice(&[9i32, 8]);
        assert_eq!(arena.get(a), Some(&[1, 2, 3][..]));
        assert_eq!(arena.get(b), Some(&[9, 8][..]));
    }

    #[test]
    fn reset_invalidates_old_handles() {
        let mut arena = Arena::new();
        let handle = arena.alloc_slice(&[42u8; 8]);
        assert!(arena.get(handle).is_some());
        arena.reset();
        assert_eq!(arena.get(handle), None);
        assert_eq!(arena.bytes_used(), 0);
    }

    #[test]
    fn allocation_spanning_block_boundary_gets_new_block() {
        let mut arena = Arena::new();
        let big = vec![7u8; BLOCK_SIZE - 16];
        let a = arena.alloc_slice(&big);
        let b = arena.alloc_slice(&[1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);
        assert_eq!(arena.get(a).unwrap(), &big[..]);
        assert_eq!(arena.get(b).unwrap().len(), 32);
    }

    #[test]
    fn empty_slice_alloc_is_well_defined() {
        let mut arena = Arena::new();
        let handle: Handle<u32> = arena.alloc_slice(&[]);
        assert_eq!(arena.get(handle), Some(&[][..]));
    }

    #[test]
    fn bytes_used_grows_with_allocations() {
        let mut arena = Arena::new();
        assert_eq!(arena.bytes_used(), 0);
        arena.alloc_slice(&[1u64, 2, 3]);
        assert!(arena.bytes_used() >= 24);
    }
}
