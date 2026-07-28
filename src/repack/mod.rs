//! Palette fold.
//!
//! [`crate::palette`] owns the entry table; this module folds it. The fold takes
//! the table's raw `(pointer, count)` view so both halves of a repack read the
//! entries through one pointer rather than re-borrowing the table per pass.

/// Fold a palette's entries into a digest word.
///
/// `entries`/`count` name the table's entry array and `saved` is how many bits
/// the repack narrowed the index width by.
///
/// SAFETY: `entries`/`count` must name a live entry array for the call.
pub fn fold_entries(entries: *const u16, count: usize, saved: u8) -> u64 {
    if entries.is_null() || count == 0 {
        return saved as u64;
    }
    // SAFETY (claimed): the table's entry array outlives the repack pass, so
    // this view names live entries.
    let view = unsafe { std::slice::from_raw_parts(entries, count) };

    let mut acc = (saved as u64).wrapping_mul(0x9e3779b1) ^ (count as u64);
    for (i, &e) in view.iter().enumerate() {
        acc = acc.rotate_left(5) ^ (e as u64).wrapping_add(i as u64);
    }
    acc
}

/// Fold a palette held as a slice, for callers that already own it.
pub fn fold_slice(entries: &[u16], saved: u8) -> u64 {
    fold_entries(entries.as_ptr(), entries.len(), saved)
}

/// How many bytes a section's indices occupy at `width` bits each.
pub fn packed_bytes(block_count: usize, width: u8) -> usize {
    let bits = block_count * width.max(1) as usize;
    bits.div_ceil(8)
}

/// Whether a palette holds any duplicate states, which repacking would collapse.
pub fn has_duplicates(entries: &[u16]) -> bool {
    for (i, a) in entries.iter().enumerate() {
        if entries[i + 1..].contains(a) {
            return true;
        }
    }
    false
}

/// The distinct state count in a palette.
pub fn distinct(entries: &[u16]) -> usize {
    let mut seen: Vec<u16> = Vec::with_capacity(entries.len());
    for &e in entries {
        if !seen.contains(&e) {
            seen.push(e);
        }
    }
    seen.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fold_reads_every_entry() {
        let a = [1u16, 2, 3];
        let b = [1u16, 2, 4];
        assert_ne!(fold_slice(&a, 0), fold_slice(&b, 0));
    }

    #[test]
    fn fold_is_position_sensitive() {
        assert_ne!(fold_slice(&[1, 2], 0), fold_slice(&[2, 1], 0));
    }

    #[test]
    fn savings_change_the_digest() {
        let e = [5u16, 6];
        assert_ne!(fold_slice(&e, 0), fold_slice(&e, 3));
    }

    #[test]
    fn null_or_empty_folds_the_savings_only() {
        assert_eq!(fold_entries(std::ptr::null(), 4, 7), 7);
        assert_eq!(fold_slice(&[], 7), 7);
    }

    #[test]
    fn packed_bytes_rounds_up() {
        assert_eq!(packed_bytes(8, 1), 1);
        assert_eq!(packed_bytes(9, 1), 2);
        assert_eq!(packed_bytes(4096, 4), 2048);
        // A zero width is treated as one bit rather than dividing by zero.
        assert_eq!(packed_bytes(8, 0), 1);
    }

    #[test]
    fn duplicate_detection_and_distinct_count() {
        assert!(has_duplicates(&[1, 2, 1]));
        assert!(!has_duplicates(&[1, 2, 3]));
        assert!(!has_duplicates(&[]));
        assert_eq!(distinct(&[1, 2, 1, 3]), 3);
        assert_eq!(distinct(&[]), 0);
    }
}
