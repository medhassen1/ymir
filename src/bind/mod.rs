//! Component interpretation.
//!
//! [`crate::entity`] owns the region-lifetime component arena; this module
//! knows how to read a component out of it. The reader takes a raw `(pointer,
//! len)` pair rather than a borrowed slice, so a region-wide fold walks every
//! retained component through one pointer instead of re-borrowing the arena
//! per entity.

use crate::entity::Kind;

/// Read one component and fold it into a digest word.
///
/// `cursor` addresses the component arena; `offset` and `len` locate the slot
/// inside it; `kind` fixes how the bytes are interpreted.
///
/// SAFETY: `cursor` must address an arena in which `offset .. offset + len` is
/// live and initialised for the duration of the call.
pub fn read_component(cursor: *const u8, offset: usize, len: usize, kind: Kind) -> u64 {
    if cursor.is_null() || len == 0 {
        return kind.tag();
    }
    // SAFETY: guaranteed by the precondition documented above.
    let bytes = unsafe { std::slice::from_raw_parts(cursor.add(offset), len) };
    match kind {
        Kind::Transform | Kind::Velocity => fold_triple(bytes) ^ kind.tag(),
        Kind::Health => fold_health(bytes) ^ kind.tag(),
        Kind::Inventory => fold_inventory(bytes) ^ kind.tag(),
        Kind::Blob => fold_opaque(bytes) ^ kind.tag(),
    }
}

/// Fold three big-endian `i32` axes.
fn fold_triple(bytes: &[u8]) -> u64 {
    let mut acc = 0x2545f491u64;
    for chunk in bytes.chunks_exact(4) {
        let v = i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        acc = acc.rotate_left(9) ^ (v as i64 as u64);
    }
    acc
}

/// Fold a current/maximum hit-point pair.
fn fold_health(bytes: &[u8]) -> u64 {
    if bytes.len() < 4 {
        return 0;
    }
    let cur = u16::from_be_bytes([bytes[0], bytes[1]]);
    let max = u16::from_be_bytes([bytes[2], bytes[3]]);
    ((cur as u64) << 16) | (max as u64).rotate_left(3)
}

/// Fold a variable-length inventory: a slot count then that many stacks.
fn fold_inventory(bytes: &[u8]) -> u64 {
    if bytes.is_empty() {
        return 0;
    }
    let slots = bytes[0] as usize;
    let mut acc = (slots as u64).wrapping_mul(0x9e3779b1);
    for (i, chunk) in bytes[1..].chunks_exact(4).enumerate().take(slots) {
        let stack = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        acc = acc.rotate_left(5) ^ (stack as u64).wrapping_add(i as u64);
    }
    acc
}

/// Fold an uninterpreted payload.
fn fold_opaque(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325u64, |a, &b| {
        (a ^ b as u64).wrapping_mul(0x100000001b3)
    })
}

/// How many bytes a slot of `kind` and wire length `len` actually occupies.
pub fn footprint(kind: Kind, len: usize) -> usize {
    match kind.stride() {
        Some(fixed) => fixed,
        None => len,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read(bytes: &[u8], kind: Kind) -> u64 {
        read_component(bytes.as_ptr(), 0, bytes.len(), kind)
    }

    #[test]
    fn null_or_empty_yields_the_kind_tag() {
        assert_eq!(read_component(std::ptr::null(), 0, 4, Kind::Health), Kind::Health.tag());
        assert_eq!(read(&[], Kind::Blob), Kind::Blob.tag());
    }

    #[test]
    fn transform_folds_three_axes() {
        let mut bytes = Vec::new();
        for v in [1i32, -2, 3] {
            bytes.extend_from_slice(&v.to_be_bytes());
        }
        let a = read(&bytes, Kind::Transform);
        // A different third axis must change the digest.
        bytes[11] = 4;
        assert_ne!(a, read(&bytes, Kind::Transform));
    }

    #[test]
    fn transform_and_velocity_differ_by_tag() {
        let bytes = [0u8; 12];
        assert_ne!(read(&bytes, Kind::Transform), read(&bytes, Kind::Velocity));
    }

    #[test]
    fn health_reads_current_and_max() {
        let bytes = [0x00, 0x0a, 0x00, 0x14];
        let a = read(&bytes, Kind::Health);
        let swapped = [0x00, 0x14, 0x00, 0x0a];
        assert_ne!(a, read(&swapped, Kind::Health));
    }

    #[test]
    fn health_shorter_than_four_bytes_is_zero_folded() {
        assert_eq!(read(&[1, 2], Kind::Health), Kind::Health.tag());
    }

    #[test]
    fn inventory_respects_its_slot_count() {
        // Two declared slots, three present: the third must be ignored.
        let mut bytes = vec![2u8];
        for s in [1u32, 2, 3] {
            bytes.extend_from_slice(&s.to_be_bytes());
        }
        let a = read(&bytes, Kind::Inventory);
        bytes[9..13].copy_from_slice(&99u32.to_be_bytes());
        assert_eq!(a, read(&bytes, Kind::Inventory));
    }

    #[test]
    fn footprint_prefers_the_fixed_stride() {
        assert_eq!(footprint(Kind::Transform, 999), 12);
        assert_eq!(footprint(Kind::Blob, 7), 7);
    }
}
