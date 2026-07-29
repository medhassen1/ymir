//! Property rendering.
//!
//! [`crate::tile`] owns the region-lifetime name arena; this module formats a
//! property against it. The renderer takes a property's own `(pointer, len)`
//! name span rather than a borrowed `&str`, so a whole tree renders through
//! each property's own pointer instead of re-borrowing a name store.

use crate::tile::{Prop, ValueKind};

/// Fold one property into a digest word, reading its name through its own
/// span.
///
/// SAFETY: `prop.name_ptr` must address `prop.name_len` live bytes for the
/// call — true for a span read back before enough further names have been
/// interned to grow the name arena past the run it was handed out of.
pub fn render_prop(prop: &Prop) -> u64 {
    let mut acc = kind_seed(prop.kind) ^ (prop.depth as u64).rotate_left(11);
    if !prop.name_ptr.is_null() && prop.name_len != 0 {
        // SAFETY: guaranteed by the precondition documented above.
        let name = unsafe { std::slice::from_raw_parts(prop.name_ptr, prop.name_len) };
        for &b in name {
            acc = acc.rotate_left(5) ^ (b as u64);
        }
    }
    acc.wrapping_mul(0x100000001b3) ^ (prop.value as i64 as u64)
}

/// The digest seed for a value kind.
fn kind_seed(kind: ValueKind) -> u64 {
    match kind {
        ValueKind::Int => 0x2545f4914f6cdd1d,
        ValueKind::Text => 0x9e3779b97f4a7c15,
        ValueKind::Compound => 0xff51afd7ed558ccd,
        ValueKind::List => 0xc4ceb9fe1a85ec53,
    }
}

/// Fold a compound closed, once everything nested inside it has been decoded.
///
/// A compound's digest is the name it was opened under together with how much
/// it turned out to contain, which is only known on the way back out of the
/// subtree. `contained` is the digest the nested decode accumulated. The name is
/// read through the compound's own span, the same way [`render_prop`] reads it.
///
/// SAFETY: `prop.name_ptr` must address `prop.name_len` live bytes for the
/// call.
pub fn close_compound(prop: &Prop, contained: u64) -> u64 {
    let mut acc = contained.rotate_left(17) ^ kind_seed(prop.kind);
    if !prop.name_ptr.is_null() && prop.name_len != 0 {
        // SAFETY: guaranteed by the precondition documented above.
        let name = unsafe { std::slice::from_raw_parts(prop.name_ptr, prop.name_len) };
        for &b in name {
            acc = acc.rotate_left(3) ^ ((b as u64) << 7);
        }
    }
    acc ^ (prop.depth as u64).rotate_left(29)
}

/// Fold a whole tree's properties in order.
pub fn render_all(props: &[Prop]) -> u64 {
    let mut acc = 0xffu64;
    for p in props {
        acc = acc.wrapping_mul(0x100000001b3) ^ render_prop(p);
    }
    acc
}

/// The total name bytes a property set refers to, counting repeats.
pub fn name_footprint(props: &[Prop]) -> usize {
    props.iter().map(|p| p.name_len).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::NameStore;

    fn prop(ptr: *const u8, len: usize, kind: ValueKind, value: i32, depth: u32) -> Prop {
        Prop { name_ptr: ptr, name_len: len, kind, value, depth }
    }

    #[test]
    fn render_reads_the_name_span() {
        let mut s = NameStore::new();
        let (ptr, len) = s.intern("Items");
        let p = prop(ptr, len, ValueKind::Int, 5, 0);
        let a = render_prop(&p);

        let mut t = NameStore::new();
        let (ptr2, len2) = t.intern("Other");
        let q = prop(ptr2, len2, ValueKind::Int, 5, 0);
        assert_ne!(a, render_prop(&q));
    }

    #[test]
    fn render_separates_kinds() {
        let a = prop(std::ptr::null(), 0, ValueKind::Int, 1, 0);
        let b = prop(std::ptr::null(), 0, ValueKind::List, 1, 0);
        assert_ne!(render_prop(&a), render_prop(&b));
    }

    #[test]
    fn render_separates_depth_and_value() {
        let a = prop(std::ptr::null(), 0, ValueKind::Int, 1, 0);
        let b = prop(std::ptr::null(), 0, ValueKind::Int, 1, 1);
        let c = prop(std::ptr::null(), 0, ValueKind::Int, 2, 0);
        assert_ne!(render_prop(&a), render_prop(&b));
        assert_ne!(render_prop(&a), render_prop(&c));
    }

    #[test]
    fn null_cursor_folds_the_value_only() {
        let p = prop(std::ptr::null(), 4, ValueKind::Int, 9, 0);
        let a = render_prop(&p);
        let empty = prop(std::ptr::null(), 0, ValueKind::Int, 9, 0);
        assert_eq!(a, render_prop(&empty));
    }

    #[test]
    fn close_compound_covers_the_name_and_what_it_held() {
        let mut s = NameStore::new();
        let (ptr, len) = s.intern("Items");
        let p = prop(ptr, len, ValueKind::Compound, 0, 1);
        let base = close_compound(&p, 7);
        assert_ne!(base, close_compound(&p, 8), "contents must move the digest");
        let mut t = NameStore::new();
        let (other, other_len) = t.intern("Other");
        let q = prop(other, other_len, ValueKind::Compound, 0, 1);
        assert_ne!(base, close_compound(&q, 7), "the name must move it too");
        // An unnamed compound folds its contents and depth only.
        let anon = prop(std::ptr::null(), 4, ValueKind::Compound, 0, 1);
        assert_eq!(
            close_compound(&anon, 7),
            close_compound(&prop(std::ptr::null(), 0, ValueKind::Compound, 0, 1), 7)
        );
    }

    #[test]
    fn render_all_is_order_sensitive() {
        let a = prop(std::ptr::null(), 0, ValueKind::Int, 1, 0);
        let b = prop(std::ptr::null(), 0, ValueKind::Int, 2, 0);
        assert_ne!(render_all(&[a, b]), render_all(&[b, a]));
    }

    #[test]
    fn footprint_counts_repeats() {
        let props = [
            prop(std::ptr::null(), 3, ValueKind::Int, 0, 0),
            prop(std::ptr::null(), 5, ValueKind::Int, 0, 0),
        ];
        assert_eq!(name_footprint(&props), 8);
    }
}
