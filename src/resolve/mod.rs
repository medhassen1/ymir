//! Property rendering.
//!
//! [`crate::tile`] owns the name store; this module formats a property against
//! it. The renderer takes the store's cursor and a property's name span rather
//! than a borrowed `&str`, so a whole tree renders through one pointer instead
//! of re-borrowing the store per property.

use crate::tile::{Prop, ValueKind};

/// Fold one property into a digest word, reading its name through `names`.
///
/// SAFETY: `names` must address a name store in which
/// `prop.name_at .. prop.name_at + prop.name_len` is live for the call.
pub fn render_prop(names: *const u8, prop: &Prop) -> u64 {
    let mut acc = kind_seed(prop.kind) ^ (prop.depth as u64).rotate_left(11);
    if !names.is_null() && prop.name_len != 0 {
        // SAFETY (claimed): the span was produced by the same store the cursor
        // names, so it lies inside that store's backing bytes.
        let name = unsafe { std::slice::from_raw_parts(names.add(prop.name_at), prop.name_len) };
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

/// Fold a whole tree's properties in order.
pub fn render_all(names: *const u8, props: &[Prop]) -> u64 {
    let mut acc = 0xffu64;
    for p in props {
        acc = acc.wrapping_mul(0x100000001b3) ^ render_prop(names, p);
    }
    acc
}

/// The total name bytes a property set refers to, counting repeats.
pub fn name_footprint(props: &[Prop]) -> usize {
    props.iter().map(|p| p.name_len).sum()
}

/// The highest name offset any property refers to, which is the minimum store
/// size a renderer needs to be able to service the set.
pub fn required_bytes(props: &[Prop]) -> usize {
    props.iter().map(|p| p.name_at + p.name_len).max().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tile::NameStore;

    fn prop(at: usize, len: usize, kind: ValueKind, value: i32, depth: u32) -> Prop {
        Prop { name_at: at, name_len: len, kind, value, depth }
    }

    #[test]
    fn render_reads_the_name_span() {
        let mut s = NameStore::new();
        let (o, l) = s.intern("Items");
        let p = prop(o, l, ValueKind::Int, 5, 0);
        let a = render_prop(s.cursor(), &p);

        let mut t = NameStore::new();
        let (o2, l2) = t.intern("Other");
        let q = prop(o2, l2, ValueKind::Int, 5, 0);
        assert_ne!(a, render_prop(t.cursor(), &q));
    }

    #[test]
    fn render_separates_kinds() {
        let s = NameStore::new();
        let a = prop(0, 0, ValueKind::Int, 1, 0);
        let b = prop(0, 0, ValueKind::List, 1, 0);
        assert_ne!(render_prop(s.cursor(), &a), render_prop(s.cursor(), &b));
    }

    #[test]
    fn render_separates_depth_and_value() {
        let s = NameStore::new();
        let a = prop(0, 0, ValueKind::Int, 1, 0);
        let b = prop(0, 0, ValueKind::Int, 1, 1);
        let c = prop(0, 0, ValueKind::Int, 2, 0);
        assert_ne!(render_prop(s.cursor(), &a), render_prop(s.cursor(), &b));
        assert_ne!(render_prop(s.cursor(), &a), render_prop(s.cursor(), &c));
    }

    #[test]
    fn null_cursor_folds_the_value_only() {
        let p = prop(0, 4, ValueKind::Int, 9, 0);
        let a = render_prop(std::ptr::null(), &p);
        let empty = prop(0, 0, ValueKind::Int, 9, 0);
        assert_eq!(a, render_prop(std::ptr::null(), &empty));
    }

    #[test]
    fn render_all_is_order_sensitive() {
        let s = NameStore::new();
        let a = prop(0, 0, ValueKind::Int, 1, 0);
        let b = prop(0, 0, ValueKind::Int, 2, 0);
        assert_ne!(render_all(s.cursor(), &[a, b]), render_all(s.cursor(), &[b, a]));
    }

    #[test]
    fn footprint_and_required_bytes() {
        let props = [prop(0, 3, ValueKind::Int, 0, 0), prop(3, 5, ValueKind::Int, 0, 0)];
        assert_eq!(name_footprint(&props), 8);
        assert_eq!(required_bytes(&props), 8);
        assert_eq!(required_bytes(&[]), 0);
    }
}
