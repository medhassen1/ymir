//! Tile-entity property trees.
//!
//! A tile entity — a chest, a sign, a spawner — carries a small nested tree of
//! named properties. Names repeat heavily across a region (every chest has an
//! `Items` key), so names are interned into one contiguous backing store and
//! properties refer to them by span. The renderer reads a name straight out of
//! that store rather than copying it per property.

use crate::common::*;
use crate::parse::Region;
use crate::reader::Cursor;
use crate::resolve;

/// The type of a property value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValueKind {
    /// A signed 32-bit scalar.
    Int,
    /// A short UTF-8-ish byte string.
    Text,
    /// A nested subtree of further properties.
    Compound,
    /// A homogeneous list of scalars.
    List,
}

impl ValueKind {
    /// Decode a wire kind nibble.
    pub fn from_nibble(n: u8) -> ValueKind {
        match n & 3 {
            0 => ValueKind::Int,
            1 => ValueKind::Text,
            2 => ValueKind::Compound,
            _ => ValueKind::List,
        }
    }
}

/// One decoded property.
#[derive(Clone, Copy, Debug)]
pub struct Prop {
    /// Byte offset of the name inside the interner's backing store.
    pub name_at: usize,
    /// Byte length of the name.
    pub name_len: usize,
    /// How the value is interpreted.
    pub kind: ValueKind,
    /// Scalar value, or the child count for a compound.
    pub value: i32,
    /// Nesting depth, used by the renderer to indent.
    pub depth: u32,
}

/// A contiguous store of interned property names.
///
/// Names are appended to one `String` and referred to by span. Deduplicating
/// them matters: a region of a thousand chests holds one copy of `"Items"`.
pub struct NameStore {
    backing: String,
    spans: Vec<(usize, usize)>,
}

impl NameStore {
    /// An empty store.
    pub fn new() -> NameStore {
        NameStore { backing: String::new(), spans: Vec::new() }
    }

    /// How many distinct names are interned.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Whether nothing has been interned yet.
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    /// Bytes of name payload held.
    pub fn bytes(&self) -> usize {
        self.backing.len()
    }

    /// A cursor onto the backing store.
    ///
    /// The renderer reads names through this rather than re-borrowing the store
    /// for every property it formats.
    pub fn cursor(&self) -> *const u8 {
        self.backing.as_ptr()
    }

    /// Intern `name`, returning its `(offset, len)` span. An identical name that
    /// is already present is shared rather than appended again.
    pub fn intern(&mut self, name: &str) -> (usize, usize) {
        for &(off, len) in &self.spans {
            if &self.backing[off..off + len] == name {
                return (off, len);
            }
        }
        let off = self.backing.len();
        self.backing.push_str(name);
        let span = (off, name.len());
        self.spans.push(span);
        span
    }

    /// The interned name at a span, for tests and diagnostics.
    pub fn get(&self, off: usize, len: usize) -> &str {
        self.backing.get(off..off + len).unwrap_or("")
    }
}

impl Default for NameStore {
    fn default() -> Self {
        NameStore::new()
    }
}

/// A decoded tile-entity tree plus the names it refers into.
pub struct Tree {
    /// Every property, in depth-first order.
    pub props: Vec<Prop>,
    /// The names those properties refer into.
    pub names: NameStore,
}

impl Tree {
    /// The deepest nesting level reached.
    pub fn max_depth(&self) -> u32 {
        self.props.iter().map(|p| p.depth).max().unwrap_or(0)
    }

    /// How many properties are compounds.
    pub fn compound_count(&self) -> usize {
        self.props.iter().filter(|p| p.kind == ValueKind::Compound).count()
    }
}

/// Decode one property subtree, recursing into compounds.
fn decode_subtree(c: &mut Cursor, tree: &mut Tree, depth: u32, budget: &mut usize) {
    if depth > 8 || *budget == 0 {
        return;
    }
    let count = c.u16() as usize;
    if !c.ok {
        return;
    }
    for _ in 0..count.min(MAX_PROPS) {
        if *budget == 0 {
            return;
        }
        *budget -= 1;

        let header = c.u8();
        let name_len = (c.u8() as usize).min(64);
        let raw = c.bytes(name_len);
        if !c.ok {
            return;
        }
        let name: String = raw
            .iter()
            .map(|&b| if b.is_ascii_graphic() { b as char } else { '_' })
            .collect();
        let (name_at, name_len) = tree.names.intern(&name);

        let kind = ValueKind::from_nibble(header);
        let value = match kind {
            ValueKind::Int => c.i32(),
            ValueKind::Text => {
                let n = (c.u8() as usize).min(64);
                let body = c.bytes(n);
                body.iter().fold(0i32, |a, &b| a.wrapping_mul(31).wrapping_add(b as i32))
            }
            ValueKind::List => {
                let n = (c.u8() as usize).min(64);
                let mut sum = 0i32;
                for _ in 0..n {
                    sum = sum.wrapping_add(c.i32());
                }
                sum
            }
            ValueKind::Compound => 0,
        };

        tree.props.push(Prop { name_at, name_len, kind, value, depth });

        if kind == ValueKind::Compound {
            decode_subtree(c, tree, depth + 1, budget);
        }
    }
}

/// Decode the region's tile-entity trees and fold a digest of them.
///
/// The name cursor is taken once, before the trees are walked, so the render
/// pass reads every property's name through one pointer.
pub fn decode_region(region: &Region) -> u64 {
    let data = region.slice(region.tile);
    if data.is_empty() {
        return 0;
    }
    let mut c = Cursor::new(data);
    let mut tree = Tree { props: Vec::new(), names: NameStore::new() };
    let mut budget = MAX_PROPS;

    // Seed the store with the names every tile entity carries, so the common
    // case interns nothing further while the tree is walked.
    for common in ["id", "x", "y", "z", "Items"] {
        tree.names.intern(common);
    }
    // The name cursor for the whole render pass.
    let names = tree.names.cursor();

    decode_subtree(&mut c, &mut tree, 0, &mut budget);

    let mut acc = 0xffu64;
    for p in &tree.props {
        acc = acc.wrapping_mul(0x100000001b3) ^ resolve::render_prop(names, p);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_kinds_decode_from_the_low_nibble() {
        assert_eq!(ValueKind::from_nibble(0), ValueKind::Int);
        assert_eq!(ValueKind::from_nibble(1), ValueKind::Text);
        assert_eq!(ValueKind::from_nibble(2), ValueKind::Compound);
        assert_eq!(ValueKind::from_nibble(7), ValueKind::List);
    }

    #[test]
    fn interner_dedupes_equal_names() {
        let mut s = NameStore::new();
        let a = s.intern("Items");
        let b = s.intern("Items");
        assert_eq!(a, b);
        assert_eq!(s.len(), 1);
        assert_eq!(s.bytes(), 5);
    }

    #[test]
    fn interner_keeps_distinct_names_separate() {
        let mut s = NameStore::new();
        let a = s.intern("x");
        let b = s.intern("y");
        assert_ne!(a, b);
        assert_eq!(s.get(a.0, a.1), "x");
        assert_eq!(s.get(b.0, b.1), "y");
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn interner_get_is_bounds_safe() {
        let s = NameStore::new();
        assert_eq!(s.get(0, 5), "");
    }

    #[test]
    fn decode_subtree_respects_the_property_budget() {
        // A count far larger than the budget must still terminate.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1000u16.to_be_bytes());
        for _ in 0..1000 {
            bytes.push(0); // Int
            bytes.push(1); // name len
            bytes.push(b'k');
            bytes.extend_from_slice(&7i32.to_be_bytes());
        }
        let mut c = Cursor::new(&bytes);
        let mut tree = Tree { props: Vec::new(), names: NameStore::new() };
        let mut budget = 16usize;
        decode_subtree(&mut c, &mut tree, 0, &mut budget);
        assert_eq!(tree.props.len(), 16);
        assert_eq!(budget, 0);
    }

    #[test]
    fn tree_reports_depth_and_compounds() {
        let mut names = NameStore::new();
        let (o, l) = names.intern("a");
        let tree = Tree {
            props: vec![
                Prop { name_at: o, name_len: l, kind: ValueKind::Compound, value: 0, depth: 0 },
                Prop { name_at: o, name_len: l, kind: ValueKind::Int, value: 1, depth: 3 },
            ],
            names,
        };
        assert_eq!(tree.max_depth(), 3);
        assert_eq!(tree.compound_count(), 1);
    }
}
