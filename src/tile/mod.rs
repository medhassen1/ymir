//! Tile-entity property trees.
//!
//! A tile entity — a chest, a sign, a spawner — carries a small nested tree of
//! named properties. Names repeat heavily across a region (every chest has an
//! `Items` key), so names are interned once and properties refer to them by a
//! `(pointer, length)` span rather than copying the text per property.
//!
//! The interner is region-lifetime: built once for the whole `tile` section
//! rather than once per tree, so a span handed out while decoding one property
//! stays meaningful to read back once every tree has been walked. Names live in
//! one contiguous run (see [`NameArena`]) rather than a chain of chunks, so a
//! name is always a single span and a lookup never has to care which block it
//! landed in. The run is grown against a load factor rather than allocated for
//! the worst case, which is what keeps a region naming a handful of properties
//! from reserving for one naming thousands. [`decode_region`] renders each
//! property as it is decoded, and reads a small deferred set of them once more
//! after the whole section has been walked.

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
    /// Start of the name inside the interner's region-lifetime arena.
    pub name_ptr: *const u8,
    /// Byte length of the name.
    pub name_len: usize,
    /// How the value is interpreted.
    pub kind: ValueKind,
    /// Scalar value, or the child count for a compound.
    pub value: i32,
    /// Nesting depth, used by the renderer to indent.
    pub depth: u32,
}

/// Bytes a [`NameArena`] reserves the first time anything is interned.
///
/// A tile section that names only the handful of properties every tile entity
/// carries never needs more than this.
const ARENA_INITIAL_BYTES: usize = 64;

/// Load factor, in eighths, the arena is grown at.
///
/// Growing before the run is actually full is what keeps the growth amortised:
/// a run filled to the brim would be relocated by the very next name.
const ARENA_LOAD_EIGHTHS: usize = 6;

/// Slots in [`DeferredSet`].
const DEFER_SLOTS: usize = 3;

/// A region-lifetime arena for interned property name bytes.
///
/// Built once per `tile` section rather than once per tree, so a span handed
/// out while decoding one property stays meaningful while later properties are
/// decoded — which is what lets [`Prop::name_ptr`] be read again well after its
/// own property has finished.
///
/// The arena is one contiguous run rather than a chain of chunks, so a name is
/// always a single span and a lookup never has to work out which block a name
/// landed in. Growing therefore means relocating: a larger run is allocated,
/// the bytes already interned are copied across, and the old run is released.
/// The load factor is what keeps that rare — the run is grown before it fills
/// and its size doubles, so the copies are amortised over the names interned.
struct NameArena {
    /// The interned name bytes, exactly as long as the run reserved.
    bytes: Box<[u8]>,
    /// Bytes already written.
    used: usize,
}

impl NameArena {
    fn new() -> NameArena {
        NameArena { bytes: Vec::new().into_boxed_slice(), used: 0 }
    }

    /// Bytes the arena's current run can hold.
    fn capacity(&self) -> usize {
        self.bytes.len()
    }

    /// Make sure `want` further bytes fit under the load factor, relocating the
    /// run to a larger one if they do not.
    fn reserve(&mut self, want: usize) {
        if (self.used + want) * 8 <= self.capacity() * ARENA_LOAD_EIGHTHS {
            return;
        }
        let mut cap = self.capacity().max(ARENA_INITIAL_BYTES / 2) * 2;
        while (self.used + want) * 8 > cap * ARENA_LOAD_EIGHTHS {
            cap *= 2;
        }
        let mut grown = vec![0u8; cap].into_boxed_slice();
        grown[..self.used].copy_from_slice(&self.bytes[..self.used]);
        // Releases the run the names were copied out of.
        self.bytes = grown;
    }

    /// Commit `bytes` into the arena and return the offset they landed at.
    fn commit(&mut self, bytes: &[u8]) -> usize {
        self.reserve(bytes.len());
        let at = self.used;
        self.bytes[at..at + bytes.len()].copy_from_slice(bytes);
        self.used = at + bytes.len();
        at
    }

    /// The address `at` bytes into the arena's current run.
    fn address(&self, at: usize) -> *const u8 {
        // SAFETY: `at` is an offset this arena handed out, so it is at most
        // `self.used`, which is at most the run's length — and offsetting a
        // live allocation by at most its own length is in bounds.
        unsafe { self.bytes.as_ptr().add(at) }
    }
}

/// An interner over the region-lifetime [`NameArena`].
///
/// Deduplication is checked against `known`, a set of owned copies kept
/// entirely separate from the arena, alongside each name's offset into the run.
/// Holding offsets rather than addresses is what lets the store keep answering
/// correctly across a growth: a lookup resolves against wherever the run
/// currently lives.
pub struct NameStore {
    arena: NameArena,
    known: Vec<(String, usize, usize)>,
}

impl NameStore {
    /// An empty store.
    pub fn new() -> NameStore {
        NameStore { arena: NameArena::new(), known: Vec::new() }
    }

    /// How many distinct names are interned.
    pub fn len(&self) -> usize {
        self.known.len()
    }

    /// Whether nothing has been interned yet.
    pub fn is_empty(&self) -> bool {
        self.known.is_empty()
    }

    /// Bytes of name payload interned so far, counting every distinct name
    /// once regardless of how large a run the arena currently holds.
    pub fn bytes(&self) -> usize {
        self.known.iter().map(|(s, _, _)| s.len()).sum()
    }

    /// Bytes the arena's run currently reserves.
    pub fn reserved(&self) -> usize {
        self.arena.capacity()
    }

    /// Intern `name`, returning its `(pointer, len)` span into the arena. An
    /// identical name that is already present is shared rather than committed
    /// again.
    pub fn intern(&mut self, name: &str) -> (*const u8, usize) {
        if let Some(&(_, at, len)) = self.known.iter().find(|(k, _, _)| k == name) {
            return (self.arena.address(at), len);
        }
        let at = self.arena.commit(name.as_bytes());
        self.known.push((name.to_string(), at, name.len()));
        (self.arena.address(at), name.len())
    }

    /// The interned name at a span, for tests and diagnostics.
    ///
    /// SAFETY: `ptr`/`len` must still address live bytes in the arena — true
    /// for a span read back before the arena has grown past the run it was
    /// handed out of.
    pub fn get(&self, ptr: *const u8, len: usize) -> &str {
        if ptr.is_null() || len == 0 {
            return "";
        }
        // SAFETY: guaranteed by the precondition documented above.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        std::str::from_utf8(bytes).unwrap_or("")
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

/// The properties [`decode_region`]'s closing pass renders a second time.
///
/// Rendering a property once as it is decoded and once at the end is what gives
/// the digest a term that depends on the section as a whole rather than on each
/// tree in isolation. Fixed capacity, reused round-robin: a region may carry
/// thousands of properties and the closing pass has to cost the same on all of
/// them, so the newest displace the oldest instead of the set growing with the
/// section.
struct DeferredSet {
    slots: [Option<Prop>; DEFER_SLOTS],
    next: usize,
}

impl DeferredSet {
    fn new() -> DeferredSet {
        DeferredSet { slots: [None; DEFER_SLOTS], next: 0 }
    }

    /// Put `prop` in the next slot, displacing whatever that slot held.
    fn defer(&mut self, prop: Prop) {
        self.slots[self.next] = Some(prop);
        self.next = (self.next + 1) % DEFER_SLOTS;
    }

    /// The properties currently held, oldest slot first.
    fn props(&self) -> impl Iterator<Item = &Prop> {
        self.slots.iter().flatten()
    }
}

/// The digest state a decode threads through the tree.
struct Fold {
    acc: u64,
    deferred: DeferredSet,
}

/// Decode one property subtree, recursing into compounds.
fn decode_subtree(c: &mut Cursor, tree: &mut Tree, depth: u32, budget: &mut usize, fold: &mut Fold) {
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
        // Names are free-form UTF-8, not just the printable ASCII subset: a
        // mod's display name for a tile entity may carry arbitrary Unicode, so
        // decoding lossily (rather than substituting every non-graphic byte)
        // is what lets that name round-trip instead of turning to underscores.
        let name = String::from_utf8_lossy(raw).into_owned();
        let (name_ptr, name_len) = tree.names.intern(&name);

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

        let prop = Prop { name_ptr, name_len, kind, value, depth };
        tree.props.push(prop);
        // Render the property here, against the span the interner just handed
        // back, and offer it to the deferred set for the closing pass.
        fold.acc = fold.acc.wrapping_mul(0x100000001b3) ^ resolve::render_prop(&prop);
        fold.deferred.defer(prop);

        if kind == ValueKind::Compound {
            decode_subtree(c, tree, depth + 1, budget, fold);
        }
    }
}

/// Decode the region's tile-entity trees and fold a digest of them.
///
/// Each property is rendered as it is decoded, and a small deferred set of them
/// is rendered once more after the whole section has gone through the interner
/// — giving the digest a term that covers the section as a whole rather than
/// each tree on its own.
pub fn decode_region(region: &Region) -> u64 {
    let data = region.slice(region.tile);
    if data.is_empty() {
        return 0;
    }
    let mut c = Cursor::new(data);
    let mut tree = Tree { props: Vec::new(), names: NameStore::new() };
    let mut budget = MAX_PROPS;
    let mut fold = Fold { acc: 0xffu64, deferred: DeferredSet::new() };

    // Seed the store with the names every tile entity carries, so the common
    // case interns nothing further while the tree is walked.
    for common in ["id", "x", "y", "z", "Items"] {
        tree.names.intern(common);
    }
    decode_subtree(&mut c, &mut tree, 0, &mut budget, &mut fold);

    let mut acc = fold.acc;
    for p in fold.deferred.props() {
        acc = acc.wrapping_mul(0x100000001b3) ^ resolve::render_prop(p);
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
        assert_ne!(a.0, b.0);
        assert_eq!(s.get(a.0, a.1), "x");
        assert_eq!(s.get(b.0, b.1), "y");
        assert_eq!(s.len(), 2);
    }

    #[test]
    fn interner_get_is_bounds_safe() {
        let s = NameStore::new();
        assert_eq!(s.get(std::ptr::null(), 5), "");
    }

    #[test]
    fn interner_round_trips_ascii_names() {
        let mut s = NameStore::new();
        let (ptr, len) = s.intern("Inventory");
        assert_eq!(len, "Inventory".len());
        assert_eq!(s.get(ptr, len), "Inventory");
    }

    #[test]
    fn interner_round_trips_non_ascii_names() {
        // A name whose char count differs from its byte count must still
        // round-trip in full — the interner commits raw bytes, never a
        // truncated character count.
        let mut s = NameStore::new();
        let name = "caf\u{e9}\u{2603}"; // "café☃": 5 chars, 7 bytes
        let (ptr, len) = s.intern(name);
        assert_eq!(len, name.len());
        assert_eq!(s.get(ptr, len), name);
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
        let mut fold = Fold { acc: 0xffu64, deferred: DeferredSet::new() };
        decode_subtree(&mut c, &mut tree, 0, &mut budget, &mut fold);
        assert_eq!(tree.props.len(), 16);
        assert_eq!(budget, 0);
        assert_eq!(fold.deferred.props().count(), DEFER_SLOTS);
    }

    #[test]
    fn tree_reports_depth_and_compounds() {
        let mut names = NameStore::new();
        let (ptr, len) = names.intern("a");
        let tree = Tree {
            props: vec![
                Prop { name_ptr: ptr, name_len: len, kind: ValueKind::Compound, value: 0, depth: 0 },
                Prop { name_ptr: ptr, name_len: len, kind: ValueKind::Int, value: 1, depth: 3 },
            ],
            names,
        };
        assert_eq!(tree.max_depth(), 3);
        assert_eq!(tree.compound_count(), 1);
    }

    #[test]
    fn arena_commit_writes_are_readable_back() {
        let mut arena = NameArena::new();
        let a = arena.commit(b"abcd");
        let b = arena.commit(b"ef");
        // SAFETY: the arena has not grown since either offset was handed out,
        // so both address the bytes just committed.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(arena.address(a), 4), b"abcd");
            assert_eq!(std::slice::from_raw_parts(arena.address(b), 2), b"ef");
        }
    }

    #[test]
    fn arena_reserves_its_initial_run_on_first_use() {
        let mut arena = NameArena::new();
        assert_eq!(arena.capacity(), 0, "an unused arena reserves nothing");
        arena.commit(b"abc");
        assert_eq!(arena.capacity(), ARENA_INITIAL_BYTES);
    }

    #[test]
    fn arena_grows_at_the_load_factor_and_keeps_its_bytes() {
        let mut arena = NameArena::new();
        // Fill to just under the load factor: 6/8 of 64 bytes is 48.
        let a = arena.commit(&[b'x'; 48]);
        assert_eq!(arena.capacity(), ARENA_INITIAL_BYTES);
        let b = arena.commit(b"yz");
        assert!(arena.capacity() > ARENA_INITIAL_BYTES, "the run must grow");
        // SAFETY: both offsets are resolved against the arena's current run,
        // which is the one the growth just moved the bytes into.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(arena.address(a), 48), vec![b'x'; 48]);
            assert_eq!(std::slice::from_raw_parts(arena.address(b), 2), b"yz");
        }
    }

    #[test]
    fn interned_names_survive_a_growth() {
        let mut s = NameStore::new();
        let early = s.intern("first-name-interned");
        assert_eq!(s.get(early.0, early.1), "first-name-interned");
        // Enough further names to force the run to be relocated.
        for i in 0..40 {
            s.intern(&format!("property-number-{i:03}"));
        }
        assert!(s.reserved() > ARENA_INITIAL_BYTES, "the run must have grown");
        // Resolved through the store, an interned name is still readable.
        let again = s.intern("first-name-interned");
        assert_eq!(s.get(again.0, again.1), "first-name-interned");
    }

    #[test]
    fn deferred_set_holds_only_its_newest_slots() {
        let mut d = DeferredSet::new();
        for value in 0..DEFER_SLOTS as i32 + 3 {
            d.defer(Prop {
                name_ptr: std::ptr::null(),
                name_len: 0,
                kind: ValueKind::Int,
                value,
                depth: 0,
            });
        }
        assert_eq!(d.props().count(), DEFER_SLOTS, "capacity must be fixed");
    }

    #[test]
    fn decode_region_is_deterministic() {
        use crate::format::*;

        fn region_with_tile(tile: &[u8]) -> Vec<u8> {
            let mut v = Vec::new();
            v.extend_from_slice(&MAGIC);
            v.extend_from_slice(&VERSION.to_be_bytes());
            v.extend_from_slice(&flag::TILE.to_be_bytes());
            v.extend_from_slice(&0i16.to_be_bytes());
            v.extend_from_slice(&0i16.to_be_bytes());
            v.extend_from_slice(&1u16.to_be_bytes());
            v.extend_from_slice(&3u16.to_be_bytes());
            v.extend_from_slice(&0x5EEDu32.to_be_bytes());
            v.extend_from_slice(&64u16.to_be_bytes());
            v.push(4);
            v.push(3);
            v.extend_from_slice(&0u16.to_be_bytes());
            assert_eq!(v.len(), HEADER_LEN);

            let dir_end = HEADER_LEN + 3 * DIR_ENTRY;
            let cmap_off = dir_end;
            let cmap_len = 2 * 4;
            let cdat_off = cmap_off + cmap_len;
            let cdat_len = 1usize;
            let tile_off = cdat_off + cdat_len;

            v.extend_from_slice(&tag::CMAP);
            v.extend_from_slice(&(cmap_off as u32).to_be_bytes());
            v.extend_from_slice(&(cmap_len as u32).to_be_bytes());
            v.extend_from_slice(&tag::CDAT);
            v.extend_from_slice(&(cdat_off as u32).to_be_bytes());
            v.extend_from_slice(&(cdat_len as u32).to_be_bytes());
            v.extend_from_slice(&tag::TILE);
            v.extend_from_slice(&(tile_off as u32).to_be_bytes());
            v.extend_from_slice(&(tile.len() as u32).to_be_bytes());

            v.extend_from_slice(&0u32.to_be_bytes());
            v.extend_from_slice(&0u32.to_be_bytes());
            v.push(0);
            v.extend_from_slice(tile);
            v
        }

        let mut tile = 3u16.to_be_bytes().to_vec();
        for (name, val) in [("x", 3i32), ("y", 71), ("z", -2)] {
            tile.push(0); // Int
            tile.push(name.len() as u8);
            tile.extend_from_slice(name.as_bytes());
            tile.extend_from_slice(&val.to_be_bytes());
        }
        let data = region_with_tile(&tile);
        let region = crate::parse::parse(&data).expect("valid region");
        assert_eq!(decode_region(&region), decode_region(&region));
    }
}
