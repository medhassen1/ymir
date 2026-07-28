//! Tile-entity property trees.
//!
//! A tile entity — a chest, a sign, a spawner — carries a small nested tree of
//! named properties. Names repeat heavily across a region (every chest has an
//! `Items` key), so names are interned once and properties refer to them by a
//! `(pointer, length)` span rather than copying the text per property.
//!
//! The interner is region-lifetime: built once for the whole `tile` section
//! rather than once per tree, so a span handed out while decoding one property
//! stays meaningful to read back once every tree has been walked. Names are
//! appended into fixed-size chunks (see [`NameArena`]) rather than one
//! ever-growing string, and the store bounds its resident memory the way a
//! buffer pool reclaims cold pages: once accumulated name bytes cross a
//! threshold, [`NameArena::compact`] drops the oldest chunks, because a region
//! naming many distinct properties would otherwise keep every name resident for
//! the whole decode no matter how many trees it carries. [`decode_region`]
//! renders every property's name only once the whole section has been decoded.

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

/// Bytes held by one [`NameArena`] chunk.
///
/// An ordinary property name is a handful of bytes, so a typical commit never
/// has to look past the chunk it lands in.
const ARENA_CHUNK_BYTES: usize = 512;

/// Accumulated resident bytes across an arena's live chunks that triggers
/// [`NameArena::compact`].
///
/// An ordinary region's tile section — a modest set of recurring property
/// names — never approaches this. A region naming many distinct properties
/// does, which is exactly the case the bound exists to catch: without it, a
/// region-lifetime arena would keep every name resident for the whole decode
/// no matter how many distinct names the section carries.
const COMPACT_THRESHOLD_BYTES: usize = 2048;

/// A region-lifetime arena for interned property name bytes.
///
/// Built once per `tile` section rather than once per tree, so a span handed
/// out while decoding one property stays valid while later properties are
/// decoded — which is what lets [`Prop::name_ptr`] be read again well after
/// its own property has finished. Bytes are appended into fixed-size chunks,
/// each stored as an exact-sized boxed slice; a chunk with no room left for
/// the next commit is left as-is and a fresh one takes over, so a single name
/// is never split across two chunks.
struct NameArena {
    /// Chunks holding committed name bytes, oldest first.
    chunks: Vec<Box<[u8]>>,
    /// Bytes already written into the last chunk.
    used: usize,
    /// Bytes held across all currently resident chunks.
    resident: usize,
}

impl NameArena {
    fn new() -> NameArena {
        NameArena { chunks: Vec::new(), used: 0, resident: 0 }
    }

    /// Commit `bytes` into the arena and return a pointer to where they
    /// landed.
    ///
    /// If what remains of the current chunk cannot hold `bytes`, a fresh chunk
    /// takes over first, so the returned pointer's `bytes.len()` bytes are
    /// always contiguous — addressing live memory for as long as the chunk
    /// backing them stays resident (see [`NameArena::compact`]).
    fn commit(&mut self, bytes: &[u8]) -> *const u8 {
        let len = bytes.len();
        let fits_current = self.chunks.last().is_some_and(|c| self.used + len <= c.len());
        if !fits_current {
            let cap = len.max(ARENA_CHUNK_BYTES);
            self.chunks.push(vec![0u8; cap].into_boxed_slice());
            self.used = 0;
            self.resident += cap;
        }
        let chunk = self.chunks.last_mut().expect("a chunk was just ensured above");
        chunk[self.used..self.used + len].copy_from_slice(bytes);
        // SAFETY: `chunk` is a live `Box<[u8]>` at least `self.used + len`
        // bytes long — either it already fit `bytes` past `self.used`, or a
        // chunk sized to hold at least `bytes` was just pushed — so this
        // offset and the `len` bytes from it lie inside the allocation.
        let ptr = unsafe { chunk.as_ptr().add(self.used) };
        self.used += len;
        ptr
    }

    /// Drop the oldest resident chunks until the arena's accumulated bytes
    /// fall back to `threshold`, or only the chunk currently being written to
    /// is left.
    ///
    /// This is the arena's memory bound: left unchecked, a region-lifetime
    /// arena would keep every name resident for the whole decode no matter
    /// how many distinct properties the section carries. The chunk currently
    /// being written to is never dropped, since the next commit needs
    /// somewhere to land.
    fn compact(&mut self, threshold: usize) {
        while self.resident > threshold && self.chunks.len() > 1 {
            let oldest = self.chunks.remove(0);
            self.resident -= oldest.len();
        }
    }
}

/// An interner over the region-lifetime [`NameArena`].
///
/// Deduplication is checked against `known`, a set of owned copies kept
/// entirely separate from the arena — so a lookup never depends on whether an
/// earlier name's arena chunk is still resident, only the arena-backed span it
/// hands back does.
pub struct NameStore {
    arena: NameArena,
    known: Vec<(String, *const u8, usize)>,
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
    /// once regardless of how many of the arena's chunks are still resident.
    pub fn bytes(&self) -> usize {
        self.known.iter().map(|(s, _, _)| s.len()).sum()
    }

    /// Intern `name`, returning its `(pointer, len)` span into the arena. An
    /// identical name that is already present is shared rather than committed
    /// again.
    pub fn intern(&mut self, name: &str) -> (*const u8, usize) {
        if let Some(&(_, ptr, len)) = self.known.iter().find(|(k, _, _)| k == name) {
            return (ptr, len);
        }
        let ptr = self.arena.commit(name.as_bytes());
        let len = name.len();
        self.known.push((name.to_string(), ptr, len));
        // Bound the arena's resident memory now that this name is safely
        // committed.
        self.arena.compact(COMPACT_THRESHOLD_BYTES);
        (ptr, len)
    }

    /// The interned name at a span, for tests and diagnostics.
    ///
    /// SAFETY: `ptr`/`len` must still address live bytes in the arena — true
    /// for a span read back before enough further names have been interned to
    /// compact away the chunk holding it.
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

        tree.props.push(Prop { name_ptr, name_len, kind, value, depth });

        if kind == ValueKind::Compound {
            decode_subtree(c, tree, depth + 1, budget);
        }
    }
}

/// Decode the region's tile-entity trees and fold a digest of them.
///
/// Every property is decoded — and every name interned — before any of them
/// are rendered: the render pass is one closing loop over `tree.props`, read
/// only once the whole section has gone through the interner.
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
    decode_subtree(&mut c, &mut tree, 0, &mut budget);

    let mut acc = 0xffu64;
    for p in &tree.props {
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
        decode_subtree(&mut c, &mut tree, 0, &mut budget);
        assert_eq!(tree.props.len(), 16);
        assert_eq!(budget, 0);
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
        // SAFETY: neither chunk has been compacted away, so both pointers
        // still address the bytes just committed.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(a, 4), b"abcd");
            assert_eq!(std::slice::from_raw_parts(b, 2), b"ef");
        }
    }

    #[test]
    fn arena_starts_a_new_chunk_once_the_current_one_is_full() {
        let mut arena = NameArena::new();
        let filler = vec![0u8; ARENA_CHUNK_BYTES - 4];
        arena.commit(&filler);
        assert_eq!(arena.chunks.len(), 1);
        arena.commit(&[1, 2, 3, 4, 5]);
        assert_eq!(arena.chunks.len(), 2);
    }

    #[test]
    fn compact_leaves_a_small_arena_untouched() {
        let mut arena = NameArena::new();
        arena.commit(b"abc");
        arena.compact(COMPACT_THRESHOLD_BYTES);
        assert_eq!(arena.chunks.len(), 1);
    }

    #[test]
    fn compact_drops_oldest_chunks_once_over_threshold() {
        let mut arena = NameArena::new();
        for _ in 0..6 {
            arena.commit(&vec![0u8; ARENA_CHUNK_BYTES]);
        }
        assert_eq!(arena.chunks.len(), 6);
        arena.compact(3 * ARENA_CHUNK_BYTES);
        assert!(arena.chunks.len() < 6, "compact must drop some chunks");
        assert!(arena.resident <= 3 * ARENA_CHUNK_BYTES);
    }

    #[test]
    fn compact_never_drops_the_last_chunk() {
        let mut arena = NameArena::new();
        arena.commit(b"abc");
        arena.compact(0);
        assert_eq!(arena.chunks.len(), 1, "the chunk being written to must survive");
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
