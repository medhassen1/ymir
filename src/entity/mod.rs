//! Entity component storage.
//!
//! Entities are stored structure-of-arrays style: every component's bytes live
//! in a region-lifetime arena, built once per load rather than once per entity,
//! so a handle captured while loading one record stays meaningful to read back
//! once the whole `ents` section has been walked. The arena appends into
//! fixed-size chunks (see [`ComponentArena`]) rather than one ever-growing
//! buffer, and bounds its resident memory the way a buffer pool reclaims cold
//! pages: once accumulated component bytes cross a threshold,
//! [`ComponentArena::compact`] drops the oldest chunks, because a region that
//! spawns entities heavily would otherwise keep every component resident for
//! the whole load no matter how many records it carries.
//!
//! Each distinct entity id's first spawn is captured as a [`Pending`] handle,
//! mirroring a caller that grabbed a reference to an entity's components right
//! after it appeared and held onto it rather than re-reading the store. A
//! respawn — the same id appearing again — retires that earlier incarnation
//! immediately rather than waiting for a later sweep to notice the repeat.

use crate::bind;
use crate::common::*;
use crate::parse::Region;
use crate::reader::Cursor;

/// The kind of a component, which fixes how its bytes are interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Position in world space: three `i32`.
    Transform,
    /// Linear velocity: three `i32`.
    Velocity,
    /// Item slots: a count followed by that many `u32` stacks.
    Inventory,
    /// Current and maximum hit points: two `u16`.
    Health,
    /// Opaque per-mod payload, carried but not interpreted.
    Blob,
}

impl Kind {
    /// Decode a wire kind byte.
    pub fn from_byte(b: u8) -> Kind {
        match b % 5 {
            0 => Kind::Transform,
            1 => Kind::Velocity,
            2 => Kind::Inventory,
            3 => Kind::Health,
            _ => Kind::Blob,
        }
    }

    /// The fixed byte width of this kind, or `None` when it is variable.
    pub fn stride(self) -> Option<usize> {
        match self {
            Kind::Transform | Kind::Velocity => Some(12),
            Kind::Health => Some(4),
            Kind::Inventory | Kind::Blob => None,
        }
    }

    /// A stable tag used when folding the digest.
    pub fn tag(self) -> u64 {
        match self {
            Kind::Transform => 0x11,
            Kind::Velocity => 0x22,
            Kind::Inventory => 0x33,
            Kind::Health => 0x44,
            Kind::Blob => 0x55,
        }
    }
}

/// Bytes held by one [`ComponentArena`] chunk.
///
/// An ordinary entity record's payload — at most 256 bytes for a variable-width
/// kind, far less for a fixed one — fits with room to spare, so a typical
/// commit never has to look past the chunk it lands in.
const ARENA_CHUNK_BYTES: usize = 1024;

/// Accumulated resident bytes across an arena's live chunks that triggers
/// [`ComponentArena::compact`].
///
/// An ordinary region's entity load — a modest cast, most carrying a small
/// fixed-width component — never approaches this. A region built from many
/// heavy variable-width components does, which is exactly the case the bound
/// exists to catch: without it, a region-lifetime arena would keep every
/// entity's bytes resident for the whole load no matter how many records the
/// region carries.
const COMPACT_THRESHOLD_BYTES: usize = 4096;

/// A region-lifetime arena for decoded component payload bytes.
///
/// Built once per load rather than once per entity, so a pointer handed out
/// while decoding one record stays valid while later records are decoded —
/// which is what lets a [`Pending`] handle be read again well after its own
/// record has finished. Bytes are appended into fixed-size chunks, each stored
/// as an exact-sized boxed slice; a chunk with no room left for the next commit
/// is left as-is and a fresh one takes over, so a single commit is never split
/// across two chunks.
pub(crate) struct ComponentArena {
    /// Chunks holding committed component bytes, oldest first.
    chunks: Vec<Box<[u8]>>,
    /// Bytes already written into the last chunk.
    used: usize,
    /// Bytes held across all currently resident chunks.
    resident: usize,
}

impl ComponentArena {
    fn new() -> ComponentArena {
        ComponentArena { chunks: Vec::new(), used: 0, resident: 0 }
    }

    /// Commit `bytes` into the arena and return a pointer to where they
    /// landed.
    ///
    /// If what remains of the current chunk cannot hold `bytes`, a fresh chunk
    /// takes over first, so the returned pointer's `bytes.len()` bytes are
    /// always contiguous — addressing live memory for as long as the chunk
    /// backing them stays resident (see [`ComponentArena::compact`]).
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
    /// arena would keep every record's bytes resident for the whole load no
    /// matter how many entities the region carries. The chunk currently being
    /// written to is never dropped, since the next commit needs somewhere to
    /// land.
    fn compact(&mut self, threshold: usize) {
        while self.resident > threshold && self.chunks.len() > 1 {
            let oldest = self.chunks.remove(0);
            self.resident -= oldest.len();
        }
    }
}

/// A component reference captured once, right after an id's first spawn.
///
/// The kind travels with the handle so the fold can interpret the bytes
/// without re-deriving it, and `alive` lets a later respawn of the same id
/// retire this handle without needing to touch the arena at all.
pub(crate) struct Pending {
    ptr: *const u8,
    len: usize,
    kind: Kind,
    alive: bool,
}

/// Read the entity records from the region's `ents` section.
///
/// A record whose id names an entity already captured is a respawn: the
/// earlier incarnation is retired immediately, rather than waiting for a later
/// sweep to notice the repeat. Each distinct id's first spawn is captured as a
/// [`Pending`] handle into the region-lifetime [`ComponentArena`], mirroring a
/// caller that grabbed a reference to an entity's components right after it
/// first appeared and held onto it rather than re-reading the store.
pub(crate) fn load_store(region: &Region) -> (ComponentArena, Vec<Pending>) {
    let data = region.slice(region.ents);
    let mut c = Cursor::new(data);
    let count = (c.u16() as usize).min(MAX_ENTITIES);
    let mut arena = ComponentArena::new();
    let mut pending: Vec<Pending> = Vec::new();
    // Each distinct id's index into `pending`, so a respawn can retire the
    // earlier incarnation without a second pass over the records.
    let mut first_index: Vec<(u16, usize)> = Vec::new();

    for _ in 0..count {
        let id = c.u16();
        let kind = Kind::from_byte(c.u8());
        let len = match kind.stride() {
            Some(fixed) => fixed,
            None => (c.u16() as usize).min(256),
        };
        if !c.ok {
            break;
        }
        let payload = c.bytes(len);
        if payload.len() != len {
            break;
        }

        // Every record's bytes are committed, respawn or not: a respawned id
        // still costs arena space, the same way a despawned entity's old
        // component would have before a sweep ever ran.
        let ptr = arena.commit(payload);

        if let Some(&(_, idx)) = first_index.iter().find(|&&(eid, _)| eid == id) {
            pending[idx].alive = false;
        } else {
            first_index.push((id, pending.len()));
            pending.push(Pending { ptr, len: payload.len(), kind, alive: true });
        }

        // Bound the arena's resident memory now that this record's bytes are
        // safely committed.
        arena.compact(COMPACT_THRESHOLD_BYTES);
    }
    (arena, pending)
}

/// Load the region's entities and fold a digest of the survivors.
///
/// Every surviving id's `pending` handle was captured at that id's first
/// spawn; all of them are read here, once, after the whole `ents` section has
/// been walked and the arena has had every chance to compact.
pub fn load_region(region: &Region, n: usize) -> u64 {
    let (_arena, pending) = load_store(region);
    if pending.is_empty() {
        return 0;
    }

    let mut acc = 0xffu64 ^ (n as u64);
    for p in &pending {
        if !p.alive {
            continue;
        }
        let len = bind::footprint(p.kind, p.len);
        acc = acc.wrapping_mul(0x100000001b3) ^ bind::read_component(p.ptr, 0, len, p.kind);
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kind_round_trips_from_bytes() {
        assert_eq!(Kind::from_byte(0), Kind::Transform);
        assert_eq!(Kind::from_byte(3), Kind::Health);
        assert_eq!(Kind::from_byte(4), Kind::Blob);
        // Wraps rather than panicking on an unknown byte.
        assert_eq!(Kind::from_byte(5), Kind::Transform);
    }

    #[test]
    fn strides_match_the_wire_layout() {
        assert_eq!(Kind::Transform.stride(), Some(12));
        assert_eq!(Kind::Health.stride(), Some(4));
        assert_eq!(Kind::Inventory.stride(), None);
    }

    #[test]
    fn arena_commit_writes_are_readable_back() {
        let mut arena = ComponentArena::new();
        let a = arena.commit(&[1, 2, 3, 4]);
        let b = arena.commit(&[5, 6]);
        // SAFETY: neither chunk has been compacted away, so both pointers
        // still address the bytes just committed.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(a, 4), [1, 2, 3, 4]);
            assert_eq!(std::slice::from_raw_parts(b, 2), [5, 6]);
        }
    }

    #[test]
    fn arena_starts_a_new_chunk_once_the_current_one_is_full() {
        let mut arena = ComponentArena::new();
        let filler = vec![0u8; ARENA_CHUNK_BYTES - 4];
        arena.commit(&filler);
        assert_eq!(arena.chunks.len(), 1);
        // Only 4 bytes remain in the first chunk; this does not fit.
        arena.commit(&[1, 2, 3, 4, 5]);
        assert_eq!(arena.chunks.len(), 2);
    }

    #[test]
    fn compact_leaves_a_small_arena_untouched() {
        let mut arena = ComponentArena::new();
        arena.commit(&[1, 2, 3]);
        arena.compact(COMPACT_THRESHOLD_BYTES);
        assert_eq!(arena.chunks.len(), 1);
    }

    #[test]
    fn compact_drops_oldest_chunks_once_over_threshold() {
        let mut arena = ComponentArena::new();
        // Each commit exactly fills its own chunk, so every commit pushes a
        // new one.
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
        let mut arena = ComponentArena::new();
        arena.commit(&[1, 2, 3]);
        arena.compact(0);
        assert_eq!(arena.chunks.len(), 1, "the chunk being written to must survive");
    }

    /// Build a minimal region carrying an `ents` section, for exercising
    /// [`load_region`] over more than one record.
    fn region_with_ents(ents: &[u8]) -> Vec<u8> {
        use crate::format::*;

        let mut v = Vec::new();
        v.extend_from_slice(&MAGIC);
        v.extend_from_slice(&VERSION.to_be_bytes());
        v.extend_from_slice(&flag::ENTITY.to_be_bytes());
        v.extend_from_slice(&0i16.to_be_bytes());
        v.extend_from_slice(&0i16.to_be_bytes());
        v.extend_from_slice(&1u16.to_be_bytes()); // num_chunks
        v.extend_from_slice(&3u16.to_be_bytes()); // num_sections
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
        let ents_off = cdat_off + cdat_len;

        v.extend_from_slice(&tag::CMAP);
        v.extend_from_slice(&(cmap_off as u32).to_be_bytes());
        v.extend_from_slice(&(cmap_len as u32).to_be_bytes());
        v.extend_from_slice(&tag::CDAT);
        v.extend_from_slice(&(cdat_off as u32).to_be_bytes());
        v.extend_from_slice(&(cdat_len as u32).to_be_bytes());
        v.extend_from_slice(&tag::ENTS);
        v.extend_from_slice(&(ents_off as u32).to_be_bytes());
        v.extend_from_slice(&(ents.len() as u32).to_be_bytes());

        v.extend_from_slice(&0u32.to_be_bytes());
        v.extend_from_slice(&0u32.to_be_bytes());
        v.push(0);
        v.extend_from_slice(ents);
        v
    }

    fn ent_record(id: u16, kind: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&id.to_be_bytes());
        out.push(kind);
        if kind % 5 == 2 || kind % 5 == 4 {
            out.extend_from_slice(&(payload.len() as u16).to_be_bytes());
        }
        out.extend_from_slice(payload);
        out
    }

    #[test]
    fn load_store_captures_first_spawn_per_id() {
        let mut ents = vec![0u8, 2]; // count = 2
        ents.extend(ent_record(1, 3, &[0, 10, 0, 20])); // Health
        ents.extend(ent_record(2, 0, &[0u8; 12])); // Transform
        let data = region_with_ents(&ents);
        let region = crate::parse::parse(&data).expect("valid region");
        let (_arena, pending) = load_store(&region);
        assert_eq!(pending.len(), 2);
        assert!(pending.iter().all(|p| p.alive));
    }

    #[test]
    fn a_respawn_retires_the_earlier_incarnation() {
        let mut ents = vec![0u8, 2]; // count = 2
        ents.extend(ent_record(1, 3, &[0, 1, 0, 2]));
        ents.extend(ent_record(1, 3, &[0, 3, 0, 4]));
        let data = region_with_ents(&ents);
        let region = crate::parse::parse(&data).expect("valid region");
        let (_arena, pending) = load_store(&region);
        // Only the first spawn is ever captured; the respawn just retires it.
        assert_eq!(pending.len(), 1);
        assert!(!pending[0].alive, "the respawned id's first handle must be retired");
    }

    #[test]
    fn load_region_is_deterministic() {
        let mut ents = vec![0u8, 2];
        ents.extend(ent_record(1, 0, &[0u8; 12]));
        ents.extend(ent_record(2, 3, &[0, 5, 0, 6]));
        let data = region_with_ents(&ents);
        let region = crate::parse::parse(&data).expect("valid region");
        assert_eq!(load_region(&region, 1), load_region(&region, 1));
    }

    #[test]
    fn load_region_of_no_entities_is_zero() {
        let data = region_with_ents(&[0u8, 0]);
        let region = crate::parse::parse(&data).expect("valid region");
        assert_eq!(load_region(&region, 1), 0);
    }
}
