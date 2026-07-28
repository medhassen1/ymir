//! Entity component storage.
//!
//! Entities are stored structure-of-arrays style: every component's bytes live
//! in one shared arena and each entity holds a slot naming its kind, offset and
//! length. Keeping every kind in a single allocation is what lets a rebuild walk
//! all of a region's entities in one linear pass instead of chasing a pointer
//! per component.
//!
//! Dead entities are swept between passes and the arena is compacted, because a
//! region that spawns and despawns heavily otherwise grows without bound.

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

/// Where one entity's component lives inside the arena.
#[derive(Clone, Copy, Debug)]
pub struct Slot {
    /// Entity id this component belongs to.
    pub id: u16,
    /// How the bytes are interpreted.
    pub kind: Kind,
    /// Byte offset into the arena.
    pub offset: usize,
    /// Byte length.
    pub len: usize,
    /// Cleared when the entity despawns.
    pub alive: bool,
}

/// Component storage for one region.
pub struct EntityStore {
    arena: Vec<u8>,
    slots: Vec<Slot>,
}

impl EntityStore {
    /// An empty store with room for `bytes` of component payload.
    pub fn with_capacity(bytes: usize) -> EntityStore {
        EntityStore { arena: Vec::with_capacity(bytes), slots: Vec::new() }
    }

    /// How many slots the store holds, alive or not.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the store holds no slots at all.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Bytes of component payload currently stored.
    pub fn arena_len(&self) -> usize {
        self.arena.len()
    }

    /// The slot table.
    pub fn slots(&self) -> &[Slot] {
        &self.slots
    }

    /// A cursor onto the component arena.
    ///
    /// The reader walks components through this pointer rather than re-slicing
    /// the arena per component, which keeps a region-wide sweep to one pass.
    pub fn arena_cursor(&self) -> *const u8 {
        self.arena.as_ptr()
    }

    /// Append a component, returning its slot index.
    pub fn push(&mut self, id: u16, kind: Kind, payload: &[u8]) -> usize {
        let offset = self.arena.len();
        self.arena.extend_from_slice(payload);
        self.slots.push(Slot { id, kind, offset, len: payload.len(), alive: true });
        self.slots.len() - 1
    }

    /// Mark an entity's components dead without reclaiming their bytes yet.
    pub fn kill(&mut self, id: u16) {
        for s in &mut self.slots {
            if s.id == id {
                s.alive = false;
            }
        }
    }

    /// How many slots are still alive.
    pub fn alive_count(&self) -> usize {
        self.slots.iter().filter(|s| s.alive).count()
    }

    /// Drop dead slots and compact the arena so the live payloads are
    /// contiguous again.
    ///
    /// Compaction rebuilds the arena into a fresh allocation sized to the live
    /// payload, which is what actually returns the memory a despawn wave freed.
    pub fn compact(&mut self) {
        let live: usize = self.slots.iter().filter(|s| s.alive).map(|s| s.len).sum();
        let mut fresh = Vec::with_capacity(live);
        let mut kept = Vec::with_capacity(self.slots.len());
        for s in &self.slots {
            if !s.alive {
                continue;
            }
            let start = s.offset.min(self.arena.len());
            let end = (s.offset + s.len).min(self.arena.len());
            let offset = fresh.len();
            fresh.extend_from_slice(&self.arena[start..end]);
            kept.push(Slot { offset, len: end - start, ..*s });
        }
        self.arena = fresh;
        self.slots = kept;
    }
}

/// Read the entity records from the region's `ents` section.
pub fn load_store(region: &Region) -> EntityStore {
    let data = region.slice(region.ents);
    let mut c = Cursor::new(data);
    let count = (c.u16() as usize).min(MAX_ENTITIES);
    let mut store = EntityStore::with_capacity(count * 12);

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
        store.push(id, kind, payload);
    }
    store
}

/// Load the region's entities, sweep the despawned ones, and fold a digest of
/// what remains.
///
/// The sweep happens between the two read passes: the first pass records which
/// entities the region marked dead, the second folds the survivors. Holding the
/// arena cursor across both is what keeps the second pass from re-deriving every
/// component's address.
pub fn load_region(region: &Region, n: usize) -> u64 {
    let mut store = load_store(region);
    if store.is_empty() {
        return 0;
    }

    // The arena cursor is taken once for the whole fold, so the reader walks the
    // component bytes without re-borrowing the store per entity.
    let cursor = store.arena_cursor();

    // Pass 1 — an entity whose id repeats later in the record stream has
    // despawned and respawned; the earlier incarnation is swept.
    let ids: Vec<u16> = store.slots().iter().map(|s| s.id).collect();
    for (i, id) in ids.iter().enumerate() {
        if ids[i + 1..].contains(id) {
            store.kill(*id);
        }
    }
    if store.alive_count() < store.len() {
        store.compact();
    }

    // Pass 2 — fold the surviving components through the cursor taken above.
    let mut acc = 0xffu64 ^ (n as u64);
    for s in store.slots() {
        if !s.alive {
            continue;
        }
        acc = acc.wrapping_mul(0x100000001b3)
            ^ bind::read_component(cursor, s.offset, s.len, s.kind);
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
    fn push_lays_components_out_contiguously() {
        let mut s = EntityStore::with_capacity(0);
        s.push(1, Kind::Health, &[1, 2, 3, 4]);
        s.push(2, Kind::Health, &[5, 6, 7, 8]);
        assert_eq!(s.slots()[0].offset, 0);
        assert_eq!(s.slots()[1].offset, 4);
        assert_eq!(s.arena_len(), 8);
    }

    #[test]
    fn compact_reclaims_dead_payload() {
        let mut s = EntityStore::with_capacity(0);
        s.push(1, Kind::Health, &[1, 2, 3, 4]);
        s.push(2, Kind::Health, &[5, 6, 7, 8]);
        s.push(3, Kind::Health, &[9, 9, 9, 9]);
        s.kill(2);
        s.compact();
        assert_eq!(s.len(), 2);
        assert_eq!(s.arena_len(), 8);
        assert_eq!(s.slots()[0].offset, 0);
        assert_eq!(s.slots()[1].offset, 4);
    }

    #[test]
    fn kill_marks_every_slot_of_an_entity() {
        let mut s = EntityStore::with_capacity(0);
        s.push(7, Kind::Health, &[0; 4]);
        s.push(7, Kind::Transform, &[0; 12]);
        s.push(8, Kind::Health, &[0; 4]);
        s.kill(7);
        assert_eq!(s.alive_count(), 1);
    }

    #[test]
    fn compact_on_an_all_alive_store_is_a_noop() {
        let mut s = EntityStore::with_capacity(0);
        s.push(1, Kind::Health, &[1, 2, 3, 4]);
        let before = s.arena_len();
        s.compact();
        assert_eq!(s.arena_len(), before);
        assert_eq!(s.len(), 1);
    }
}
