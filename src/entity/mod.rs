//! Entity component storage.
//!
//! Entities are stored structure-of-arrays style: every component's bytes live
//! in a region-lifetime store, built once per load rather than once per entity,
//! so a handle captured while loading one record stays meaningful to read back
//! once the whole `ents` section has been walked. Each record's payload gets its
//! own exact-sized run, addressed by a slot index (see [`ComponentStore`]).
//! Superseded runs are released in batches once the load's live payload crosses
//! a ceiling drawn from the region's own decoded shape, so a region that
//! respawns heavily reuses slots instead of growing the table with every
//! incarnation it has ever seen — and how far into a load that happens is a
//! property of what the whole region carries, not of any one record.
//!
//! Each distinct entity id's first spawn is captured as a [`Handle`], mirroring
//! a caller that grabbed a reference to an entity's components right after it
//! appeared and held onto it rather than re-reading the store.

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

    /// This kind's position in a per-kind table, in declaration order.
    pub fn index(self) -> usize {
        match self {
            Kind::Transform => 0,
            Kind::Velocity => 1,
            Kind::Inventory => 2,
            Kind::Health => 3,
            Kind::Blob => 4,
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

/// Slots in [`WatchList`].
const WATCH_SLOTS: usize = 4;

/// How many component kinds there are, and so how wide a per-kind table is.
const KIND_COUNT: usize = 5;

/// A region-lifetime slot table for decoded component payloads.
///
/// Built once per load rather than once per entity, so a slot handed out while
/// decoding one record stays meaningful while later records are decoded — which
/// is what lets a [`Handle`] be read again well after its own record has
/// finished. Each payload gets its own exact-sized run, so a component never
/// carries the slack a shared chunk would leave behind it.
///
/// Retiring an entity releases its run and returns the slot to a free list. A
/// later record takes that slot rather than extending the table, which is what
/// keeps a region that respawns the same handful of entities thousands of times
/// from paying for every incarnation at once.
pub(crate) struct ComponentStore {
    /// One entry per slot: the payload occupying it, or nothing if the slot is
    /// on the free list.
    slots: Vec<Option<Box<[u8]>>>,
    /// Slots released by a sweep, newest first.
    free: Vec<usize>,
    /// Slots superseded by a respawn but not yet released.
    pending: Vec<usize>,
    /// Bytes the occupied slots currently hold.
    live: usize,
    /// Live bytes past which a sweep releases what has been superseded.
    ceiling: usize,
}

impl ComponentStore {
    fn new(ceiling: usize) -> ComponentStore {
        ComponentStore {
            slots: Vec::new(),
            free: Vec::new(),
            pending: Vec::new(),
            live: 0,
            ceiling: ceiling.max(1),
        }
    }

    /// How many slots the table has ever needed.
    pub(crate) fn len(&self) -> usize {
        self.slots.len()
    }

    /// Store `bytes` in a free slot if there is one, or a fresh slot, and
    /// return which.
    fn insert(&mut self, bytes: &[u8]) -> usize {
        // A load that has taken on more than the region's shape budgets for
        // releases what has been superseded before taking on more.
        if self.live + bytes.len() > self.ceiling {
            self.sweep();
        }
        self.live += bytes.len();
        let run = bytes.to_vec().into_boxed_slice();
        match self.free.pop() {
            Some(slot) => {
                self.slots[slot] = Some(run);
                slot
            }
            None => {
                self.slots.push(Some(run));
                self.slots.len() - 1
            }
        }
    }

    /// Mark `slot` as superseded by a later incarnation.
    ///
    /// The payload is not released here. A region that respawns heavily would
    /// pay a free per record for slots the next sweep is going to take anyway,
    /// so the slot joins a pending list and is released in a batch (see
    /// [`ComponentStore::sweep`]).
    fn supersede(&mut self, slot: usize) {
        if self.slots.get(slot).is_some_and(Option::is_some) {
            self.pending.push(slot);
        }
    }

    /// Release every superseded slot, returning them to the free list.
    fn sweep(&mut self) {
        for slot in std::mem::take(&mut self.pending) {
            if let Some(entry) = self.slots.get_mut(slot) {
                // Releases the run this slot was holding.
                if let Some(run) = entry.take() {
                    self.live -= run.len().min(self.live);
                }
                self.free.push(slot);
            }
        }
    }

    /// The start of the payload occupying `slot`, or null if the slot is free
    /// or was never handed out.
    ///
    /// The run's own length is the store's business, not the caller's: a caller
    /// that recorded how many bytes it put in a slot walks that many from here.
    pub(crate) fn start_of(&self, slot: usize) -> *const u8 {
        match self.slots.get(slot).and_then(Option::as_ref) {
            Some(run) => run.as_ptr(),
            None => std::ptr::null(),
        }
    }
}

/// A component reference captured once, right after an id's first spawn.
///
/// The kind travels with the handle so the fold can interpret the bytes without
/// re-deriving it, and the slot is what the store is asked for when the handle
/// is finally read. Holding the slot rather than an address is what lets a
/// handle be read repeatedly while the load is still running.
pub(crate) struct Handle {
    slot: usize,
    len: usize,
    kind: Kind,
}

/// A component payload named by where it was put, for the one closing read.
///
/// The closing fold walks a handful of payloads exactly once, so it takes the
/// address the store gave out rather than going back through the slot table for
/// each — the table lookup is a branch and a bounds check per component for a
/// pointer that was already in hand.
pub(crate) struct Watched {
    base: *const u8,
    len: usize,
    kind: Kind,
}

/// The components [`load_region`]'s closing fold reads.
///
/// Fixed capacity, reused round-robin: a region may name thousands of distinct
/// entities and the closing fold has to cost the same on all of them, so the
/// newest displace the oldest instead of the list growing with the cast.
pub(crate) struct WatchList {
    slots: [Option<Watched>; WATCH_SLOTS],
    next: usize,
}

impl WatchList {
    fn new() -> WatchList {
        WatchList { slots: [None, None, None, None], next: 0 }
    }

    /// Put `watched` in the next slot, displacing whatever that slot held.
    fn watch(&mut self, watched: Watched) {
        self.slots[self.next] = Some(watched);
        self.next = (self.next + 1) % WATCH_SLOTS;
    }

    /// The components currently held, oldest slot first.
    pub(crate) fn handles(&self) -> impl Iterator<Item = &Watched> {
        self.slots.iter().flatten()
    }
}

/// Read the entity records from the region's `ents` section.
///
/// A record whose id names an entity already seen is a respawn: it supersedes
/// the earlier incarnation, whose slot is retired immediately rather than
/// waiting for a later sweep to notice the repeat. Each distinct id's first
/// spawn is captured as a [`Handle`] into the region-lifetime
/// [`ComponentStore`], mirroring a caller that grabbed a reference to an
/// entity's components right after it first appeared and held onto it rather
/// than re-reading the store.
pub(crate) fn load_store(region: &Region, n: usize) -> (ComponentStore, WatchList, u64) {
    let data = region.slice(region.ents);
    let mut c = Cursor::new(data);
    let count = (c.u16() as usize).min(MAX_ENTITIES);
    // The store's sweep ceiling tracks the region's own decoded shape, so a
    // region of dense columns carries more live component data than a sparse
    // one before anything is released.
    let mut store = ComponentStore::new(crate::budget::store_bytes(region, n));
    let mut watched = WatchList::new();
    // The slot each distinct id currently occupies, so a respawn can retire the
    // incarnation before it without a second pass over the records.
    let mut current: Vec<(u16, usize)> = Vec::new();
    // The most recent record of each component kind. A component means little
    // on its own — a velocity is interesting against the last velocity, not in
    // isolation — so each record is bound against its predecessor of the same
    // kind as it is decoded.
    let mut latest: [Option<Handle>; KIND_COUNT] = Default::default();
    let mut acc = 0xffu64;

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

        // Every record gets its own slot, respawn or not: the new incarnation
        // is a distinct set of component bytes and has to be readable
        // alongside whatever else the load has already produced.
        let slot = store.insert(payload);

        // Bind this record against the last one of its kind, before it becomes
        // that predecessor itself.
        let bucket = kind.index();
        if let Some(prev) = latest[bucket].as_ref() {
            let prev_len = bind::footprint(prev.kind, prev.len);
            let here_len = bind::footprint(kind, payload.len());
            acc = acc.wrapping_mul(0x9e3779b97f4a7c15)
                ^ bind::fold_succession(
                    store.start_of(prev.slot),
                    prev_len,
                    store.start_of(slot),
                    here_len,
                    kind,
                );
        }
        latest[bucket] = Some(Handle { slot, len: payload.len(), kind });

        match current.iter_mut().find(|(eid, _)| *eid == id) {
            Some(entry) => {
                // A respawn: the incarnation before it is superseded, and its
                // slot goes back for a later record once the store sweeps.
                let previous = std::mem::replace(&mut entry.1, slot);
                store.supersede(previous);
            }
            None => {
                current.push((id, slot));
                watched.watch(Watched { base: store.start_of(slot), len: payload.len(), kind });
            }
        }
    }
    (store, watched, acc)
}

/// Load the region's entities and fold a digest of them.
///
/// Every watched handle was captured at some id's first spawn; all of them are
/// read here, once, after the whole `ents` section has been walked. The
/// succession term the load itself accumulated — each record against its
/// predecessor of the same kind — folds in alongside them.
pub fn load_region(region: &Region, n: usize) -> u64 {
    let (store, watched, succession) = load_store(region, n);
    if store.len() == 0 {
        return 0;
    }

    let mut acc = (0xffu64 ^ (n as u64)).wrapping_mul(0x100000001b3) ^ succession;
    for w in watched.handles() {
        let len = bind::footprint(w.kind, w.len);
        acc = acc.wrapping_mul(0x100000001b3) ^ bind::read_component(w.base, 0, len, w.kind);
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
    fn store_writes_are_readable_back_through_their_slot() {
        let mut store = ComponentStore::new(1 << 20);
        let a = store.insert(&[1, 2, 3, 4]);
        let b = store.insert(&[5, 6]);
        assert_ne!(a, b);
        // SAFETY: both slots are occupied by the runs just inserted, and each
        // run is exactly as long as the payload put into it.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(store.start_of(a), 4), [1, 2, 3, 4]);
            assert_eq!(std::slice::from_raw_parts(store.start_of(b), 2), [5, 6]);
        }
    }

    /// Slot reuse is the store's bound. A region that respawns the same cast
    /// over and over must keep taking the same slots back rather than growing
    /// the table with every incarnation it has ever seen.
    #[test]
    fn the_slot_table_stays_flat_under_heavy_respawning() {
        // A ceiling a handful of records crosses, so the sweep runs often.
        let mut store = ComponentStore::new(64);
        let mut held = 0usize;
        for i in 0..500u16 {
            let slot = store.insert(&[i as u8; 8]);
            store.supersede(held);
            held = slot;
        }
        assert!(
            store.len() <= 16,
            "the table grew to {} slots for one live entity",
            store.len()
        );
    }

    #[test]
    fn a_swept_slot_reads_back_as_nothing() {
        let mut store = ComponentStore::new(1 << 20);
        let a = store.insert(&[1, 2, 3, 4]);
        store.supersede(a);
        store.sweep();
        assert!(store.start_of(a).is_null());
        assert!(store.start_of(99).is_null(), "a slot never handed out is empty too");
    }

    #[test]
    fn a_swept_slot_is_handed_to_the_next_record() {
        let mut store = ComponentStore::new(1 << 20);
        let a = store.insert(&[1, 2, 3, 4]);
        let b = store.insert(&[5, 6]);
        store.supersede(a);
        store.sweep();
        let c = store.insert(&[7]);
        assert_eq!(c, a, "the free slot must be reused before the table grows");
        assert_eq!(store.len(), 2, "reuse must not grow the table");
        // SAFETY: slot `c` holds the one-byte run just inserted, and slot `b`
        // is untouched.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(store.start_of(c), 1), [7]);
            assert_eq!(std::slice::from_raw_parts(store.start_of(b), 2), [5, 6]);
        }
    }

    #[test]
    fn watch_list_holds_only_its_newest_slots() {
        let mut store = ComponentStore::new(1 << 20);
        let mut w = WatchList::new();
        for i in 0..WATCH_SLOTS + 3 {
            let slot = store.insert(&[i as u8; 4]);
            w.watch(Watched { base: store.start_of(slot), len: 4, kind: Kind::Health });
        }
        assert_eq!(w.handles().count(), WATCH_SLOTS, "capacity must be fixed");
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
        let (store, watched, _) = load_store(&region, region.worked_chunks());
        assert_eq!(watched.handles().count(), 2);
        assert_eq!(store.len(), 2, "two records, two slots");
        assert!(watched.handles().all(|w| !w.base.is_null()));
    }

    #[test]
    fn a_respawn_supersedes_the_incarnation_before_it() {
        let mut ents = vec![0u8, 2]; // count = 2
        ents.extend(ent_record(1, 3, &[0, 1, 0, 2]));
        ents.extend(ent_record(1, 3, &[0, 3, 0, 4]));
        let data = region_with_ents(&ents);
        let region = crate::parse::parse(&data).expect("valid region");
        let (store, watched, _) = load_store(&region, region.worked_chunks());
        // Only the first spawn is ever captured, and the respawn marks it
        // superseded — a load this small never comes under enough pressure to
        // sweep, so the payload is still there.
        assert_eq!(watched.handles().count(), 1);
        assert_eq!(store.len(), 2, "two records, two slots");
        assert_eq!(store.pending, vec![0], "the first incarnation is superseded");
    }

    /// Once the load does come under pressure, a superseded slot is swept and
    /// handed to a later record rather than the table growing.
    #[test]
    fn a_swept_slot_is_taken_by_a_later_record() {
        let mut store = ComponentStore::new(24);
        let a = store.insert(&[1u8; 12]);
        let b = store.insert(&[2u8; 12]);
        store.supersede(a);
        // This insert crosses the ceiling, so the sweep runs first and `a` is
        // free to take.
        let c = store.insert(&[3u8; 12]);
        assert_eq!(c, a, "the swept slot must be reused before the table grows");
        assert_eq!(store.len(), 2, "reuse must not grow the table");
        assert!(!store.start_of(b).is_null());
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
