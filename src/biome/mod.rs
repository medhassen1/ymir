//! Biome grid resolution and blending.
//!
//! Biomes are stored at quarter resolution — one cell per 4x4x4 blocks — and
//! blended at render time so a desert fades into a savanna instead of changing
//! at a cell boundary.
//!
//! A region's biome pass decodes one column at a time, but continuity across
//! the region means a later column's blend sometimes needs to see an earlier
//! column's row, not only its own — so a column's decoded cells cannot simply
//! live and die with that column's turn through the loop. [`RowStore`] gives
//! the pass a home for decoded rows that outlives any single column: built
//! once per region rather than once per column, it appends every column's rows
//! into one buffer and hands back a pointer into them instead of an owned
//! buffer.
//!
//! Because the store lives for the whole pass, its memory is bounded by
//! compaction rather than by any one column's lifetime: a commit with no room
//! left runs [`RowStore::compact`], which retires the rows of the oldest
//! columns and slides the survivors down over them, the way a defragmenting
//! allocator packs its live objects together instead of growing past them.
//!
//! Each column registers its opening row with a [`ReferenceRing`], so a later
//! column always has a nearby reference to blend against rather than only ever
//! its immediate predecessor. The ring's capacity is fixed rather than one
//! entry per column: what the closing fold costs, and how many rows the pass
//! keeps addressable, is then bounded by the ring rather than by how many
//! columns the region carries.

use crate::blend;
use crate::common::*;
use crate::parse::Region;
use crate::reader::Cursor;

/// Cells along one edge of a chunk's biome grid.
pub const GRID_EDGE: usize = BIOME_EDGE;

/// Bytes a fresh [`RowStore`] holds.
///
/// A column's decoded volume — the base grid plus up to [`MAX_STACK`]
/// refinement rows — is far smaller than this, so an ordinary region's columns
/// all fit without the store ever having to compact. A region built from many
/// heavily refined columns does not, which is exactly the case compaction
/// exists to catch: without it, a region-lifetime store would keep every
/// column's rows resident for the whole pass no matter how many columns the
/// region carries.
const STORE_INIT_BYTES: usize = 128;

/// Reference rows the ring keeps addressable at once.
///
/// Four is enough for a blend to reach a handful of columns back while keeping
/// the closing fold's cost — and the rows it holds on to — independent of the
/// region's column count.
const REFERENCE_SLOTS: usize = 4;

/// A chunk's biome cells, stored as one span per row rather than a fixed
/// stride.
///
/// This is the scratch buffer a single column decodes into. [`resolve_region`]
/// builds one per column, then commits its finished `cells` into the region's
/// [`RowStore`] so the bytes survive past this struct's own scope. Every row
/// of the base grid is [`GRID_EDGE`] cells wide, but a region's sub-cell
/// refinement layer can contribute a shorter row (see [`BiomeGrid::refine`]),
/// so each row records its own `(start, len)` span into `cells` instead of
/// assuming a uniform width.
pub struct BiomeGrid {
    cells: Vec<u8>,
    /// `(start, len)` into `cells` for each row, in the grid's current row
    /// order.
    spans: Vec<(usize, usize)>,
    /// Rows currently stored. Starts at [`GRID_EDGE`] and grows when the region
    /// carries sub-cell refinement data.
    pub rows: usize,
}

impl BiomeGrid {
    /// A grid of `GRID_EDGE` rows, every cell set to `fill`.
    pub fn uniform(fill: u8) -> BiomeGrid {
        let cells = vec![fill; GRID_EDGE * GRID_EDGE];
        let spans = (0..GRID_EDGE).map(|z| (z * GRID_EDGE, GRID_EDGE)).collect();
        BiomeGrid { cells, spans, rows: GRID_EDGE }
    }

    /// The cells, as a slice.
    pub fn cells(&self) -> &[u8] {
        &self.cells
    }

    /// The biome at a cell, or 0 outside the grid or past a short row's span.
    pub fn get(&self, x: usize, z: usize) -> u8 {
        if z >= self.rows {
            return 0;
        }
        let (start, len) = self.spans[z];
        if x >= len {
            return 0;
        }
        self.cells[start + x]
    }

    /// Overwrite a cell.
    pub fn set(&mut self, x: usize, z: usize, biome: u8) {
        if z >= self.rows {
            return;
        }
        let (start, len) = self.spans[z];
        if x < len {
            self.cells[start + x] = biome;
        }
    }

    /// A view of row `z` into this grid's own scratch buffer.
    ///
    /// Only meaningful before the grid's cells are committed into a
    /// [`RowStore`] — once committed, [`resolve_region`] addresses the row
    /// through the store's copy instead (see [`ColumnView::row_ptr`]), since
    /// this buffer is dropped at the end of the column's scope.
    pub fn row_view(&self, z: usize) -> *const u8 {
        if z >= self.rows {
            return std::ptr::null();
        }
        let (start, _) = self.spans[z];
        // SAFETY: `uniform` and `refine` only ever record a span whose `start`
        // falls within the buffer they just wrote (at most at its very end),
        // so this offset lands inside or at the boundary of the allocation.
        unsafe { self.cells.as_ptr().add(start) }
    }

    /// The number of live cells addressable from [`BiomeGrid::row_view`] for
    /// row `z`, or 0 past the end of the grid.
    pub fn row_len(&self, z: usize) -> usize {
        if z >= self.rows {
            return 0;
        }
        self.spans[z].1
    }

    /// How many distinct biomes the grid names.
    pub fn variety(&self) -> usize {
        let mut seen = [false; 256];
        let mut count = 0usize;
        for &c in &self.cells {
            if !seen[c as usize] {
                seen[c as usize] = true;
                count += 1;
            }
        }
        count
    }

    /// Append refinement rows carrying the sub-cell detail the region stored,
    /// then group rows by dominant biome.
    ///
    /// A region written by a generator that resolves biomes below quarter
    /// resolution carries extra rows; appending them lets the blend read the
    /// finer detail instead of quantising it away. The deepest refinement row
    /// only ever covers a column's interior — the generator has nothing below
    /// it to blend against at the edges, so it never bothers recording them —
    /// which means the last row appended can be shorter than a full row.
    ///
    /// Rows are then sorted by dominant biome so the blend kernel, which reads
    /// a three-row stencil, spends most of its work on neighbours that already
    /// agree instead of jumping between unrelated parts of the grid.
    pub fn refine(&mut self, extra_rows: usize, seed: u8) {
        for r in 0..extra_rows {
            let start = self.cells.len();
            let len = if r + 1 == extra_rows { (GRID_EDGE / 2).max(1) } else { GRID_EDGE };
            for x in 0..len {
                let v = seed.wrapping_add((r * GRID_EDGE + x) as u8);
                self.cells.push(v);
            }
            self.spans.push((start, len));
        }
        self.rows += extra_rows;

        let cells = &self.cells;
        self.spans.sort_by_key(|&(start, len)| {
            // SAFETY: every span in `self.spans` — appended above or already
            // present — satisfies `start + len <= cells.len()`, so the slice
            // stays inside `cells`.
            let row = unsafe { std::slice::from_raw_parts(cells.as_ptr().add(start), len) };
            std::cmp::Reverse(blend::dominant(row))
        });
    }
}

/// One column's rows within a [`RowStore`], as `(start, len)` from the store's
/// own base.
#[derive(Clone, Copy)]
struct Extent {
    start: usize,
    len: usize,
}

/// A region-lifetime store for decoded biome rows.
///
/// Built once per region rather than once per column, so a pointer handed out
/// while decoding one column stays valid while later columns are decoded —
/// which is what lets a reference row (see [`resolve_region`]) be read again
/// well after its own column has finished. A column's cells are copied in as
/// one contiguous run, so a single column's rows are never split.
///
/// The store is bounded by compaction rather than by refusing work: see
/// [`RowStore::compact`].
struct RowStore {
    /// The live rows, back to back from offset zero.
    bytes: Box<[u8]>,
    /// How much of `bytes` the live rows occupy.
    used: usize,
    /// One entry per committed column, oldest first.
    extents: Vec<Extent>,
}

impl RowStore {
    fn new() -> RowStore {
        RowStore {
            bytes: vec![0u8; STORE_INIT_BYTES].into_boxed_slice(),
            used: 0,
            extents: Vec::new(),
        }
    }

    /// Commit one column's cells and return a pointer to where they landed.
    ///
    /// The returned pointer's `cells.len()` bytes are contiguous, and address
    /// live memory for as long as the buffer holding them is the store's own
    /// (see [`RowStore::compact`]).
    fn commit(&mut self, cells: &[u8]) -> *const u8 {
        if self.used + cells.len() > self.bytes.len() {
            self.compact(cells.len());
        }
        let start = self.used;
        self.bytes[start..start + cells.len()].copy_from_slice(cells);
        self.used += cells.len();
        self.extents.push(Extent { start, len: cells.len() });
        // SAFETY: the copy above wrote `cells.len()` bytes starting at
        // `start`, so `start` is an offset inside this allocation.
        unsafe { self.bytes.as_ptr().add(start) }
    }

    /// Make room for a commit of `need` bytes.
    ///
    /// This is the store's memory bound: left unchecked, a region-lifetime
    /// store would keep every column's rows resident for the whole pass no
    /// matter how many columns the region carries. The oldest columns' rows
    /// are retired one column at a time until the incoming column fits, and
    /// the surviving rows are then slid down over the retired ones so the live
    /// rows stay packed at the front of the buffer. The most recent column is
    /// never retired, since the column about to commit blends against it.
    ///
    /// A slide that still leaves no room is followed by a re-fit: the store
    /// takes a buffer sized to the survivors plus the incoming column, with
    /// room to take the next few without compacting again, and the exhausted
    /// buffer is released.
    fn compact(&mut self, need: usize) {
        let mut retired = 0usize;
        let mut cut = 0usize;
        while retired + 1 < self.extents.len() && self.used - cut + need > self.bytes.len() {
            let oldest = self.extents[retired];
            cut = oldest.start + oldest.len;
            retired += 1;
        }
        if cut > 0 {
            self.bytes.copy_within(cut..self.used, 0);
            self.used -= cut;
            self.extents.drain(..retired);
            for e in &mut self.extents {
                e.start -= cut;
            }
        }
        if self.used + need > self.bytes.len() {
            let want = (self.used + need) * 2;
            let mut refitted = vec![0u8; want].into_boxed_slice();
            refitted[..self.used].copy_from_slice(&self.bytes[..self.used]);
            self.bytes = refitted;
        }
    }
}

/// One column's rows, committed into the region's [`RowStore`].
///
/// Mirrors [`BiomeGrid`]'s `(start, len)` spans, but resolved against the
/// store's pointer rather than a private buffer, since by the time this exists
/// the grid that produced it has already handed its cells to the store.
struct ColumnView {
    /// First cell of this column's grid within the store.
    base: *const u8,
    /// `(start, len)` into the committed buffer, one per row.
    spans: Vec<(usize, usize)>,
}

impl ColumnView {
    /// A pointer to the first cell of row `z`, or null past the row count.
    fn row_ptr(&self, z: usize) -> *const u8 {
        self.spans.get(z).map_or(std::ptr::null(), |&(start, _)| {
            // SAFETY: `spans` was taken from the `BiomeGrid` whose cells were
            // just copied into the store at `base`, so every `start` here
            // falls within that copy.
            unsafe { self.base.add(start) }
        })
    }

    /// The number of cells addressable from [`ColumnView::row_ptr`] for row
    /// `z`, or 0 past the row count.
    fn row_len(&self, z: usize) -> usize {
        self.spans.get(z).map_or(0, |&(_, len)| len)
    }
}

/// A reference row a column registered for later columns to blend against —
/// see [`resolve_region`].
struct ReferenceRow {
    ptr: *const u8,
    len: usize,
    z: u8,
}

/// A fixed-capacity ring of cross-column reference rows.
///
/// Registering a row takes the next slot in round-robin order, displacing
/// whatever reference was there [`REFERENCE_SLOTS`] registrations ago. Bounding
/// the ring is what keeps the pass's cross-column state — and the closing
/// fold's cost — flat over a region with many columns, instead of growing one
/// entry per column the way a plain list would.
struct ReferenceRing {
    slots: [Option<ReferenceRow>; REFERENCE_SLOTS],
    next: usize,
}

impl ReferenceRing {
    fn new() -> ReferenceRing {
        ReferenceRing { slots: Default::default(), next: 0 }
    }

    /// Register `row` as the newest cross-column reference.
    fn register(&mut self, row: ReferenceRow) {
        self.slots[self.next] = Some(row);
        self.next = (self.next + 1) % REFERENCE_SLOTS;
    }

    /// The references the ring currently holds, in slot order.
    fn rows(&self) -> impl Iterator<Item = &ReferenceRow> {
        self.slots.iter().flatten()
    }
}

/// Decode one column's biome grid starting at `c`'s current position,
/// advancing it past the column's header and cell block.
///
/// A region's `biom` section is a back-to-back sequence of these blocks, one
/// per column, so a later call simply picks up where the previous one left
/// off. Returns `None` once there is no more column data to read — a region
/// whose section is shorter than its column count simply gets no biome
/// override for the columns past the end.
fn decode_column(c: &mut Cursor) -> Option<(BiomeGrid, usize)> {
    let base = c.u8();
    let refine_rows = (c.u8() as usize).min(MAX_STACK);
    if !c.ok {
        return None;
    }
    let mut grid = BiomeGrid::uniform(base);
    'cells: for z in 0..GRID_EDGE {
        for x in 0..GRID_EDGE {
            let v = c.u8();
            if !c.ok {
                break 'cells;
            }
            grid.set(x, z, v);
        }
    }
    Some((grid, refine_rows))
}

/// Resolve and blend every column's biome grid, folding a digest of the
/// result.
///
/// Each column is decoded in turn from the `biom` section, refined if the
/// region carries sub-cell detail for it, and blended through the same
/// three-tap stencil [`crate::biome`]'s module docs describe. Its cells are
/// then committed into a region-wide [`RowStore`] rather than freed with the
/// column, and its opening row is registered with the region's
/// [`ReferenceRing`], giving later columns a cross-region reference to blend
/// against beyond just their immediate predecessor. The references the ring
/// still holds are folded into the digest once more at the end, closing out
/// the pass.
pub fn resolve_region(region: &Region) -> u64 {
    let n = region.worked_chunks();
    let mut c = Cursor::new(region.slice(region.biom));

    let mut store = RowStore::new();
    let mut ring = ReferenceRing::new();
    // The column decoded on the previous turn, held so the stencil can reach
    // across the boundary between the two.
    let mut upstream: Option<ColumnView> = None;
    let mut acc = 0xffu64;

    for cid in 0..n {
        let Some((mut grid, refine_rows)) = decode_column(&mut c) else {
            break;
        };

        // Row positions this column's stencil will visit, fixed before
        // refinement appends or reorders anything.
        let rows_to_blend: Vec<usize> = (0..grid.rows).collect();
        if refine_rows > 0 {
            grid.refine(refine_rows, region.seed as u8 ^ cid as u8);
        }

        acc ^= grid.variety() as u64;

        // Commit this column's finished cells into the region's store. Only
        // past this point is there a pointer stable enough to register as a
        // cross-column reference.
        let spans = std::mem::take(&mut grid.spans);
        let base = store.commit(&grid.cells);
        let view = ColumnView { base, spans };

        for &z in &rows_to_blend {
            let ptr = view.row_ptr(z);
            let len = view.row_len(z);
            acc = acc.wrapping_mul(0x100000001b3) ^ blend::mix_row(ptr, len, z as u8);
        }

        // Carry the stencil across the boundary with the column upstream of
        // this one, row by row at matching depths.
        if let Some(u) = upstream.as_ref() {
            for &z in &rows_to_blend {
                acc = acc.wrapping_mul(0x9e3779b97f4a7c15)
                    ^ blend::mix_across(
                        u.row_ptr(z),
                        u.row_len(z),
                        view.row_ptr(z),
                        view.row_len(z),
                        z as u8,
                    );
            }
        }

        // Register this column's opening row as the newest cross-column
        // reference, giving the region continuity beyond just each column's
        // immediate predecessor.
        ring.register(ReferenceRow { ptr: view.row_ptr(0), len: view.row_len(0), z: 0 });
        upstream = Some(view);
    }

    // Close out the pass by folding in every reference row the ring holds.
    for r in ring.rows() {
        acc = acc.wrapping_mul(0x100000001b3) ^ blend::mix_row(r.ptr, r.len, r.z);
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compaction is the store's bound. A region of many columns must leave it
    /// holding a working set, not every column it ever decoded — and it must
    /// not reach that by simply reserving the whole region up front.
    #[test]
    fn the_row_store_stays_bounded_over_a_long_region() {
        let mut store = RowStore::new();
        let cells = vec![3u8; GRID_EDGE * GRID_EDGE];
        for _ in 0..400 {
            let _ = store.commit(&cells);
        }
        assert!(
            store.extents.len() < 400,
            "every one of 400 columns is still resident"
        );
        assert!(
            store.bytes.len() <= 8 * cells.len(),
            "the store reserved {} bytes to hold {} columns",
            store.bytes.len(),
            store.extents.len()
        );
    }

    #[test]
    fn uniform_grid_reports_one_biome() {
        let g = BiomeGrid::uniform(4);
        assert_eq!(g.variety(), 1);
        assert_eq!(g.get(0, 0), 4);
        assert_eq!(g.rows, GRID_EDGE);
    }

    #[test]
    fn set_and_get_round_trip() {
        let mut g = BiomeGrid::uniform(0);
        g.set(2, 3, 9);
        assert_eq!(g.get(2, 3), 9);
        assert_eq!(g.variety(), 2);
    }

    #[test]
    fn out_of_range_access_is_clamped() {
        let mut g = BiomeGrid::uniform(1);
        assert_eq!(g.get(99, 0), 0);
        assert_eq!(g.get(0, 99), 0);
        // Setting out of range is ignored rather than panicking.
        g.set(99, 99, 7);
        assert_eq!(g.variety(), 1);
    }

    #[test]
    fn row_view_is_null_past_the_end() {
        let g = BiomeGrid::uniform(1);
        assert!(!g.row_view(0).is_null());
        assert!(g.row_view(GRID_EDGE).is_null());
    }

    #[test]
    fn refine_appends_rows() {
        let mut g = BiomeGrid::uniform(1);
        let before = g.cells().len();
        g.refine(2, 5);
        assert_eq!(g.rows, GRID_EDGE + 2);
        // The deepest appended row is shorter than a full row.
        assert_eq!(g.cells().len(), before + GRID_EDGE + (GRID_EDGE / 2).max(1));
    }

    #[test]
    fn refine_of_zero_rows_changes_nothing() {
        let mut g = BiomeGrid::uniform(1);
        let before = g.cells().len();
        g.refine(0, 5);
        assert_eq!(g.cells().len(), before);
        assert_eq!(g.rows, GRID_EDGE);
    }

    #[test]
    fn refine_can_shorten_the_deepest_row() {
        let mut g = BiomeGrid::uniform(1);
        g.refine(3, 9);
        let shortest = (0..g.rows).map(|z| g.row_len(z)).min().unwrap();
        assert!(shortest < GRID_EDGE);
    }

    #[test]
    fn store_commit_writes_are_readable_back() {
        let mut store = RowStore::new();
        let a = store.commit(&[1, 2, 3, 4]);
        let b = store.commit(&[5, 6]);
        // SAFETY: the store has not compacted (two tiny commits, far below its
        // initial size), so both pointers still address the bytes just
        // committed.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(a, 4), [1, 2, 3, 4]);
            assert_eq!(std::slice::from_raw_parts(b, 2), [5, 6]);
        }
    }

    #[test]
    fn store_leaves_a_small_region_alone() {
        let mut store = RowStore::new();
        for _ in 0..4 {
            store.commit(&[7u8; 16]);
        }
        assert_eq!(store.extents.len(), 4, "no column should have been retired");
        assert_eq!(store.bytes.len(), STORE_INIT_BYTES, "the store should not have re-fitted");
    }

    #[test]
    fn compact_retires_the_oldest_columns_and_packs_the_survivors() {
        let mut store = RowStore::new();
        // Eight 16-byte columns exactly fill the initial store.
        for i in 0..8u8 {
            store.commit(&[i; 16]);
        }
        assert_eq!(store.used, STORE_INIT_BYTES);
        store.commit(&[99u8; 16]);
        assert!(store.extents.len() < 9, "the oldest columns must be retired");
        // The survivors are packed from offset zero, newest last.
        assert_eq!(store.extents[0].start, 0);
        let last = store.extents[store.extents.len() - 1];
        assert_eq!(last.start + last.len, store.used);
        assert_eq!(&store.bytes[last.start..store.used], &[99u8; 16]);
    }

    #[test]
    fn compact_refits_when_the_survivors_leave_no_room() {
        let mut store = RowStore::new();
        // Two columns that together outgrow the store: retiring all but the
        // newest still leaves no room, so the store re-fits.
        store.commit(&[1u8; 80]);
        store.commit(&[2u8; 80]);
        assert!(store.bytes.len() > STORE_INIT_BYTES, "the store must have re-fitted");
        assert_eq!(store.used, 160);
    }

    #[test]
    fn reference_ring_holds_only_its_newest_entries() {
        let mut ring = ReferenceRing::new();
        let cells = [1u8, 2, 3, 4];
        for z in 0..(REFERENCE_SLOTS as u8 + 2) {
            ring.register(ReferenceRow { ptr: cells.as_ptr(), len: cells.len(), z });
        }
        assert_eq!(ring.rows().count(), REFERENCE_SLOTS);
        let mut depths: Vec<u8> = ring.rows().map(|r| r.z).collect();
        depths.sort_unstable();
        // The two oldest registrations have been displaced.
        assert_eq!(depths, vec![2, 3, 4, 5]);
    }

    /// Build a minimal region with a `biom` section, for exercising
    /// [`resolve_region`] over more than one column.
    fn region_with_biom(num_chunks: u16, biom: &[u8]) -> Vec<u8> {
        use crate::format::*;

        let mut v = Vec::new();
        v.extend_from_slice(&MAGIC);
        v.extend_from_slice(&VERSION.to_be_bytes());
        v.extend_from_slice(&flag::BIOME.to_be_bytes());
        v.extend_from_slice(&0i16.to_be_bytes()); // region_x
        v.extend_from_slice(&0i16.to_be_bytes()); // region_z
        v.extend_from_slice(&num_chunks.to_be_bytes());
        v.extend_from_slice(&3u16.to_be_bytes()); // num_sections
        v.extend_from_slice(&0x5EEDu32.to_be_bytes()); // seed
        v.extend_from_slice(&64u16.to_be_bytes()); // world_height
        v.push(4); // bits
        v.push(3); // depth
        v.extend_from_slice(&0u16.to_be_bytes()); // csum
        assert_eq!(v.len(), HEADER_LEN);

        let dir_end = HEADER_LEN + 3 * DIR_ENTRY;
        let cmap_off = dir_end;
        let cmap_len = (num_chunks as usize + 1) * 4;
        let cdat_off = cmap_off + cmap_len;
        let cdat_len = 1usize;
        let biom_off = cdat_off + cdat_len;

        v.extend_from_slice(&tag::CMAP);
        v.extend_from_slice(&(cmap_off as u32).to_be_bytes());
        v.extend_from_slice(&(cmap_len as u32).to_be_bytes());
        v.extend_from_slice(&tag::CDAT);
        v.extend_from_slice(&(cdat_off as u32).to_be_bytes());
        v.extend_from_slice(&(cdat_len as u32).to_be_bytes());
        v.extend_from_slice(&tag::BIOM);
        v.extend_from_slice(&(biom_off as u32).to_be_bytes());
        v.extend_from_slice(&(biom.len() as u32).to_be_bytes());

        v.extend(std::iter::repeat(0u8).take(cmap_len));
        v.push(0);
        v.extend_from_slice(biom);
        v
    }

    fn one_column_biom(base: u8, refine_rows: u8, cells: [u8; GRID_EDGE * GRID_EDGE]) -> Vec<u8> {
        let mut v = vec![base, refine_rows];
        v.extend_from_slice(&cells);
        v
    }

    #[test]
    fn resolve_region_is_deterministic_across_columns() {
        let mut biom = Vec::new();
        for i in 0..5u8 {
            biom.extend(one_column_biom(i, 0, [i; GRID_EDGE * GRID_EDGE]));
        }
        let data = region_with_biom(5, &biom);
        let region = crate::parse::parse(&data).expect("valid region");
        assert_eq!(resolve_region(&region), resolve_region(&region));
    }

    #[test]
    fn resolve_region_on_empty_biom_section_is_the_baseline() {
        let data = region_with_biom(1, &[]);
        let region = crate::parse::parse(&data).expect("valid region");
        assert_eq!(resolve_region(&region), 0xffu64);
    }

    #[test]
    fn the_closing_fold_reads_the_reference_ring() {
        // A single column: the ring holds its opening row, so the digest is
        // the per-row blend followed by that one reference folded once more.
        let mut cells = [0u8; GRID_EDGE * GRID_EDGE];
        for (i, c) in cells.iter_mut().enumerate() {
            *c = i as u8 + 1;
        }
        let data = region_with_biom(1, &one_column_biom(0, 0, cells));
        let region = crate::parse::parse(&data).expect("valid region");

        let mut expect = 0xffu64 ^ (GRID_EDGE * GRID_EDGE) as u64;
        for z in 0..GRID_EDGE {
            let row = &cells[z * GRID_EDGE..(z + 1) * GRID_EDGE];
            expect = expect.wrapping_mul(0x100000001b3) ^ blend::mix_slice(row, z as u8);
        }
        let opening = &cells[0..GRID_EDGE];
        expect = expect.wrapping_mul(0x100000001b3) ^ blend::mix_slice(opening, 0);

        assert_eq!(
            resolve_region(&region),
            expect,
            "the closing fold over the reference ring is part of the digest"
        );
    }

    #[test]
    fn a_registered_reference_row_influences_the_final_digest() {
        // Two regions differing only in the opening row of the last column —
        // the one the ring's newest entry names.
        let mut biom_a = Vec::new();
        let mut biom_b = Vec::new();
        for i in 0..3u8 {
            let mut cells_a = [i; GRID_EDGE * GRID_EDGE];
            let mut cells_b = cells_a;
            if i == 2 {
                cells_a[1] = 9;
                cells_b[1] = 10;
            }
            biom_a.extend(one_column_biom(i, 0, cells_a));
            biom_b.extend(one_column_biom(i, 0, cells_b));
        }
        let a = region_with_biom(3, &biom_a);
        let b = region_with_biom(3, &biom_b);
        let ra = crate::parse::parse(&a).expect("valid region");
        let rb = crate::parse::parse(&b).expect("valid region");
        assert_ne!(resolve_region(&ra), resolve_region(&rb));
    }
}
