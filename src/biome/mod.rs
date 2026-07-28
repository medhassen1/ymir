//! Biome grid resolution and blending.
//!
//! Biomes are stored at quarter resolution — one cell per 4x4x4 blocks — and
//! blended at render time so a desert fades into a savanna instead of changing
//! at a cell boundary.
//!
//! A region's biome pass decodes one column at a time, but continuity across
//! the region means a later column's blend sometimes needs to see an earlier
//! column's row, not only its own — so a column's decoded cells cannot simply
//! live and die with that column's turn through the loop. [`CellArena`] gives
//! the pass a home for decoded rows that outlives any single column: built
//! once per region rather than once per column, it appends every column's
//! cells into fixed-size chunks and hands back a view into them instead of an
//! owned buffer. A handful of columns, spaced out across the region, register
//! their opening row in a retained list, so a later column always has a
//! nearby reference to blend against rather than only ever its immediate
//! predecessor. Because the arena lives for the whole pass, its memory is
//! bounded separately from any one column's lifetime: once the accumulated
//! volume crosses a threshold, [`CellArena::compact`] drops the oldest
//! chunks, the way a buffer pool reclaims cold pages instead of growing
//! without bound on a region with many columns.

use crate::blend;
use crate::common::*;
use crate::parse::Region;
use crate::reader::Cursor;

/// Cells along one edge of a chunk's biome grid.
pub const GRID_EDGE: usize = BIOME_EDGE;

/// Bytes held by one [`CellArena`] chunk.
///
/// A column's decoded volume — the base grid plus up to [`MAX_STACK`]
/// refinement rows — fits with room to spare, so an ordinary column's commit
/// never has to look past the chunk it lands in.
const ARENA_CHUNK_BYTES: usize = 1024;

/// Accumulated resident bytes across an arena's live chunks that triggers
/// [`CellArena::compact`].
///
/// An ordinary region's biome section — a modest number of columns, most
/// carrying little or no refinement — never approaches this. A region built
/// from many heavily refined columns does, which is exactly the case the
/// bound exists to catch: without it, a region-lifetime arena would keep
/// every column's rows resident for the whole pass no matter how many
/// columns the region carries.
const COMPACT_THRESHOLD_BYTES: usize = 4096;

/// Spacing between columns that register a retained reference row.
///
/// The first column has no earlier neighbour to blend against, so retention
/// starts at the first multiple of the stride past it.
const RETAIN_STRIDE: usize = 8;

/// A chunk's biome cells, stored as one span per row rather than a fixed
/// stride.
///
/// This is the scratch buffer a single column decodes into. [`resolve_region`]
/// builds one per column, then commits its finished `cells` into the region's
/// [`CellArena`] so the bytes survive past this struct's own scope. Every row
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
    /// [`CellArena`] — once committed, [`resolve_region`] addresses the row
    /// through the arena's copy instead (see [`ColumnView::row_ptr`]), since
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

/// A region-lifetime arena for decoded biome rows.
///
/// Built once per region rather than once per column, so a view handed out
/// while decoding one column stays valid while later columns are decoded —
/// which is what lets a retained reference row (see [`resolve_region`]) be
/// read again well after its own column has finished. Cells are appended into
/// fixed-size chunks, each stored as an exact-sized boxed slice; a chunk with
/// no room left for the next commit is left as-is and a fresh one takes over,
/// so a single commit is never split across two chunks.
struct CellArena {
    /// Chunks holding committed column bytes, oldest first.
    chunks: Vec<Box<[u8]>>,
    /// Bytes already written into the last chunk.
    used: usize,
    /// Bytes held across all currently resident chunks.
    resident: usize,
}

impl CellArena {
    fn new() -> CellArena {
        CellArena { chunks: Vec::new(), used: 0, resident: 0 }
    }

    /// Commit `bytes` into the arena and return a pointer to where they
    /// landed.
    ///
    /// If what remains of the current chunk cannot hold `bytes`, a fresh
    /// chunk takes over first, so the returned pointer's `bytes.len()` cells
    /// are always contiguous — addressing live memory for as long as the
    /// chunk backing them stays resident (see [`CellArena::compact`]).
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
    /// arena would keep every column's rows resident for the whole pass no
    /// matter how many columns the region carries. The chunk currently being
    /// written to is never dropped, since the next commit needs somewhere to
    /// land.
    fn compact(&mut self, threshold: usize) {
        while self.resident > threshold && self.chunks.len() > 1 {
            let oldest = self.chunks.remove(0);
            self.resident -= oldest.len();
        }
    }
}

/// One column's rows, committed into the region's [`CellArena`].
///
/// Mirrors [`BiomeGrid`]'s `(start, len)` spans, but resolved against the
/// arena's pointer rather than a private buffer, since by the time this
/// exists the grid that produced it has already handed its cells to the
/// arena.
struct ColumnView {
    /// First cell of this column's grid within the arena.
    base: *const u8,
    /// `(start, len)` into the committed buffer, one per row.
    spans: Vec<(usize, usize)>,
}

impl ColumnView {
    /// A pointer to the first cell of row `z`, or null past the row count.
    fn row_ptr(&self, z: usize) -> *const u8 {
        self.spans.get(z).map_or(std::ptr::null(), |&(start, _)| {
            // SAFETY: `spans` was taken from the `BiomeGrid` whose cells were
            // just copied into the arena at `base`, so every `start` here
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

/// A reference row retained past its own column, for a later column's blend
/// to read — see [`resolve_region`].
struct ReferenceRow {
    ptr: *const u8,
    len: usize,
    z: u8,
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
/// then committed into a region-wide [`CellArena`] rather than freed with the
/// column: a spaced-out subset of columns keep their opening row registered
/// in a retained list for the rest of the pass, giving later columns a
/// cross-region reference to blend against beyond just their immediate
/// predecessor. The retained rows are folded into the digest once more at the
/// end, closing out the pass.
pub fn resolve_region(region: &Region) -> u64 {
    let n = region.worked_chunks();
    let mut c = Cursor::new(region.slice(region.biom));

    let mut arena = CellArena::new();
    let mut retained: Vec<ReferenceRow> = Vec::new();
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

        // Commit this column's finished cells into the region's arena. Only
        // past this point is there a pointer stable enough to retain past
        // this column's own scope.
        let spans = std::mem::take(&mut grid.spans);
        let base_ptr = arena.commit(&grid.cells);
        let view = ColumnView { base: base_ptr, spans };

        for &z in &rows_to_blend {
            let ptr = view.row_ptr(z);
            let len = view.row_len(z);
            acc = acc.wrapping_mul(0x100000001b3) ^ blend::mix_row(ptr, len, z as u8);
        }

        // Every `RETAIN_STRIDE`-th column past the first keeps its opening
        // row alive as a reference for later columns to blend against,
        // giving the region continuity beyond just each column's immediate
        // predecessor.
        if cid > 0 && cid % RETAIN_STRIDE == 0 {
            retained.push(ReferenceRow { ptr: view.row_ptr(0), len: view.row_len(0), z: 0 });
        }

        // Bound the arena's resident memory now that this column's bytes are
        // safely committed.
        arena.compact(COMPACT_THRESHOLD_BYTES);
    }

    // Close out the pass by folding in every retained reference row once.
    for r in &retained {
        acc = acc.wrapping_mul(0x100000001b3) ^ blend::mix_row(r.ptr, r.len, r.z);
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn arena_commit_writes_are_readable_back() {
        let mut arena = CellArena::new();
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
        let mut arena = CellArena::new();
        let filler = vec![0u8; ARENA_CHUNK_BYTES - 4];
        arena.commit(&filler);
        assert_eq!(arena.chunks.len(), 1);
        // Only 4 bytes remain in the first chunk; this does not fit.
        arena.commit(&[1, 2, 3, 4, 5]);
        assert_eq!(arena.chunks.len(), 2);
    }

    #[test]
    fn compact_leaves_a_small_arena_untouched() {
        let mut arena = CellArena::new();
        arena.commit(&[1, 2, 3]);
        arena.compact(COMPACT_THRESHOLD_BYTES);
        assert_eq!(arena.chunks.len(), 1);
    }

    #[test]
    fn compact_drops_oldest_chunks_once_over_threshold() {
        let mut arena = CellArena::new();
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
        let mut arena = CellArena::new();
        arena.commit(&[1, 2, 3]);
        arena.compact(0);
        assert_eq!(arena.chunks.len(), 1, "the chunk being written to must survive");
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
    fn retained_reference_row_influences_the_final_digest() {
        // Nine columns puts one (column 8) past the first retain stride,
        // while staying far below the compaction threshold, so its row is
        // still live when the closing fold reads it.
        let mut biom_a = Vec::new();
        let mut biom_b = Vec::new();
        for i in 0..9u8 {
            let fill = if i == 8 { 1 } else { i };
            biom_a.extend(one_column_biom(i, 0, [i; GRID_EDGE * GRID_EDGE]));
            biom_b.extend(one_column_biom(i, 0, [fill; GRID_EDGE * GRID_EDGE]));
        }
        let a = region_with_biom(9, &biom_a);
        let b = region_with_biom(9, &biom_b);
        let ra = crate::parse::parse(&a).expect("valid region");
        let rb = crate::parse::parse(&b).expect("valid region");
        assert_ne!(
            resolve_region(&ra),
            resolve_region(&rb),
            "changing the retained column's cells must change the digest"
        );
    }
}
