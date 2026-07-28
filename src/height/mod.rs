//! Surface heightmap rebuild.
//!
//! The heightmap records, for every column of a chunk, the world Y of the
//! highest non-air block. It drives sky-light seeding and terrain queries, so it
//! is rebuilt whenever a region's blocks change. Columns are stored in one flat
//! array and the profiler reads them through a cursor, because the smoothing
//! pass touches each column several times and re-indexing dominated the profile.
//!
//! A region's height pass rebuilds one column's map at a time, but a slope
//! reading is more useful with a neighbour to compare against, so recent
//! columns keep their map's span registered as landmarks for later columns to
//! measure against — see [`LandmarkRing`]. That only works if a column's map
//! outlives the column itself, which is what [`HeightArena`] is for: built once
//! per region rather than once per column, it bump-appends every column's
//! finished map into fixed-size chunks and hands back a pointer into them
//! instead of an owned buffer.
//!
//! The pass runs in waves: a column whose map needed no overflow rows is flat
//! terrain and closes out the wave the taller columns before it opened. The
//! arena takes a watermark at each such boundary and rewinds its bump pointer
//! there when the next one arrives, handing back the chunks the intervening
//! columns forced — see [`HeightArena::rewind`]. That is the arena's bound: a
//! region of ordinary terrain never carries more than a wave's worth of maps.

use crate::chunk::{self, linear_index, Column};
use crate::common::*;
use crate::parse::Region;
use crate::profile;
use crate::reader::Cursor;

/// Columns along one chunk edge.
pub const MAP_EDGE: usize = SECTION_EDGE;
/// Columns in one chunk's heightmap.
pub const MAP_AREA: usize = MAP_EDGE * MAP_EDGE;

/// A chunk's surface heightmap.
pub struct HeightMap {
    columns: Vec<u16>,
    /// World Y the map is measured from.
    pub base_y: i32,
}

impl HeightMap {
    /// A flat map at `base_y` with every column empty.
    pub fn flat(base_y: i32) -> HeightMap {
        HeightMap { columns: vec![0u16; MAP_AREA], base_y }
    }

    /// The number of columns held.
    pub fn len(&self) -> usize {
        self.columns.len()
    }

    /// How many rows of columns the map currently holds.
    ///
    /// A flat map holds exactly `MAP_EDGE` rows; a column whose stack recorded
    /// overflow rows holds more, so the profiler walks however many rows the
    /// map actually grew to.
    pub fn rows(&self) -> usize {
        self.columns.len() / MAP_EDGE
    }

    /// Whether the map holds no columns.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    /// The recorded height of a column.
    pub fn get(&self, x: usize, z: usize) -> u16 {
        self.columns.get(z * MAP_EDGE + x).copied().unwrap_or(0)
    }

    /// Record a column's height, keeping the greater of the two.
    pub fn raise(&mut self, x: usize, z: usize, y: u16) {
        if let Some(slot) = self.columns.get_mut(z * MAP_EDGE + x) {
            if y > *slot {
                *slot = y;
            }
        }
    }

    /// A cursor onto the column array.
    ///
    /// The smoothing and profiling passes read columns through this rather than
    /// bounds-checking each of the four neighbours they touch per column.
    pub fn cursor(&self) -> *const u16 {
        self.columns.as_ptr()
    }

    /// The columns, as a slice.
    pub fn columns(&self) -> &[u16] {
        &self.columns
    }

    /// The tallest column recorded.
    pub fn peak(&self) -> u16 {
        self.columns.iter().copied().max().unwrap_or(0)
    }

    /// Extend the map upward so it can record heights in a taller world.
    ///
    /// A region whose columns reach past the current span needs more resolution
    /// than the flat map allocated, so the array grows to carry the overflow
    /// rows the taller sections occupy.
    pub fn extend_span(&mut self, extra_rows: usize) {
        let want = MAP_AREA + extra_rows * MAP_EDGE;
        if want > self.columns.len() {
            self.columns.resize(want, 0);
        }
    }
}

/// Columns held by one [`HeightArena`] chunk.
///
/// An ordinary column's map — the flat map plus up to [`MAX_STACK`] overflow
/// rows — fits with room to spare, so an ordinary commit never has to look
/// past the chunk it lands in.
const ARENA_CHUNK_UNITS: usize = 4 * MAP_AREA;

/// Chunks a single wave may reach before [`HeightArena::rewind`] gives up.
///
/// Rewinding reuses the space above the watermark for the next wave, which pays
/// only while a wave fits inside the chunk the watermark sits in. A region
/// whose waves have needed a chunk beyond it has outgrown the scheme — every
/// rewind would be undone by the very next commit — so the arena stops and
/// keeps what it has.
const REWIND_MAX_CHUNKS: usize = 1;

/// Slots in [`LandmarkRing`].
const LANDMARK_SLOTS: usize = 4;

/// A region-lifetime arena for finished column heightmaps.
///
/// Built once per region rather than once per column, so a span handed out
/// while rebuilding one column's map stays meaningful while later columns are
/// rebuilt — which is what lets a landmark (see [`rebuild_region`]) be read
/// again well after its own column has finished. Columns are bump-appended into
/// fixed-size chunks, each stored as an exact-sized boxed slice; a chunk with no
/// room left for the next commit is left as-is and a fresh one takes over, so a
/// single column's map is never split across two chunks.
struct HeightArena {
    /// Chunks holding committed column data, oldest first.
    chunks: Vec<Box<[u16]>>,
    /// Columns already written into the last chunk.
    used: usize,
    /// The bump position the next [`HeightArena::rewind`] returns to: how many
    /// chunks were open at the last wave boundary, and how far into the last of
    /// them the arena had written.
    mark: (usize, usize),
    /// The most chunks the arena has ever held at once.
    peak_chunks: usize,
}

impl HeightArena {
    fn new() -> HeightArena {
        HeightArena { chunks: Vec::new(), used: 0, mark: (0, 0), peak_chunks: 0 }
    }

    /// Columns held across every chunk the arena currently holds.
    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.chunks.iter().map(|c| c.len()).sum()
    }

    /// Commit a column's map into the arena and return a pointer to where it
    /// landed.
    ///
    /// If what remains of the current chunk cannot hold `cols`, a fresh chunk
    /// takes over first, so the returned pointer's `cols.len()` entries are
    /// always contiguous — addressing live memory for as long as the arena's
    /// bump pointer stays past them (see [`HeightArena::rewind`]).
    fn commit(&mut self, cols: &[u16]) -> *const u16 {
        let len = cols.len();
        let fits_current = self.chunks.last().is_some_and(|c| self.used + len <= c.len());
        if !fits_current {
            let cap = len.max(ARENA_CHUNK_UNITS);
            self.chunks.push(vec![0u16; cap].into_boxed_slice());
            self.used = 0;
            self.peak_chunks = self.peak_chunks.max(self.chunks.len());
        }
        let chunk = self.chunks.last_mut().expect("a chunk was just ensured above");
        chunk[self.used..self.used + len].copy_from_slice(cols);
        // SAFETY: `chunk` is a live `Box<[u16]>` at least `self.used + len`
        // entries long — either it already fit `cols` past `self.used`, or a
        // chunk sized to hold at least `cols` was just pushed — so this
        // offset and the `len` entries from it lie inside the allocation.
        let ptr = unsafe { chunk.as_ptr().add(self.used) };
        self.used += len;
        ptr
    }

    /// Take a watermark here: this is where the next wave starts, and where a
    /// later [`HeightArena::rewind`] returns to.
    fn mark(&mut self) {
        self.mark = (self.chunks.len(), self.used);
    }

    /// Return the bump pointer to the last watermark, releasing every chunk
    /// opened past it.
    ///
    /// This is the arena's bound. The maps of a finished wave have had their
    /// slopes folded and nothing after the wave asks for them again, so the next
    /// wave writes over the same span instead of the arena growing once per
    /// column for the length of the region.
    ///
    /// A wave that needed a chunk beyond the watermark's has outgrown the
    /// scheme — rewinding would be undone by the next commit — so past
    /// [`REWIND_MAX_CHUNKS`] the arena keeps its chunks and simply carries on.
    fn rewind(&mut self) {
        if self.peak_chunks > REWIND_MAX_CHUNKS {
            return;
        }
        // Releases every chunk opened since the watermark was taken.
        self.chunks.truncate(self.mark.0);
        self.used = self.mark.1;
    }
}

/// A column's map span, named for later columns and the closing profile pass to
/// measure slope against.
struct Landmark {
    ptr: *const u16,
    rows: usize,
}

/// The map spans [`rebuild_region`] profiles once more at the end of the pass.
///
/// Fixed capacity, reused round-robin: every column offers its map as a
/// landmark, and the ring keeps the most recent few so the closing pass costs
/// the same on a region of four columns and one of four thousand.
struct LandmarkRing {
    slots: [Option<Landmark>; LANDMARK_SLOTS],
    next: usize,
}

impl LandmarkRing {
    fn new() -> LandmarkRing {
        LandmarkRing { slots: [None, None, None, None], next: 0 }
    }

    /// Put `landmark` in the next slot, displacing whatever that slot held.
    fn offer(&mut self, landmark: Landmark) {
        self.slots[self.next] = Some(landmark);
        self.next = (self.next + 1) % LANDMARK_SLOTS;
    }

    /// The landmarks currently held, oldest slot first.
    fn landmarks(&self) -> impl Iterator<Item = &Landmark> {
        self.slots.iter().flatten()
    }
}

/// Scan one column's sections and record the highest solid block per position.
pub fn scan_column(col: &Column) -> HeightMap {
    let mut map = HeightMap::flat(col.base_y);
    for s in &col.sections {
        if s.is_empty() {
            continue;
        }
        for y in 0..SECTION_EDGE {
            for z in 0..SECTION_EDGE {
                for x in 0..SECTION_EDGE {
                    if s.state_at(linear_index(x, y, z)) != 0 {
                        let world_y = s.base_y + y as i32;
                        let rel = (world_y - col.base_y).max(0) as u16;
                        map.raise(x, z, rel);
                    }
                }
            }
        }
    }
    map
}

/// How many extra rows a column's sections need beyond the flat map.
///
/// A column whose stack reaches past one section's worth of height records its
/// overflow in additional rows, so the profiler can distinguish a tall spire
/// from a clipped one.
pub fn overflow_rows(col: &Column, world_height: usize) -> usize {
    let span = col.sections.len() * SECTION_EDGE;
    if span <= SECTION_EDGE {
        return 0;
    }
    let over = (span - SECTION_EDGE) / SECTION_EDGE;
    over.min(world_height / SECTION_EDGE).min(MAX_STACK)
}

/// Whether any section of `col` could actually raise a column's recorded
/// height.
///
/// A stack whose every section is empty cannot change what [`scan_column`]
/// records anywhere: every reading stays at the flat map's base value. Growing
/// the map's storage for such a column's predicted overflow would only add
/// rows holding the same zero the flat map already starts with.
fn overflow_is_solid(col: &Column) -> bool {
    col.sections.iter().any(|s| !s.is_empty())
}

/// Rebuild the heightmap for every column and fold a digest of the result.
///
/// Each column's finished map is committed into a region-wide [`HeightArena`]
/// rather than freed with the column, and offered to the [`LandmarkRing`],
/// giving a later column a neighbour to measure slope against beyond just its
/// immediate predecessor. The row count handed to the profiler is always the
/// map's actual row count — never a predicted span the map's storage may not
/// have actually grown to — so the profiler never reads past what a column
/// really committed. The ring's landmarks are profiled once more at the end,
/// closing out the pass.
///
/// A column that needed no overflow rows is flat terrain and closes out the
/// wave the taller columns before it opened, so the arena rewinds to the
/// watermark taken at the previous boundary and takes a fresh one here.
pub fn rebuild_region(region: &Region, n: usize) -> u64 {
    // The `hgts` section supplies a per-region bias applied to every column.
    let bias = {
        let mut c = Cursor::new(region.slice(region.hgts));
        c.i16() as i32
    };

    let mut arena = HeightArena::new();
    let mut landmarks = LandmarkRing::new();
    let mut acc = 0xffu64 ^ (bias as i64 as u64);

    for cid in 0..n {
        let col = match chunk::decode(region, cid) {
            Ok(c) => c,
            Err(_) => continue,
        };
        if col.sections.is_empty() {
            continue;
        }
        let mut map = scan_column(&col);

        // Tall columns record their overflow in extra rows before profiling.
        // Growing the map costs a reallocation, which is skipped when the
        // overflow cannot hold a solid block in the first place.
        let extra = overflow_rows(&col, region.world_height);
        if extra > 0 && overflow_is_solid(&col) {
            map.extend_span(extra);
        }

        // A map that stayed at its base span is flat terrain: the wave the
        // taller columns before it opened ends here, so the arena returns to
        // the watermark that wave started from.
        let wave_boundary = map.rows() == MAP_EDGE;
        if wave_boundary {
            arena.rewind();
        }

        // Commit this column's finished map into the region's arena. Only
        // past this point is there a pointer stable enough to name past this
        // column's own scope. The row count committed is the map's own —
        // whatever it actually grew to, never a larger predicted span.
        let rows = map.rows();
        let ptr = arena.commit(map.columns());
        acc = acc.wrapping_mul(0x100000001b3) ^ profile::slope_sum(ptr, rows, MAP_EDGE, bias);
        landmarks.offer(Landmark { ptr, rows });

        if wave_boundary {
            // The next wave starts from here.
            arena.mark();
        }
    }

    // Close out the pass by profiling every landmark the ring holds.
    for l in landmarks.landmarks() {
        acc = acc.wrapping_mul(0x100000001b3) ^ profile::slope_sum(l.ptr, l.rows, MAP_EDGE, bias);
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::SectionData;

    fn column_with(sections: usize, solid: bool) -> Column {
        let secs = (0..sections)
            .map(|i| SectionData {
                palette: vec![0, 1],
                blocks: vec![if solid { 1 } else { 0 }; SECTION_VOLUME],
                flags: 0,
                base_y: (i * 16) as i32,
            })
            .collect();
        Column { sections: secs, base_y: 0, flags: 0, cid: 0 }
    }

    #[test]
    fn flat_map_starts_empty() {
        let m = HeightMap::flat(0);
        assert_eq!(m.len(), MAP_AREA);
        assert_eq!(m.peak(), 0);
        assert!(!m.is_empty());
    }

    #[test]
    fn rows_reflects_the_flat_and_extended_span() {
        let mut m = HeightMap::flat(0);
        assert_eq!(m.rows(), MAP_EDGE);
        m.extend_span(2);
        assert_eq!(m.rows(), MAP_EDGE + 2);
    }

    #[test]
    fn raise_keeps_the_greater_height() {
        let mut m = HeightMap::flat(0);
        m.raise(1, 2, 10);
        m.raise(1, 2, 4);
        assert_eq!(m.get(1, 2), 10);
        m.raise(1, 2, 12);
        assert_eq!(m.get(1, 2), 12);
    }

    #[test]
    fn get_out_of_range_is_zero() {
        let m = HeightMap::flat(0);
        assert_eq!(m.get(99, 99), 0);
    }

    #[test]
    fn scan_records_the_top_of_a_solid_column() {
        let col = column_with(2, true);
        let m = scan_column(&col);
        // Two sections of solid blocks: the top is at relative y 31.
        assert_eq!(m.get(0, 0), 31);
        assert_eq!(m.peak(), 31);
    }

    #[test]
    fn scan_of_air_records_nothing() {
        let col = column_with(1, false);
        let m = scan_column(&col);
        assert_eq!(m.peak(), 0);
    }

    #[test]
    fn overflow_rows_tracks_stack_height() {
        assert_eq!(overflow_rows(&column_with(1, true), 256), 0);
        assert_eq!(overflow_rows(&column_with(3, true), 256), 2);
        // Bounded by the world height.
        assert_eq!(overflow_rows(&column_with(9, true), 32), 2);
    }

    fn empty_flagged_column(sections: usize) -> Column {
        let secs = (0..sections)
            .map(|i| SectionData {
                palette: vec![0],
                blocks: Vec::new(),
                flags: crate::format::sec::EMPTY,
                base_y: (i * 16) as i32,
            })
            .collect();
        Column { sections: secs, base_y: 0, flags: 0, cid: 0 }
    }

    #[test]
    fn overflow_is_solid_ignores_an_all_empty_stack() {
        assert!(!overflow_is_solid(&empty_flagged_column(3)));
        assert!(overflow_is_solid(&column_with(3, true)));
        // A single solid section among empty ones is still enough.
        let mut mixed = empty_flagged_column(2);
        mixed.sections.push(column_with(1, true).sections.remove(0));
        assert!(overflow_is_solid(&mixed));
    }

    #[test]
    fn extend_span_only_grows() {
        let mut m = HeightMap::flat(0);
        m.extend_span(2);
        assert_eq!(m.len(), MAP_AREA + 2 * MAP_EDGE);
        m.extend_span(1);
        assert_eq!(m.len(), MAP_AREA + 2 * MAP_EDGE, "extend must not shrink");
    }

    #[test]
    fn arena_commit_writes_are_readable_back() {
        let mut arena = HeightArena::new();
        let a = arena.commit(&[1, 2, 3, 4]);
        let b = arena.commit(&[5, 6]);
        // SAFETY: the bump pointer has only moved forward since both were
        // committed, so both still address the entries just written.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(a, 4), [1, 2, 3, 4]);
            assert_eq!(std::slice::from_raw_parts(b, 2), [5, 6]);
        }
    }

    #[test]
    fn arena_starts_a_new_chunk_once_the_current_one_is_full() {
        let mut arena = HeightArena::new();
        let filler = vec![0u16; ARENA_CHUNK_UNITS - 4];
        arena.commit(&filler);
        assert_eq!(arena.chunks.len(), 1);
        arena.commit(&[1, 2, 3, 4, 5]);
        assert_eq!(arena.chunks.len(), 2);
    }

    #[test]
    fn rewinding_without_a_watermark_empties_the_arena() {
        let mut arena = HeightArena::new();
        arena.commit(&[1, 2, 3]);
        assert_eq!(arena.capacity(), ARENA_CHUNK_UNITS);
        arena.rewind();
        assert_eq!(arena.capacity(), 0, "nothing was marked, so nothing is kept");
        assert_eq!(arena.used, 0);
    }

    #[test]
    fn rewinding_returns_the_bump_pointer_to_the_watermark() {
        let mut arena = HeightArena::new();
        arena.commit(&[1, 2, 3]);
        arena.mark();
        arena.commit(&[4, 5, 6, 7]);
        assert_eq!(arena.used, 7);
        arena.rewind();
        assert_eq!(arena.used, 3, "the wave after the watermark is reusable space");
        assert_eq!(arena.chunks.len(), 1);
    }

    #[test]
    fn rewinding_stops_once_a_wave_outgrows_its_chunk() {
        let mut arena = HeightArena::new();
        arena.commit(&[1, 2, 3]);
        arena.mark();
        // A wave that needs a second chunk has outgrown the scheme.
        arena.commit(&vec![0u16; ARENA_CHUNK_UNITS]);
        assert_eq!(arena.chunks.len(), 2);
        arena.rewind();
        assert_eq!(arena.chunks.len(), 2, "the arena must keep what it has");
    }

    #[test]
    fn landmark_ring_holds_only_its_newest_slots() {
        let mut ring = LandmarkRing::new();
        let cols = [1u16, 2, 3, 4];
        for _ in 0..LANDMARK_SLOTS + 3 {
            ring.offer(Landmark { ptr: cols.as_ptr(), rows: 1 });
        }
        assert_eq!(ring.landmarks().count(), LANDMARK_SLOTS, "capacity must be fixed");
    }

    /// Build a minimal region with a flat, single-section column repeated
    /// `num_chunks` times, for exercising [`rebuild_region`] over more than
    /// one column.
    fn region_with_flat_columns(num_chunks: u16) -> Vec<u8> {
        use crate::format::*;

        fn column_record() -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&1u16.to_be_bytes()); // one section
            out.extend_from_slice(&0i16.to_be_bytes());
            out.push(0);
            out.push(0);
            out.push(crate::format::sec::UNIFORM);
            out.extend_from_slice(&2u16.to_be_bytes());
            out.extend_from_slice(&0u16.to_be_bytes());
            out.extend_from_slice(&5u16.to_be_bytes());
            out.extend_from_slice(&1u16.to_be_bytes()); // uniform index
            out
        }

        let cols: Vec<Vec<u8>> = (0..num_chunks).map(|_| column_record()).collect();
        let cdat: Vec<u8> = cols.iter().flatten().copied().collect();

        let mut v = Vec::new();
        v.extend_from_slice(&MAGIC);
        v.extend_from_slice(&VERSION.to_be_bytes());
        v.extend_from_slice(&flag::HEIGHT.to_be_bytes());
        v.extend_from_slice(&0i16.to_be_bytes());
        v.extend_from_slice(&0i16.to_be_bytes());
        v.extend_from_slice(&num_chunks.to_be_bytes());
        v.extend_from_slice(&3u16.to_be_bytes());
        v.extend_from_slice(&0x5EEDu32.to_be_bytes());
        v.extend_from_slice(&64u16.to_be_bytes());
        v.push(4);
        v.push(3);
        v.extend_from_slice(&0u16.to_be_bytes());
        assert_eq!(v.len(), HEADER_LEN);

        let dir_end = HEADER_LEN + 3 * DIR_ENTRY;
        let cmap_off = dir_end;
        let cmap_len = (num_chunks as usize + 1) * 4;
        let cdat_off = cmap_off + cmap_len;
        let hgts_off = cdat_off + cdat.len();
        let hgts = 3i16.to_be_bytes();

        v.extend_from_slice(&tag::CMAP);
        v.extend_from_slice(&(cmap_off as u32).to_be_bytes());
        v.extend_from_slice(&(cmap_len as u32).to_be_bytes());
        v.extend_from_slice(&tag::CDAT);
        v.extend_from_slice(&(cdat_off as u32).to_be_bytes());
        v.extend_from_slice(&(cdat.len() as u32).to_be_bytes());
        v.extend_from_slice(&tag::HGTS);
        v.extend_from_slice(&(hgts_off as u32).to_be_bytes());
        v.extend_from_slice(&(hgts.len() as u32).to_be_bytes());

        let mut at = 0u32;
        for c in &cols {
            v.extend_from_slice(&at.to_be_bytes());
            at += c.len() as u32;
        }
        v.extend_from_slice(&at.to_be_bytes());
        v.extend_from_slice(&cdat);
        v.extend_from_slice(&hgts);
        v
    }

    #[test]
    fn rebuild_region_is_deterministic_across_columns() {
        let data = region_with_flat_columns(9);
        let region = crate::parse::parse(&data).expect("valid region");
        let n = region.worked_chunks();
        assert_eq!(rebuild_region(&region, n), rebuild_region(&region, n));
    }
}
