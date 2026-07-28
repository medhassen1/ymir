//! Parse a `.ymr` region into a [`Region`]: validate the header, walk the
//! section directory, and expose each section as a byte slice.
//!
//! This layer only proves that the sections are present and in bounds. Decoding
//! the contents of a section is the job of the module that owns it —
//! [`crate::chunk`], [`crate::palette`], [`crate::entity`] and so on.

use crate::common::*;
use crate::format::*;
use crate::reader::Cursor;

/// A located section: an absolute offset and length into the region buffer.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct Section {
    /// Absolute byte offset of the section within the region.
    pub offset: usize,
    /// Section length in bytes.
    pub len: usize,
}

impl Section {
    /// Whether the directory named this section at all.
    pub fn is_present(&self) -> bool {
        self.len != 0
    }
}

/// A parsed, structurally valid region. Section bytes are borrowed from the
/// original buffer.
pub struct Region<'a> {
    /// The whole region buffer.
    pub data: &'a [u8],
    /// Format version; always [`VERSION`] for a region that parsed.
    pub version: u16,
    /// Stage-selector bits; see [`crate::format::flag`].
    pub flags: u16,
    /// Region coordinate on the X axis.
    pub region_x: i16,
    /// Region coordinate on the Z axis.
    pub region_z: i16,
    /// Number of chunk columns stored.
    pub num_chunks: usize,
    /// World generation seed, carried for structure placement.
    pub seed: u32,
    /// World height in blocks.
    pub world_height: usize,
    /// Palette index width, in bits.
    pub bits: u8,
    /// Structure-template nesting budget.
    pub depth: u8,
    /// Advisory header checksum, not validated during decode.
    pub csum: u16,

    /// Chunk offset table.
    pub cmap: Section,
    /// Chunk section stacks.
    pub cdat: Section,
    /// Block-state palettes.
    pub palt: Section,
    /// Entity component records.
    pub ents: Section,
    /// Tile-entity property trees.
    pub tile: Section,
    /// Light source list.
    pub lgts: Section,
    /// Per-column heightmap.
    pub hgts: Section,
    /// Biome cell grid.
    pub biom: Section,
    /// Structure template instances.
    pub strc: Section,
    /// Scheduled tick queue.
    pub tick: Section,
}

impl<'a> Region<'a> {
    /// The bytes of a located section, or an empty slice if it was absent or
    /// the directory named a range this buffer does not cover.
    pub fn slice(&self, s: Section) -> &'a [u8] {
        self.data.get(s.offset..s.offset + s.len).unwrap_or(&[])
    }

    /// How many chunk columns this region will actually process, after the
    /// per-region work cap.
    pub fn worked_chunks(&self) -> usize {
        self.num_chunks.min(WORK_CAP)
    }
}

/// Parse and structurally validate a region.
pub fn parse(data: &[u8]) -> Result<Region<'_>, Status> {
    if data.len() < HEADER_LEN || data.len() > REGION_MAX {
        return Err(Status::Malformed);
    }

    let mut c = Cursor::new(data);
    if c.tag() != MAGIC {
        return Err(Status::Malformed);
    }
    let version = c.u16();
    let flags = c.u16();
    let region_x = c.i16();
    let region_z = c.i16();
    let num_chunks = c.u16() as usize;
    let num_sections = c.u16() as usize;
    let seed = c.u32();
    let world_height = c.u16() as usize;
    let bits = c.u8();
    let depth = c.u8();
    let csum = c.u16();

    if !c.ok
        || version != VERSION
        || num_chunks == 0
        || num_chunks > MAX_CHUNKS
        || num_sections == 0
        || num_sections > MAX_SECTIONS
        || world_height == 0
        || world_height > MAX_WORLD_HEIGHT
        || !valid_palette_bits(bits)
    {
        return Err(Status::Malformed);
    }

    let dir_end = HEADER_LEN + num_sections * DIR_ENTRY;
    if dir_end > data.len() {
        return Err(Status::Malformed);
    }

    let mut located = [Section::default(); 10];
    let mut dir = Cursor::at(data, HEADER_LEN);
    for _ in 0..num_sections {
        let t = dir.tag();
        let offset = dir.u32() as usize;
        let len = dir.u32() as usize;
        // A section must start after the directory and lie wholly inside the
        // buffer. The subtraction is safe because `offset <= data.len()`.
        if offset < dir_end || offset > data.len() || len > data.len() - offset {
            return Err(Status::Malformed);
        }
        let section = Section { offset, len };
        let idx = match t {
            tag::CMAP => 0,
            tag::CDAT => 1,
            tag::PALT => 2,
            tag::ENTS => 3,
            tag::TILE => 4,
            tag::LGTS => 5,
            tag::HGTS => 6,
            tag::BIOM => 7,
            tag::STRC => 8,
            tag::TICK => 9,
            // Unknown tags are reserved for later versions and skipped.
            _ => continue,
        };
        located[idx] = section;
    }
    if !dir.ok {
        return Err(Status::Malformed);
    }

    // The chunk offset table must hold one offset per chunk plus a terminator,
    // and there has to be chunk data for those offsets to point into.
    if located[0].len < (num_chunks + 1) * 4 || located[1].len == 0 {
        return Err(Status::Malformed);
    }

    Ok(Region {
        data,
        version,
        flags,
        region_x,
        region_z,
        num_chunks,
        seed,
        world_height,
        bits,
        depth,
        csum,
        cmap: located[0],
        cdat: located[1],
        palt: located[2],
        ents: located[3],
        tile: located[4],
        lgts: located[5],
        hgts: located[6],
        biom: located[7],
        strc: located[8],
        tick: located[9],
    })
}

/// The `[start, end)` byte range of chunk `cid` within the `cdat` section, read
/// from the `cmap` offset table.
///
/// Returns `None` for a chunk whose entry is empty or inverted, which is how an
/// unpopulated column is stored.
pub fn chunk_range(region: &Region, cid: usize) -> Option<(usize, usize)> {
    if cid >= region.num_chunks {
        return None;
    }
    let cmap = region.slice(region.cmap);
    let mut c = Cursor::at(cmap, cid * 4);
    let start = c.u32() as usize;
    let end = c.u32() as usize;
    if !c.ok || end <= start || end > region.cdat.len {
        return None;
    }
    Some((start, end))
}

/// The raw bytes of chunk `cid`, or an empty slice if the column is unpopulated.
pub fn chunk_bytes<'a>(region: &Region<'a>, cid: usize) -> &'a [u8] {
    match chunk_range(region, cid) {
        Some((s, e)) => {
            let cdat = region.slice(region.cdat);
            cdat.get(s..e).unwrap_or(&[])
        }
        None => &[],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal region: header + directory + cmap + cdat.
    fn minimal() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&MAGIC);
        v.extend_from_slice(&VERSION.to_be_bytes());
        v.extend_from_slice(&0u16.to_be_bytes()); // flags
        v.extend_from_slice(&0i16.to_be_bytes()); // region_x
        v.extend_from_slice(&0i16.to_be_bytes()); // region_z
        v.extend_from_slice(&1u16.to_be_bytes()); // num_chunks
        v.extend_from_slice(&2u16.to_be_bytes()); // num_sections
        v.extend_from_slice(&7u32.to_be_bytes()); // seed
        v.extend_from_slice(&64u16.to_be_bytes()); // world_height
        v.push(4); // bits
        v.push(2); // depth
        v.extend_from_slice(&0u16.to_be_bytes()); // csum
        debug_assert_eq!(v.len(), HEADER_LEN);

        let dir_end = HEADER_LEN + 2 * DIR_ENTRY;
        let cmap_off = dir_end;
        let cdat_off = cmap_off + 8;
        v.extend_from_slice(&tag::CMAP);
        v.extend_from_slice(&(cmap_off as u32).to_be_bytes());
        v.extend_from_slice(&8u32.to_be_bytes());
        v.extend_from_slice(&tag::CDAT);
        v.extend_from_slice(&(cdat_off as u32).to_be_bytes());
        v.extend_from_slice(&4u32.to_be_bytes());

        v.extend_from_slice(&0u32.to_be_bytes()); // cmap[0]
        v.extend_from_slice(&4u32.to_be_bytes()); // cmap[1]
        v.extend_from_slice(&[0xaa, 0xbb, 0xcc, 0xdd]); // cdat
        v
    }

    #[test]
    fn parses_a_minimal_region() {
        let data = minimal();
        let r = parse(&data).expect("valid region");
        assert_eq!(r.num_chunks, 1);
        assert_eq!(r.seed, 7);
        assert_eq!(r.world_height, 64);
        assert_eq!(r.bits, 4);
        assert!(r.cmap.is_present());
        assert!(r.cdat.is_present());
        assert!(!r.palt.is_present());
    }

    #[test]
    fn chunk_range_and_bytes_round_trip() {
        let data = minimal();
        let r = parse(&data).unwrap();
        assert_eq!(chunk_range(&r, 0), Some((0, 4)));
        assert_eq!(chunk_bytes(&r, 0), &[0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!(chunk_range(&r, 1), None);
        assert_eq!(chunk_bytes(&r, 1), &[] as &[u8]);
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let mut data = minimal();
        data[0] = b'X';
        assert_eq!(parse(&data).err(), Some(Status::Malformed));

        let mut data = minimal();
        data[5] = 9; // version low byte
        assert_eq!(parse(&data).err(), Some(Status::Malformed));
    }

    #[test]
    fn rejects_out_of_range_palette_bits() {
        // Layout: magic 0..4, version 4..6, flags 6..8, rx 8..10, rz 10..12,
        // chunks 12..14, sections 14..16, seed 16..20, height 20..22, bits 22.
        let mut data = minimal();
        data[22] = 0;
        assert_eq!(parse(&data).err(), Some(Status::Malformed));
        let mut data = minimal();
        data[22] = MAX_PALETTE_BITS + 1;
        assert_eq!(parse(&data).err(), Some(Status::Malformed));
    }

    #[test]
    fn rejects_section_overrunning_the_buffer() {
        let mut data = minimal();
        // Inflate the cdat length past the end of the file.
        let cdat_len_at = HEADER_LEN + DIR_ENTRY + 8;
        data[cdat_len_at..cdat_len_at + 4].copy_from_slice(&0xffff_u32.to_be_bytes());
        assert_eq!(parse(&data).err(), Some(Status::Malformed));
    }

    #[test]
    fn rejects_short_cmap() {
        let mut data = minimal();
        // Claim 2 chunks but keep the 8-byte cmap, which only fits 1 + 1.
        data[12..14].copy_from_slice(&2u16.to_be_bytes());
        assert_eq!(parse(&data).err(), Some(Status::Malformed));
    }

    #[test]
    fn rejects_truncated_header() {
        let data = minimal();
        assert_eq!(parse(&data[..HEADER_LEN - 1]).err(), Some(Status::Malformed));
    }

    #[test]
    fn unknown_tags_are_skipped_not_fatal() {
        let mut data = minimal();
        // Rewrite the first directory tag to something unknown; parse must then
        // fail on the missing cmap rather than on the tag itself.
        data[HEADER_LEN..HEADER_LEN + 4].copy_from_slice(b"zzzz");
        assert_eq!(parse(&data).err(), Some(Status::Malformed));
    }
}
