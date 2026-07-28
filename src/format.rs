//! The `.ymr` wire format: a region container for persisted voxel worlds.
//!
//! A region file stores one square of chunk columns. The header names which
//! optional rebuild stages the region wants; the directory locates each section;
//! the sections themselves may appear in any order.
//!
//! ```text
//! header      : magic[4]="YMIR", version u16, flags u16, region_x i16,
//!               region_z i16, num_chunks u16, num_sections u16, seed u32,
//!               world_height u16, bits u8, depth u8, csum u16
//! directory   : num_sections * { tag[4], offset u32, len u32 }
//! sections    : referenced by the directory, in any order
//!   cmap      : (num_chunks+1) * u32  -- byte offsets into `cdat`
//!   cdat      : per-chunk section stacks (see `chunk`)
//!   palt      : block-state palettes (see `palette`)
//!   ents      : entity component records (see `entity`)
//!   tile      : tile-entity property trees (see `tile`)
//!   lgts      : light source list (see `light`)
//!   hgts      : per-column heightmap (see `height`)
//!   biom      : biome cell grid (see `biome`)
//!   strc      : structure template instances (see `structure`)
//!   tick      : scheduled tick queue (see `tick`)
//! ```
//!
//! Every multi-byte field is big-endian. The `csum` field is advisory: it is
//! recorded on write and surfaced by the inspector, but decoding never depends
//! on it, so a region with a stale checksum still loads.

/// Magic at the head of every region file.
pub const MAGIC: [u8; 4] = *b"YMIR";
/// The only format version this build understands.
pub const VERSION: u16 = 1;
/// Fixed header length in bytes.
pub const HEADER_LEN: usize = 26;
/// One directory entry: tag + offset + length.
pub const DIR_ENTRY: usize = 12;

/// Four-byte section tags.
pub mod tag {
    /// Chunk offset table: one `u32` per chunk plus a terminating offset.
    pub const CMAP: [u8; 4] = *b"cmap";
    /// Chunk section stacks.
    pub const CDAT: [u8; 4] = *b"cdat";
    /// Block-state palettes.
    pub const PALT: [u8; 4] = *b"palt";
    /// Entity component records.
    pub const ENTS: [u8; 4] = *b"ents";
    /// Tile-entity property trees.
    pub const TILE: [u8; 4] = *b"tile";
    /// Light source list.
    pub const LGTS: [u8; 4] = *b"lgts";
    /// Per-column heightmap.
    pub const HGTS: [u8; 4] = *b"hgts";
    /// Biome cell grid.
    pub const BIOM: [u8; 4] = *b"biom";
    /// Structure template instances.
    pub const STRC: [u8; 4] = *b"strc";
    /// Scheduled tick queue.
    pub const TICK: [u8; 4] = *b"tick";
}

/// Header `flags` bits. Each selects an optional rebuild stage.
///
/// The stages are mutually exclusive in practice: [`crate::rebuild`] tests them
/// in the priority order listed in that function, so a region that sets several
/// runs only the highest-priority one. A region that sets none runs the default
/// full pipeline through the section cache.
pub mod flag {
    /// Rebuild the render mesh with greedy face merging.
    pub const MESH: u16 = 0x0001;
    /// Propagate block light from the source list.
    pub const LIGHT: u16 = 0x0002;
    /// Load entities into component storage.
    pub const ENTITY: u16 = 0x0004;
    /// Decode tile-entity property trees.
    pub const TILE: u16 = 0x0008;
    /// Rebuild the surface heightmap.
    pub const HEIGHT: u16 = 0x0010;
    /// Resolve and blend the biome grid.
    pub const BIOME: u16 = 0x0020;
    /// Instance structure templates into the world.
    pub const STRUCT: u16 = 0x0040;
    /// Drain the scheduled tick queue.
    pub const TICK: u16 = 0x0080;
    /// Run an incremental relight over dirty columns.
    pub const RELIGHT: u16 = 0x0100;
    /// Re-pack chunk palettes to the narrowest sufficient width.
    pub const PALETTE: u16 = 0x0200;
    /// Verify the advisory header checksum. Recorded but not acted on.
    pub const VERIFY: u16 = 0x0400;
}

/// Per-section flags inside a `cdat` chunk record.
pub mod sec {
    /// The section is uniform (a single block state, no packed index array).
    pub const UNIFORM: u8 = 0x01;
    /// The section reuses the previous section's palette.
    pub const SHARED_PALETTE: u8 = 0x02;
    /// The section carries its own per-block light nibbles.
    pub const HAS_LIGHT: u8 = 0x04;
    /// The section is entirely air and may be skipped by the mesher.
    pub const EMPTY: u8 = 0x08;
}

/// Marker stored in place of a section's palette length when the section
/// inherits the column's shared palette.
pub const INHERIT_PALETTE: u16 = 0xFFFF;

/// Marker stored in place of a structure instance's template id when the
/// instance nests another template list rather than naming a leaf template.
pub const NESTED_TEMPLATE: u16 = 0xFFFE;

/// The tag of a section, as a printable four-character string, for diagnostics.
pub fn tag_name(t: [u8; 4]) -> String {
    t.iter()
        .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
        .collect()
}

/// Whether `flags` selects any optional stage at all.
pub fn has_stage(flags: u16) -> bool {
    const ANY: u16 = flag::MESH
        | flag::LIGHT
        | flag::ENTITY
        | flag::TILE
        | flag::HEIGHT
        | flag::BIOME
        | flag::STRUCT
        | flag::TICK
        | flag::RELIGHT
        | flag::PALETTE;
    flags & ANY != 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_len_matches_field_layout() {
        // magic 4 + version 2 + flags 2 + rx 2 + rz 2 + chunks 2 + sections 2
        // + seed 4 + height 2 + bits 1 + depth 1 + csum 2
        assert_eq!(4 + 2 + 2 + 2 + 2 + 2 + 2 + 4 + 2 + 1 + 1 + 2, HEADER_LEN);
    }

    #[test]
    fn section_tags_are_distinct() {
        let tags = [
            tag::CMAP,
            tag::CDAT,
            tag::PALT,
            tag::ENTS,
            tag::TILE,
            tag::LGTS,
            tag::HGTS,
            tag::BIOM,
            tag::STRC,
            tag::TICK,
        ];
        for (i, a) in tags.iter().enumerate() {
            for b in &tags[i + 1..] {
                assert_ne!(a, b);
            }
        }
    }

    #[test]
    fn stage_flags_are_single_bits_and_distinct() {
        let flags = [
            flag::MESH,
            flag::LIGHT,
            flag::ENTITY,
            flag::TILE,
            flag::HEIGHT,
            flag::BIOME,
            flag::STRUCT,
            flag::TICK,
            flag::RELIGHT,
            flag::PALETTE,
            flag::VERIFY,
        ];
        let mut seen = 0u16;
        for f in flags {
            assert_eq!(f.count_ones(), 1, "flag {f:#06x} is not a single bit");
            assert_eq!(seen & f, 0, "flag {f:#06x} collides");
            seen |= f;
        }
    }

    #[test]
    fn has_stage_ignores_verify() {
        assert!(!has_stage(0));
        assert!(!has_stage(flag::VERIFY));
        assert!(has_stage(flag::MESH));
        assert!(has_stage(flag::VERIFY | flag::TICK));
    }

    #[test]
    fn tag_name_is_printable() {
        assert_eq!(tag_name(tag::CMAP), "cmap");
        assert_eq!(tag_name([0, 1, b'a', 0xff]), "..a.");
    }

    #[test]
    fn markers_do_not_collide_with_real_lengths() {
        assert!(INHERIT_PALETTE > crate::common::MAX_PALETTE as u16);
        assert_ne!(INHERIT_PALETTE, NESTED_TEMPLATE);
    }
}
