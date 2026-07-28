//! Shared status codes and hard limits for the world store.
//!
//! Every fallible entry point in the crate returns [`Status`]. The limits below
//! are deliberately generous — they exist to bound work on hostile input, not to
//! express what a well-formed region actually contains.

/// The result of parsing, decoding, or rebuilding a region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// Everything succeeded.
    Ok,
    /// The region is structurally malformed (bad magic, out-of-range offset).
    Malformed,
    /// The region is well-formed but names a feature this build rejects.
    Unsupported,
    /// A record ran past the end of its section.
    Truncated,
    /// A palette index named an entry that does not exist.
    BadPalette,
    /// A structure template recursed past the depth budget.
    DepthExceeded,
}

impl Status {
    /// Whether this status represents success.
    pub fn is_ok(self) -> bool {
        matches!(self, Status::Ok)
    }

    /// A short, stable identifier, used by the inspector CLI.
    pub fn code(self) -> &'static str {
        match self {
            Status::Ok => "ok",
            Status::Malformed => "malformed",
            Status::Unsupported => "unsupported",
            Status::Truncated => "truncated",
            Status::BadPalette => "bad-palette",
            Status::DepthExceeded => "depth-exceeded",
        }
    }
}

impl core::fmt::Display for Status {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.code())
    }
}

/// Largest accepted region file.
pub const REGION_MAX: usize = 16 * 1024 * 1024;
/// Largest number of chunk records a region may declare.
pub const MAX_CHUNKS: usize = 32_768;
/// Largest number of entries in the section directory.
pub const MAX_SECTIONS: usize = 64;
/// Largest number of sections stacked in one chunk column.
pub const MAX_STACK: usize = 64;
/// Edge length of a chunk section, in blocks.
pub const SECTION_EDGE: usize = 16;
/// Blocks in one chunk section (`SECTION_EDGE^3`).
pub const SECTION_VOLUME: usize = SECTION_EDGE * SECTION_EDGE * SECTION_EDGE;
/// Largest number of distinct block states in one palette.
pub const MAX_PALETTE: usize = 4096;
/// Widest palette index, in bits.
pub const MAX_PALETTE_BITS: u8 = 12;
/// Largest world height, in blocks.
pub const MAX_WORLD_HEIGHT: usize = 1024;
/// Deepest structure-template nesting.
pub const MAX_TEMPLATE_DEPTH: u32 = 6;
/// Largest number of entities in one region.
pub const MAX_ENTITIES: usize = 8192;
/// Largest number of properties on one tile entity.
pub const MAX_PROPS: usize = 512;
/// Largest number of scheduled ticks in the queue.
pub const MAX_TICKS: usize = 16_384;
/// Largest number of light sources considered per relight pass.
pub const MAX_LIGHTS: usize = 4096;
/// Highest light level a source may emit.
pub const MAX_LIGHT_LEVEL: u8 = 15;
/// Number of biome cells along one chunk edge (biomes are 4x4x4 blocks).
pub const BIOME_EDGE: usize = 4;
/// Cap on records processed per region, to bound work.
pub const WORK_CAP: usize = 4096;

/// Clamp a `usize` into `0..=MAX_STACK`.
#[inline]
pub fn clamp_stack(n: usize) -> usize {
    n.min(MAX_STACK)
}

/// Whether `bits` is a palette width this build can unpack.
#[inline]
pub fn valid_palette_bits(bits: u8) -> bool {
    (1..=MAX_PALETTE_BITS).contains(&bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_is_ok_only_for_ok() {
        assert!(Status::Ok.is_ok());
        assert!(!Status::Malformed.is_ok());
        assert!(!Status::Truncated.is_ok());
        assert!(!Status::BadPalette.is_ok());
    }

    #[test]
    fn status_codes_are_distinct() {
        let all = [
            Status::Ok,
            Status::Malformed,
            Status::Unsupported,
            Status::Truncated,
            Status::BadPalette,
            Status::DepthExceeded,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.code(), b.code());
            }
        }
    }

    #[test]
    fn status_display_matches_code() {
        assert_eq!(Status::BadPalette.to_string(), "bad-palette");
        assert_eq!(Status::Ok.to_string(), "ok");
    }

    #[test]
    fn section_volume_is_cubic() {
        assert_eq!(SECTION_VOLUME, 4096);
        assert_eq!(SECTION_EDGE * SECTION_EDGE * SECTION_EDGE, SECTION_VOLUME);
    }

    #[test]
    fn clamp_stack_bounds() {
        assert_eq!(clamp_stack(0), 0);
        assert_eq!(clamp_stack(MAX_STACK), MAX_STACK);
        assert_eq!(clamp_stack(MAX_STACK + 100), MAX_STACK);
    }

    #[test]
    fn palette_bits_range() {
        assert!(!valid_palette_bits(0));
        assert!(valid_palette_bits(1));
        assert!(valid_palette_bits(MAX_PALETTE_BITS));
        assert!(!valid_palette_bits(MAX_PALETTE_BITS + 1));
    }
}
