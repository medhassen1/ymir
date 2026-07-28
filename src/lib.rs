//! Ymir — a persistent voxel world store.
//!
//! A `.ymr` region file holds a square of chunk columns, each a vertical stack
//! of palette-compressed 16³ sections. The crate parses a region, decodes its
//! columns, and rebuilds whichever derived data the region asks for: a render
//! mesh, propagated light, entity component storage, tile-entity property
//! trees, heightmaps, biome blends, structure instances, or the scheduled tick
//! queue.
//!
//! # Entry points
//!
//! * [`verify`] — parse and decode every column, reporting the first failure.
//!   Structural validation only; no derived data is rebuilt.
//! * [`rebuild`] — run the region's selected stage end to end and return a
//!   digest of the result.
//!
//! # Stage selection
//!
//! The header's `flags` field selects one rebuild stage. [`rebuild`] tests the
//! bits in a fixed priority order, so a region that sets several runs only the
//! first; a region that sets none runs the default pipeline, which resolves
//! every column through the shared section cache. This keeps each stage
//! independently reachable from a single region file.
//!
//! ```no_run
//! let bytes = std::fs::read("region.ymr").unwrap();
//! if ymir::verify(&bytes).is_ok() {
//!     let digest = ymir::rebuild(&bytes);
//!     println!("rebuilt: {digest:#018x}");
//! }
//! ```

// Each rebuild stage owns its working buffers and hands its consumer a
// `(pointer, length)` view of them rather than a borrowed slice, so a
// whole-region pass reads through one pointer instead of re-borrowing per
// element. Every such consumer states its precondition in a `SAFETY:` note on
// the signature, and the owning stage is responsible for meeting it.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

pub mod anneal;
pub mod bind;
pub mod biome;
pub mod blend;
pub mod budget;
pub mod chunk;
pub mod common;
pub mod diffuse;
pub mod drain;
pub mod emit;
pub mod entity;
pub mod format;
pub mod gather;
pub mod graft;
pub mod height;
pub mod light;
pub mod mesh;
pub mod palette;
pub mod parse;
pub mod profile;
pub mod reader;
pub mod relight;
pub mod repack;
pub mod resolve;
pub mod seccache;
pub mod structure;
pub mod tick;
pub mod tile;
pub mod util;
pub mod weld;

use common::Status;
use format::flag;

/// Parse and structurally validate a region without rebuilding anything.
///
/// Returns the first failing status, or [`Status::Ok`] if every column decodes.
/// A column that names an unsupported feature is skipped rather than failing the
/// whole region, so a newer writer's optional sections stay forward compatible.
pub fn verify(data: &[u8]) -> Status {
    let region = match parse::parse(data) {
        Ok(r) => r,
        Err(e) => return e,
    };
    let n = region.worked_chunks();
    for cid in 0..n {
        match chunk::decode(&region, cid) {
            Ok(_) => {}
            Err(Status::Unsupported) => {}
            Err(e) => return e,
        }
    }
    // Structure templates are validated separately: they are the one section
    // whose own nesting can be malformed independently of the columns.
    structure::validate(&region)
}

/// Rebuild the region's selected stage and return a digest of the result.
///
/// The digest exists so the whole pipeline is observable from one value — every
/// stage folds its output into it, which keeps the fuzz harness from optimising
/// the work away and gives the golden tests something byte-stable to assert.
pub fn rebuild(data: &[u8]) -> u64 {
    let region = match parse::parse(data) {
        Ok(r) => r,
        Err(_) => return 0,
    };
    let n = region.worked_chunks();

    // Priority order. A region setting several stage bits runs only the first.
    if region.flags & flag::MESH != 0 {
        return mesh::build_region(&region, n);
    }
    if region.flags & flag::LIGHT != 0 {
        return light::propagate_region(&region, n);
    }
    if region.flags & flag::ENTITY != 0 {
        return entity::load_region(&region, n);
    }
    if region.flags & flag::TILE != 0 {
        return tile::decode_region(&region);
    }
    if region.flags & flag::HEIGHT != 0 {
        return height::rebuild_region(&region, n);
    }
    if region.flags & flag::BIOME != 0 {
        return biome::resolve_region(&region);
    }
    if region.flags & flag::STRUCT != 0 {
        return structure::instance_region(&region, n);
    }
    if region.flags & flag::TICK != 0 {
        return tick::drain_region(&region, n);
    }
    if region.flags & flag::RELIGHT != 0 {
        return relight::incremental(&region, n);
    }
    if region.flags & flag::PALETTE != 0 {
        return palette::repack_region(&region, n);
    }

    // Default: resolve every column through the shared section cache and fold
    // the resident views.
    seccache::resolve_region(&region, n)
}

/// The stage name a region's flags select, for the inspector CLI.
pub fn stage_name(flags: u16) -> &'static str {
    if flags & flag::MESH != 0 {
        "mesh"
    } else if flags & flag::LIGHT != 0 {
        "light"
    } else if flags & flag::ENTITY != 0 {
        "entity"
    } else if flags & flag::TILE != 0 {
        "tile"
    } else if flags & flag::HEIGHT != 0 {
        "height"
    } else if flags & flag::BIOME != 0 {
        "biome"
    } else if flags & flag::STRUCT != 0 {
        "struct"
    } else if flags & flag::TICK != 0 {
        "tick"
    } else if flags & flag::RELIGHT != 0 {
        "relight"
    } else if flags & flag::PALETTE != 0 {
        "palette"
    } else {
        "seccache"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garbage_does_not_parse() {
        assert_eq!(verify(b""), Status::Malformed);
        assert_eq!(verify(b"not a region file at all"), Status::Malformed);
        assert_eq!(rebuild(b"not a region file at all"), 0);
    }

    #[test]
    fn stage_name_follows_priority_order() {
        assert_eq!(stage_name(0), "seccache");
        assert_eq!(stage_name(flag::MESH), "mesh");
        assert_eq!(stage_name(flag::PALETTE), "palette");
        // MESH outranks PALETTE when both are set.
        assert_eq!(stage_name(flag::MESH | flag::PALETTE), "mesh");
        // VERIFY alone selects no stage.
        assert_eq!(stage_name(flag::VERIFY), "seccache");
    }

    #[test]
    fn every_stage_flag_has_a_name() {
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
        ];
        let mut names: Vec<&str> = flags.iter().map(|&f| stage_name(f)).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), flags.len(), "stage names must be distinct");
    }
}
