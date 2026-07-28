//! Byte-stable rebuild goldens over valid, non-crashing regions.
//!
//! One fixture per rebuild stage, each a well-formed region that exercises that
//! stage end to end. The digests below pin the *behaviour* of the pipeline, not
//! just its liveness: a genuine fix to any defect preserves them, whereas
//! amputating a stage — returning early, disabling a pass, skipping a buffer —
//! changes the digest and fails here.
//!
//! Regenerate the fixtures with:
//!
//! ```text
//! python3 tools/mkfixtures.py tests/fixtures
//! ```

use std::path::PathBuf;

fn fixture(name: &str) -> Vec<u8> {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("tests/fixtures");
    path.push(format!("{name}.ymr"));
    std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()))
}

fn digest(name: &str) -> u64 {
    ymir::rebuild(&fixture(name))
}

/// Every fixture must be structurally valid; a fix that starts rejecting
/// well-formed regions fails here before any digest is compared.
#[test]
fn every_fixture_verifies() {
    for name in [
        "seccache_ok",
        "mesh_ok",
        "light_ok",
        "entity_ok",
        "tile_ok",
        "height_ok",
        "biome_ok",
        "struct_ok",
        "tick_ok",
        "relight_ok",
        "palette_ok",
        "verify_ok",
    ] {
        assert!(
            ymir::verify(&fixture(name)).is_ok(),
            "fixture {name} must verify cleanly"
        );
    }
}

/// Each fixture must actually route to the stage it is named for, so a golden
/// cannot silently start covering a different code path.
#[test]
fn every_fixture_selects_its_stage() {
    for (name, stage) in [
        ("seccache_ok", "seccache"),
        ("mesh_ok", "mesh"),
        ("light_ok", "light"),
        ("entity_ok", "entity"),
        ("tile_ok", "tile"),
        ("height_ok", "height"),
        ("biome_ok", "biome"),
        ("struct_ok", "struct"),
        ("tick_ok", "tick"),
        ("relight_ok", "relight"),
        ("palette_ok", "palette"),
        // VERIFY selects no stage, so it falls through to the default.
        ("verify_ok", "seccache"),
    ] {
        let data = fixture(name);
        let flags = u16::from_be_bytes([data[6], data[7]]);
        assert_eq!(ymir::stage_name(flags), stage, "fixture {name}");
    }
}

/// Rebuilding is a pure function of the bytes.
#[test]
fn rebuild_is_deterministic() {
    for name in ["mesh_ok", "light_ok", "struct_ok", "tick_ok"] {
        assert_eq!(digest(name), digest(name), "fixture {name} must be stable");
    }
}

#[test]
fn golden_seccache() {
    assert_eq!(digest("seccache_ok"), 0xa6a73bfb15d4f789);
}

#[test]
fn golden_mesh() {
    assert_eq!(digest("mesh_ok"), 0x9a3376c4b9e012e4);
}

#[test]
fn golden_light() {
    assert_eq!(digest("light_ok"), 0x66b0629756d0d7fb);
}

#[test]
fn golden_entity() {
    assert_eq!(digest("entity_ok"), 0x0ce5bc93350d5502);
}

#[test]
fn golden_tile() {
    assert_eq!(digest("tile_ok"), 0xc3e0d2e5add3aebf);
}

#[test]
fn golden_height() {
    assert_eq!(digest("height_ok"), 0x034e260002ced7a1);
}

#[test]
fn golden_biome() {
    assert_eq!(digest("biome_ok"), 0x49cb7d0e3c130f6b);
}

#[test]
fn golden_struct() {
    assert_eq!(digest("struct_ok"), 0x033badf4655239a1);
}

#[test]
fn golden_tick() {
    assert_eq!(digest("tick_ok"), 0x01adbd85aab683f1);
}

#[test]
fn golden_relight() {
    assert_eq!(digest("relight_ok"), 0x03619a0002de93d7);
}

#[test]
fn golden_palette() {
    assert_eq!(digest("palette_ok"), 0x09f5cbfca1d9a328);
}

/// The VERIFY bit selects no stage, so it must rebuild exactly as the default
/// pipeline does on the same columns.
#[test]
fn golden_verify_falls_through_to_default() {
    assert_eq!(digest("verify_ok"), digest("seccache_ok"));
}

/// Garbage must be rejected rather than rebuilt.
#[test]
fn malformed_input_rebuilds_to_zero() {
    assert_eq!(ymir::rebuild(b""), 0);
    assert_eq!(ymir::rebuild(b"YMIR but truncated"), 0);
    assert_eq!(ymir::rebuild(&[0u8; 64]), 0);
}
