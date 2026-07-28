# ymir

A persistent voxel world store, in Rust.

A `.ymr` region file holds a square of chunk columns, each a vertical stack of
palette-compressed 16×16×16 sections. `ymir` parses a region, decodes its
columns, and rebuilds whichever derived data the region asks for — a render
mesh, propagated light, entity component storage, tile-entity property trees,
heightmaps, biome blends, structure instances, or the scheduled tick queue.

No dependencies. `std` only.

## Usage

```rust
let bytes = std::fs::read("region.ymr")?;

// Structural validation: parse and decode every column.
if ymir::verify(&bytes).is_ok() {
    // Run the region's selected rebuild stage.
    let digest = ymir::rebuild(&bytes);
    println!("rebuilt {} -> {digest:#018x}", ymir::stage_name(0));
}
```

There is also a CLI:

```console
$ cargo run --bin ymir-inspect -- tests/fixtures/mesh_ok.ymr --columns
tests/fixtures/mesh_ok.ymr  (92 bytes)
  version      1
  flags        0x0001  -> stage `mesh`
  region       (0, 0)
  chunks       1
  ...
  verify       ok
  digest       0x6f7ab85f4ac1f876
```

## The format

Every multi-byte field is big-endian.

```text
header      : magic[4]="YMIR", version u16, flags u16, region_x i16,
              region_z i16, num_chunks u16, num_sections u16, seed u32,
              world_height u16, bits u8, depth u8, csum u16
directory   : num_sections * { tag[4], offset u32, len u32 }
sections    : referenced by the directory, in any order
  cmap      : (num_chunks+1) * u32  -- byte offsets into `cdat`
  cdat      : per-chunk section stacks
  palt      : block-state palettes
  ents      : entity component records
  tile      : tile-entity property trees
  lgts      : light source list
  hgts      : per-column heightmap
  biom      : biome cell grid
  strc      : structure template instances
  tick      : scheduled tick queue
```

A chunk record is a stack of sections. Each section carries a block-state
palette and a run-length encoded array of palette indices — a section of solid
stone is one run, and a section of air is flagged empty and carries none at all.
A section may inherit the previous section's palette instead of repeating it.

`csum` is advisory: it is recorded on write and surfaced by the inspector, but
decoding never depends on it, so a region with a stale checksum still loads.

## Rebuild stages

The header's `flags` field selects one rebuild stage. `rebuild()` tests the bits
in a fixed priority order, so a region that sets several runs only the first; a
region that sets none runs the default pipeline, which resolves every column
through the shared section cache.

| Bit | Flag | What it rebuilds |
|---|---|---|
| `0x0001` | `MESH` | render mesh with greedy face merging |
| `0x0002` | `LIGHT` | block-light propagation from the source list |
| `0x0004` | `ENTITY` | entity component storage |
| `0x0008` | `TILE` | tile-entity property trees |
| `0x0010` | `HEIGHT` | surface heightmap |
| `0x0020` | `BIOME` | biome grid resolution and blending |
| `0x0040` | `STRUCT` | structure template instancing |
| `0x0080` | `TICK` | scheduled tick queue drain |
| `0x0100` | `RELIGHT` | incremental relight over dirty columns |
| `0x0200` | `PALETTE` | palette repack to the narrowest width |
| `0x0400` | `VERIFY` | verify the advisory checksum (recorded, not acted on) |
| — | *none* | default pipeline through the section cache |

## Layout

```text
src/
  lib.rs          stage dispatch; verify() and rebuild()
  format.rs       wire format constants
  common.rs       status codes and hard limits
  reader.rs       bounds-checked cursor
  parse/          header + section directory
  chunk/          column and section decoding
  budget/         memory budgets derived from decoded shape
  mesh/ light/ entity/ tile/ height/ biome/
  structure/ tick/ relight/ palette/ seccache/    rebuild stages
  emit/ weld/ diffuse/ bind/ resolve/ profile/
  blend/ graft/ drain/ anneal/ repack/ gather/    stage consumers
  util/           56 dependency-free building blocks
fuzz/             three libFuzzer harnesses + per-harness seed corpora
tools/            fixture generator
tests/            golden digests over one fixture per stage
```

`src/util/` holds the engine's self-contained primitives: noise and PRNGs
(SplitMix64, xoroshiro128++, PCG32, Perlin, simplex, Worley, fBm), hashing
(xxHash32, FNV-1a, Murmur3, CRC-32, SipHash-2-4), spatial math (Vec3, Mat4,
quaternions, AABB, ray casts, frustum culling, Morton codes, octrees, DDA voxel
traversal), encoding (LEB128, zigzag, bit packing, RLE, delta, LZ77), containers
(bitset, slotmap, ring buffer, binary heap, small vector, arena, interner, LRU),
and the voxel/render helpers the mesher and lighter are built on.

## Building and testing

```console
$ cargo build
$ cargo test
$ cargo clippy --all-targets
```

The test suite is deterministic: no clocks, no ambient randomness, no I/O
outside `tests/fixtures/`, no network. Regenerate the fixtures with:

```console
$ python3 tools/mkfixtures.py tests/fixtures
```

`tests/goldens.rs` pins a byte-stable rebuild digest for one fixture per stage.
Those digests pin *behaviour*, not just liveness — a change that disables a
stage or returns early will still compile and still not crash, but it will
change a digest and fail there.

## Fuzzing

Three libFuzzer harnesses, wired for ClusterFuzzLite:

| Harness | Entry point |
|---|---|
| `a_rebuild_fuzzer` | `rebuild()` — the header's flags steer it into any stage |
| `b_verify_fuzzer` | `verify()` — parse, decode, walk template recursion |
| `c_column_fuzzer` | column passes directly, bypassing stage selection |

```console
$ cargo fuzz run a_rebuild_fuzzer fuzz/corpus/a_rebuild_fuzzer
```

The library itself has no dependencies at all; `libfuzzer-sys` is used only by
the fuzz targets, which live in their own crate under `fuzz/` and are never
built as part of `cargo build` or `cargo test`.

`.clusterfuzzlite/build.sh` compiles all three harnesses into `$OUT` and
packages each one's seed corpus alongside it.

## Limitations

- Regions are read-only; there is no writer. Fixtures are produced by
  `tools/mkfixtures.py`.
- `world_height` is capped at 1024 blocks and a column at 64 sections.
- Structure template nesting is capped at 6 levels and 256 templates per region.
- The advisory header checksum is never validated during decode.
- Biome blending is a 3-tap horizontal kernel only; there is no vertical blend.

## License

MIT OR Apache-2.0.
