#!/usr/bin/env python3
"""Generate `.ymr` region fixtures.

Writes one valid, non-crashing region per rebuild stage. The same files serve
two purposes: they are the fuzzer's seed corpus (so mutation starts from a
region that already reaches deep into a stage) and the inputs the golden tests
assert byte-stable digests over.

Usage:
    python3 tools/mkfixtures.py [outdir]      # default: tests/fixtures
"""

import os
import struct
import sys

MAGIC = b"YMIR"
VERSION = 1
HEADER_LEN = 26
DIR_ENTRY = 12

# Stage-selector bits, matching src/format.rs.
FLAG = {
    "mesh": 0x0001,
    "light": 0x0002,
    "entity": 0x0004,
    "tile": 0x0008,
    "height": 0x0010,
    "biome": 0x0020,
    "struct": 0x0040,
    "tick": 0x0080,
    "relight": 0x0100,
    "palette": 0x0200,
    "verify": 0x0400,
    "seccache": 0x0000,
}

# Per-section flags, matching src/format.rs `sec`.
SEC_UNIFORM = 0x01
SEC_SHARED_PALETTE = 0x02
SEC_HAS_LIGHT = 0x04
SEC_EMPTY = 0x08

INHERIT_PALETTE = 0xFFFF
NESTED_TEMPLATE = 0xFFFE

u8 = lambda v: struct.pack(">B", v & 0xFF)
u16 = lambda v: struct.pack(">H", v & 0xFFFF)
i16 = lambda v: struct.pack(">h", v)
u32 = lambda v: struct.pack(">I", v & 0xFFFFFFFF)
i32 = lambda v: struct.pack(">i", v)


def section(palette, *, flags=0, uniform_index=None, runs=None):
    """Encode one 16^3 chunk section."""
    out = bytearray()
    out += u8(flags)
    if palette is None:
        out += u16(INHERIT_PALETTE)
    else:
        out += u16(len(palette))
        for state in palette:
            out += u16(state)
    if flags & SEC_EMPTY:
        return bytes(out)
    if flags & SEC_UNIFORM:
        out += u16(uniform_index or 0)
        return bytes(out)
    runs = runs or []
    out += u16(len(runs))
    for count, index in runs:
        out += u16(count) + u16(index)
    return bytes(out)


def chunk(sections, *, base_y=0, cflags=0):
    """Encode one chunk column record."""
    out = bytearray()
    out += u16(len(sections))
    out += i16(base_y)
    out += u8(cflags)
    out += u8(0)  # reserved
    for s in sections:
        out += s
    return bytes(out)


def region(flags, chunks, extras=None, *, seed=0x5EED, world_height=64,
           bits=4, depth=3, region_x=0, region_z=0):
    """Assemble a whole region file from its chunk records and extra sections."""
    extras = extras or {}

    cdat = b"".join(chunks)
    offsets = []
    at = 0
    for c in chunks:
        offsets.append(at)
        at += len(c)
    offsets.append(at)
    cmap = b"".join(u32(o) for o in offsets)

    sections = [(b"cmap", cmap), (b"cdat", cdat)]
    for tag, body in extras.items():
        sections.append((tag, body))

    num_sections = len(sections)
    dir_end = HEADER_LEN + num_sections * DIR_ENTRY

    directory = bytearray()
    payload = bytearray()
    at = dir_end
    for tag, body in sections:
        directory += tag + u32(at) + u32(len(body))
        payload += body
        at += len(body)

    header = bytearray()
    header += MAGIC
    header += u16(VERSION)
    header += u16(flags)
    header += i16(region_x)
    header += i16(region_z)
    header += u16(len(chunks))
    header += u16(num_sections)
    header += u32(seed)
    header += u16(world_height)
    header += u8(bits)
    header += u8(depth)
    header += u16(0)  # advisory checksum
    assert len(header) == HEADER_LEN, len(header)

    return bytes(header + directory + payload)


# --------------------------------------------------------------------------
# Section payload builders
# --------------------------------------------------------------------------

def lgts(sources):
    """Light source list: (cid, at, level, sky)."""
    out = bytearray(u16(len(sources)))
    for cid, at, level, sky in sources:
        out += u16(cid) + u16(at) + u8((level & 0x0F) | (0x80 if sky else 0))
    return bytes(out)


def ents(records):
    """Entity records: (id, kind, payload). Variable kinds carry a u16 length."""
    out = bytearray(u16(len(records)))
    for eid, kind, payload in records:
        out += u16(eid) + u8(kind)
        if kind % 5 in (2, 4):  # Inventory / Blob are variable width
            out += u16(len(payload))
        out += payload
    return bytes(out)


def tile(props):
    """Property tree: (kind, name, value).

    A Compound (kind % 4 == 2) carries its children as its value, encoded as
    a nested subtree.
    """
    out = bytearray(u16(len(props)))
    for kind, name, value in props:
        out += u8(kind) + u8(len(name)) + name.encode()
        if kind % 4 == 0:      # Int
            out += i32(value)
        elif kind % 4 == 1:    # Text
            body = str(value).encode()
            out += u8(len(body)) + body
        elif kind % 4 == 2:    # Compound: children follow as a subtree
            out += tile(value)
        elif kind % 4 == 3:    # List
            out += u8(len(value))
            for v in value:
                out += i32(v)
    return bytes(out)


def hgts(bias):
    return i16(bias)


def biom(base, refine_rows, cells):
    out = bytearray(u8(base) + u8(refine_rows))
    for c in cells:
        out += u8(c)
    return bytes(out)


def template(tid, x, y, z, rot, cells):
    out = bytearray()
    out += u16(tid) + i16(x) + i16(y) + i16(z) + u8(rot) + u16(len(cells))
    for state, dy in cells:
        out += u16(state) + i16(dy)
    return bytes(out)


def strc(templates):
    return u16(len(templates)) + b"".join(templates)


def tick(now, entries):
    out = bytearray(u32(now) + u16(len(entries)))
    for at_tick, target, priority in entries:
        out += u32(at_tick) + u16(target) + u8(priority)
    return bytes(out)


# --------------------------------------------------------------------------
# One valid, non-crashing fixture per stage
# --------------------------------------------------------------------------

def simple_column():
    """A three-section column exercising all three section encodings.

    Every palette entry is referenced by at least one block, which is what a
    settled region looks like: nothing has been broken since the last repack, so
    the palettes are already minimal.
    """
    # Uniform bedrock over a single-entry palette.
    bedrock = section([12], flags=SEC_UNIFORM, uniform_index=0)
    # Layered stone/dirt/air, referencing every entry of its own palette.
    layered = section([0, 12, 13], runs=[(2048, 1), (1024, 2), (1024, 0)])
    # Sparse surface inheriting the layered palette, again touching all three.
    surface = section(None, flags=SEC_SHARED_PALETTE,
                      runs=[(64, 2), (64, 0), (32, 1)])
    return chunk([bedrock, layered, surface], base_y=0)


def shallow_column():
    """A single-section column, for stages whose fixtures should stay flat."""
    layered = section([0, 12, 13], runs=[(2048, 1), (1024, 2), (1024, 0)])
    return chunk([layered], base_y=0)


def lit_column(state=5):
    """A column flagged as carrying light, for the relight stage.

    `state` sets how opaque the lit half is.
    """
    body = section([0, state], flags=SEC_HAS_LIGHT, runs=[(128, 1), (128, 0)])
    return chunk([body], base_y=0)


def fixtures():
    col = simple_column()
    flat = shallow_column()

    return {
        "seccache_ok": region(FLAG["seccache"], [col]),
        # Two flat columns.
        "mesh_ok": region(FLAG["mesh"], [flat, flat]),
        "light_ok": region(
            FLAG["light"], [col],
            {b"lgts": lgts([(0, 0x888, 12, False)])},
        ),
        # Three records: two Transforms and a Health.
        "entity_ok": region(
            FLAG["entity"], [col],
            {b"ents": ents([
                (1, 0, i32(4) + i32(70) + i32(9)),      # Transform
                (2, 3, u16(18) + u16(20)),              # Health
                (3, 0, i32(9) + i32(66) + i32(-4)),     # Transform
            ])},
        ),
        # A compound property, nesting only names the store already carries.
        "tile_ok": region(
            FLAG["tile"], [col],
            {b"tile": tile([
                (0, "x", 3),
                (0, "y", 71),
                (2, "Items", [(1, "id", "chest"), (0, "z", 12)]),
            ])},
        ),
        # Flat, tall, flat.
        "height_ok": region(FLAG["height"], [flat, col, flat], {b"hgts": hgts(6)}),
        # Two columns.
        "biome_ok": region(
            FLAG["biome"], [col, col],
            {b"biom": (
                biom(4, 0, [4, 4, 5, 5, 4, 5, 5, 6, 5, 5, 6, 6, 5, 6, 6, 6])
                + biom(5, 0, [5, 5, 6, 6, 5, 6, 6, 7, 6, 6, 7, 7, 6, 7, 7, 7])
            )},
        ),
        # Two templates.
        "struct_ok": region(
            FLAG["struct"], [col],
            {b"strc": strc([
                template(1, 2, 70, 3, 1, [(12, 0), (12, 1), (13, 2), (13, 3)]),
                template(2, 9, 68, 5, 2, [(14, 0), (14, 2), (15, 3), (15, 4)]),
            ])},
        ),
        "tick_ok": region(
            FLAG["tick"], [col],
            # Eight ticks, all already due, in four widely spaced pairs.
            {b"tick": tick(2000, [
                (0, 0x101, 0), (100, 0x102, 1),
                (300, 0x202, 1), (400, 0x203, 2),
                (600, 0x303, 2), (700, 0x304, 0),
                (1100, 0x404, 1), (1300, 0x405, 0),
            ])},
        ),
        # Three dirty columns of differing exposure.
        "relight_ok": region(
            FLAG["relight"], [lit_column(5), lit_column(11), lit_column(2)],
        ),
        "palette_ok": region(FLAG["palette"], [col]),
        "verify_ok": region(FLAG["verify"], [col]),
    }


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures"
    os.makedirs(outdir, exist_ok=True)
    written = fixtures()
    for name, blob in sorted(written.items()):
        path = os.path.join(outdir, f"{name}.ymr")
        with open(path, "wb") as fh:
            fh.write(blob)
        print(f"{len(blob):6d}  {path}")
    print(f"\n{len(written)} fixtures written to {outdir}")


if __name__ == "__main__":
    main()
