//! Light-level attenuation and the packed block/sky light nibble byte.
//!
//! Light is stored per block as two 4-bit channels packed into one byte.
//! This module provides the level-to-intensity falloff (level 15 is full,
//! each step down multiplies by 0.8) and the block/sky combine and pack rules.

/// Highest light level a block or sky channel can hold.
pub const MAX_LIGHT_LEVEL: u8 = 15;

/// Per-level multiplicative falloff: each level below 15 is 80% as bright
/// as the one above it.
pub const LIGHT_FALLOFF: f32 = 0.8;

const fn build_attenuation() -> [f32; 16] {
    let mut table = [0.0f32; 16];
    let mut i = 0;
    let mut value = 1.0f32;
    while i < 16 {
        table[15 - i] = value;
        value *= LIGHT_FALLOFF;
        i += 1;
    }
    table
}

/// Level -> intensity lookup, built once at compile time: `ATTENUATION[15]`
/// is `1.0`, `ATTENUATION[14]` is `0.8`, down to `ATTENUATION[0] = 0.8^15`.
static ATTENUATION: [f32; 16] = build_attenuation();

/// Look up the intensity multiplier for a light level, clamped to
/// `0..=MAX_LIGHT_LEVEL`.
#[inline]
pub fn attenuation(level: u8) -> f32 {
    let idx = level.min(MAX_LIGHT_LEVEL) as usize;
    // SAFETY: `idx` is `level.min(MAX_LIGHT_LEVEL)` cast to `usize`, and
    // `MAX_LIGHT_LEVEL` is 15, so `idx` is always in `0..=15`. `ATTENUATION`
    // has exactly 16 elements, so the index is always in bounds.
    unsafe { *ATTENUATION.get_unchecked(idx) }
}

/// Combine a block-light level and a sky-light level into the single level
/// a renderer should shade by. Light does not add across sources — a block
/// lit by both a torch and open sky is as bright as its brightest source,
/// not brighter — so this is a clamped maximum.
#[inline]
pub fn combine_light(block: u8, sky: u8) -> u8 {
    block.max(sky).min(MAX_LIGHT_LEVEL)
}

/// Combine block and sky levels directly into a shading intensity in
/// `[0, 1]`, taking the brighter of the two attenuated channels.
#[inline]
pub fn combined_intensity(block: u8, sky: u8) -> f32 {
    attenuation(block).max(attenuation(sky))
}

/// Pack a block-light and sky-light level (each clamped to 4 bits) into one
/// byte: sky in the high nibble, block in the low nibble.
#[inline]
pub fn pack_light(block: u8, sky: u8) -> u8 {
    let b = block.min(MAX_LIGHT_LEVEL) & 0x0F;
    let s = sky.min(MAX_LIGHT_LEVEL) & 0x0F;
    (s << 4) | b
}

/// Inverse of [`pack_light`]: returns `(block, sky)`.
#[inline]
pub fn unpack_light(byte: u8) -> (u8, u8) {
    (byte & 0x0F, (byte >> 4) & 0x0F)
}

/// Unpack a whole light plane (one byte per block) into parallel block-light
/// and sky-light arrays. Used when handing a section's light plane to a
/// mesher pass that wants the two channels as separate contiguous buffers.
/// Builds both output buffers with one pre-sized allocation each instead of
/// growing them through repeated `push`.
pub fn unpack_light_batch(bytes: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let n = bytes.len();
    let mut block: Vec<u8> = Vec::with_capacity(n);
    let mut sky: Vec<u8> = Vec::with_capacity(n);
    let block_ptr = block.as_mut_ptr();
    let sky_ptr = sky.as_mut_ptr();
    for (i, &byte) in bytes.iter().enumerate() {
        let (b, s) = unpack_light(byte);
        // SAFETY: `block_ptr` and `sky_ptr` come from `Vec::with_capacity(n)`,
        // so each has room for `n` elements starting uninitialized; `i`
        // ranges over `0..n` (the length of `bytes`), so `add(i)` stays
        // within the allocated (though not yet logically initialized)
        // capacity of each vector, and every index `0..n` is written exactly
        // once by this loop before either vector's length is adjusted below.
        unsafe {
            block_ptr.add(i).write(b);
            sky_ptr.add(i).write(s);
        }
    }
    // SAFETY: the loop above wrote every index `0..n` of both `block` and
    // `sky` through their respective raw pointers (each obtained from the
    // same allocation these vectors own), so both are now fully initialized
    // up to length `n`, which does not exceed either vector's capacity.
    unsafe {
        block.set_len(n);
        sky.set_len(n);
    }
    (block, sky)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attenuation_top_level_is_full_brightness() {
        assert_eq!(attenuation(MAX_LIGHT_LEVEL), 1.0);
    }

    #[test]
    fn attenuation_bottom_level_matches_falloff_power() {
        let expected = LIGHT_FALLOFF.powi(15);
        assert!((attenuation(0) - expected).abs() < 1e-6);
    }

    #[test]
    fn attenuation_is_monotonically_increasing_with_level() {
        let mut prev = attenuation(0);
        for level in 1..=MAX_LIGHT_LEVEL {
            let cur = attenuation(level);
            assert!(cur > prev, "not increasing at level {level}");
            prev = cur;
        }
    }

    #[test]
    fn attenuation_clamps_above_max_level() {
        assert_eq!(attenuation(255), attenuation(MAX_LIGHT_LEVEL));
    }

    #[test]
    fn combine_light_picks_the_brighter_channel() {
        assert_eq!(combine_light(3, 10), 10);
        assert_eq!(combine_light(12, 4), 12);
        assert_eq!(combine_light(0, 0), 0);
        assert_eq!(combine_light(200, 3), MAX_LIGHT_LEVEL);
    }

    #[test]
    fn pack_unpack_round_trip_covers_all_nibble_pairs() {
        for block in 0..=MAX_LIGHT_LEVEL {
            for sky in 0..=MAX_LIGHT_LEVEL {
                let packed = pack_light(block, sky);
                assert_eq!(unpack_light(packed), (block, sky));
            }
        }
    }

    #[test]
    fn batch_unpack_matches_scalar_unpack() {
        let bytes: Vec<u8> = (0..=255u8).step_by(7).collect();
        let (blocks, skies) = unpack_light_batch(&bytes);
        assert_eq!(blocks.len(), bytes.len());
        assert_eq!(skies.len(), bytes.len());
        for (i, &byte) in bytes.iter().enumerate() {
            let (b, s) = unpack_light(byte);
            assert_eq!(blocks[i], b);
            assert_eq!(skies[i], s);
        }
    }
}
