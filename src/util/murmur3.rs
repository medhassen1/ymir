//! MurmurHash3 (x86, 32-bit variant): a fast, well-distributed
//! non-cryptographic hash with an explicit seed.
//!
//! `ymir` uses the seeded form to derive independent hash families from
//! the same bytes — for example hashing a block-coordinate key once per
//! spatial partition "shard" with a different seed per shard, so the same
//! coordinate doesn't land in the same relative bucket in every shard.

/// First block-mixing constant (chosen for good bit-avalanche, per the
/// original MurmurHash3 reference implementation).
const C1: u32 = 0xcc9e_2d51;
/// Second block-mixing constant.
const C2: u32 = 0x1b87_3593;

/// Reads a little-endian `u32` out of `bytes` starting at `offset`.
#[inline]
fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    assert!(offset + 4 <= bytes.len(), "read_u32_le out of bounds");
    // SAFETY: the assert above guarantees `offset + 4 <= bytes.len()`, so
    // the 4-byte read starting at `bytes.as_ptr().add(offset)` stays
    // within `bytes`'s allocation. `read_unaligned` does not require the
    // pointer to be 4-byte aligned.
    let raw = unsafe { (bytes.as_ptr().add(offset) as *const u32).read_unaligned() };
    u32::from_le(raw)
}

/// Computes the MurmurHash3 x86_32 digest of `data` with the given `seed`.
pub fn murmur3_x86_32(data: &[u8], seed: u32) -> u32 {
    let len = data.len();
    let nblocks = len / 4;
    let mut h1 = seed;

    for i in 0..nblocks {
        // Each full 4-byte block is scrambled with the two odd
        // multipliers and a rotation before being folded into `h1`, then
        // `h1` itself is rotated and scaled so successive blocks cannot
        // simply cancel each other out.
        let k1 = read_u32_le(data, i * 4);
        let k1 = k1.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(13);
        h1 = h1.wrapping_mul(5).wrapping_add(0xe654_6b64);
    }

    let tail_start = nblocks * 4;
    let tail_len = len & 3;
    let mut k1 = 0u32;
    if tail_len >= 3 {
        // SAFETY: `tail_len >= 3` means `data` has at least
        // `tail_start + 3` bytes (since `tail_len == len - tail_start`),
        // so index `tail_start + 2` is in bounds.
        k1 ^= unsafe { *data.get_unchecked(tail_start + 2) as u32 } << 16;
    }
    if tail_len >= 2 {
        // SAFETY: `tail_len >= 2` guarantees index `tail_start + 1` is in
        // bounds, by the same reasoning as above.
        k1 ^= unsafe { *data.get_unchecked(tail_start + 1) as u32 } << 8;
    }
    if tail_len >= 1 {
        k1 ^= data[tail_start] as u32;
        k1 = k1.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h1 ^= k1;
    }

    h1 ^= len as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85eb_ca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2_ae35);
    h1 ^= h1 >> 16;
    h1
}

/// A newtype wrapping a fixed seed, for callers who want a `.hash(...)`
/// method rather than passing the seed at every call site (e.g. one
/// instance per spatial shard, stored alongside the shard's other state).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Murmur3Seed(pub u32);

impl Murmur3Seed {
    /// Hashes `data` with this instance's fixed seed.
    pub fn hash(&self, data: &[u8]) -> u32 {
        murmur3_x86_32(data, self.0)
    }
}

/// Hashes a 3D block coordinate directly, without the caller needing to
/// build a byte buffer first. Packs `x`, `y`, and `z` as little-endian
/// `i32`s into a 12-byte stack buffer and hashes that, which is the usual
/// way `ymir` derives a shard index for a block position.
pub fn hash_coords(x: i32, y: i32, z: i32, seed: u32) -> u32 {
    let mut buf = [0u8; 12];
    buf[0..4].copy_from_slice(&x.to_le_bytes());
    buf[4..8].copy_from_slice(&y.to_le_bytes());
    buf[8..12].copy_from_slice(&z.to_le_bytes());
    murmur3_x86_32(&buf, seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_empty_input() {
        assert_eq!(murmur3_x86_32(b"", 0), 0);
        assert_eq!(murmur3_x86_32(b"", 1), 0x514e_28b7);
    }

    #[test]
    fn known_answer_test_and_hello_world() {
        assert_eq!(murmur3_x86_32(b"test", 0), 0xba6b_d213);
        assert_eq!(murmur3_x86_32(b"Hello, world!", 0), 0xc036_3e43);
    }

    #[test]
    fn known_answers_for_short_tails() {
        // Exercises the 1, 2, 3, and 4-byte tail-handling paths.
        assert_eq!(murmur3_x86_32(b"a", 0), 0x3c25_69b2);
        assert_eq!(murmur3_x86_32(b"abcd", 0), 0x43ed_676a);
    }

    #[test]
    fn seed_changes_the_digest() {
        assert_ne!(murmur3_x86_32(b"test", 0), murmur3_x86_32(b"test", 1));
    }

    #[test]
    fn murmur3seed_matches_the_free_function() {
        let shard = Murmur3Seed(1234);
        assert_eq!(shard.hash(b"chunk:0,0"), murmur3_x86_32(b"chunk:0,0", 1234));
    }

    #[test]
    fn different_shards_scatter_the_same_key_differently() {
        let a = Murmur3Seed(1);
        let b = Murmur3Seed(2);
        assert_ne!(a.hash(b"chunk:5,-3"), b.hash(b"chunk:5,-3"));
    }

    #[test]
    fn hash_coords_matches_a_manually_packed_buffer() {
        let mut buf = [0u8; 12];
        buf[0..4].copy_from_slice(&5i32.to_le_bytes());
        buf[4..8].copy_from_slice(&(-3i32).to_le_bytes());
        buf[8..12].copy_from_slice(&12i32.to_le_bytes());
        assert_eq!(hash_coords(5, -3, 12, 7), murmur3_x86_32(&buf, 7));
    }

    #[test]
    fn hash_coords_is_sensitive_to_every_axis() {
        let base = hash_coords(1, 2, 3, 0);
        assert_ne!(base, hash_coords(2, 2, 3, 0));
        assert_ne!(base, hash_coords(1, 3, 3, 0));
        assert_ne!(base, hash_coords(1, 2, 4, 0));
    }
}
