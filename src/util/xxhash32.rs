//! xxHash32: a fast, high-quality non-cryptographic hash.
//!
//! `ymir` uses this to fingerprint serialized chunk sections and region
//! file blobs for integrity checks and change detection: it is much
//! faster than a cryptographic hash and, unlike a plain sum, is very
//! unlikely to collide on the kind of structured, repetitive data a
//! palette-compressed chunk section contains.

const PRIME1: u32 = 0x9E37_79B1;
const PRIME2: u32 = 0x85EB_CA77;
const PRIME3: u32 = 0xC2B2_AE3D;
const PRIME4: u32 = 0x27D4_EB2F;
const PRIME5: u32 = 0x1656_67B1;

#[inline]
fn round(acc: u32, input: u32) -> u32 {
    acc.wrapping_add(input.wrapping_mul(PRIME2))
        .rotate_left(13)
        .wrapping_mul(PRIME1)
}

/// Reads a little-endian `u32` out of `bytes` starting at `offset`.
#[inline]
fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    assert!(offset + 4 <= bytes.len(), "read_u32_le out of bounds");
    // SAFETY: the assert above guarantees `offset + 4 <= bytes.len()`, so
    // `bytes.as_ptr().add(offset)` points at 4 in-bounds bytes belonging
    // to `bytes`. `read_unaligned` does not require 4-byte alignment,
    // which a byte slice offset cannot generally guarantee.
    let raw = unsafe { (bytes.as_ptr().add(offset) as *const u32).read_unaligned() };
    u32::from_le(raw)
}

/// Computes the xxHash32 digest of `data` in one call.
pub fn xxhash32(data: &[u8], seed: u32) -> u32 {
    let mut hasher = XxHash32::with_seed(seed);
    hasher.write(data);
    hasher.finish()
}

/// A streaming xxHash32 hasher, for hashing data that arrives in pieces
/// (e.g. a chunk section's sub-arrays written one at a time) without
/// concatenating them into one buffer first.
#[derive(Debug, Clone)]
pub struct XxHash32 {
    seed: u32,
    total_len: u64,
    v1: u32,
    v2: u32,
    v3: u32,
    v4: u32,
    buf: [u8; 16],
    buf_len: usize,
}

impl XxHash32 {
    /// Creates a fresh streaming hasher seeded with `seed`.
    pub fn with_seed(seed: u32) -> Self {
        XxHash32 {
            seed,
            total_len: 0,
            v1: seed.wrapping_add(PRIME1).wrapping_add(PRIME2),
            v2: seed.wrapping_add(PRIME2),
            v3: seed,
            v4: seed.wrapping_sub(PRIME1),
            buf: [0; 16],
            buf_len: 0,
        }
    }

    /// Feeds more bytes into the hasher.
    pub fn write(&mut self, mut data: &[u8]) {
        self.total_len += data.len() as u64;

        if self.buf_len > 0 {
            let need = 16 - self.buf_len;
            if data.len() < need {
                self.buf[self.buf_len..self.buf_len + data.len()].copy_from_slice(data);
                self.buf_len += data.len();
                return;
            }
            self.buf[self.buf_len..16].copy_from_slice(&data[..need]);
            let block = self.buf;
            self.process_block(&block);
            data = &data[need..];
            self.buf_len = 0;
        }

        while data.len() >= 16 {
            self.process_block(&data[..16]);
            data = &data[16..];
        }

        if !data.is_empty() {
            self.buf[..data.len()].copy_from_slice(data);
            self.buf_len = data.len();
        }
    }

    fn process_block(&mut self, block: &[u8]) {
        self.v1 = round(self.v1, read_u32_le(block, 0));
        self.v2 = round(self.v2, read_u32_le(block, 4));
        self.v3 = round(self.v3, read_u32_le(block, 8));
        self.v4 = round(self.v4, read_u32_le(block, 12));
    }

    /// Finalizes the hasher and returns the digest, without consuming it
    /// (further `write` calls after `finish` are not meaningful, but
    /// `finish` itself can be called repeatedly for the same result).
    pub fn finish(&self) -> u32 {
        let mut h32 = if self.total_len >= 16 {
            self.v1
                .rotate_left(1)
                .wrapping_add(self.v2.rotate_left(7))
                .wrapping_add(self.v3.rotate_left(12))
                .wrapping_add(self.v4.rotate_left(18))
        } else {
            self.seed.wrapping_add(PRIME5)
        };
        h32 = h32.wrapping_add(self.total_len as u32);

        let mut idx = 0;
        while idx + 4 <= self.buf_len {
            let lane = read_u32_le(&self.buf, idx);
            h32 = h32.wrapping_add(lane.wrapping_mul(PRIME3));
            h32 = h32.rotate_left(17).wrapping_mul(PRIME4);
            idx += 4;
        }
        while idx < self.buf_len {
            h32 = h32.wrapping_add((self.buf[idx] as u32).wrapping_mul(PRIME5));
            h32 = h32.rotate_left(11).wrapping_mul(PRIME1);
            idx += 1;
        }

        h32 ^= h32 >> 15;
        h32 = h32.wrapping_mul(PRIME2);
        h32 ^= h32 >> 13;
        h32 = h32.wrapping_mul(PRIME3);
        h32 ^= h32 >> 16;
        h32
    }

    /// Finalizes the hasher and writes the digest into `out` as little-endian
    /// bytes, avoiding an intermediate `u32` at call sites that already work
    /// with byte buffers (e.g. appending a checksum to a serialized record).
    pub fn finish_into(&self, out: &mut [u8; 4]) {
        let bytes = self.finish().to_le_bytes();
        // SAFETY: `bytes` and `out` are both exactly 4-byte arrays on
        // distinct stack allocations, so the regions cannot overlap and
        // both are valid for a 4-byte copy.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr(), 4);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_empty_input() {
        assert_eq!(xxhash32(b"", 0), 0x02CC_5D05);
        assert_eq!(xxhash32(b"", 1), 0x0B2C_B792);
    }

    #[test]
    fn known_answer_abc() {
        assert_eq!(xxhash32(b"abc", 0), 0x32D1_53FF);
    }

    #[test]
    fn streaming_matches_one_shot_for_various_chunkings() {
        let data: Vec<u8> = (0u8..200).collect();
        let expected = xxhash32(&data, 42);

        let mut whole = XxHash32::with_seed(42);
        whole.write(&data);
        assert_eq!(whole.finish(), expected);

        for chunk_size in [1usize, 3, 7, 16, 31] {
            let mut hasher = XxHash32::with_seed(42);
            for chunk in data.chunks(chunk_size) {
                hasher.write(chunk);
            }
            assert_eq!(hasher.finish(), expected, "chunk size {chunk_size}");
        }
    }

    #[test]
    fn different_seeds_produce_different_hashes() {
        assert_ne!(xxhash32(b"ymir chunk section", 0), xxhash32(b"ymir chunk section", 1));
    }

    #[test]
    fn finish_into_matches_finish() {
        let hasher = XxHash32::with_seed(7);
        let mut out = [0u8; 4];
        hasher.finish_into(&mut out);
        assert_eq!(out, hasher.finish().to_le_bytes());
    }

    #[test]
    fn empty_write_calls_do_not_change_the_result() {
        let mut a = XxHash32::with_seed(5);
        a.write(b"hello world");
        let mut b = XxHash32::with_seed(5);
        b.write(b"");
        b.write(b"hello world");
        b.write(b"");
        assert_eq!(a.finish(), b.finish());
    }
}
