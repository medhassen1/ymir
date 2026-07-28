//! FNV-1a: a tiny, dependency-free hash for cases where xxHash/Murmur are
//! more machinery than needed.
//!
//! `ymir` uses FNV-1a for lightweight, in-memory lookups that never leave
//! the process — bucketing loaded chunks by coordinate key, deduplicating
//! palette entries within a chunk section — anywhere a hash is wanted but
//! the input is small and speed of the hasher itself barely matters
//! compared to its simplicity.

/// The 32-bit FNV offset basis (the hash of the empty string).
pub const FNV1A_32_OFFSET: u32 = 0x811c_9dc5;
/// The 32-bit FNV prime.
pub const FNV1A_32_PRIME: u32 = 0x0100_0193;
/// The 64-bit FNV offset basis (the hash of the empty string).
pub const FNV1A_64_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// The 64-bit FNV prime.
pub const FNV1A_64_PRIME: u64 = 0x0000_0100_0000_01B3;

/// Computes the 32-bit FNV-1a hash of `data` in one call.
pub fn fnv1a32(data: &[u8]) -> u32 {
    let mut h = Fnv1a32::new();
    h.write(data);
    h.finish()
}

/// Computes the 64-bit FNV-1a hash of `data` in one call.
pub fn fnv1a64(data: &[u8]) -> u64 {
    let mut h = Fnv1a64::new();
    h.write(data);
    h.finish()
}

/// An incremental 32-bit FNV-1a hasher.
#[derive(Debug, Clone, Copy)]
pub struct Fnv1a32 {
    state: u32,
}

impl Default for Fnv1a32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv1a32 {
    /// Starts a new hash from the FNV offset basis.
    pub fn new() -> Self {
        Fnv1a32 { state: FNV1A_32_OFFSET }
    }

    /// Feeds more bytes into the hash: XOR each byte in, then multiply by
    /// the FNV prime, in that order (the "a" in FNV-1a).
    pub fn write(&mut self, mut data: &[u8]) {
        while data.len() >= 4 {
            // SAFETY: the loop condition just checked `data.len() >= 4`,
            // so reading 4 bytes from `data.as_ptr()` is in-bounds;
            // `read_unaligned` does not require 4-byte alignment.
            let word = unsafe { (data.as_ptr() as *const u32).read_unaligned() };
            for byte in word.to_le_bytes() {
                self.state ^= byte as u32;
                self.state = self.state.wrapping_mul(FNV1A_32_PRIME);
            }
            data = &data[4..];
        }
        for &byte in data {
            self.state ^= byte as u32;
            self.state = self.state.wrapping_mul(FNV1A_32_PRIME);
        }
    }

    /// Returns the current digest without consuming the hasher.
    pub fn finish(&self) -> u32 {
        self.state
    }
}

/// An incremental 64-bit FNV-1a hasher.
#[derive(Debug, Clone, Copy)]
pub struct Fnv1a64 {
    state: u64,
}

impl Default for Fnv1a64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Fnv1a64 {
    /// Starts a new hash from the FNV offset basis.
    pub fn new() -> Self {
        Fnv1a64 { state: FNV1A_64_OFFSET }
    }

    /// Feeds more bytes into the hash.
    pub fn write(&mut self, mut data: &[u8]) {
        while data.len() >= 8 {
            // SAFETY: the loop condition just checked `data.len() >= 8`,
            // so reading 8 bytes from `data.as_ptr()` is in-bounds;
            // `read_unaligned` does not require 8-byte alignment.
            let word = unsafe { (data.as_ptr() as *const u64).read_unaligned() };
            for byte in word.to_le_bytes() {
                self.state ^= byte as u64;
                self.state = self.state.wrapping_mul(FNV1A_64_PRIME);
            }
            data = &data[8..];
        }
        for &byte in data {
            self.state ^= byte as u64;
            self.state = self.state.wrapping_mul(FNV1A_64_PRIME);
        }
    }

    /// Returns the current digest without consuming the hasher.
    pub fn finish(&self) -> u64 {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answers_32_bit() {
        assert_eq!(fnv1a32(b""), 0x811c_9dc5);
        assert_eq!(fnv1a32(b"a"), 0xe40c_292c);
        assert_eq!(fnv1a32(b"abc"), 0x1a47_e90b);
        assert_eq!(fnv1a32(b"foobar"), 0xbf9c_f968);
    }

    #[test]
    fn known_answers_64_bit() {
        assert_eq!(fnv1a64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a64(b"abc"), 0xe71f_a219_0541_574b);
        assert_eq!(fnv1a64(b"foobar"), 0x8594_4171_f739_67e8);
    }

    #[test]
    fn incremental_matches_one_shot_across_chunk_boundaries() {
        let data: Vec<u8> = (0u8..137).collect();
        let expected32 = fnv1a32(&data);
        let expected64 = fnv1a64(&data);
        for chunk_size in [1usize, 3, 4, 5, 8, 9, 64] {
            let mut h32 = Fnv1a32::new();
            let mut h64 = Fnv1a64::new();
            for chunk in data.chunks(chunk_size) {
                h32.write(chunk);
                h64.write(chunk);
            }
            assert_eq!(h32.finish(), expected32, "32-bit chunk size {chunk_size}");
            assert_eq!(h64.finish(), expected64, "64-bit chunk size {chunk_size}");
        }
    }

    #[test]
    fn default_matches_new() {
        assert_eq!(Fnv1a32::default().finish(), Fnv1a32::new().finish());
        assert_eq!(Fnv1a64::default().finish(), Fnv1a64::new().finish());
    }

    #[test]
    fn different_inputs_usually_differ() {
        assert_ne!(fnv1a32(b"chunk:0,0,0"), fnv1a32(b"chunk:0,0,1"));
        assert_ne!(fnv1a64(b"chunk:0,0,0"), fnv1a64(b"chunk:0,0,1"));
    }

    #[test]
    fn empty_write_is_a_no_op() {
        let mut a = Fnv1a32::new();
        a.write(b"");
        assert_eq!(a.finish(), FNV1A_32_OFFSET);
    }
}
