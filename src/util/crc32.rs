//! CRC-32 (IEEE 802.3, reflected polynomial `0xEDB88320`).
//!
//! Region files store many independently-compressed chunk section blobs;
//! `ymir` checksums each one with CRC-32 so a truncated write or a flipped
//! disk sector is caught on load instead of silently corrupting the
//! decoded voxel data. It is not cryptographically secure, but it is
//! extremely cheap and very good at catching the kind of damage a
//! filesystem or disk actually produces.

/// Builds the standard 256-entry CRC-32 lookup table at compile time.
const fn build_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut bit = 0;
        while bit < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            bit += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

/// The CRC-32 lookup table, one entry per possible byte value.
pub const TABLE: [u32; 256] = build_table();

/// An incremental CRC-32 accumulator.
#[derive(Debug, Clone, Copy)]
pub struct Crc32 {
    state: u32,
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32 {
    /// Starts a new checksum. The internal state is pre- and
    /// post-inverted, per the IEEE CRC-32 definition.
    pub fn new() -> Self {
        Crc32 { state: 0xFFFF_FFFF }
    }

    /// Feeds more bytes into the checksum.
    pub fn update(&mut self, bytes: &[u8]) {
        for &b in bytes {
            let idx = ((self.state ^ b as u32) & 0xFF) as usize;
            // SAFETY: `idx` is masked with `0xFF` (255), so it is always
            // in `0..256`, matching `TABLE`'s fixed length of 256.
            let entry = unsafe { *TABLE.get_unchecked(idx) };
            self.state = entry ^ (self.state >> 8);
        }
    }

    /// Finalizes the checksum (undoing the initial bit-inversion).
    pub fn finalize(&self) -> u32 {
        self.state ^ 0xFFFF_FFFF
    }
}

/// Computes the CRC-32 checksum of `data` in one call.
pub fn crc32(data: &[u8]) -> u32 {
    let mut c = Crc32::new();
    c.update(data);
    c.finalize()
}

/// Computes the CRC-32 checksum of several byte slices as if they had
/// been concatenated, without actually allocating a concatenated buffer.
/// Useful for checksumming a region-file record that is naturally split
/// into a header and a payload held in separate buffers.
pub fn crc32_of_parts(parts: &[&[u8]]) -> u32 {
    let mut c = Crc32::new();
    for part in parts {
        c.update(part);
    }
    c.finalize()
}

/// A variant of [`crc32`] that indexes `data` directly with
/// [`slice::get_unchecked`] in its inner loop rather than iterating, for
/// callers who have already paid for a bounds check on `data.len()` and
/// want to skip the per-byte one the `for &b in bytes` form still lets
/// the optimizer elide in most, but not guaranteed all, cases.
pub fn crc32_indexed(data: &[u8]) -> u32 {
    let mut state = 0xFFFF_FFFFu32;
    let len = data.len();
    for i in 0..len {
        // SAFETY: `i` ranges over `0..len` where `len == data.len()`, so
        // the index is always in bounds.
        let b = unsafe { *data.get_unchecked(i) };
        let idx = ((state ^ b as u32) & 0xFF) as usize;
        // SAFETY: identical reasoning to `Crc32::update`: masked with
        // `0xFF`, always in `0..256`.
        let entry = unsafe { *TABLE.get_unchecked(idx) };
        state = entry ^ (state >> 8);
    }
    state ^ 0xFFFF_FFFF
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_answer_check_value() {
        // The standard CRC-32/IEEE "check" value for the ASCII string
        // "123456789", published alongside the polynomial definition.
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn empty_input_hashes_to_zero() {
        assert_eq!(crc32(b""), 0);
    }

    #[test]
    fn incremental_matches_one_shot() {
        let data = b"the quick brown fox jumps over the lazy dog";
        let expected = crc32(data);

        let mut c = Crc32::new();
        c.update(&data[..10]);
        c.update(&data[10..]);
        assert_eq!(c.finalize(), expected);
    }

    #[test]
    fn indexed_variant_matches_the_table_driven_one() {
        for data in [&b""[..], b"a", b"123456789", b"the quick brown fox"] {
            assert_eq!(crc32_indexed(data), crc32(data));
        }
    }

    #[test]
    fn table_matches_the_well_known_first_entries() {
        // Widely published values for the reflected CRC-32/IEEE table.
        assert_eq!(TABLE[0], 0x0000_0000);
        assert_eq!(TABLE[1], 0x7707_3096);
        assert_eq!(TABLE[2], 0xEE0E_612C);
    }

    #[test]
    fn different_inputs_produce_different_checksums() {
        assert_ne!(crc32(b"chunk:0,0,0"), crc32(b"chunk:0,0,1"));
    }

    #[test]
    fn crc32_of_parts_matches_a_concatenated_buffer() {
        let header = b"HDR1";
        let payload = b"the rest of the record";
        let mut concatenated = Vec::new();
        concatenated.extend_from_slice(header);
        concatenated.extend_from_slice(payload);
        assert_eq!(crc32_of_parts(&[header, payload]), crc32(&concatenated));
    }

    #[test]
    fn crc32_of_parts_handles_an_empty_part_list() {
        assert_eq!(crc32_of_parts(&[]), crc32(b""));
    }
}
