//! SipHash-2-4 with a 128-bit key.
//!
//! Chunk and entity lookups in `ymir` are keyed by coordinates or entity
//! IDs that, in a networked or modded setting, an adversary could try to
//! craft to force many keys into the same hash bucket (an algorithmic
//! complexity attack). SipHash is a keyed pseudo-random function designed
//! specifically to resist that: without knowing the key, an attacker
//! cannot predict which bucket a chosen input lands in.

#[inline]
fn rotl(x: u64, b: u32) -> u64 {
    x.rotate_left(b)
}

/// One SipHash mixing round (`SIPROUND` in the reference implementation).
#[inline]
fn sipround(v0: &mut u64, v1: &mut u64, v2: &mut u64, v3: &mut u64) {
    *v0 = v0.wrapping_add(*v1);
    *v1 = rotl(*v1, 13);
    *v1 ^= *v0;
    *v0 = rotl(*v0, 32);
    *v2 = v2.wrapping_add(*v3);
    *v3 = rotl(*v3, 16);
    *v3 ^= *v2;
    *v0 = v0.wrapping_add(*v3);
    *v3 = rotl(*v3, 21);
    *v3 ^= *v0;
    *v2 = v2.wrapping_add(*v1);
    *v1 = rotl(*v1, 17);
    *v1 ^= *v2;
    *v2 = rotl(*v2, 32);
}

/// Reads a little-endian `u64` out of `bytes` starting at `offset`.
#[inline]
fn read_u64_le(bytes: &[u8], offset: usize) -> u64 {
    assert!(offset + 8 <= bytes.len(), "read_u64_le out of bounds");
    // SAFETY: the assert above guarantees `offset + 8 <= bytes.len()`, so
    // the 8-byte read starting at `bytes.as_ptr().add(offset)` stays
    // within `bytes`'s allocation; `read_unaligned` does not require
    // 8-byte alignment.
    let raw = unsafe { (bytes.as_ptr().add(offset) as *const u64).read_unaligned() };
    u64::from_le(raw)
}

/// A SipHash-2-4 instance keyed with a 128-bit key.
#[derive(Debug, Clone, Copy)]
pub struct SipHash24 {
    k0: u64,
    k1: u64,
}

impl SipHash24 {
    /// Builds a hasher from a 16-byte key (interpreted as two
    /// little-endian `u64` halves, per the reference implementation).
    pub fn new(key: [u8; 16]) -> Self {
        SipHash24 {
            k0: read_u64_le(&key, 0),
            k1: read_u64_le(&key, 8),
        }
    }

    /// Hashes `data`, returning a 64-bit output.
    pub fn hash(&self, data: &[u8]) -> u64 {
        let mut v0 = 0x736f_6d65_7073_6575 ^ self.k0;
        let mut v1 = 0x646f_7261_6e64_6f6d ^ self.k1;
        let mut v2 = 0x6c79_6765_6e65_7261 ^ self.k0;
        let mut v3 = 0x7465_6462_7974_6573 ^ self.k1;

        let len = data.len();
        let nblocks = len / 8;
        for i in 0..nblocks {
            let mi = read_u64_le(data, i * 8);
            v3 ^= mi;
            sipround(&mut v0, &mut v1, &mut v2, &mut v3);
            sipround(&mut v0, &mut v1, &mut v2, &mut v3);
            v0 ^= mi;
        }

        let tail_start = nblocks * 8;
        let tail_len = len & 7;
        let mut b: u64 = (len as u64 & 0xFF) << 56;
        for shift in 0..tail_len {
            // SAFETY: `shift` ranges over `0..tail_len`, and `tail_len ==
            // len - tail_start` is the number of bytes remaining after
            // `tail_start`, so `tail_start + shift` is always a valid
            // index into `data`.
            let byte = unsafe { *data.get_unchecked(tail_start + shift) };
            b |= (byte as u64) << (8 * shift);
        }

        v3 ^= b;
        sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        v0 ^= b;
        v2 ^= 0xff;
        for _ in 0..4 {
            sipround(&mut v0, &mut v1, &mut v2, &mut v3);
        }

        v0 ^ v1 ^ v2 ^ v3
    }
}

/// Hashes `data` with `key` in one call.
pub fn siphash24(key: [u8; 16], data: &[u8]) -> u64 {
    SipHash24::new(key).hash(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key and message convention used by the SipHash reference test
    /// vectors: key bytes `0..16`, message bytes `0..n`.
    fn ref_key() -> [u8; 16] {
        let mut k = [0u8; 16];
        for (i, b) in k.iter_mut().enumerate() {
            *b = i as u8;
        }
        k
    }

    #[test]
    fn known_answer_vectors() {
        // First entries of the reference `vectors_sip64` table from the
        // original SipHash reference implementation.
        let key = ref_key();
        let cases: [(usize, u64); 4] = [
            (0, 0x726f_db47_dd0e_0e31),
            (1, 0x74f8_39c5_93dc_67fd),
            (8, 0x93f5_f579_9a93_2462),
            (15, 0xa129_ca61_49be_45e5),
        ];
        for (n, expected) in cases {
            let msg: Vec<u8> = (0..n as u8).collect();
            assert_eq!(siphash24(key, &msg), expected, "message length {n}");
        }
    }

    #[test]
    fn short_message_vector() {
        let key = ref_key();
        let msg = [0u8, 1];
        assert_eq!(siphash24(key, &msg), 0x0d6c_8009_d9a9_4f5a);
    }

    #[test]
    fn deterministic_for_same_key_and_input() {
        let key = ref_key();
        let h = SipHash24::new(key);
        for data in [&b""[..], b"a", b"chunk:1,2,3", b"a much longer message than a block"] {
            assert_eq!(h.hash(data), h.hash(data));
        }
    }

    #[test]
    fn different_keys_scatter_the_same_message() {
        let mut key2 = ref_key();
        key2[0] = 0xFF;
        let a = SipHash24::new(ref_key());
        let b = SipHash24::new(key2);
        assert_ne!(a.hash(b"same message"), b.hash(b"same message"));
    }

    #[test]
    fn different_messages_usually_differ() {
        let h = SipHash24::new(ref_key());
        assert_ne!(h.hash(b"chunk:0,0,0"), h.hash(b"chunk:0,0,1"));
    }
}
