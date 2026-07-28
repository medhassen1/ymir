//! LEB128 variable-length integer encoding: region file offsets and chunk
//! header counts are usually tiny but occasionally large, and varints store
//! the common case in one byte while still allowing the full 64-bit range.

/// Maximum number of bytes a LEB128-encoded `u64` can occupy.
pub const MAX_VARINT_LEN: usize = 10;

/// Appends the LEB128 encoding of `value` to `out` and returns the number of
/// bytes written.
///
/// Each byte carries 7 payload bits plus a continuation bit in the high
/// position; encoding stops as soon as the remaining value is zero.
pub fn encode_u64(value: u64, out: &mut Vec<u8>) -> usize {
    let mut buf = [0u8; MAX_VARINT_LEN];
    let mut len = 0usize;
    let mut v = value;
    loop {
        let mut byte = (v & 0x7f) as u8;
        v >>= 7;
        if v != 0 {
            byte |= 0x80;
        }
        // SAFETY: `len` starts at 0 and increments by exactly one per loop
        // iteration; a 64-bit value needs at most ceil(64/7) = 10 groups of
        // 7 bits, matching `MAX_VARINT_LEN`, so `len` never reaches the
        // array length before the loop terminates (`v == 0`).
        unsafe {
            *buf.get_unchecked_mut(len) = byte;
        }
        len += 1;
        if v == 0 {
            break;
        }
    }
    out.extend_from_slice(&buf[..len]);
    len
}

/// Decodes a LEB128 `u64` from `buf` starting at `*cursor`, advancing the
/// cursor past the consumed bytes on success.
///
/// Returns `None` on truncated input (the buffer ends mid-varint) or on a
/// value that would overflow a `u64` (more than 10 continuation groups).
pub fn decode_u64(buf: &[u8], cursor: &mut usize) -> Option<u64> {
    let mut result: u64 = 0;
    let mut shift: u32 = 0;
    let mut pos = *cursor;
    for _ in 0..MAX_VARINT_LEN {
        if pos >= buf.len() {
            return None; // truncated: ran off the end mid-varint
        }
        // SAFETY: the check above guarantees `pos < buf.len()` for this
        // access, so reading the byte at `pos` is in bounds.
        let byte = unsafe { *buf.get_unchecked(pos) };
        pos += 1;
        let payload = (byte & 0x7f) as u64;
        if shift < 64 {
            result |= payload << shift;
        }
        if byte & 0x80 == 0 {
            *cursor = pos;
            return Some(result);
        }
        shift += 7;
    }
    None // continuation bit set for too many groups: malformed / overflow
}

/// Encodes a signed `i64` using zigzag mapping followed by LEB128, so small
/// magnitude negatives stay compact instead of encoding as near-`u64::MAX`.
pub fn encode_i64(value: i64, out: &mut Vec<u8>) -> usize {
    let zz = ((value << 1) ^ (value >> 63)) as u64;
    encode_u64(zz, out)
}

/// Decodes a zigzag+LEB128-encoded signed integer written by [`encode_i64`].
pub fn decode_i64(buf: &[u8], cursor: &mut usize) -> Option<i64> {
    let zz = decode_u64(buf, cursor)?;
    let signed_shifted = (zz >> 1) as i64;
    Some(signed_shifted ^ -((zz & 1) as i64))
}

/// Returns the number of bytes [`encode_u64`] would write for `value`,
/// without doing the encoding, so callers can pre-size a buffer.
pub fn varint_len(value: u64) -> usize {
    let mut v = value;
    let mut len = 1;
    while v >= 0x80 {
        v >>= 7;
        len += 1;
    }
    len
}

/// Encodes a whole slice of `u64` values back-to-back and returns the
/// concatenated bytes, pre-sized exactly to avoid any reallocation.
pub fn encode_all(values: &[u64]) -> Vec<u8> {
    let total: usize = values.iter().map(|v| varint_len(*v)).sum();
    let mut out = Vec::with_capacity(total);
    for &v in values {
        encode_u64(v, &mut out);
    }
    out
}

/// Decodes exactly `count` back-to-back varints from `buf`, returning `None`
/// if the stream runs out before `count` values are read.
///
/// The result vector is allocated with the exact known capacity and filled
/// through a raw pointer, which avoids the bounds checks and capacity growth
/// checks a `push`-based loop would otherwise pay for a size we already know.
pub fn decode_all(buf: &[u8], count: usize) -> Option<Vec<u64>> {
    let mut out: Vec<u64> = Vec::with_capacity(count);
    let ptr = out.as_mut_ptr();
    let mut cursor = 0usize;
    for i in 0..count {
        let value = decode_u64(buf, &mut cursor)?;
        // SAFETY: `ptr` was obtained from a `Vec` allocated with capacity
        // `count`, and `i` ranges over `0..count`, so `ptr.add(i)` stays
        // within the allocation. Each slot is written at most once (`i` is
        // unique per iteration) before `set_len` below makes the vector
        // observe them, so no uninitialized memory is ever read.
        unsafe {
            ptr.add(i).write(value);
        }
    }
    // SAFETY: the loop above wrote exactly `count` elements at indices
    // `0..count` with no early return past this point, so all `count` slots
    // are initialized and `set_len(count)` reflects reality.
    unsafe {
        out.set_len(count);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_u64_boundaries() {
        let values = [
            0u64,
            1,
            127,
            128,
            16383,
            16384,
            u32::MAX as u64,
            u64::MAX,
            u64::MAX - 1,
        ];
        for &v in &values {
            let mut buf = Vec::new();
            let written = encode_u64(v, &mut buf);
            assert_eq!(written, buf.len());
            let mut cursor = 0;
            assert_eq!(decode_u64(&buf, &mut cursor), Some(v));
            assert_eq!(cursor, buf.len());
        }
    }

    #[test]
    fn round_trip_i64_sign_boundaries() {
        for &v in &[0i64, 1, -1, 63, -64, i32::MIN as i64, i64::MIN, i64::MAX] {
            let mut buf = Vec::new();
            encode_i64(v, &mut buf);
            let mut cursor = 0;
            assert_eq!(decode_i64(&buf, &mut cursor), Some(v));
        }
    }

    #[test]
    fn truncated_input_is_rejected() {
        let mut buf = Vec::new();
        encode_u64(1_000_000, &mut buf);
        buf.truncate(buf.len() - 1);
        let mut cursor = 0;
        assert_eq!(decode_u64(&buf, &mut cursor), None);
    }

    #[test]
    fn single_byte_values_stay_one_byte() {
        let mut buf = Vec::new();
        let n = encode_u64(42, &mut buf);
        assert_eq!(n, 1);
        assert_eq!(buf, vec![42]);
    }

    #[test]
    fn varint_len_matches_actual_encoding() {
        for v in [0u64, 5, 127, 128, 300, 1 << 20, u64::MAX] {
            let mut buf = Vec::new();
            let written = encode_u64(v, &mut buf);
            assert_eq!(written, varint_len(v));
        }
    }

    #[test]
    fn batch_encode_decode_round_trip() {
        let values: Vec<u64> = (0..50).map(|i| i * i * 977).collect();
        let bytes = encode_all(&values);
        let decoded = decode_all(&bytes, values.len()).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn decode_all_reports_short_stream() {
        let values = [1u64, 2, 3];
        let bytes = encode_all(&values);
        assert!(decode_all(&bytes, 10).is_none());
    }
}
