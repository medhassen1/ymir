//! A bounds-checked cursor over a region buffer.
//!
//! Every read is range-checked; once a read runs past the end the cursor latches
//! `ok = false` and yields zeros from then on. That lets a decoder read a whole
//! record's fields straight through and validate once at the end, instead of
//! branching on every field. Region fields are big-endian.

/// A latching, bounds-checked reader over a byte buffer.
pub struct Cursor<'a> {
    /// The buffer being read.
    pub buf: &'a [u8],
    /// The current read offset.
    pub pos: usize,
    /// Cleared permanently by the first out-of-range read.
    pub ok: bool,
}

impl<'a> Cursor<'a> {
    /// A cursor at the start of `buf`.
    pub fn new(buf: &'a [u8]) -> Self {
        Cursor { buf, pos: 0, ok: true }
    }

    /// A cursor positioned at an absolute offset into `buf`.
    ///
    /// An offset past the end starts the cursor already latched.
    pub fn at(buf: &'a [u8], pos: usize) -> Self {
        Cursor { buf, pos, ok: pos <= buf.len() }
    }

    /// Take `n` bytes, latching and returning `None` if they are not available.
    #[inline]
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        if !self.ok || n > self.buf.len() - self.pos.min(self.buf.len()) {
            self.ok = false;
            return None;
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Some(s)
    }

    /// Read one byte.
    pub fn u8(&mut self) -> u8 {
        self.take(1).map(|s| s[0]).unwrap_or(0)
    }

    /// Read one signed byte.
    pub fn i8(&mut self) -> i8 {
        self.u8() as i8
    }

    /// Read a big-endian `u16`.
    pub fn u16(&mut self) -> u16 {
        self.take(2).map(|s| u16::from_be_bytes([s[0], s[1]])).unwrap_or(0)
    }

    /// Read a big-endian `i16`.
    pub fn i16(&mut self) -> i16 {
        self.u16() as i16
    }

    /// Read a big-endian `u32`.
    pub fn u32(&mut self) -> u32 {
        self.take(4)
            .map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
            .unwrap_or(0)
    }

    /// Read a big-endian `i32`.
    pub fn i32(&mut self) -> i32 {
        self.u32() as i32
    }

    /// Read a big-endian `u64`.
    pub fn u64(&mut self) -> u64 {
        let hi = self.u32() as u64;
        let lo = self.u32() as u64;
        (hi << 32) | lo
    }

    /// Read a four-byte section tag.
    pub fn tag(&mut self) -> [u8; 4] {
        self.take(4).map(|s| [s[0], s[1], s[2], s[3]]).unwrap_or([0; 4])
    }

    /// Borrow `n` raw bytes, advancing the cursor.
    pub fn bytes(&mut self, n: usize) -> &'a [u8] {
        self.take(n).unwrap_or(&[])
    }

    /// Skip `n` bytes.
    pub fn skip(&mut self, n: usize) {
        let _ = self.take(n);
    }

    /// Bytes left before the end of the buffer.
    pub fn remaining(&self) -> usize {
        self.buf.len().saturating_sub(self.pos)
    }

    /// Whether the cursor has consumed the whole buffer.
    pub fn at_end(&self) -> bool {
        self.pos >= self.buf.len()
    }

    /// Reposition to an absolute offset, latching if it is past the end.
    pub fn seek(&mut self, pos: usize) {
        if pos > self.buf.len() {
            self.ok = false;
        } else {
            self.pos = pos;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_big_endian_fields() {
        let data = [0x12, 0x34, 0x56, 0x78, 0x9a];
        let mut c = Cursor::new(&data);
        assert_eq!(c.u16(), 0x1234);
        assert_eq!(c.u16(), 0x5678);
        assert_eq!(c.u8(), 0x9a);
        assert!(c.ok);
        assert!(c.at_end());
    }

    #[test]
    fn latches_on_overrun_and_yields_zero() {
        let data = [0x01, 0x02];
        let mut c = Cursor::new(&data);
        assert_eq!(c.u16(), 0x0102);
        assert!(c.ok);
        assert_eq!(c.u32(), 0);
        assert!(!c.ok);
        // Latched: even an otherwise-valid read now fails.
        c.pos = 0;
        assert_eq!(c.u8(), 0);
        assert!(!c.ok);
    }

    #[test]
    fn take_does_not_overflow_on_huge_n() {
        let data = [0u8; 4];
        let mut c = Cursor::new(&data);
        assert_eq!(c.bytes(usize::MAX).len(), 0);
        assert!(!c.ok);
    }

    #[test]
    fn at_past_end_starts_latched() {
        let data = [0u8; 2];
        let c = Cursor::at(&data, 3);
        assert!(!c.ok);
        let d = Cursor::at(&data, 2);
        assert!(d.ok);
    }

    #[test]
    fn u64_composes_two_words() {
        let data = [0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02];
        let mut c = Cursor::new(&data);
        assert_eq!(c.u64(), 0x0000_0001_0000_0002);
    }

    #[test]
    fn signed_reads_sign_extend() {
        let data = [0xff, 0xff, 0xfe];
        let mut c = Cursor::new(&data);
        assert_eq!(c.i16(), -1);
        assert_eq!(c.i8(), -2);
    }

    #[test]
    fn seek_and_remaining() {
        let data = [0u8; 8];
        let mut c = Cursor::new(&data);
        c.skip(3);
        assert_eq!(c.remaining(), 5);
        c.seek(8);
        assert_eq!(c.remaining(), 0);
        c.seek(9);
        assert!(!c.ok);
    }
}
