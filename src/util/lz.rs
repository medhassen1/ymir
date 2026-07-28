//! A small LZ77-style compressor for byte buffers. Region files store many
//! chunk sections that repeat structure (terrain layers, air padding,
//! symmetric worldgen structures); a bounded-window LZ pass catches that
//! redundancy for arbitrary payloads that palette or RLE coders miss.

/// Matches must reach at least this length before they are worth encoding
/// as a back-reference instead of literal bytes.
const MIN_MATCH: usize = 4;
/// Longest match length a single token can represent.
const MAX_MATCH: usize = 255 + MIN_MATCH;
/// Furthest back a match can point.
const WINDOW: usize = 4096;
/// Hash table size (power of two) for the 4-byte match finder.
const HASH_BITS: u32 = 12;
const HASH_SIZE: usize = 1 << HASH_BITS;
const HASH_MASK: usize = HASH_SIZE - 1;

/// One unit of the compressed token stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Token {
    /// A single literal byte, copied verbatim.
    Literal(u8),
    /// A back-reference: copy `length` bytes starting `distance` bytes
    /// before the current output position.
    Match { distance: u16, length: u16 },
}

fn hash4(bytes: &[u8]) -> usize {
    let w = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    ((w.wrapping_mul(2654435761)) >> (32 - HASH_BITS)) as usize
}

/// Compresses `input` into a stream of [`Token`]s using a hash-chain match
/// finder bounded to a `WINDOW`-byte lookback and `MAX_MATCH`-byte matches.
pub fn compress(input: &[u8]) -> Vec<Token> {
    let len = input.len();
    let mut tokens = Vec::new();
    if len < MIN_MATCH {
        for &b in input {
            tokens.push(Token::Literal(b));
        }
        return tokens;
    }

    // `head[h]` is the most recent position whose 4-byte hash was `h`, or
    // `u32::MAX` if none. `prev[i]` chains position `i` back to the
    // previous position with the same hash, forming a singly linked list
    // per bucket that the match finder walks backwards through.
    let mut head = vec![u32::MAX; HASH_SIZE];
    let mut prev = vec![u32::MAX; len];

    let mut i = 0usize;
    while i < len {
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if i + MIN_MATCH <= len {
            let h = hash4(&input[i..]) & HASH_MASK;
            // SAFETY: `h` is masked with `HASH_MASK = HASH_SIZE - 1` and
            // `head.len() == HASH_SIZE`, so `h` is always a valid index
            // into `head` regardless of the hash function's output range.
            let mut candidate = unsafe { *head.get_unchecked(h) };
            let max_dist = i.min(WINDOW);
            let mut steps = 0;
            while candidate != u32::MAX
                && i - candidate as usize <= max_dist
                && steps < 64
            {
                let cand = candidate as usize;
                let max_len = (len - i).min(MAX_MATCH);
                let mut match_len = 0usize;
                while match_len < max_len && input[cand + match_len] == input[i + match_len] {
                    match_len += 1;
                }
                if match_len > best_len {
                    best_len = match_len;
                    best_dist = i - cand;
                }
                // SAFETY: `cand = candidate as usize` is always `< i <=
                // prev.len()` here: every value ever stored into `head` or
                // chained through `prev` is a position that was itself
                // visited earlier in this same left-to-right scan (i.e.
                // strictly less than the current `i`), and `prev` has one
                // slot per input byte.
                candidate = unsafe { *prev.get_unchecked(cand) };
                steps += 1;
            }
            // SAFETY: identical reasoning to the `head` access above: `h`
            // is masked into `0..HASH_SIZE`, matching `head.len()`.
            unsafe {
                *prev.get_unchecked_mut(i) = *head.get_unchecked(h);
                *head.get_unchecked_mut(h) = i as u32;
            }
        }

        if best_len >= MIN_MATCH {
            tokens.push(Token::Match { distance: best_dist as u16, length: best_len as u16 });
            // Insert the hashes of the skipped positions too, so future
            // matches can still find them as candidates.
            let end = i + best_len;
            i += 1;
            while i < end && i + MIN_MATCH <= len {
                let h = hash4(&input[i..]) & HASH_MASK;
                prev[i] = head[h];
                head[h] = i as u32;
                i += 1;
            }
            i = end;
        } else {
            tokens.push(Token::Literal(input[i]));
            i += 1;
        }
    }
    tokens
}

/// Decompresses a token stream produced by [`compress`] back into bytes.
///
/// Returns `None` if a match references a distance of 0 or a distance
/// greater than the number of bytes decoded so far, which would read
/// outside the reconstructed output.
pub fn decompress(tokens: &[Token]) -> Option<Vec<u8>> {
    let mut out: Vec<u8> = Vec::new();
    for token in tokens {
        match *token {
            Token::Literal(b) => out.push(b),
            Token::Match { distance, length } => {
                let distance = distance as usize;
                let length = length as usize;
                if distance == 0 || distance > out.len() {
                    return None;
                }
                let start = out.len() - distance;
                for k in 0..length {
                    // SAFETY: `start = out.len() - distance` was computed
                    // before this loop began growing `out`, and `start + k`
                    // for `k < length` always names a byte at or after
                    // `start`, which was already valid output at loop
                    // entry; overlapping matches (`distance < length`) read
                    // bytes this same loop already pushed on an earlier
                    // `k`, which is exactly the classic LZ77 overlap-copy
                    // behavior. `start + k < out.len()` holds at the time
                    // of each read because `out` has grown by `k` pushes
                    // since `start` was computed.
                    let byte = unsafe { *out.get_unchecked(start + k) };
                    out.push(byte);
                }
            }
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(data: &[u8]) {
        let tokens = compress(data);
        let restored = decompress(&tokens).expect("valid token stream");
        assert_eq!(restored, data, "round trip mismatch for {} bytes", data.len());
    }

    #[test]
    fn round_trip_highly_repetitive_data() {
        let data = b"abcdabcdabcdabcdabcdabcdabcdabcdabcdabcd".to_vec();
        round_trip(&data);
    }

    #[test]
    fn round_trip_pseudo_random_data() {
        let mut data = Vec::with_capacity(2000);
        let mut state: u32 = 12345;
        for _ in 0..2000 {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            data.push((state >> 24) as u8);
        }
        round_trip(&data);
    }

    #[test]
    fn round_trip_empty_and_tiny_inputs() {
        round_trip(&[]);
        round_trip(&[1]);
        round_trip(&[1, 2, 3]);
    }

    #[test]
    fn repetitive_input_compresses_smaller_than_literal_stream() {
        let data = vec![b'x'; 1000];
        let tokens = compress(&data);
        assert!(tokens.len() < data.len() / 2);
        assert_eq!(decompress(&tokens).unwrap(), data);
    }

    #[test]
    fn overlapping_match_extends_correctly() {
        // "aaaa" then a match with distance < length exercises the
        // overlap-copy path in `decompress`.
        let data = b"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_vec();
        round_trip(&data);
    }

    #[test]
    fn decompress_rejects_out_of_range_distance() {
        let tokens = vec![Token::Literal(b'a'), Token::Match { distance: 5, length: 1 }];
        assert_eq!(decompress(&tokens), None);
    }
}
