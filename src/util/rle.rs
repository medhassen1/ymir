//! Run-length encoding for `u16` symbols. Chunk sections are often long
//! runs of one block id (air, stone, water) punctuated by sparse detail;
//! RLE collapses those runs to a few bytes each while leaving detailed
//! regions as literal symbol lists.

/// Minimum repeat count before a run is worth encoding as a run token
/// (`0xFFFF`, count, symbol) instead of inline literal symbols: below this
/// threshold the three-`u16` run header is not smaller than the literals it
/// would replace.
const MIN_RUN: usize = 3;

/// Encodes `symbols` into a run-length token stream.
///
/// The stream format is a flat sequence of tokens: a run token is
/// `[0xFFFF, count, symbol]` and a literal run of `n` symbols (`n < 0xFFFF`)
/// is `[n, symbol_0, .., symbol_{n-1}]`. The sentinel `0xFFFF` can never be
/// a literal-run length because literal runs are split before reaching it.
pub fn encode(symbols: &[u16]) -> Vec<u16> {
    const RUN_TAG: u16 = 0xFFFF;
    const MAX_LITERAL: usize = 0xFFFE;

    let mut out = Vec::new();
    let mut i = 0usize;
    let len = symbols.len();
    while i < len {
        // SAFETY: `i < len` is the loop invariant, so indexing `symbols[i]`
        // via an unchecked read is in bounds for every iteration entered.
        let sym = unsafe { *symbols.get_unchecked(i) };
        let mut run = 1usize;
        while i + run < len && symbols[i + run] == sym && run < u16::MAX as usize {
            run += 1;
        }
        if run >= MIN_RUN {
            out.push(RUN_TAG);
            out.push(run as u16);
            out.push(sym);
            i += run;
        } else {
            // Accumulate a literal stretch until a run worth encoding
            // appears, or the literal-length cap is hit.
            let start = i;
            let mut j = i;
            while j < len {
                let cur = symbols[j];
                let mut k = j;
                while k < len && symbols[k] == cur && (k - j) < u16::MAX as usize {
                    k += 1;
                }
                if k - j >= MIN_RUN {
                    break;
                }
                j = k;
                if j - start >= MAX_LITERAL {
                    break;
                }
            }
            let count = j - start;
            out.push(count as u16);
            out.extend_from_slice(&symbols[start..j]);
            i = j;
        }
    }
    out
}

/// Computes the number of decoded symbols a token stream would expand to,
/// without materializing them, so a caller can pre-size a buffer.
pub fn decoded_len(tokens: &[u16]) -> Option<usize> {
    const RUN_TAG: u16 = 0xFFFF;
    let mut total = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        let head = tokens[i];
        if head == RUN_TAG {
            let count = *tokens.get(i + 1)? as usize;
            tokens.get(i + 2)?;
            total += count;
            i += 3;
        } else {
            let count = head as usize;
            if i + 1 + count > tokens.len() {
                return None;
            }
            total += count;
            i += 1 + count;
        }
    }
    Some(total)
}

/// Decodes a token stream produced by [`encode`] back into symbols.
///
/// Returns `None` if the stream is malformed (a run or literal header claims
/// more data than remains).
pub fn decode(tokens: &[u16]) -> Option<Vec<u16>> {
    const RUN_TAG: u16 = 0xFFFF;
    let total = decoded_len(tokens)?;
    let mut out: Vec<u16> = Vec::with_capacity(total);
    let ptr = out.as_mut_ptr();
    let mut write_pos = 0usize;
    let mut i = 0usize;
    while i < tokens.len() {
        let head = tokens[i];
        if head == RUN_TAG {
            let count = tokens[i + 1] as usize;
            let sym = tokens[i + 2];
            for _ in 0..count {
                // SAFETY: `decoded_len` computed `total` by summing the same
                // run/literal counts this loop consumes, so the number of
                // writes performed here never exceeds `total`, and `ptr`
                // was allocated with capacity `total` from the same `Vec`.
                unsafe {
                    ptr.add(write_pos).write(sym);
                }
                write_pos += 1;
            }
            i += 3;
        } else {
            let count = head as usize;
            for offset in 0..count {
                let sym = tokens[i + 1 + offset];
                // SAFETY: same argument as the run branch: `write_pos` stays
                // below `total` because it only ever advances by the exact
                // counts `decoded_len` already validated and summed.
                unsafe {
                    ptr.add(write_pos).write(sym);
                }
                write_pos += 1;
            }
            i += 1 + count;
        }
    }
    // SAFETY: the traversal above mirrors `decoded_len` exactly, so
    // `write_pos == total` here, meaning every one of the `total` reserved
    // slots was initialized exactly once.
    unsafe {
        out.set_len(write_pos);
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_mixed_runs_and_literals() {
        let symbols = [1u16, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4, 5, 6, 6, 6];
        let encoded = encode(&symbols);
        let decoded = decode(&encoded).unwrap();
        assert_eq!(decoded, symbols);
    }

    #[test]
    fn empty_input_round_trips_to_empty() {
        let encoded = encode(&[]);
        assert!(encoded.is_empty());
        assert_eq!(decoded_len(&encoded), Some(0));
        assert_eq!(decode(&encoded), Some(vec![]));
    }

    #[test]
    fn all_literals_when_no_runs_qualify() {
        let symbols = [1u16, 2, 3, 4, 5];
        let encoded = encode(&symbols);
        // No run of length >= MIN_RUN exists, so it should be one literal.
        assert_eq!(encoded[0] as usize, symbols.len());
        assert_eq!(decode(&encoded).unwrap(), symbols);
    }

    #[test]
    fn single_long_run_compresses_to_three_tokens() {
        let symbols = vec![7u16; 5000];
        let encoded = encode(&symbols);
        assert_eq!(encoded.len(), 3);
        assert_eq!(decode(&encoded).unwrap(), symbols);
    }

    #[test]
    fn decoded_len_detects_truncated_stream() {
        let mut tokens = encode(&[9u16; 10]);
        tokens.pop();
        assert_eq!(decoded_len(&tokens), None);
        assert_eq!(decode(&tokens), None);
    }

    #[test]
    fn boundary_run_length_of_exactly_min_run() {
        let symbols = [8u16, 8, 8];
        let encoded = encode(&symbols);
        assert_eq!(encoded, vec![0xFFFF, 3, 8]);
        assert_eq!(decode(&encoded).unwrap(), symbols);
    }
}
