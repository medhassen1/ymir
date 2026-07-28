//! Delta and double-delta encoding for `i32` sequences. Heightmaps and
//! light-level profiles change slowly from one entry to the next; delta
//! coding rewrites each value as the difference from its predecessor, and
//! double-delta repeats that once more, turning smooth data into small
//! numbers a varint or bit-packed coder can store cheaply.

/// Applies forward delta coding in place: `data[0]` is left unchanged and
/// each subsequent element becomes `data[i] - data[i-1]`.
///
/// The subtraction saturates at `i32::MIN`/`i32::MAX` instead of wrapping or
/// panicking: an out-of-range delta is clamped, so [`delta_decode_in_place`]
/// will not perfectly reconstruct the original value in that rare,
/// pathological case, trading exactness at extreme magnitudes for a total,
/// panic-free function.
pub fn delta_encode_in_place(data: &mut [i32]) {
    let len = data.len();
    if len < 2 {
        return;
    }
    // SAFETY: `prev` is a local copy of the original value at the previous
    // index, captured before that slot is overwritten, so overwriting
    // `data[i]` in place never corrupts a value this loop still needs to
    // read. All indices `1..len` are within `data`'s bounds by definition.
    let mut prev = data[0];
    for i in 1..len {
        unsafe {
            let cur = *data.get_unchecked(i);
            *data.get_unchecked_mut(i) = cur.saturating_sub(prev);
            prev = cur;
        }
    }
}

/// Inverse of [`delta_encode_in_place`]: reconstructs original values from
/// deltas in place, saturating on overflow.
pub fn delta_decode_in_place(data: &mut [i32]) {
    let len = data.len();
    if len < 2 {
        return;
    }
    let mut running = data[0];
    for i in 1..len {
        // SAFETY: `i` ranges over `1..len`, strictly inside `data`'s bounds,
        // and `running` always holds the just-reconstructed value at
        // `i - 1`, computed before this iteration overwrites index `i`.
        unsafe {
            running = running.saturating_add(*data.get_unchecked(i));
            *data.get_unchecked_mut(i) = running;
        }
    }
}

/// Applies double-delta coding in place: first a forward delta pass, then a
/// second forward delta pass over the result (excluding the leading anchor
/// each time). Equivalent to encoding second differences.
pub fn double_delta_encode_in_place(data: &mut [i32]) {
    delta_encode_in_place(data);
    if data.len() > 2 {
        delta_encode_in_place(&mut data[1..]);
    }
}

/// Inverse of [`double_delta_encode_in_place`].
pub fn double_delta_decode_in_place(data: &mut [i32]) {
    if data.len() > 2 {
        delta_decode_in_place(&mut data[1..]);
    }
    delta_decode_in_place(data);
}

/// Delta-encodes `src` into a freshly allocated vector of the same length,
/// leaving `src` untouched.
pub fn delta_encode(src: &[i32]) -> Vec<i32> {
    let len = src.len();
    let mut out: Vec<i32> = Vec::with_capacity(len);
    let ptr = out.as_mut_ptr();
    // SAFETY: `ptr` points at storage freshly allocated for exactly `len`
    // elements. The block below writes index 0 (when `len > 0`) and then
    // indices `1..len`, i.e. every index in `0..len` exactly once, so no
    // write lands outside the reserved capacity and no slot is written
    // twice.
    unsafe {
        if len > 0 {
            ptr.write(src[0]);
        }
        for i in 1..len {
            let d = src[i].saturating_sub(src[i - 1]);
            ptr.add(i).write(d);
        }
    }
    // SAFETY: the block above initialized every index in `0..len`, so it is
    // sound to tell the vector its length is now `len`.
    unsafe {
        out.set_len(len);
    }
    out
}

/// Reconstructs original values from a delta-encoded slice produced by
/// [`delta_encode`], returning a freshly allocated vector.
pub fn delta_decode(deltas: &[i32]) -> Vec<i32> {
    let mut out = deltas.to_vec();
    delta_decode_in_place(&mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_round_trip_smooth_sequence() {
        let original: Vec<i32> = (0..100).map(|i| (i * i) / 3 - 500).collect();
        let mut buf = original.clone();
        delta_encode_in_place(&mut buf);
        delta_decode_in_place(&mut buf);
        assert_eq!(buf, original);
    }

    #[test]
    fn double_delta_round_trip() {
        let original: Vec<i32> = vec![10, 12, 15, 19, 24, 30, 37, 45];
        let mut buf = original.clone();
        double_delta_encode_in_place(&mut buf);
        double_delta_decode_in_place(&mut buf);
        assert_eq!(buf, original);
    }

    #[test]
    fn constant_sequence_deltas_are_zero() {
        let mut data = vec![42i32; 10];
        delta_encode_in_place(&mut data);
        assert_eq!(&data[1..], &vec![0; 9][..]);
        assert_eq!(data[0], 42);
    }

    #[test]
    fn empty_and_singleton_are_no_ops() {
        let mut empty: Vec<i32> = vec![];
        delta_encode_in_place(&mut empty);
        assert!(empty.is_empty());

        let mut single = vec![7i32];
        delta_encode_in_place(&mut single);
        assert_eq!(single, vec![7]);
        delta_decode_in_place(&mut single);
        assert_eq!(single, vec![7]);
    }

    #[test]
    fn saturating_behavior_at_extremes() {
        let mut data = vec![i32::MIN, i32::MAX];
        delta_encode_in_place(&mut data);
        // i32::MAX - i32::MIN overflows; must saturate rather than panic.
        assert_eq!(data[1], i32::MAX);

        let mut data2 = vec![i32::MAX, i32::MIN];
        delta_encode_in_place(&mut data2);
        assert_eq!(data2[1], i32::MIN);
    }

    #[test]
    fn allocating_helpers_match_in_place_versions() {
        let original = vec![5, 5, 8, 2, 2, 2, 100, -50];
        let encoded = delta_encode(&original);
        let mut in_place = original.clone();
        delta_encode_in_place(&mut in_place);
        assert_eq!(encoded, in_place);
        assert_eq!(delta_decode(&encoded), original);
    }
}
