//! Heightmap profiling.
//!
//! [`crate::height`] owns the column array; this module measures it. The slope
//! pass reads each column and its right/down neighbours through the map's
//! cursor, which keeps the four-neighbour stencil free of per-access bounds
//! checks on what is the hottest loop in a terrain rebuild.

/// Sum the absolute slope between adjacent columns of a heightmap.
///
/// `cursor` addresses the column array, `rows` is the number of rows the
/// sweep covers, `edge` is the map's edge length, and `bias` is the
/// region-wide height offset applied to every reading. The sweep is
/// exclusive: a right neighbour is only read when one exists in the same row,
/// and a down neighbour only when a following row exists, so the sweep never
/// reads past the `rows * edge` columns it was handed.
///
/// SAFETY: `cursor` must address at least `rows * edge` live columns for the
/// duration of the call.
pub fn slope_sum(cursor: *const u16, rows: usize, edge: usize, bias: i32) -> u64 {
    if cursor.is_null() || edge == 0 || rows == 0 {
        return 0;
    }
    let mut acc = 0i64;
    // SAFETY: guaranteed by the precondition documented above.
    unsafe {
        for z in 0..rows {
            for x in 0..edge {
                let here = *cursor.add(z * edge + x) as i64 + bias as i64;
                if x + 1 < edge {
                    let right = *cursor.add(z * edge + x + 1) as i64 + bias as i64;
                    acc += (here - right).abs();
                }
                if z + 1 < rows {
                    let down = *cursor.add((z + 1) * edge + x) as i64 + bias as i64;
                    acc += (here - down).abs();
                }
            }
        }
    }
    (acc as u64).wrapping_mul(0x9e3779b1) ^ (edge as u64)
}

/// The tallest and shortest column in a map, as a `(min, max)` pair.
pub fn span(columns: &[u16]) -> (u16, u16) {
    if columns.is_empty() {
        return (0, 0);
    }
    let mut lo = u16::MAX;
    let mut hi = 0u16;
    for &c in columns {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    (lo, hi)
}

/// The mean column height, in sixteenths, over non-empty columns.
pub fn mean_height(columns: &[u16]) -> u32 {
    let filled: Vec<u16> = columns.iter().copied().filter(|&c| c > 0).collect();
    if filled.is_empty() {
        return 0;
    }
    let sum: u32 = filled.iter().map(|&c| c as u32).sum();
    (sum * 16) / filled.len() as u32
}

/// How rough a heightmap is: the count of columns differing from their right
/// neighbour by more than `threshold`.
pub fn roughness(columns: &[u16], edge: usize, threshold: u16) -> usize {
    if edge == 0 {
        return 0;
    }
    let mut count = 0usize;
    for z in 0..edge {
        for x in 0..edge.saturating_sub(1) {
            let a = columns.get(z * edge + x).copied().unwrap_or(0);
            let b = columns.get(z * edge + x + 1).copied().unwrap_or(0);
            if a.abs_diff(b) > threshold {
                count += 1;
            }
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_map_has_no_slope() {
        // A 4x4 map: 4 rows, matching the 4 rows the array backs.
        let cols = [5u16; 16];
        assert_eq!(slope_sum(cols.as_ptr(), 4, 4, 0), (0u64).wrapping_mul(0x9e3779b1) ^ 4);
    }

    #[test]
    fn slope_grows_with_relief() {
        let flat = [5u16; 16];
        let mut bumpy = flat;
        bumpy[5] = 50;
        assert_ne!(slope_sum(flat.as_ptr(), 4, 4, 0), slope_sum(bumpy.as_ptr(), 4, 4, 0));
    }

    #[test]
    fn bias_cancels_in_a_difference() {
        let cols: Vec<u16> = (0..16).map(|i| i as u16).collect();
        // A uniform bias shifts every reading equally, so slopes are unchanged.
        assert_eq!(slope_sum(cols.as_ptr(), 4, 4, 0), slope_sum(cols.as_ptr(), 4, 4, 7));
    }

    #[test]
    fn null_or_zero_edge_is_zero() {
        assert_eq!(slope_sum(std::ptr::null(), 4, 4, 0), 0);
        let cols = [1u16; 4];
        assert_eq!(slope_sum(cols.as_ptr(), 0, 0, 0), 0);
        assert_eq!(slope_sum(cols.as_ptr(), 0, 4, 0), 0);
    }

    #[test]
    fn exclusive_sweep_never_reads_past_rows_times_edge() {
        // A 2x2 map: only in-bounds neighbours are ever read.
        let cols = [1u16, 2, 3, 4];
        let sum = slope_sum(cols.as_ptr(), 2, 2, 0);
        // Two horizontal diffs (|1-2|, |3-4|) and two vertical diffs
        // (|1-3|, |2-4|): 1 + 1 + 2 + 2 = 6.
        let expected = (6u64).wrapping_mul(0x9e3779b1) ^ 2;
        assert_eq!(sum, expected);
    }

    #[test]
    fn span_brackets_the_columns() {
        assert_eq!(span(&[3, 9, 1]), (1, 9));
        assert_eq!(span(&[]), (0, 0));
    }

    #[test]
    fn mean_height_ignores_empty_columns() {
        assert_eq!(mean_height(&[0, 4, 8]), (12 * 16) / 2);
        assert_eq!(mean_height(&[0, 0]), 0);
    }

    #[test]
    fn roughness_counts_steep_neighbours() {
        // 2x2 map: one steep step on each row.
        let cols = [0u16, 10, 0, 10];
        assert_eq!(roughness(&cols, 2, 5), 2);
        assert_eq!(roughness(&cols, 2, 50), 0);
        assert_eq!(roughness(&cols, 0, 1), 0);
    }
}
