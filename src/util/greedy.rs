//! Pure 2D greedy quad merging over a face-id mask.
//!
//! Naive meshing emits one quad per visible face, a flat wall's worth of
//! redundant geometry. This sweeps a 2D face-id mask and merges runs into
//! the largest rectangles, widening then heightening, mask-only.

/// A merged rectangle of identical face ids, in mask-local cell coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    /// Left edge (inclusive), in cells.
    pub x: usize,
    /// Top edge (inclusive), in cells.
    pub y: usize,
    /// Width, in cells.
    pub w: usize,
    /// Height, in cells.
    pub h: usize,
    /// The face id every cell in this rectangle shares.
    pub id: u32,
}

impl Rect {
    /// Area of the rectangle, in cells.
    pub fn area(&self) -> usize {
        self.w * self.h
    }
}

/// Greedily merge a `width x height` mask (row-major, index `y * width + x`)
/// into the fewest rectangles of matching face id. Consumes the mask: every
/// cell that ends up in an emitted rectangle is cleared to `None`, so a
/// second call on the same buffer returns nothing.
pub fn merge_mask(mask: &mut [Option<u32>], width: usize, height: usize) -> Vec<Rect> {
    assert_eq!(mask.len(), width * height, "mask size must be width * height");
    if width == 0 || height == 0 {
        return Vec::new();
    }

    let mut rects = Vec::new();
    for y in 0..height {
        let mut x = 0;
        while x < width {
            let start = y * width + x;
            let id = match mask[start] {
                Some(id) => id,
                None => {
                    x += 1;
                    continue;
                }
            };

            // Grow the run rightward while the next cell in this row still
            // carries the same id.
            let mut w = 1;
            while x + w < width {
                let idx = y * width + (x + w);
                // SAFETY: the loop condition `x + w < width` combined with
                // `y < height` (the outer `for` bound) gives
                // `idx = y * width + (x + w) < y * width + width
                // = (y + 1) * width <= height * width = mask.len()`.
                let cell = unsafe { *mask.get_unchecked(idx) };
                if cell == Some(id) {
                    w += 1;
                } else {
                    break;
                }
            }

            // Grow the run downward while every cell in the next row, under
            // the whole width-`w` span found above, still matches.
            let mut h = 1;
            'grow_h: while y + h < height {
                for k in 0..w {
                    let idx = (y + h) * width + (x + k);
                    // SAFETY: `y + h < height` (this `while`'s own
                    // condition) and `k < w`, and the width-growth loop
                    // above already established `x + w <= width`, so
                    // `x + k < x + w <= width`. Hence
                    // `idx = (y + h) * width + (x + k)
                    // < (y + h) * width + width = (y + h + 1) * width
                    // <= height * width = mask.len()`.
                    let cell = unsafe { *mask.get_unchecked(idx) };
                    if cell != Some(id) {
                        break 'grow_h;
                    }
                }
                h += 1;
            }

            // Clear every cell in the merged rectangle so later sweeps (and
            // any accidental reprocessing of this same run) see it as empty.
            for row in 0..h {
                for col in 0..w {
                    let idx = (y + row) * width + (x + col);
                    // SAFETY: `row < h` and `y + h <= height` (established
                    // by the height-growth loop's own termination
                    // condition), so `y + row < height`; `col < w` and
                    // `x + w <= width` (established by the width-growth
                    // loop), so `x + col < width`. Together these give
                    // `idx < height * width = mask.len()`, matching the
                    // reasoning used for the reads above.
                    unsafe {
                        *mask.get_unchecked_mut(idx) = None;
                    }
                }
            }

            rects.push(Rect { x, y, w, h, id });
            x += w;
        }
    }
    rects
}

#[cfg(test)]
mod tests {
    use super::*;

    fn count_filled(mask: &[Option<u32>]) -> usize {
        mask.iter().filter(|c| c.is_some()).count()
    }

    #[test]
    fn empty_mask_yields_no_rects() {
        let mut mask = vec![None; 16];
        let rects = merge_mask(&mut mask, 4, 4);
        assert!(rects.is_empty());
    }

    #[test]
    fn full_uniform_mask_merges_into_one_rect() {
        let mut mask = vec![Some(7u32); 16];
        let rects = merge_mask(&mut mask, 4, 4);
        assert_eq!(rects.len(), 1);
        assert_eq!(rects[0], Rect { x: 0, y: 0, w: 4, h: 4, id: 7 });
    }

    #[test]
    fn merged_area_always_equals_filled_cell_count() {
        // A non-trivial, irregular pattern: two ids and a hole.
        #[rustfmt::skip]
        let cells: [Option<u32>; 25] = [
            Some(1), Some(1), Some(1), None,    Some(2),
            Some(1), Some(1), Some(1), None,    Some(2),
            None,    None,    None,    None,    Some(2),
            Some(3), Some(3), Some(1), Some(1), Some(2),
            Some(3), Some(3), Some(1), Some(1), Some(2),
        ];
        let filled_before = count_filled(&cells);
        let mut mask = cells.to_vec();
        let rects = merge_mask(&mut mask, 5, 5);
        let merged_area: usize = rects.iter().map(Rect::area).sum();
        assert_eq!(merged_area, filled_before);
        // And the mask should now be fully cleared.
        assert_eq!(count_filled(&mask), 0);
    }

    #[test]
    fn does_not_merge_across_different_ids() {
        let mut mask = vec![Some(1u32), Some(2u32)];
        let rects = merge_mask(&mut mask, 2, 1);
        assert_eq!(rects.len(), 2);
        assert!(rects.contains(&Rect { x: 0, y: 0, w: 1, h: 1, id: 1 }));
        assert!(rects.contains(&Rect { x: 1, y: 0, w: 1, h: 1, id: 2 }));
    }

    #[test]
    fn stops_height_growth_at_first_mismatched_row() {
        #[rustfmt::skip]
        let cells: [Option<u32>; 6] = [
            Some(9), Some(9),
            Some(9), Some(9),
            Some(0), Some(9),
        ];
        let mut mask = cells.to_vec();
        let rects = merge_mask(&mut mask, 2, 3);
        // Top two rows merge into one 2x2 rect; the bottom row can only
        // merge its second cell (the first is a different id), giving a
        // separate 1x1 rect.
        assert!(rects.contains(&Rect { x: 0, y: 0, w: 2, h: 2, id: 9 }));
        assert!(rects.contains(&Rect { x: 1, y: 2, w: 1, h: 1, id: 9 }));
        assert!(rects.contains(&Rect { x: 0, y: 2, w: 1, h: 1, id: 0 }));
    }

    #[test]
    fn calling_merge_twice_on_the_same_buffer_finds_nothing_the_second_time() {
        let mut mask = vec![Some(4u32); 9];
        let first = merge_mask(&mut mask, 3, 3);
        assert_eq!(first.len(), 1);
        let second = merge_mask(&mut mask, 3, 3);
        assert!(second.is_empty());
    }
}
