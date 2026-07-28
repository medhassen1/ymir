//! Axis-aligned 2D integer rectangle, used for chunk-column and region
//! extents.
//!
//! Loading, saving, and streaming all operate on rectangular ranges of
//! chunk or region coordinates ("everything within 8 chunks of the
//! player", "this region's 32x32 chunk grid"). `Bounds2` gives those
//! operations one small, well-tested type instead of four loose integers
//! passed around and compared ad hoc at every call site.

/// An axis-aligned rectangle over integer coordinates. Both corners are
/// inclusive: `(min_x, min_y)` and `(max_x, max_y)` are themselves inside
/// the rectangle. A rectangle is empty when `min_x > max_x` or
/// `min_y > max_y`; [`Bounds2::EMPTY`] is the canonical empty value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bounds2 {
    pub min_x: i32,
    pub min_y: i32,
    pub max_x: i32,
    pub max_y: i32,
}

impl Bounds2 {
    /// The canonical empty rectangle: contains no points, and is the
    /// identity value for [`Bounds2::union`].
    pub const EMPTY: Bounds2 = Bounds2 { min_x: 1, min_y: 1, max_x: 0, max_y: 0 };

    /// Builds a rectangle from any two opposite corners, in either order.
    pub fn from_corners(a: (i32, i32), b: (i32, i32)) -> Bounds2 {
        Bounds2 {
            min_x: a.0.min(b.0),
            min_y: a.1.min(b.1),
            max_x: a.0.max(b.0),
            max_y: a.1.max(b.1),
        }
    }

    /// Whether this rectangle contains no points.
    pub fn is_empty(&self) -> bool {
        self.min_x > self.max_x || self.min_y > self.max_y
    }

    /// The number of distinct integer `x` values contained, `0` if empty.
    pub fn width(&self) -> i64 {
        if self.is_empty() {
            0
        } else {
            self.max_x as i64 - self.min_x as i64 + 1
        }
    }

    /// The number of distinct integer `y` values contained, `0` if empty.
    pub fn height(&self) -> i64 {
        if self.is_empty() {
            0
        } else {
            self.max_y as i64 - self.min_y as i64 + 1
        }
    }

    /// The total number of integer coordinates contained.
    pub fn area(&self) -> i64 {
        self.width() * self.height()
    }

    /// Whether `(x, y)` lies within this rectangle, corners included.
    pub fn contains(&self, x: i32, y: i32) -> bool {
        (self.min_x..=self.max_x).contains(&x) && (self.min_y..=self.max_y).contains(&y)
    }

    /// Whether this rectangle shares at least one point with `other`.
    /// Always `false` if either rectangle is empty.
    pub fn intersects(&self, other: &Bounds2) -> bool {
        if self.is_empty() || other.is_empty() {
            return false;
        }
        self.min_x <= other.max_x
            && other.min_x <= self.max_x
            && self.min_y <= other.max_y
            && other.min_y <= self.max_y
    }

    /// The smallest rectangle containing both `self` and `other`. An empty
    /// operand is the identity: unioning with it returns the other side
    /// unchanged.
    pub fn union(&self, other: &Bounds2) -> Bounds2 {
        if self.is_empty() {
            return *other;
        }
        if other.is_empty() {
            return *self;
        }
        Bounds2 {
            min_x: self.min_x.min(other.min_x),
            min_y: self.min_y.min(other.min_y),
            max_x: self.max_x.max(other.max_x),
            max_y: self.max_y.max(other.max_y),
        }
    }

    /// Clamps `(x, y)` into this rectangle. Returns the (arbitrary but
    /// deterministic) `(min_x, min_y)` corner if the rectangle is empty,
    /// since there is then no valid point to clamp into.
    pub fn clamp_point(&self, x: i32, y: i32) -> (i32, i32) {
        if self.is_empty() {
            return (self.min_x, self.min_y);
        }
        (x.clamp(self.min_x, self.max_x), y.clamp(self.min_y, self.max_y))
    }

    /// Computes the smallest rectangle containing every point in `points`.
    /// Returns `None` for an empty slice, since there is no smallest
    /// rectangle containing zero points.
    pub fn from_points(points: &[(i32, i32)]) -> Option<Bounds2> {
        if points.is_empty() {
            return None;
        }
        // SAFETY: the `is_empty` check above guarantees `points` has at
        // least one element, so index `0` is in bounds.
        let first = unsafe { *points.get_unchecked(0) };
        let mut bounds = Bounds2::from_corners(first, first);
        for &(x, y) in &points[1..] {
            bounds = bounds.union(&Bounds2::from_corners((x, y), (x, y)));
        }
        Some(bounds)
    }

    /// An iterator over every integer coordinate contained, in row-major
    /// order (`y` outermost, `x` innermost).
    pub fn iter(&self) -> Bounds2Iter {
        Bounds2Iter { bounds: *self, x: self.min_x, y: self.min_y }
    }

    /// Collects every coordinate contained into a freshly allocated `Vec`,
    /// in the same order as [`Bounds2::iter`], writing directly into a
    /// pre-sized buffer instead of growing one push at a time.
    pub fn to_vec(&self) -> Vec<(i32, i32)> {
        let count = self.area() as usize;
        let mut out: Vec<(i32, i32)> = Vec::with_capacity(count);
        let ptr = out.as_mut_ptr();
        let mut i = 0usize;
        for y in self.min_y..=self.max_y {
            for x in self.min_x..=self.max_x {
                // SAFETY: `count == self.area()` is exactly the number of
                // `(x, y)` pairs these two nested inclusive ranges
                // produce (empty on either axis contributes zero), so `i`
                // never reaches `count`, keeping `ptr.add(i)` within the
                // buffer reserved by `Vec::with_capacity(count)`.
                unsafe {
                    ptr.add(i).write((x, y));
                }
                i += 1;
            }
        }
        // SAFETY: the loop above wrote exactly `count` elements at
        // indices `0..count`, matching the reserved capacity, so they
        // are all initialized.
        unsafe {
            out.set_len(count);
        }
        out
    }
}

/// Row-major iterator over the coordinates in a [`Bounds2`]. See
/// [`Bounds2::iter`].
pub struct Bounds2Iter {
    bounds: Bounds2,
    x: i32,
    y: i32,
}

impl Iterator for Bounds2Iter {
    type Item = (i32, i32);

    fn next(&mut self) -> Option<(i32, i32)> {
        if self.bounds.is_empty() || self.y > self.bounds.max_y {
            return None;
        }
        let item = (self.x, self.y);
        if self.x >= self.bounds.max_x {
            self.x = self.bounds.min_x;
            self.y += 1;
        } else {
            self.x += 1;
        }
        Some(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_corners_normalizes_either_order() {
        let a = Bounds2::from_corners((5, 5), (1, 1));
        let b = Bounds2::from_corners((1, 1), (5, 5));
        assert_eq!(a, b);
        assert_eq!(a, Bounds2 { min_x: 1, min_y: 1, max_x: 5, max_y: 5 });
    }

    #[test]
    fn width_height_area_match_expected_counts() {
        let b = Bounds2::from_corners((0, 0), (3, 1));
        assert_eq!(b.width(), 4);
        assert_eq!(b.height(), 2);
        assert_eq!(b.area(), 8);

        let point = Bounds2::from_corners((7, 7), (7, 7));
        assert_eq!(point.area(), 1);

        assert_eq!(Bounds2::EMPTY.width(), 0);
        assert_eq!(Bounds2::EMPTY.area(), 0);
    }

    #[test]
    fn contains_includes_corners_and_excludes_outside_points() {
        let b = Bounds2::from_corners((0, 0), (4, 4));
        assert!(b.contains(0, 0));
        assert!(b.contains(4, 4));
        assert!(b.contains(2, 2));
        assert!(!b.contains(5, 4));
        assert!(!b.contains(-1, 0));
        assert!(!Bounds2::EMPTY.contains(0, 0));
    }

    #[test]
    fn intersects_detects_overlap_and_rejects_disjoint_or_empty() {
        let a = Bounds2::from_corners((0, 0), (4, 4));
        let overlapping = Bounds2::from_corners((3, 3), (10, 10));
        let disjoint = Bounds2::from_corners((10, 10), (20, 20));
        assert!(a.intersects(&overlapping));
        assert!(!a.intersects(&disjoint));
        assert!(!a.intersects(&Bounds2::EMPTY));
    }

    #[test]
    fn union_grows_to_cover_both_and_treats_empty_as_identity() {
        let a = Bounds2::from_corners((0, 0), (2, 2));
        let b = Bounds2::from_corners((5, -1), (6, 1));
        let u = a.union(&b);
        assert_eq!(u, Bounds2 { min_x: 0, min_y: -1, max_x: 6, max_y: 2 });
        assert_eq!(a.union(&Bounds2::EMPTY), a);
        assert_eq!(Bounds2::EMPTY.union(&a), a);
    }

    #[test]
    fn clamp_point_pulls_outside_points_to_the_nearest_edge() {
        let b = Bounds2::from_corners((0, 0), (4, 4));
        assert_eq!(b.clamp_point(-3, 2), (0, 2));
        assert_eq!(b.clamp_point(2, 99), (2, 4));
        assert_eq!(b.clamp_point(2, 2), (2, 2));
    }

    #[test]
    fn iter_and_to_vec_agree_and_cover_the_whole_rectangle() {
        let b = Bounds2::from_corners((-1, -1), (1, 1));
        let via_iter: Vec<(i32, i32)> = b.iter().collect();
        let via_vec = b.to_vec();
        assert_eq!(via_iter, via_vec);
        assert_eq!(via_iter.len(), b.area() as usize);
        for &(x, y) in &via_iter {
            assert!(b.contains(x, y));
        }
        assert_eq!(Bounds2::EMPTY.iter().count(), 0);
    }

    #[test]
    fn from_points_matches_a_manual_union_sweep() {
        let points = [(3, 4), (-2, 10), (0, 0), (5, -5)];
        let bounds = Bounds2::from_points(&points).unwrap();
        assert_eq!(bounds, Bounds2 { min_x: -2, min_y: -5, max_x: 5, max_y: 10 });
        assert!(Bounds2::from_points(&[]).is_none());
    }
}
