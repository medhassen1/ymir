//! Voxel vertex ambient occlusion, the classic 4-neighbour scheme.
//!
//! Flat faces need cheap fake shadowing: sample the two side neighbours and
//! the diagonal corner around each quad corner, darken by how many are
//! solid, and decide when to flip the quad's diagonal to hide the seam.

/// The 8 neighbour cells surrounding a quad in its own plane, in compass
/// order starting from the top: `[N, NE, E, SE, S, SW, W, NW]`.
pub type QuadNeighbors = [bool; 8];

/// For each of the 4 quad corners (`0`=top-left, `1`=top-right,
/// `2`=bottom-right, `3`=bottom-left), the `[side1, side2, corner]` indices
/// into a [`QuadNeighbors`] array. Every entry here is a literal in `0..8`,
/// verifiable by inspection, which is what makes the unchecked neighbour
/// reads in [`corner_ao`] sound.
const CORNER_NEIGHBORS: [[usize; 3]; 4] = [
    [0, 6, 7], // top-left:     N, W, NW
    [0, 2, 1], // top-right:    N, E, NE
    [4, 2, 3], // bottom-right: S, E, SE
    [4, 6, 5], // bottom-left:  S, W, SW
];

/// The classic per-vertex AO term: given whether the two orthogonal side
/// neighbours and the diagonal corner neighbour are solid, return an
/// occlusion level from `0` (fully occluded) to `3` (fully lit). When both
/// sides are solid the corner is irrelevant — the vertex is maximally
/// occluded regardless of what's in the diagonal cell.
#[inline]
pub fn vertex_ao(side1: bool, side2: bool, corner: bool) -> u8 {
    if side1 && side2 {
        return 0;
    }
    3 - (side1 as u8 + side2 as u8 + corner as u8)
}

/// Compute the AO term for one corner of a quad (`corner` in `0..4`,
/// matching the order documented on [`CORNER_NEIGHBORS`]), reading directly
/// out of the full 8-neighbour set.
pub fn corner_ao(neighbors: &QuadNeighbors, corner: usize) -> u8 {
    assert!(corner < 4, "quad corner index out of range");
    // SAFETY: the assert above guarantees `corner < 4`, and
    // `CORNER_NEIGHBORS` has exactly 4 rows, so this read is in bounds.
    let idx = unsafe { *CORNER_NEIGHBORS.get_unchecked(corner) };
    // SAFETY: every value ever stored in `CORNER_NEIGHBORS` is a literal in
    // `0..8` (see its definition above), and `neighbors` is a `[bool; 8]`,
    // so all three indices are in bounds regardless of which row `idx` came
    // from.
    let (side1, side2, corner_bit) = unsafe {
        (
            *neighbors.get_unchecked(idx[0]),
            *neighbors.get_unchecked(idx[1]),
            *neighbors.get_unchecked(idx[2]),
        )
    };
    vertex_ao(side1, side2, corner_bit)
}

/// Compute the AO term for all four corners of a quad at once, in the order
/// `[top-left, top-right, bottom-right, bottom-left]`.
pub fn quad_ao(neighbors: &QuadNeighbors) -> [u8; 4] {
    [
        corner_ao(neighbors, 0),
        corner_ao(neighbors, 1),
        corner_ao(neighbors, 2),
        corner_ao(neighbors, 3),
    ]
}

/// Decide whether a quad's two triangles should be flipped to run along the
/// `(top-right, bottom-left)` diagonal instead of the default
/// `(top-left, bottom-right)` one. This matters only when the AO is
/// asymmetric: interpolating AO across the "wrong" diagonal produces a
/// visible seam, so the mesher should pick whichever diagonal connects the
/// two corners with the closer combined AO.
pub fn should_flip_quad(ao: [u8; 4]) -> bool {
    let main_diagonal = ao[0] as u16 + ao[2] as u16;
    let other_diagonal = ao[1] as u16 + ao[3] as u16;
    main_diagonal > other_diagonal
}

/// Whether every corner of a quad has the same AO term. A uniform quad
/// needs no per-vertex AO shading at all (a flat multiply is enough), and
/// its two triangulations are interchangeable, so the mesher can skip both
/// the diagonal-flip check and the vertex-color interpolation entirely.
#[inline]
pub fn is_uniform(ao: [u8; 4]) -> bool {
    ao[0] == ao[1] && ao[1] == ao[2] && ao[2] == ao[3]
}

/// Average the 4 corner AO terms into one scalar in `[0, 3]`, useful as a
/// cheap fallback shade for quads a caller has decided not to interpolate
/// per-vertex (e.g. distant chunks rendered at reduced detail).
pub fn average_ao(ao: [u8; 4]) -> f32 {
    ao.iter().map(|&v| v as u32).sum::<u32>() as f32 / 4.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_ao_no_occlusion_is_full_brightness() {
        assert_eq!(vertex_ao(false, false, false), 3);
    }

    #[test]
    fn vertex_ao_both_sides_solid_is_fully_occluded_regardless_of_corner() {
        assert_eq!(vertex_ao(true, true, false), 0);
        assert_eq!(vertex_ao(true, true, true), 0);
    }

    #[test]
    fn vertex_ao_single_contributions_step_down_by_one() {
        assert_eq!(vertex_ao(true, false, false), 2);
        assert_eq!(vertex_ao(false, true, false), 2);
        assert_eq!(vertex_ao(false, false, true), 2);
        assert_eq!(vertex_ao(true, false, true), 1);
        assert_eq!(vertex_ao(false, true, true), 1);
    }

    #[test]
    fn quad_ao_open_quad_is_uniformly_bright() {
        let neighbors: QuadNeighbors = [false; 8];
        let ao = quad_ao(&neighbors);
        assert_eq!(ao, [3, 3, 3, 3]);
        assert_eq!(average_ao(ao), 3.0);
    }

    #[test]
    fn quad_ao_isolates_each_corner_correctly() {
        // Solid only at NW (index 7): should darken exactly the top-left
        // corner (which reads N, W, NW) and leave the rest untouched.
        let mut neighbors: QuadNeighbors = [false; 8];
        neighbors[7] = true;
        let ao = quad_ao(&neighbors);
        assert_eq!(ao, [2, 3, 3, 3]);
    }

    #[test]
    fn equal_diagonal_sums_never_flip_even_when_corners_differ() {
        assert!(!should_flip_quad([3, 3, 3, 3]));
        // Diagonal (0, 2) sums to 2 + 1 = 3; diagonal (1, 3) also sums to
        // 2 + 1 = 3, so despite unequal corners neither diagonal wins.
        assert!(!should_flip_quad([2, 2, 1, 1]));
        assert!(is_uniform([3, 3, 3, 3]));
        assert!(!is_uniform([2, 2, 1, 1]));
    }

    #[test]
    fn asymmetric_ao_flips_towards_the_heavier_diagonal_sum() {
        // Diagonal (0, 2) sums to 6, diagonal (1, 3) sums to 0: the main
        // diagonal strictly outweighs the other, so the quad flips.
        assert!(should_flip_quad([3, 0, 3, 0]));
        // The mirror image (weight on the other diagonal instead) must not
        // flip, since the flip rule is asymmetric by design.
        assert!(!should_flip_quad([0, 3, 0, 3]));
        assert!(!is_uniform([3, 0, 3, 0]));
        assert_eq!(average_ao([0, 1, 2, 3]), 1.5);
    }
}

