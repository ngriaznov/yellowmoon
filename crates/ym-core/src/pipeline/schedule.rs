//! The scheduler's registry of kinds (`docs/23` section 5.12).
//!
//! Every kind of node and every kind of mask row says here which footprint it has: per pixel or spatial, how far it
//! reads and what it reads (`ym_color::schedule`). Both functions are **exhaustive `match`es with no wildcard arm**:
//! adding a variant to [`NodeKind`] or [`MaskKind`] does not compile until the new kind is given its step kind, which
//! in turn has to declare its footprint (`StepKind::declared`, `OpKind::declared` in `ym-color`). The scheduler itself
//! (`ColorChain::schedule`, `Schedule::infer`) never looks at a kind, only at the footprints.
//!
//! The random-graph test (`tests/any_graph.rs`) draws its kinds from [`NodeKind::ALL`] and [`MaskKind::ALL`] and fails
//! when a kind that is not marked as excluded there never appeared in a run.

use ym_color::mask::OpKind;
use ym_color::mask_raster::Area;
use ym_color::schedule::{Declared, StepKind};

use crate::settings::{MaskKind, NodeKind};

/// The plan step a node kind runs as, or `None` for the Input and the Output, which are not plan steps.
#[must_use]
pub fn step_kind(kind: NodeKind) -> Option<StepKind> {
    match kind {
        NodeKind::Input | NodeKind::Output => None,
        NodeKind::Corrector => Some(StepKind::Corrector),
        NodeKind::Film => Some(StepKind::Film),
    }
}

/// The prepared row kind a mask kind runs as.
#[must_use]
pub fn op_kind(kind: MaskKind) -> OpKind {
    match kind {
        MaskKind::Object
        | MaskKind::Subject
        | MaskKind::People
        | MaskKind::Background
        | MaskKind::Brush => OpKind::Raster,
        MaskKind::Linear => OpKind::Linear,
        MaskKind::Radial => OpKind::Radial,
        MaskKind::ColorRange => OpKind::Color,
        MaskKind::BrightnessRange => OpKind::Brightness,
    }
}

/// The footprint of a node kind with `reach_px` of its own (a Film node's halo), or `None` for the Input and the Output.
#[must_use]
pub fn node_footprint(kind: NodeKind, reach_px: usize) -> Option<Declared> {
    step_kind(kind).map(|k| k.declared(reach_px))
}

/// The footprint of a mask kind at a render whose raster planes reach `raster_reach_px`.
#[must_use]
pub fn mask_footprint(kind: MaskKind, raster_reach_px: usize) -> Declared {
    op_kind(kind).declared(raster_reach_px)
}

// ---- Sizes and cuts ---------------------------------------------------------------------------------------------------
//
// The scheduler's areas are the only source of sizes (`docs/23` section 5.12, *Limits*): a candidate window's areas give
// the pictures a render of it holds, [`picture_bytes`] and [`camera_bytes`] give their bindings, and [`search_cut`] finds
// the cut of a window into bands or tiles for which every binding fits. Nothing else in the render route estimates a size.

/// The bytes of a picture of four planes of f32 over `w x h` pixels, an even number of columns (two codes share a word):
/// the colour pass's frame, the wider picture of the raster masks and the Film nodes, the stage's frame.
#[must_use]
pub(crate) fn picture_bytes(w: usize, h: usize) -> u64 {
    16 * w.next_multiple_of(2) as u64 * h as u64
}

/// The bytes of the camera RGB (three planes of f32) over `w x h` pixels, as uploaded.
#[must_use]
pub(crate) fn camera_bytes(w: usize, h: usize) -> u64 {
    12 * w.next_multiple_of(2) as u64 * h as u64
}

/// The bytes of the picture over an area of the scheduler.
#[must_use]
pub(crate) fn area_bytes(area: &Area) -> u64 {
    picture_bytes(area.w, area.h)
}

/// The most tiles a window is cut into. More is a device the window does not suit (a binding of a few hundred KiB): the CPU
/// answers, with its reason, rather than a frame of thousands of passes.
pub(crate) const MAX_TILES: usize = 256;

/// How a window is cut.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Shape {
    /// One piece.
    Whole,
    /// Bands of this many rows each, the full width.
    Rows(usize),
    /// Tiles of this many columns and rows.
    Tiles(usize, usize),
}

/// The tiles of `w x h` pixels a `(width, height)` window is cut into, row by row: `(x, y, w, h)`.
#[must_use]
pub(crate) fn tiles_of(
    width: usize,
    height: usize,
    tw: usize,
    th: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let mut tiles = Vec::new();
    for y in (0..height).step_by(th.max(1)) {
        for x in (0..width).step_by(tw.max(1)) {
            tiles.push((x, y, tw.min(width - x), th.min(height - y)));
        }
    }
    tiles
}

impl Shape {
    /// The tiles of a `(width, height)` window in this shape.
    #[must_use]
    pub(crate) fn tiles(self, width: usize, height: usize) -> Vec<(usize, usize, usize, usize)> {
        match self {
            Self::Whole => vec![(0, 0, width, height)],
            Self::Rows(rows) => tiles_of(width, height, width, rows),
            Self::Tiles(tw, th) => tiles_of(width, height, tw, th),
        }
    }
}

/// The coarsest cut of a `width x height` window for which `fits` holds, or `None` when not even the smallest tile fits (a
/// resource limit of the device: the reach of the spatial steps alone fills a binding).
///
/// One rule for every window (a view, the raster planes of a view, an export's tile): the window whole; then bands of rows,
/// starting at `rows_guess` (the rows the widest picture of one band allows, from its area; a start only, `fits` decides)
/// and one band more each time; then, with one row not enough, tiles with the longer side halved in turn (the width stays
/// even, so that a tile starts on an even column: two codes share a word). `fits` is given the tiles of the candidate shape
/// and says whether every binding of each fits; it may test a representative tile (the one with the most photo around it)
/// when the tiles are alike. `columns` allows the cut in columns.
///
/// # Errors
///
/// The error of `fits`.
pub(crate) fn search_cut<E>(
    width: usize,
    height: usize,
    rows_guess: Option<usize>,
    columns: bool,
    fits: &mut dyn FnMut(Shape, &[(usize, usize, usize, usize)]) -> Result<bool, E>,
) -> Result<Option<Shape>, E> {
    let mut rows = rows_guess.unwrap_or(height).clamp(1, height.max(1));
    loop {
        let shape = if rows >= height {
            Shape::Whole
        } else {
            Shape::Rows(rows)
        };
        let tiles = shape.tiles(width, height);
        if fits(shape, &tiles)? {
            return Ok(Some(shape));
        }
        if rows == 1 {
            break;
        }
        // One more band each time: the fewest bands that fit, the same rows for every machine.
        rows = height.div_ceil(tiles.len() + 1).min(rows - 1);
    }
    if !columns {
        return Ok(None);
    }
    // Tiles: from the whole window, the longer side halved in turn, so that the tiles are as square as the window allows
    // (the wider picture of a tile is the tile grown by the reach on every side, so a square tile has the least of it).
    let (mut tw, mut th) = (width.max(2) & !1, height.max(1));
    loop {
        let shape = Shape::Tiles(tw, th);
        let tiles = shape.tiles(width, height);
        if tiles.len() > MAX_TILES {
            return Ok(None);
        }
        if fits(shape, &tiles)? {
            return Ok(Some(shape));
        }
        if tw >= th && tw > 2 {
            tw = (tw / 2).max(2) & !1;
        } else if th > 1 {
            th = (th / 2).max(1);
        } else if tw > 2 {
            tw = (tw / 2).max(2) & !1;
        } else {
            return Ok(None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_color::schedule::Footprint;

    /// The wider picture of a piece of a `w x h` window grown by `reach` on every side must fit `limit` bytes.
    fn cut(w: usize, h: usize, reach: usize, limit: u64, columns: bool) -> Option<Shape> {
        search_cut(w, h, None, columns, &mut |_, tiles| {
            Ok::<_, ()>(
                tiles
                    .iter()
                    .all(|&(_, _, tw, th)| picture_bytes(tw + 2 * reach, th + 2 * reach) <= limit),
            )
        })
        .expect("no error")
    }

    #[test]
    fn a_window_that_fits_one_binding_is_not_cut() {
        // 128 MiB holds 8,388,608 pixels of four f32 planes.
        assert_eq!(cut(2880, 1920, 17, 128 << 20, true), Some(Shape::Whole));
        assert_eq!(cut(4000, 2000, 0, 128 << 20, true), Some(Shape::Whole));
    }

    #[test]
    fn a_window_that_does_not_is_cut_in_rows_whose_wider_pictures_fit() {
        let Some(Shape::Rows(rows)) = cut(5120, 2880, 17, 128 << 20, true) else {
            panic!("a 5K frame is cut in rows");
        };
        assert!(rows < 2880);
        assert!(picture_bytes(5120 + 34, rows + 34) <= 128 << 20);
        // The fewest bands: one band fewer would not fit.
        let bands = 2880usize.div_ceil(rows);
        let fewer = 2880usize.div_ceil(bands - 1);
        assert!(picture_bytes(5120 + 34, fewer + 34) > 128 << 20);
    }

    #[test]
    fn a_reach_wider_than_the_rows_allow_is_cut_in_tiles_or_not_at_all() {
        // One row of 1332 + 2 x 300 columns is 16 x 1932 x 601 = 18.6 MB: over 16 MiB, so the columns are cut too.
        let limit = 16 << 20;
        let Some(Shape::Tiles(tw, th)) = cut(1332, 888, 300, limit, true) else {
            panic!("tiles");
        };
        assert!(picture_bytes(tw + 600, th + 600) <= limit);
        assert_eq!(tw % 2, 0);
        // Without the columns, or when the reach alone fills the binding, nothing fits.
        assert_eq!(cut(1332, 888, 300, limit, false), None);
        assert_eq!(cut(1332, 888, 300, 1 << 20, true), None);
        // Never a band of no rows, however small the limit.
        assert_eq!(cut(64, 5, 0, 1, false), None);
    }

    #[test]
    fn tiles_cover_the_window_once() {
        for shape in [Shape::Whole, Shape::Rows(7), Shape::Tiles(10, 6)] {
            let tiles = shape.tiles(33, 20);
            let mut seen = vec![0u8; 33 * 20];
            for (x, y, w, h) in tiles {
                for row in y..y + h {
                    for col in x..x + w {
                        seen[row * 33 + col] += 1;
                    }
                }
            }
            assert!(seen.iter().all(|&n| n == 1), "{shape:?}");
        }
    }

    #[test]
    fn every_kind_has_a_footprint() {
        for k in NodeKind::ALL {
            let d = node_footprint(k, 7);
            assert_eq!(
                d.is_some(),
                !matches!(k, NodeKind::Input | NodeKind::Output)
            );
        }
        for k in MaskKind::ALL {
            let spatial = matches!(mask_footprint(k, 5).footprint, Footprint::Spatial { .. });
            assert_eq!(
                spatial,
                matches!(
                    k,
                    MaskKind::Object
                        | MaskKind::Subject
                        | MaskKind::People
                        | MaskKind::Background
                        | MaskKind::Brush
                ),
                "{k:?}"
            );
        }
    }
}
