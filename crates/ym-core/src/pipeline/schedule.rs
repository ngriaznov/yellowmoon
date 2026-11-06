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

/// The most pieces [`search_cut`] chooses among by cost. A window that does not fit in this many is cut at the largest piece
/// that fits ([`MAX_TILES`] at most): never a cut of dozens of thin pieces that each process nearly the whole wider picture.
pub(crate) const MAX_COSTED_PIECES: usize = 16;

/// What a piece costs besides its pixels, in pixels of its wider picture: the passes and the submission every piece has
/// whatever its size (a few hundred dispatches, the sampler and the readback of a piece), as the pixels they would take
/// (512 x 512). The cost of a cut is the sum of its pieces' wider pictures in pixels plus this for each piece.
pub(crate) const PIECE_COST_PX: u64 = 512 * 512;

/// `tile_percent` of [`search_cut`]: no grid of tiles among the costed candidates (the ladder after them still cuts tiles when
/// no band fits).
pub(crate) const NO_TILES: u64 = u64::MAX;

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

/// What [`search_cut`] asks of a caller: for the tiles of a candidate shape, `Some(pixels)` when every binding of each fits,
/// with the pixels the shape's pieces process together (each piece's wider picture; for a shape of tiles that are alike a
/// representative tile's, times the tiles), or `None` when one does not fit.
pub(crate) type FitsFn<'a, E> = dyn FnMut(Shape, &[Tile]) -> Result<Option<u64>, E> + 'a;

/// A tile of a window: `(x, y, w, h)`.
pub(crate) type Tile = (usize, usize, usize, usize);

/// The tiles of `w x h` pixels a `(width, height)` window is cut into, row by row.
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

    /// The cost of the shape: `pixels` (what [`FitsFn`] said its pieces process) and a fixed cost for each of its
    /// `pieces` ([`PIECE_COST_PX`]).
    #[must_use]
    pub(crate) fn cost(pixels: u64, pieces: usize) -> u64 {
        pixels.saturating_add(PIECE_COST_PX.saturating_mul(pieces as u64))
    }
}

/// The cheapest cut of a `width x height` window for which `fits` holds, or `None` when not even the smallest tile fits (a
/// resource limit of the device: the reach of the spatial steps alone fills a binding).
///
/// One rule for every window (a view, the raster planes of a view, an export's tile):
///
/// 1. The window whole, when it fits.
/// 2. Otherwise the **cheapest** of the cuts into at most [`MAX_COSTED_PIECES`] pieces, bands of rows and (with `columns`)
///    grids of tiles: the cost is the pixels the pieces process together (each is its window grown by the reach of the
///    spatial steps, so thin pieces each process nearly the whole wider picture) plus [`PIECE_COST_PX`] for each piece
///    ([`Shape::cost`]). Candidates are tried by their number of pieces, and the search ends one count after the first that
///    fits: more pieces only add overlap. Not the fewest pieces: a grid of tiles whose pieces are smaller can beat bands
///    that are thin.
/// 3. Otherwise (not even 16 pieces fit) one cut at the largest piece that fits: bands of rows, starting at `rows_guess`
///    (the rows the widest picture of one band allows, from its area; a start only, `fits` decides) and one band more each
///    time; then, with one row not enough, tiles with the longer side halved in turn (the width stays even, so that a
///    tile starts on an even column: two codes share a word).
///
/// `tile_percent` scales the cost of a grid of tiles (100: as it is). 0: tiles are tried only when no band fits in 16 pieces.
/// [`NO_TILES`]: not at all before the ladder. A sharpened view is cut in bands when it can be: its tiles are finished by the
/// CPU stage over the whole display frame, where the device's fused tail needs bands and gives the view in one piece bit for bit.
///
/// `fits` is given the tiles of the candidate shape; it may test a representative tile (the one with the most photo around
/// it) when the tiles are alike. `columns` allows the cut in columns.
///
/// # Errors
///
/// The error of `fits`.
pub(crate) fn search_cut<E>(
    width: usize,
    height: usize,
    rows_guess: Option<usize>,
    columns: bool,
    tile_percent: u64,
    fits: &mut FitsFn<'_, E>,
) -> Result<Option<Shape>, E> {
    if fits(Shape::Whole, &Shape::Whole.tiles(width, height))?.is_some() {
        return Ok(Some(Shape::Whole));
    }
    // Candidates by their number of pieces: bands and (unless `tile_percent` is 0) grids of tiles together, then, with
    // `tile_percent` 0, the tiles alone when no band fits.
    let mut best: Option<(u64, Shape)> = None;
    let mut tried: Vec<Shape> = Vec::new();
    for tiles_only in [false, true] {
        if tiles_only && (best.is_some() || !columns || tile_percent != 0) {
            break;
        }
        let mut first_fit: Option<usize> = None;
        for n in 2..=MAX_COSTED_PIECES {
            if first_fit.is_some_and(|f| n > f + 1) {
                break;
            }
            let mut candidates = Vec::new();
            if !tiles_only {
                // The bands the caller's own start gives are the first candidate of the count they make (the rows its
                // area allows: the cut the fewest-bands rule always made, so that its frames keep their bits).
                if n == 2
                    && let Some(rows) = rows_guess.filter(|&r| r >= 1 && r < height)
                {
                    candidates.push(Shape::Rows(rows));
                }
                candidates.push(Shape::Rows(height.div_ceil(n).max(1)));
            }
            if columns && (tiles_only || (tile_percent != 0 && tile_percent != NO_TILES)) {
                for c in 2..=n.min(width / 2) {
                    if n % c == 0 {
                        let r = n / c;
                        candidates.push(Shape::Tiles(
                            (width.div_ceil(c).max(2) + 1) & !1,
                            height.div_ceil(r).max(1),
                        ));
                    }
                }
            }
            for shape in candidates {
                let tiles = shape.tiles(width, height);
                // The same shape from two counts (a window of few rows), or more pieces than asked after the rounding.
                if tried.contains(&shape) || tiles.len() > MAX_COSTED_PIECES {
                    continue;
                }
                tried.push(shape);
                if let Some(pixels) = fits(shape, &tiles)? {
                    // The search ends one count after the fewest pieces that fit (a start the caller guessed can fit with
                    // many more than the count it is tried under).
                    first_fit = Some(first_fit.map_or(tiles.len(), |f| f.min(tiles.len())));
                    let mut cost = Shape::cost(pixels, tiles.len());
                    if matches!(shape, Shape::Tiles(..)) {
                        cost = cost.saturating_mul(tile_percent.max(100)) / 100;
                    }
                    if best.is_none_or(|(c, _)| cost < c) {
                        best = Some((cost, shape));
                    }
                }
            }
        }
    }
    if let Some((_, shape)) = best {
        return Ok(Some(shape));
    }
    // More than 16 pieces: one cut at the largest piece that fits.
    let mut rows = rows_guess.unwrap_or(height).clamp(1, height.max(1));
    loop {
        let shape = if rows >= height {
            Shape::Whole
        } else {
            Shape::Rows(rows)
        };
        let tiles = shape.tiles(width, height);
        if fits(shape, &tiles)?.is_some() {
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
        if fits(shape, &tiles)?.is_some() {
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
        search_cut(w, h, None, columns, 100, &mut |_, tiles| {
            let fits = tiles
                .iter()
                .all(|&(_, _, tw, th)| picture_bytes(tw + 2 * reach, th + 2 * reach) <= limit);
            Ok::<_, ()>(fits.then(|| {
                tiles
                    .iter()
                    .map(|&(_, _, tw, th)| ((tw + 2 * reach) * (th + 2 * reach)) as u64)
                    .sum()
            }))
        })
        .expect("no error")
    }

    #[test]
    fn a_window_that_fits_one_binding_is_not_cut() {
        // 128 MiB holds 8,388,608 pixels of four f32 planes.
        assert_eq!(cut(2880, 1920, 17, 128 << 20, true), Some(Shape::Whole));
        assert_eq!(cut(4000, 2000, 0, 128 << 20, true), Some(Shape::Whole));
    }

    /// The pixels the pieces of `shape` process together: each piece grown by `reach` on every side.
    fn pixels(w: usize, h: usize, reach: usize, shape: Shape) -> u64 {
        shape
            .tiles(w, h)
            .iter()
            .map(|&(_, _, tw, th)| ((tw + 2 * reach) * (th + 2 * reach)) as u64)
            .sum()
    }

    fn fits(w: usize, h: usize, reach: usize, limit: u64, shape: Shape) -> bool {
        shape
            .tiles(w, h)
            .iter()
            .all(|&(_, _, tw, th)| picture_bytes(tw + 2 * reach, th + 2 * reach) <= limit)
    }

    #[test]
    fn a_window_that_does_not_is_cut_in_pieces_whose_wider_pictures_fit_at_the_least_cost() {
        let (w, h, reach, limit) = (5120, 2880, 17, 128u64 << 20);
        let shape = cut(w, h, reach, limit, true).expect("a 5K frame is cut");
        assert_ne!(shape, Shape::Whole);
        let tiles = shape.tiles(w, h);
        assert!(tiles.len() <= MAX_COSTED_PIECES);
        assert!(fits(w, h, reach, limit, shape));
        // No cut into at most 16 bands or tiles that fits costs less.
        let cost = Shape::cost(pixels(w, h, reach, shape), tiles.len());
        for n in 2..=MAX_COSTED_PIECES {
            let rows = Shape::Rows(h.div_ceil(n));
            if fits(w, h, reach, limit, rows) {
                let c = Shape::cost(pixels(w, h, reach, rows), rows.tiles(w, h).len());
                assert!(cost <= c, "{shape:?} costs {cost}, {rows:?} costs {c}");
            }
        }
    }

    /// The owner's case (G-12): a 2080 x 1387 viewport at 116 % reads the 100 % picture under it, 1793 x 1196 pixels, and the
    /// Film node's reach (655 pixels) and the raster masks' (75) grow every piece: 3073 x 2488 pixels is 122 MB at 16 bytes
    /// a pixel. At the lavapipe's 128 MiB (76.7 MB for the raster planes' 28 bytes) it was cut into 78 thin pieces that each
    /// processed nearly the whole wider picture; now into a few, and not at all where the device holds it.
    #[test]
    fn the_owners_zoomed_view_is_one_piece_on_a_big_device_and_a_few_on_a_small_one() {
        let (w, h, reach) = (1793, 1196, 730);
        // A Mac: a binding of a gigabyte or more, 4 GiB of budget: 170 MB a picture.
        assert_eq!(cut(w, h, reach, 178 << 20, true), Some(Shape::Whole));
        // Lavapipe, with raster planes: 76.7 MB.
        let limit = (128u64 << 20) / 28 * 16;
        let shape = cut(w, h, reach, limit, true).expect("a cut");
        let pieces = shape.tiles(w, h).len();
        assert!((2..=MAX_COSTED_PIECES).contains(&pieces), "{shape:?}");
        assert!(fits(w, h, reach, limit, shape));
        // The pixels processed: a few times the whole picture's, not dozens.
        let whole = ((w + 2 * reach) * (h + 2 * reach)) as u64;
        assert!(
            pixels(w, h, reach, shape) < 4 * whole,
            "{shape:?}: {} against {whole}",
            pixels(w, h, reach, shape)
        );
    }

    #[test]
    fn beyond_sixteen_pieces_the_cut_is_the_largest_piece_that_fits() {
        // 5120 x 2880 with a reach of 17 at 8 MB a picture: more than 16 pieces, bands of the fewest that fit.
        let limit = 8u64 << 20;
        let Some(Shape::Rows(rows)) = cut(5120, 2880, 17, limit, true) else {
            panic!("bands");
        };
        assert!(2880usize.div_ceil(rows) > MAX_COSTED_PIECES);
        assert!(picture_bytes(5120 + 34, rows + 34) <= limit);
        let bands = 2880usize.div_ceil(rows);
        let fewer = 2880usize.div_ceil(bands - 1);
        assert!(picture_bytes(5120 + 34, fewer + 34) > limit);
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

    /// A start the caller guessed that fits with many pieces does not end the search for the cheapest cut at its count.
    #[test]
    fn a_guess_with_many_pieces_does_not_hide_a_cheaper_cut() {
        let (w, h, reach) = (2000usize, 1400usize, 500usize);
        let limit = 60u64 << 20;
        let shape = search_cut(w, h, Some(40), true, 100, &mut |_, tiles| {
            let fits = tiles
                .iter()
                .all(|&(_, _, tw, th)| picture_bytes(tw + 2 * reach, th + 2 * reach) <= limit);
            Ok::<_, ()>(fits.then(|| {
                tiles
                    .iter()
                    .map(|&(_, _, tw, th)| ((tw + 2 * reach) * (th + 2 * reach)) as u64)
                    .sum()
            }))
        })
        .expect("no error")
        .expect("a cut");
        // Bands of 40 rows (35 of them) fit; a few tiles fit and cost far less.
        assert!(shape.tiles(w, h).len() <= 8, "{shape:?}");
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
