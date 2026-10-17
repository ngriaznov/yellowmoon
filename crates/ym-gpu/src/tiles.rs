//! Tile scheduler: tile rectangles, halos computed by walking the stages backwards, crop-not-blend
//! reassembly and row-wise completion (`docs/01-architecture.md` section GPU tiling,
//! `docs/05-raw-processing.md` section Tiling and halos).
//!
//! A 50 MP frame in RGBA f32 is 0.8 GiB, more than wgpu's default buffer and texture limits, so the
//! GPU chain works on tiles of 1024 or 2048 px ([`TILE_SIZES`]) with a halo of context on each
//! side. A tile's halo is not guessed: [`TilePlan::new`] asks the CPU reference for the input
//! rectangle each tile needs by walking the chain's stages from the last to the first
//! ([`crate::cpu::tiling::required_input_rect`], which uses the measured receptive fields of
//! `docs/05` section Tiling and halos). Mosaic-domain rectangles start on even coordinates, so the
//! Bayer phase and the R5 superpixel grid are the same in a tile as in the frame.
//!
//! Tiles are cropped, not blended: the kernels run over the whole `compute` rectangle (core plus
//! halo), mirror at the buffer edge like the CPU reference, and only the core is kept. The outer
//! ring of a tile's output, within one halo of the buffer edge, is therefore never used.
//!
//! Tiles are independent after the raw data is uploaded, so export can stream: [`RowCollector`]
//! gathers the cores of one tile row and releases them together once the last one is in, and
//! [`Band`] assembles them into full-width rows for the encoder.
//!
//! # Chunked work
//!
//! The device thread never pre-empts a running job (`crate::queue`), so a whole-frame export run
//! as one job would keep every viewport frame waiting until it ends. [`TilePlan::tile_rows`] and
//! [`TilePlan::tile_range`] cut a plan down to some of its tiles (same frame, same tile numbers),
//! so an export can be submitted as one job per tile row or per tile at its own priority and the
//! viewport jobs queued meanwhile run between them. Feed the finished tiles of all the jobs to one
//! [`RowCollector`] built from the *full* plan.

use std::ops::Range;

use crate::cpu::tiling::{Halo, Rect, Stage, required_input_rect, tile_grid};

/// The tile sizes the scheduler normally uses, in pixels (`docs/01`: "tiles of 1024 or 2048 px").
pub const TILE_SIZES: [usize; 2] = [1024, 2048];

/// The storage binding size every kernel is designed for: 128 MiB, what Mesa lavapipe reports (the
/// CI adapter; `ym doctor` there prints "storage binding 128 MiB" although the device asks for up to
/// 2 GiB, `docs/17-gpu.md` section 4). Rectangle splitting, slab sizes and the `max_binding_bytes`
/// tests use this constant, never the device's own limit, so the app and `ym` cut identically on
/// every machine; a device that offers more simply does not need the room.
pub const DESIGN_BINDING_BYTES: u64 = 128 << 20;

/// Whether a buffer of `bytes` bytes fits one binding of [`DESIGN_BINDING_BYTES`].
pub const fn fits_design_binding(bytes: u64) -> bool {
    bytes <= DESIGN_BINDING_BYTES
}

/// One tile of a plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tile {
    /// Position in raster order.
    pub index: usize,
    /// Tile row (0 at the top).
    pub row: usize,
    /// Tile column (0 at the left).
    pub col: usize,
    /// The part of the output this tile is responsible for.
    pub core: Rect,
    /// The part of the first stage's input the tile needs (core plus the halos of all stages,
    /// clamped to the frame, left and top edges even): the region held in GPU buffers.
    pub compute: Rect,
}

impl Tile {
    /// Offset of the core inside the compute rectangle, `(dx, dy)`.
    pub fn core_offset(&self) -> (usize, usize) {
        (self.core.x - self.compute.x, self.core.y - self.compute.y)
    }

    /// The core in the compute rectangle's own coordinates.
    pub fn core_in_compute(&self) -> Rect {
        self.core.relative_to(&self.compute)
    }
}

/// All tiles of one frame.
#[derive(Clone, Debug)]
pub struct TilePlan {
    frame: Rect,
    tile_px: usize,
    cols: usize,
    rows: usize,
    tiles: Vec<Tile>,
}

impl TilePlan {
    /// Plans the tiles of a `frame_w x frame_h` frame: cores of at most `tile_px` (rounded down to
    /// an even number, at least 2), inputs from the backwards walk over `stages`.
    ///
    /// `stages` is the chain in processing order; an empty slice means pointwise processing with no
    /// halo. The output rectangle equals the input frame (no lens crop); for a chain that ends in
    /// the lens stage the cores are in the cropped output's coordinates, as in
    /// [`required_input_rect`].
    pub fn new(frame_w: usize, frame_h: usize, tile_px: usize, stages: &[Stage<'_>]) -> TilePlan {
        let frame = Rect::whole(frame_w, frame_h);
        let tile_px = (tile_px.max(2) / 2) * 2;
        let cores = tile_grid(frame_w, frame_h, tile_px, tile_px, 2);
        let cols = frame_w.div_ceil(tile_px);
        let rows = frame_h.div_ceil(tile_px);
        let tiles = cores
            .into_iter()
            .enumerate()
            .map(|(index, core)| Tile {
                index,
                row: index / cols.max(1),
                col: index % cols.max(1),
                core,
                compute: required_input_rect(stages, core, frame),
            })
            .collect();
        TilePlan {
            frame,
            tile_px,
            cols,
            rows,
            tiles,
        }
    }

    /// A plan for pointwise processing with a fixed symmetric halo of `halo_px`.
    pub fn with_halo(frame_w: usize, frame_h: usize, tile_px: usize, halo_px: usize) -> TilePlan {
        let mut plan = TilePlan::new(frame_w, frame_h, tile_px, &[]);
        let frame = plan.frame;
        for t in &mut plan.tiles {
            t.compute = t.core.grow_aligned(Halo::uniform(halo_px), frame, 2);
        }
        plan
    }

    /// The frame rectangle.
    pub fn frame(&self) -> Rect {
        self.frame
    }

    /// The core size in pixels.
    pub fn tile_px(&self) -> usize {
        self.tile_px
    }

    /// Tiles per row.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Tile rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// All tiles in raster order.
    pub fn tiles(&self) -> &[Tile] {
        &self.tiles
    }

    /// The tiles of row `row` (none if the row is outside the plan or was cut out of it).
    pub fn row(&self, row: usize) -> &[Tile] {
        // Tiles are in raster order, so those of one row are contiguous.
        let start = self.tiles.partition_point(|t| t.row < row);
        let end = self.tiles.partition_point(|t| t.row <= row);
        &self.tiles[start..end]
    }

    /// A plan with only the tiles of the tile rows in `rows`. The frame, the tile size and every
    /// kept tile (index, row, column, core, compute rectangle) are unchanged, so the result can be
    /// run like the whole plan but as a smaller job (see the module notes on chunked work).
    #[must_use]
    pub fn tile_rows(&self, rows: Range<usize>) -> TilePlan {
        let mut plan = self.clone();
        plan.tiles.retain(|t| rows.contains(&t.row));
        plan
    }

    /// A plan with only the tiles whose raster index is in `indices` (see
    /// [`tile_rows`](Self::tile_rows)).
    #[must_use]
    pub fn tile_range(&self, indices: Range<usize>) -> TilePlan {
        let mut plan = self.clone();
        plan.tiles.retain(|t| indices.contains(&t.index));
        plan
    }

    /// The largest compute rectangle area in pixels (what buffers must be sized for).
    pub fn max_compute_area(&self) -> usize {
        self.tiles
            .iter()
            .map(|t| t.compute.area())
            .max()
            .unwrap_or(0)
    }

    /// The largest compute rectangle width and height.
    pub fn max_compute_size(&self) -> (usize, usize) {
        self.tiles
            .iter()
            .fold((0, 0), |(w, h), t| (w.max(t.compute.w), h.max(t.compute.h)))
    }
}

/// The largest tile size from `candidates` (descending order is tried first) whose working set fits
/// in half the budget with `in_flight` tiles at once.
///
/// `bytes_per_pixel` is the device memory one compute-rectangle pixel needs across all the
/// intermediate buffers of the chain (see [`crate::kernels::spike::bytes_per_pixel`]), `halo_px`
/// the total halo. Returns the smallest candidate when nothing fits, so the caller still gets a
/// plan and the pool reports the shortfall.
pub fn tile_size_for_budget(
    budget_bytes: u64,
    bytes_per_pixel: u64,
    halo_px: usize,
    in_flight: usize,
    candidates: &[usize],
) -> usize {
    let mut sizes: Vec<usize> = candidates.to_vec();
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    let allowed = budget_bytes / 2;
    for &tile in &sizes {
        let side = (tile + 2 * halo_px) as u64;
        let need = side * side * bytes_per_pixel * in_flight.max(1) as u64;
        if need <= allowed {
            return tile;
        }
    }
    sizes.last().copied().unwrap_or(256)
}

/// Gathers finished tile cores of one frame and releases them a tile row at a time.
///
/// Tiles may arrive in any order (a pipelined or parallel scheduler need not finish them in raster
/// order); [`RowCollector::push`] returns a row's tiles, ordered left to right, as soon as the last
/// of them has arrived. Rows themselves are released in completion order; the sequential scheduler
/// completes them top to bottom, which is what a streaming encoder needs.
#[derive(Debug)]
pub struct RowCollector<T> {
    cols: usize,
    rows: Vec<Vec<Option<(Tile, T)>>>,
    filled: Vec<usize>,
}

impl<T> RowCollector<T> {
    /// A collector for `plan`.
    pub fn new(plan: &TilePlan) -> Self {
        Self {
            cols: plan.cols(),
            rows: (0..plan.rows())
                .map(|_| (0..plan.cols()).map(|_| None).collect())
                .collect(),
            filled: vec![0; plan.rows()],
        }
    }

    /// Adds a finished tile. Returns the whole tile row (left to right) when this was its last
    /// missing tile. A tile pushed twice replaces the first and is not counted again.
    pub fn push(&mut self, tile: Tile, data: T) -> Option<Vec<(Tile, T)>> {
        let slot = self.rows.get_mut(tile.row)?.get_mut(tile.col)?;
        if slot.replace((tile, data)).is_none() {
            self.filled[tile.row] += 1;
        }
        if self.filled[tile.row] == self.cols {
            let row = std::mem::take(&mut self.rows[tile.row]);
            self.filled[tile.row] = usize::MAX; // released
            return Some(row.into_iter().flatten().collect());
        }
        None
    }

    /// Tile rows not yet released.
    pub fn rows_pending(&self) -> usize {
        self.filled.iter().filter(|&&f| f != usize::MAX).count()
    }
}

/// A full-width horizontal band of the output, `height` rows from row `y0`, planar
/// (`channels` planes one after the other, each `width * height`).
#[derive(Clone, Debug, PartialEq)]
pub struct Band<T> {
    /// First row of the band in the frame.
    pub y0: usize,
    /// Frame width.
    pub width: usize,
    /// Rows in the band.
    pub height: usize,
    /// Number of planes.
    pub channels: usize,
    /// `channels * width * height` samples.
    pub data: Vec<T>,
}

impl<T: Copy + Default> Band<T> {
    /// An empty (default-filled) band.
    pub fn new(y0: usize, width: usize, height: usize, channels: usize) -> Self {
        Self {
            y0,
            width,
            height,
            channels,
            data: vec![T::default(); width * height * channels],
        }
    }

    /// Pastes a planar tile core (`channels` planes of `core.w x core.h`) at its place in the band.
    ///
    /// Panics if the core does not lie inside the band or the data is too short; callers pass
    /// rectangles from a [`TilePlan`] and data cropped to the core.
    pub fn paste_planar(&mut self, core: Rect, tile: &[T]) {
        assert!(
            core.y >= self.y0 && core.y1() <= self.y0 + self.height && core.x1() <= self.width,
            "tile core {core:?} outside the band"
        );
        let plane = core.w * core.h;
        assert!(tile.len() >= plane * self.channels, "tile data too short");
        let band_plane = self.width * self.height;
        for c in 0..self.channels {
            for y in 0..core.h {
                let dst = c * band_plane + (core.y - self.y0 + y) * self.width + core.x;
                let src = c * plane + y * core.w;
                self.data[dst..dst + core.w].copy_from_slice(&tile[src..src + core.w]);
            }
        }
    }
}

/// Assembles one released tile row into a [`Band`]. `cores` are planar tile cores with
/// `channels` planes each.
pub fn band_of_row<T: Copy + Default>(
    frame_width: usize,
    channels: usize,
    row: &[(Tile, Vec<T>)],
) -> Option<Band<T>> {
    let first = row.first()?.0.core;
    let mut band = Band::new(first.y, frame_width, first.h, channels);
    for (tile, data) in row {
        band.paste_planar(tile.core, data);
    }
    Some(band)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::demosaic::Method;

    fn chain() -> Vec<Stage<'static>> {
        vec![
            Stage::Normalize,
            Stage::Gains,
            Stage::Demosaic(Method::Bilinear),
        ]
    }

    #[test]
    fn cores_cover_the_frame_exactly_once_and_inputs_contain_their_cores() {
        let plan = TilePlan::new(1001, 777, 256, &chain());
        assert_eq!((plan.cols(), plan.rows()), (4, 4));
        assert_eq!(plan.tiles().len(), 16);
        let mut seen = vec![0u8; 1001 * 777];
        for t in plan.tiles() {
            assert!(t.compute.contains(&t.core), "{t:?}");
            assert_eq!(
                (t.compute.x % 2, t.compute.y % 2),
                (0, 0),
                "even origin {t:?}"
            );
            assert_eq!((t.core.x % 2, t.core.y % 2), (0, 0));
            for y in t.core.y..t.core.y1() {
                for x in t.core.x..t.core.x1() {
                    seen[y * 1001 + x] += 1;
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1));
    }

    #[test]
    fn halos_come_from_the_backwards_walk_and_clamp_at_the_frame() {
        let plan = TilePlan::new(1024, 1024, 512, &chain());
        // Bilinear needs one pixel; the left/top edge is rounded down to an even coordinate.
        let t = plan.tiles()[3]; // bottom right
        assert_eq!(t.core, Rect::new(512, 512, 512, 512));
        assert_eq!(t.compute, Rect::new(510, 510, 514, 514));
        let t = plan.tiles()[0];
        assert_eq!(
            t.compute,
            Rect::new(0, 0, 513, 513),
            "clamped at the frame edge"
        );
        assert_eq!(t.core_offset(), (0, 0));
        assert_eq!(plan.tiles()[3].core_offset(), (2, 2));
        assert_eq!(plan.tiles()[3].core_in_compute(), Rect::new(2, 2, 512, 512));
    }

    #[test]
    fn a_deep_chain_gets_the_summed_halo() {
        let stages = [
            Stage::Normalize,
            Stage::Highlights,
            Stage::Demosaic(Method::Menon),
            Stage::Denoise { radius_px: 2 },
        ];
        let plan = TilePlan::new(2000, 2000, 1000, &stages);
        let t = plan.tiles()[3];
        let halo = 4 + Method::Menon.halo() + 2;
        assert_eq!(t.compute.x, 1000 - halo);
        assert_eq!(t.compute.x1(), 2000);
        let fixed = TilePlan::with_halo(2000, 2000, 1000, 14);
        assert_eq!(fixed.tiles()[3].compute, t.compute);
    }

    #[test]
    fn the_largest_compute_rectangle_is_the_interior_tile() {
        let plan = TilePlan::new(3000, 3000, 1024, &chain());
        // One pixel of halo each side; the left/top edge goes down to an even coordinate (2 px) and
        // the right/bottom edge does not move (1 px): 1024 + 2 + 1.
        assert_eq!(plan.max_compute_size(), (1027, 1027));
        assert_eq!(plan.max_compute_area(), 1027 * 1027);
        assert_eq!(plan.row(1).len(), 3);
        assert!(plan.row(9).is_empty());
    }

    #[test]
    fn a_plan_can_be_cut_into_rows_and_single_tiles_that_together_are_the_plan() {
        let plan = TilePlan::new(1001, 777, 256, &chain());
        let rows: Vec<TilePlan> = (0..plan.rows()).map(|r| plan.tile_rows(r..r + 1)).collect();
        let singles: Vec<TilePlan> = (0..plan.tiles().len())
            .map(|i| plan.tile_range(i..i + 1))
            .collect();
        for cut in [&rows, &singles] {
            let joined: Vec<Tile> = cut.iter().flat_map(|p| p.tiles().iter().copied()).collect();
            assert_eq!(
                joined,
                plan.tiles(),
                "the pieces are the plan, tile for tile"
            );
            for p in cut.iter() {
                assert_eq!(p.frame(), plan.frame());
                assert_eq!((p.cols(), p.rows(), p.tile_px()), (4, 4, 256));
            }
        }
        assert_eq!(rows[2].row(2), plan.row(2));
        assert!(rows[2].row(1).is_empty() && rows[2].row(3).is_empty());
        assert_eq!(plan.tile_rows(1..3).tiles().len(), 8);
        assert!(plan.tile_rows(9..12).tiles().is_empty());
        // A collector built from the full plan takes the tiles of the pieces in any order.
        let mut rc = RowCollector::new(&plan);
        let mut released = 0;
        for piece in singles.iter().rev() {
            for t in piece.tiles() {
                released += usize::from(rc.push(*t, ()).is_some());
            }
        }
        assert_eq!((released, rc.rows_pending()), (4, 0));
    }

    #[test]
    fn the_design_binding_is_the_lavapipe_figure() {
        assert_eq!(DESIGN_BINDING_BYTES, 134_217_728);
        assert!(fits_design_binding(DESIGN_BINDING_BYTES));
        assert!(!fits_design_binding(DESIGN_BINDING_BYTES + 1));
        // The one plane of a 61 MP body (9504 x 6336, two bytes a sample) is the largest plane
        // kept resident; it fits, a 61 MP f32 quantisation-step plane does not.
        assert!(fits_design_binding(9504 * 6336 * 2));
        assert!(!fits_design_binding(9504 * 6336 * 4));
    }

    #[test]
    fn tile_size_follows_the_budget() {
        let c = [1024, 2048];
        // 4 GiB budget, 30 bytes per pixel, 2 tiles in flight: 2048 fits comfortably.
        assert_eq!(tile_size_for_budget(4 << 30, 30, 8, 2, &c), 2048);
        // 256 MiB: 2048 needs (2064^2 * 30 * 2) = 255 MB > 128 MiB allowed; 1024 needs 1040^2*60 = 65 MB.
        assert_eq!(tile_size_for_budget(256 << 20, 30, 8, 2, &c), 1024);
        // Nothing fits: the smallest candidate.
        assert_eq!(tile_size_for_budget(1 << 20, 30, 8, 2, &c), 1024);
    }

    #[test]
    fn rows_are_released_when_their_last_tile_arrives_in_any_order() {
        let plan = TilePlan::new(8, 8, 4, &[]);
        let mut rc = RowCollector::new(&plan);
        let t = |i: usize| plan.tiles()[i];
        assert!(rc.push(t(1), 'b').is_none());
        assert!(rc.push(t(2), 'c').is_none());
        let row0 = rc.push(t(0), 'a').expect("row 0 complete");
        assert_eq!(row0.iter().map(|(_, c)| *c).collect::<String>(), "ab");
        assert_eq!(rc.rows_pending(), 1);
        let row1 = rc.push(t(3), 'd').expect("row 1 complete");
        assert_eq!(row1.iter().map(|(t, _)| t.col).collect::<Vec<_>>(), [0, 1]);
        assert_eq!(rc.rows_pending(), 0);
    }

    #[test]
    fn bands_assemble_planar_cores_into_full_width_rows() {
        let plan = TilePlan::new(6, 4, 4, &[]);
        let mut rc = RowCollector::new(&plan);
        let mut out = None;
        for t in plan.tiles() {
            // Two planes; value encodes plane, x and y of the frame.
            let mut data = Vec::new();
            for c in 0..2u32 {
                for y in 0..t.core.h {
                    for x in 0..t.core.w {
                        data.push(c * 1000 + ((t.core.y + y) * 10 + t.core.x + x) as u32);
                    }
                }
            }
            if let Some(row) = rc.push(*t, data) {
                out = band_of_row(6, 2, &row);
            }
        }
        let band = out.expect("a band");
        assert_eq!((band.y0, band.width, band.height), (0, 6, 4));
        for c in 0..2usize {
            for y in 0..4usize {
                for x in 0..6usize {
                    assert_eq!(
                        band.data[c * 24 + y * 6 + x],
                        c as u32 * 1000 + (y * 10 + x) as u32
                    );
                }
            }
        }
    }
}
