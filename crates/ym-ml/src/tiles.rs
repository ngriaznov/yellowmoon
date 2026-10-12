//! Tiling of half-resolution frames for inference: halos from the receptive field, crop (never blend),
//! seam-free reassembly (`docs/07-ml-models.md` section jdd architecture, `docs/05-raw-processing.md`
//! section Tiling and halos, `docs/01-architecture.md` section GPU tiling).
//!
//! A [`Frame`] is the whole network input at half resolution, planar `7 x h x w` ([`crate::model`]).
//! [`run_tiled`] cuts it into tiles of the model's fixed [`TileShape`]: each tile's *core* (the part
//! whose output is kept) is `shape - 2 * halo` pixels, the surrounding halo comes from the neighbouring
//! pixels, and outside the frame it is filled by symmetric reflection of the packed planes (the Bayer
//! phase is preserved because the planes are separate channels). Only the core is written to the output:
//! tiles are cropped, not blended, and a test requires the tiled result to equal the whole-frame result
//! to within 1e-4 (in practice they are bit-identical).
//!
//! Tile origins are multiples of the core size, which is a multiple of 4, so every tile sees the same
//! stride-2 phases as a whole-frame run would. Cores at the right and bottom edges are cropped to what is
//! left of the frame; the model still gets a full-shape input there.

use crate::model::{
    ExecContext, IN_CHANNELS, MlError, OUT_CHANNELS, OUT_SCALE, Result, TILE_MULTIPLE, TileInputs,
    TileModel, TileOutput, TileShape,
};

/// Whole-frame network input at half resolution, planar `7 x height x width`.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    /// Columns (half-resolution pixels).
    pub width: usize,
    /// Rows (half-resolution pixels).
    pub height: usize,
    data: Vec<f32>,
}

impl Frame {
    /// Wrap planar data of `7 * height * width` values.
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(IN_CHANNELS));
        if width == 0 || height == 0 || expected != Some(data.len()) {
            return Err(MlError::Shape(format!(
                "frame {width}x{height} needs {IN_CHANNELS} x {width} x {height} values, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// An all-zero frame.
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; IN_CHANNELS * width * height],
        }
    }

    /// All values, channel-major.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Mutable access to all values.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Channel `c` as a row-major slice.
    pub fn plane(&self, c: usize) -> &[f32] {
        let n = self.width * self.height;
        &self.data[c * n..(c + 1) * n]
    }

    /// Mutable channel `c`.
    pub fn plane_mut(&mut self, c: usize) -> &mut [f32] {
        let n = self.width * self.height;
        &mut self.data[c * n..(c + 1) * n]
    }
}

/// Full-resolution result of a frame: planar `3 x (2 * height) x (2 * width)`.
#[derive(Clone, Debug, PartialEq)]
pub struct FrameOutput {
    /// Columns (full-resolution pixels).
    pub width: usize,
    /// Rows (full-resolution pixels).
    pub height: usize,
    data: Vec<f32>,
}

impl FrameOutput {
    /// An all-zero output for a frame of `frame_width x frame_height` half-resolution pixels.
    pub fn zeros(frame_width: usize, frame_height: usize) -> Self {
        let (w, h) = (OUT_SCALE * frame_width, OUT_SCALE * frame_height);
        Self {
            width: w,
            height: h,
            data: vec![0.0; OUT_CHANNELS * w * h],
        }
    }

    /// Wrap planar full-resolution data of `3 * width * height` values.
    pub fn from_planar(width: usize, height: usize, data: Vec<f32>) -> Result<Self> {
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(OUT_CHANNELS));
        if expected != Some(data.len()) {
            return Err(MlError::Shape(format!(
                "output {width}x{height} needs {OUT_CHANNELS} x {width} x {height} values, got {}",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// All values, channel-major.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Consume into the raw planar vector.
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }

    /// Colour channel `c` as a row-major slice.
    pub fn plane(&self, c: usize) -> &[f32] {
        let n = self.width * self.height;
        &self.data[c * n..(c + 1) * n]
    }
}

/// One tile of a frame: its core rectangle in half-resolution frame coordinates (already clipped to the
/// frame).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileRect {
    /// Left edge of the core.
    pub x0: usize,
    /// Top edge of the core.
    pub y0: usize,
    /// Core width (at most the nominal core width; smaller at the right edge).
    pub width: usize,
    /// Core height (smaller at the bottom edge).
    pub height: usize,
}

/// The nominal core size (`shape - 2 * halo`) for a model, or an error if the halo leaves no core or the
/// shape is invalid.
pub fn core_size(shape: TileShape, halo: usize) -> Result<(usize, usize)> {
    shape.validate()?;
    if !halo.is_multiple_of(TILE_MULTIPLE) {
        return Err(MlError::Shape(format!(
            "halo {halo} must be a multiple of {TILE_MULTIPLE}"
        )));
    }
    if shape.height <= 2 * halo || shape.width <= 2 * halo {
        return Err(MlError::Shape(format!(
            "tile {shape} is not larger than twice the halo ({halo})"
        )));
    }
    Ok((shape.width - 2 * halo, shape.height - 2 * halo))
}

/// Cover a `frame_width x frame_height` frame with tile cores of `core_w x core_h` (row-major order).
/// A zero core size covers nothing and returns no tiles (it would never advance).
pub fn plan_tiles(
    frame_width: usize,
    frame_height: usize,
    core_w: usize,
    core_h: usize,
) -> Vec<TileRect> {
    let mut tiles = Vec::new();
    if core_w == 0 || core_h == 0 {
        return tiles;
    }
    let mut y0 = 0;
    while y0 < frame_height {
        let mut x0 = 0;
        while x0 < frame_width {
            tiles.push(TileRect {
                x0,
                y0,
                width: core_w.min(frame_width - x0),
                height: core_h.min(frame_height - y0),
            });
            x0 += core_w;
        }
        y0 += core_h;
    }
    tiles
}

/// Index of the source pixel for coordinate `i` on an axis of length `n` under symmetric reflection
/// (`... 2 1 0 | 0 1 2 ... n-1 | n-1 n-2 ...`), valid for any distance outside the frame.
pub fn reflect_index(i: isize, n: usize) -> usize {
    let n = n as isize;
    let m = i.rem_euclid(2 * n);
    (if m >= n { 2 * n - 1 - m } else { m }) as usize
}

/// Copy the tile belonging to `rect` into `out`: the window of `out.shape()` whose top-left corner is
/// `halo` pixels up and to the left of the core, with reflection outside the frame.
pub fn extract_tile(frame: &Frame, rect: &TileRect, halo: usize, out: &mut TileInputs) {
    let shape = out.shape();
    let (fw, fh) = (frame.width, frame.height);
    let cols: Vec<usize> = (0..shape.width)
        .map(|j| reflect_index(rect.x0 as isize - halo as isize + j as isize, fw))
        .collect();
    for c in 0..IN_CHANNELS {
        let src = frame.plane(c);
        let dst = out.plane_mut(c);
        for i in 0..shape.height {
            let sy = reflect_index(rect.y0 as isize - halo as isize + i as isize, fh);
            let srow = &src[sy * fw..(sy + 1) * fw];
            for (d, &sx) in dst[i * shape.width..(i + 1) * shape.width]
                .iter_mut()
                .zip(&cols)
            {
                *d = srow[sx];
            }
        }
    }
}

/// Write the core of a tile output (full resolution) into the frame output.
fn place_core(tile: &TileOutput, rect: &TileRect, halo: usize, out: &mut FrameOutput) {
    let shape = tile.shape();
    let (tw, th) = (shape.out_width(), shape.out_height());
    let (ox, oy) = (OUT_SCALE * rect.x0, OUT_SCALE * rect.y0);
    let (cw, ch) = (OUT_SCALE * rect.width, OUT_SCALE * rect.height);
    let h2 = OUT_SCALE * halo;
    let plane_len = out.width * out.height;
    for c in 0..OUT_CHANNELS {
        let src = &tile.as_slice()[c * tw * th..(c + 1) * tw * th];
        for i in 0..ch {
            let s = &src[(h2 + i) * tw + h2..(h2 + i) * tw + h2 + cw];
            let d0 = c * plane_len + (oy + i) * out.width + ox;
            out.data[d0..d0 + cw].copy_from_slice(s);
        }
    }
}

/// Run `model` over a whole frame tile by tile and reassemble the full-resolution result.
///
/// Tiles are processed one after another (the model parallelises internally); `on_tile` is called after
/// each tile with `(done, total)` so callers can report progress or check for cancellation (return
/// `false` to stop early with [`MlError::Cancelled`]).
pub fn run_tiled_with_progress(
    model: &dyn TileModel,
    frame: &Frame,
    ctx: &ExecContext<'_>,
    mut on_tile: impl FnMut(usize, usize) -> bool,
) -> Result<FrameOutput> {
    let shape = model.input_shape();
    let halo = model.halo_half_px();
    let (core_w, core_h) = core_size(shape, halo)?;
    let tiles = plan_tiles(frame.width, frame.height, core_w, core_h);
    let mut out = FrameOutput::zeros(frame.width, frame.height);
    let mut tin = TileInputs::zeros(shape);
    let mut tout = TileOutput::zeros(shape);
    for (i, rect) in tiles.iter().enumerate() {
        extract_tile(frame, rect, halo, &mut tin);
        model.run(&tin, &mut tout, ctx)?;
        place_core(&tout, rect, halo, &mut out);
        if !on_tile(i + 1, tiles.len()) {
            return Err(MlError::Cancelled);
        }
    }
    Ok(out)
}

/// [`run_tiled_with_progress`] without a progress callback.
pub fn run_tiled(
    model: &dyn TileModel,
    frame: &Frame,
    ctx: &ExecContext<'_>,
) -> Result<FrameOutput> {
    run_tiled_with_progress(model, frame, ctx, |_, _| true)
}

/// Pad a frame by symmetric reflection: `left`/`top`/`right`/`bottom` extra half-resolution pixels.
pub fn reflect_pad(frame: &Frame, left: usize, top: usize, right: usize, bottom: usize) -> Frame {
    let (w, h) = (frame.width + left + right, frame.height + top + bottom);
    let mut out = Frame::zeros(w, h);
    for c in 0..IN_CHANNELS {
        let src = frame.plane(c);
        let dst = out.plane_mut(c);
        for y in 0..h {
            let sy = reflect_index(y as isize - top as isize, frame.height);
            for x in 0..w {
                let sx = reflect_index(x as isize - left as isize, frame.width);
                dst[y * w + x] = src[sy * frame.width + sx];
            }
        }
    }
    out
}

/// Number of tiles [`run_tiled`] needs for a frame (useful for cost estimates).
pub fn tile_count(
    frame_width: usize,
    frame_height: usize,
    shape: TileShape,
    halo: usize,
) -> Result<usize> {
    let (cw, ch) = core_size(shape, halo)?;
    Ok(frame_width.div_ceil(cw) * frame_height.div_ceil(ch))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_is_symmetric_and_handles_far_outside() {
        let n = 5;
        let idx: Vec<usize> = (-7..12).map(|i| reflect_index(i, n)).collect();
        assert_eq!(
            idx,
            vec![3, 4, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 4, 3, 2, 1, 0, 0, 1]
        );
        assert_eq!(reflect_index(0, 1), 0);
        assert_eq!(reflect_index(-3, 1), 0);
        assert_eq!(reflect_index(100, 7), reflect_index(100 - 14 * 7, 7));
    }

    #[test]
    fn plan_covers_the_frame_exactly_once() {
        let tiles = plan_tiles(100, 70, 32, 32);
        assert_eq!(tiles.len(), 4 * 3);
        let area: usize = tiles.iter().map(|t| t.width * t.height).sum();
        assert_eq!(area, 100 * 70);
        assert_eq!(
            tiles.last().copied(),
            Some(TileRect {
                x0: 96,
                y0: 64,
                width: 4,
                height: 6
            })
        );
        assert!(plan_tiles(0, 10, 8, 8).is_empty());
        // a zero core never advances: no tiles instead of an endless loop
        assert!(plan_tiles(10, 10, 0, 8).is_empty());
        assert!(plan_tiles(10, 10, 8, 0).is_empty());
    }

    #[test]
    fn size_checks_do_not_overflow() {
        assert!(Frame::new(usize::MAX, 2, vec![0.0; 4]).is_err());
        assert!(Frame::new(0, 4, Vec::new()).is_err());
        assert!(FrameOutput::from_planar(usize::MAX / 2, 4, vec![0.0; 4]).is_err());
        assert!(Frame::new(2, 2, vec![0.0; 28]).is_ok());
    }

    #[test]
    fn core_size_checks() {
        assert_eq!(core_size(TileShape::square(96), 32).ok(), Some((32, 32)));
        assert!(core_size(TileShape::square(64), 32).is_err());
        assert!(core_size(TileShape::square(96), 30).is_err());
        assert!(core_size(TileShape::square(95), 32).is_err());
        assert_eq!(
            tile_count(100, 70, TileShape::square(96), 32).ok(),
            Some(12)
        );
    }

    #[test]
    fn extract_tile_reflects_outside_the_frame() {
        let (w, h) = (6, 5);
        let mut f = Frame::zeros(w, h);
        for c in 0..IN_CHANNELS {
            for (i, v) in f.plane_mut(c).iter_mut().enumerate() {
                *v = (c * 1000 + i) as f32;
            }
        }
        let mut t = TileInputs::zeros(TileShape {
            height: 8,
            width: 8,
        });
        extract_tile(
            &f,
            &TileRect {
                x0: 0,
                y0: 0,
                width: 4,
                height: 4,
            },
            2,
            &mut t,
        );
        // tile pixel (2, 2) is frame pixel (0, 0); tile (0, 0) reflects to frame (1, 1)
        assert_eq!(t.plane(3)[2 * 8 + 2], f.plane(3)[0]);
        assert_eq!(t.plane(3)[0], f.plane(3)[w + 1]);
        assert_eq!(t.plane(3)[8 + 1], f.plane(3)[0]);
        let p = reflect_pad(&f, 2, 2, 0, 0);
        assert_eq!((p.width, p.height), (8, 7));
        assert_eq!(p.plane(0)[2 * 8 + 2], f.plane(0)[0]);
    }
}
