//! Lossless M and S reduced path: YCbCr tiles to unbalanced camera RGB (`docs/04-raw-decoding.md` section Lossless M and S).
//!
//! # What the file holds
//!
//! Lossless M and S files are not mosaics. The camera has demosaiced the image, applied its
//! white-balance gains, converted to YCbCr, subsampled the chroma (4:2:0 or 4:2:2) and stored the
//! result as 512 x 512 lossless-JPEG tiles with three components (TIFF compression 7, photometric
//! YCbCr). Y uses `Hi x Vi` = 2 x 2 (4:2:0) or 2 x 1 (4:2:2), Cb and Cr 1 x 1. Cb and Cr are
//! stored with an offset of 16383, which LibRaw's `ycc2rgb` and rawler's `ycbcr_to_rgb` both use
//! (and the black pedestal of 1024 they report says the samples are on a doubled, 15-bit scale).
//!
//! # What this module returns
//!
//! 1. The tiles are decoded (in parallel) into full-frame Y, Cb and Cr planes. 4:2:0 tiles use
//!    Sony's variant of the first-column prediction ([`super::ljpeg::Options::sony_420_quirk`]).
//! 2. The chroma planes are upsampled to full resolution, either by replication (what LibRaw does)
//!    or by co-sited linear interpolation (what rawler does, and the default here: it avoids
//!    blocky chroma). The chroma sample belongs to the left or upper luma sample of its group.
//! 3. YCbCr becomes RGB with the full-range BT.601 inverse that LibRaw's `ycc2rgb` uses, in
//!    single precision with truncation, so [`ChromaUpsampling::Replicate`] without the white
//!    balance step reproduces LibRaw's pseudo-raw image exactly:
//!    `R = Y + 1.402 Cr`, `G = Y - 0.34414 Cb - 0.71414 Cr`, `B = Y + 1.772 Cb`, with `Cb` and
//!    `Cr` taken relative to 16383, clamped to `[0, 65535]`.
//! 4. The camera's white balance is undone: `v = (v_balanced - black) / gain + black`, where the
//!    gains are the as-shot levels `wb_as_shot` (SR2 tag 0x7313) divided by their green, and the
//!    black pedestal is twice the SR2 black level (1024 for the usual 512). Both are what rawler
//!    (`black * 2` for three-component data) and LibRaw (`black = 1024`) use. The result is
//!    unbalanced camera RGB, the same domain as a demosaiced mosaic, so the colour pipeline
//!    treats it like any other file but skips demosaicing (`Layout::Ycbcr`).
//!
//! # Levels
//!
//! The balanced data ends at the ceiling LibRaw measured on over-exposed frames (17536 above a
//! pedestal of 1024: `LIBRAW_MAXIMUM`). After the gains are undone, a channel therefore clips at
//! `black + (17536 - 1024) / gain`; that is the `white` this module reports per channel. It is
//! LibRaw's estimate, not something the file states, and the scale of the camera's YCbCr data
//! has not been checked against real Lossless M or S files (there is none in the test
//! environment). The levels are ordered red, green, green, blue.
//!
//! # Output layout
//!
//! `rgb` is a [`Plane<u16>`] with three interleaved samples per pixel: its `width` is
//! `3 * pixel width` and its `height` the number of rows. The pixel width is
//! `rgb.width / 3`, also `RawMetadata::width`.

use rayon::prelude::*;

use super::ljpeg::{self, Options as LjpegOptions};
use crate::error::RawError;
use crate::layout::{CfaColor, CfaPattern, ChromaSubsampling, Plane, check_dimensions};
use crate::metadata::RawMetadata;
use crate::tiff::ByteSource;

/// The offset Cb and Cr are stored with.
pub const CHROMA_CENTER: i32 = 16383;

/// The pedestal LibRaw uses for the pseudo-raw RGB data (`black = 1024`), twice the usual
/// 14-bit sensor black level of 512.
pub const LIBRAW_BLACK: f32 = 1024.0;

/// LibRaw's `maximum` for the pseudo-raw RGB data, "estimated on over-exposed ISO100 frame".
pub const LIBRAW_MAXIMUM: f32 = 17536.0;

/// How the subsampled chroma planes are brought to full resolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ChromaUpsampling {
    /// Every luma sample takes the chroma of its group (LibRaw's behaviour).
    Replicate,
    /// Luma samples between two chroma samples take their rounded average, horizontally and then
    /// vertically (rawler's behaviour); chroma is co-sited with the left and upper luma sample.
    #[default]
    Linear,
}

/// Switches of the reduced path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct YcbcrOptions {
    /// Chroma upsampling method.
    pub chroma: ChromaUpsampling,
    /// Divide the camera's white balance out again (the default). Turn it off to get exactly
    /// what LibRaw's pseudo-raw loader returns.
    pub undo_white_balance: bool,
}

impl Default for YcbcrOptions {
    fn default() -> Self {
        Self {
            chroma: ChromaUpsampling::Linear,
            undo_white_balance: true,
        }
    }
}

/// A decoded Lossless M or S image.
#[derive(Clone, Debug)]
pub struct YcbcrImage {
    /// Interleaved RGB, three `u16` samples per pixel (see the module documentation).
    pub rgb: Plane<u16>,
    /// The chroma subsampling found in the tiles.
    pub subsampling: ChromaSubsampling,
    /// Pedestal per channel in the order red, green, green, blue.
    pub black: [f32; 4],
    /// Clip level per channel in the same order.
    pub white: [f32; 4],
    /// True when the white-balance gains were divided out.
    pub white_balance_undone: bool,
}

/// One decoded tile at its own resolution.
#[derive(Clone, Debug)]
pub struct TilePlanes {
    /// Luma samples per row (a multiple of the sampling factor).
    pub y_width: usize,
    /// Luma rows.
    pub y_height: usize,
    /// Luma.
    pub y: Vec<u16>,
    /// Chroma samples per row.
    pub c_width: usize,
    /// Chroma rows.
    pub c_height: usize,
    /// Cb.
    pub cb: Vec<u16>,
    /// Cr.
    pub cr: Vec<u16>,
    /// Horizontal luma sampling factor (chroma subsampling in x).
    pub h: usize,
    /// Vertical luma sampling factor (chroma subsampling in y).
    pub v: usize,
}

/// Decodes one tile stream of `tile_width` x `tile_length` pixels.
pub fn decode_tile(
    bytes: &[u8],
    tile_width: usize,
    tile_length: usize,
) -> Result<TilePlanes, RawError> {
    let options = LjpegOptions {
        max_samples: tile_width * tile_length * 3 + 64,
        sony_420_quirk: true,
    };
    let bad = |what: &str| RawError::corrupt(format!("Lossless M/S tile: {what}"));
    // Check the declared geometry before anything is allocated.
    let header = ljpeg::parse_header(bytes)?;
    if header.components.len() != 3 {
        return Err(bad("the stream does not have three components"));
    }
    if header.width != tile_width || header.height != tile_length {
        return Err(bad("the stream size is not the tile size"));
    }
    let (y_info, cb_info, cr_info) = (
        header.components[0],
        header.components[1],
        header.components[2],
    );
    let (h, v) = (usize::from(y_info.h), usize::from(y_info.v));
    if !matches!((h, v), (1, 1) | (2, 1) | (2, 2))
        || (cb_info.h, cb_info.v, cr_info.h, cr_info.v) != (1, 1, 1, 1)
    {
        return Err(bad("the sampling factors are not 4:4:4, 4:2:2 or 4:2:0"));
    }
    if !tile_width.is_multiple_of(h) || !tile_length.is_multiple_of(v) {
        return Err(bad("the tile size does not fit the chroma subsampling"));
    }
    let decoded = ljpeg::decode_scan(&header, bytes, &options)?;
    let mut planes = decoded.planes.into_iter();
    let (Some(y), Some(cb), Some(cr)) = (planes.next(), planes.next(), planes.next()) else {
        return Err(bad("the stream lost a component"));
    };
    Ok(TilePlanes {
        y_width: y.width,
        y_height: y.height,
        y: y.data,
        c_width: cb.width,
        c_height: cb.height,
        cb: cb.data,
        cr: cr.data,
        h,
        v,
    })
}

/// The red, green 1, green 2 and blue values of a four-value array in CFA raster order.
fn rggb_of(pattern: CfaPattern, positions: [f32; 4]) -> [f32; 4] {
    let mut out = [0.0; 4];
    let mut greens = 0;
    for (color, value) in pattern.0.iter().zip(positions) {
        match color {
            CfaColor::Red => out[0] = value,
            CfaColor::Green => {
                greens += 1;
                out[greens] = value;
            }
            CfaColor::Blue => out[3] = value,
        }
    }
    out
}

/// White-balance gains relative to green (red, green, blue) from the as-shot levels, or `None`
/// when they are missing or unusable.
fn gains_from(meta: &RawMetadata) -> Option<[f32; 3]> {
    let [r, g1, g2, b] = rggb_of(meta.cfa, meta.wb_as_shot?);
    let g = (g1 + g2) / 2.0;
    let gains = [r / g, 1.0, b / g];
    gains
        .iter()
        .all(|v| v.is_finite() && *v > 0.05 && *v < 50.0)
        .then_some(gains)
}

/// Decodes the Lossless M or S image described by `meta`.
pub fn decode(
    src: &dyn ByteSource,
    meta: &RawMetadata,
    options: &YcbcrOptions,
) -> Result<YcbcrImage, RawError> {
    let (width, height) = (meta.width as usize, meta.height as usize);
    check_dimensions(width, height)?;
    let (grid, offsets, counts) = ljpeg::tile_grid(meta)?;
    let (tw, tl) = (grid.tile_width, grid.tile_length);

    // The first tile tells the subsampling; every other tile must agree.
    let first_bytes = ljpeg::read_tile(src, offsets[0], counts[0], tile_limit(&grid))?;
    let first = decode_tile(&first_bytes, tw, tl)?;
    let (h, v) = (first.h, first.v);
    let subsampling = match (h, v) {
        (1, 1) => ChromaSubsampling::Yuv444,
        (2, 1) => ChromaSubsampling::Yuv422,
        _ => ChromaSubsampling::Yuv420,
    };
    let (cw, ch) = (width.div_ceil(h), height.div_ceil(v));
    let (ctw, ctl) = (tw / h, tl / v);

    let mut y_plane = vec![0u16; width * height];
    let mut cb_plane = vec![0u16; cw * ch];
    let mut cr_plane = vec![0u16; cw * ch];

    let first = std::sync::Mutex::new(Some(first));
    y_plane
        .par_chunks_mut(width * tl)
        .zip(cb_plane.par_chunks_mut(cw * ctl))
        .zip(cr_plane.par_chunks_mut(cw * ctl))
        .enumerate()
        .try_for_each(|(tile_row, ((yb, cbb), crb))| -> Result<(), RawError> {
            let tiles: Vec<TilePlanes> = (0..grid.across)
                .into_par_iter()
                .map(|tile_col| {
                    let k = tile_row * grid.across + tile_col;
                    if k == 0
                        && let Some(done) = first.lock().ok().and_then(|mut g| g.take())
                    {
                        return Ok(done);
                    }
                    let bytes = ljpeg::read_tile(src, offsets[k], counts[k], tile_limit(&grid))?;
                    let tile = decode_tile(&bytes, tw, tl)?;
                    if (tile.h, tile.v) != (h, v) {
                        return Err(RawError::corrupt(
                            "Lossless M/S tiles disagree about the chroma subsampling",
                        ));
                    }
                    Ok(tile)
                })
                .collect::<Result<_, _>>()?;
            for (tile_col, tile) in tiles.iter().enumerate() {
                let x0 = tile_col * tw;
                let copy = tw.min(width - x0);
                for row in 0..yb.len() / width {
                    yb[row * width + x0..row * width + x0 + copy]
                        .copy_from_slice(&tile.y[row * tile.y_width..row * tile.y_width + copy]);
                }
                let cx0 = tile_col * ctw;
                let ccopy = ctw.min(cw - cx0);
                for row in 0..cbb.len() / cw {
                    let from = row * tile.c_width..row * tile.c_width + ccopy;
                    cbb[row * cw + cx0..row * cw + cx0 + ccopy]
                        .copy_from_slice(&tile.cb[from.clone()]);
                    crb[row * cw + cx0..row * cw + cx0 + ccopy].copy_from_slice(&tile.cr[from]);
                }
            }
            Ok(())
        })?;

    // Levels and gains.
    let sensor_black = rggb_of(meta.cfa, meta.black);
    let pedestal = [
        2.0 * sensor_black[0],
        sensor_black[1].min(sensor_black[2]) * 2.0,
        2.0 * sensor_black[3],
    ];
    let gains = if options.undo_white_balance {
        gains_from(meta)
    } else {
        None
    };
    let gain = gains.unwrap_or([1.0; 3]);
    let range = LIBRAW_MAXIMUM - LIBRAW_BLACK;
    let channel_white = |c: usize| pedestal[c] + range / gain[c];
    let black = [pedestal[0], pedestal[1], pedestal[1], pedestal[2]];
    let white = [
        channel_white(0),
        channel_white(1),
        channel_white(1),
        channel_white(2),
    ];

    // Upsample, convert and (optionally) undo the white balance, row by row.
    // The size limit counts pixels, not samples: a 120 MP image has three samples per pixel here.
    let mut rgb = Plane::from_vec(width * 3, height, vec![0u16; width * 3 * height])
        .ok_or_else(|| RawError::corrupt("internal size mismatch"))?;
    let geometry = Geometry {
        cw,
        ch,
        h,
        v,
        mode: options.chroma,
    };
    let undo = gains.is_some();
    rgb.data
        .par_chunks_mut(width * 3)
        .enumerate()
        .for_each_init(
            || (vec![0i32; width], vec![0i32; width], vec![0i32; width]),
            |(cb_row, cr_row, scratch), (y, out)| {
                geometry.chroma_row(&cb_plane, y, cb_row, scratch);
                geometry.chroma_row(&cr_plane, y, cr_row, scratch);
                let luma = &y_plane[y * width..(y + 1) * width];
                let pixels = out
                    .chunks_exact_mut(3)
                    .zip(luma)
                    .zip(cb_row.iter().zip(cr_row.iter()));
                if undo {
                    for ((px, &l), (&cb, &cr)) in pixels {
                        let balanced = ycc_to_rgb(l, cb, cr);
                        for ((slot, value), (&black, &gain)) in
                            px.iter_mut().zip(balanced).zip(pedestal.iter().zip(&gain))
                        {
                            let raw = (f32::from(value) - black) / gain + black;
                            *slot = raw.round().clamp(0.0, 65535.0) as u16;
                        }
                    }
                } else {
                    for ((px, &l), (&cb, &cr)) in pixels {
                        px.copy_from_slice(&ycc_to_rgb(l, cb, cr));
                    }
                }
            },
        );

    Ok(YcbcrImage {
        rgb,
        subsampling,
        black,
        white,
        white_balance_undone: gains.is_some(),
    })
}

fn tile_limit(grid: &ljpeg::TileGrid) -> u64 {
    (grid.tile_width * grid.tile_length * 3 * 4) as u64 + (1 << 16)
}

/// Full-frame chroma geometry.
struct Geometry {
    cw: usize,
    ch: usize,
    h: usize,
    v: usize,
    mode: ChromaUpsampling,
}

impl Geometry {
    /// Expands one row of a chroma plane to the full width, as raw sample values.
    fn horizontal(&self, row: &[u16], out: &mut [i32]) {
        if self.h == 1 {
            for (slot, &c) in out.iter_mut().zip(row) {
                *slot = i32::from(c);
            }
        } else if self.mode == ChromaUpsampling::Replicate {
            for (pair, &c) in out.chunks_mut(self.h).zip(row) {
                pair.fill(i32::from(c));
            }
        } else {
            // Co-sited: the even luma sample takes the chroma sample, the odd one the rounded
            // average of it and the next (the last one repeats).
            for (cx, pair) in out.chunks_mut(self.h).enumerate() {
                let a = i32::from(row[cx]);
                pair[0] = a;
                if let Some(second) = pair.get_mut(1) {
                    let b = i32::from(row[(cx + 1).min(self.cw - 1)]);
                    *second = (a + b + 1) / 2;
                }
            }
        }
    }

    /// One chroma plane sampled at luma row `y`, at full width, as signed offsets from the centre.
    /// `scratch` is a full-width buffer for the second row of a vertical interpolation.
    fn chroma_row(&self, plane: &[u16], y: usize, out: &mut [i32], scratch: &mut [i32]) {
        let cy = y / self.v;
        self.horizontal(&plane[cy * self.cw..(cy + 1) * self.cw], out);
        if self.v > 1 && self.mode == ChromaUpsampling::Linear && !y.is_multiple_of(self.v) {
            let cy2 = (cy + 1).min(self.ch - 1);
            self.horizontal(&plane[cy2 * self.cw..(cy2 + 1) * self.cw], scratch);
            for (slot, &second) in out.iter_mut().zip(scratch.iter()) {
                *slot = (*slot + second + 1) / 2;
            }
        }
        for slot in out.iter_mut() {
            *slot -= CHROMA_CENTER;
        }
    }
}

/// Full-range BT.601 YCbCr to RGB in single precision with truncation, exactly as LibRaw's
/// `ycc2rgb`. `cb` and `cr` are signed offsets from [`CHROMA_CENTER`].
#[inline(always)]
pub fn ycc_to_rgb(y: u16, cb: i32, cr: i32) -> [u16; 3] {
    let (y, cb, cr) = (f32::from(y), cb as f32, cr as f32);
    let r = y + 1.402_f32 * cr;
    let g = y - 0.344_14_f32 * cb - 0.714_14_f32 * cr;
    let b = y + 1.772_f32 * cb;
    let clip = |q: f32| q.clamp(0.0, 65535.0) as u32 as u16;
    [clip(r), clip(g), clip(b)]
}

/// Inverse of [`ycc_to_rgb`] for the test encoder: RGB to YCbCr samples (Cb and Cr include the
/// 16383 offset), rounded to integers and clamped to 16 bits.
#[cfg(any(test, feature = "testgen"))]
pub fn rgb_to_ycc(rgb: [u16; 3]) -> [u16; 3] {
    let (r, g, b) = (f64::from(rgb[0]), f64::from(rgb[1]), f64::from(rgb[2]));
    let y = 0.299 * r + 0.587 * g + 0.114 * b;
    let cb = (b - y) / 1.772;
    let cr = (r - y) / 1.402;
    let clip = |q: f64| q.round().clamp(0.0, 65535.0) as u16;
    [
        clip(y),
        clip(cb + f64::from(CHROMA_CENTER)),
        clip(cr + f64::from(CHROMA_CENTER)),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grey_stays_grey_and_primaries_come_back_within_a_step() {
        for v in [0u16, 1000, 16000, 30000] {
            let ycc = rgb_to_ycc([v, v, v]);
            assert_eq!(ycc[1], CHROMA_CENTER as u16);
            assert_eq!(ycc[2], CHROMA_CENTER as u16);
            let back = ycc_to_rgb(
                ycc[0],
                i32::from(ycc[1]) - CHROMA_CENTER,
                i32::from(ycc[2]) - CHROMA_CENTER,
            );
            assert!(back.iter().all(|c| c.abs_diff(v) <= 1), "{back:?} vs {v}");
        }
        let ycc = rgb_to_ycc([20000, 4000, 9000]);
        let back = ycc_to_rgb(
            ycc[0],
            i32::from(ycc[1]) - CHROMA_CENTER,
            i32::from(ycc[2]) - CHROMA_CENTER,
        );
        for (a, b) in back.iter().zip([20000u16, 4000, 9000]) {
            assert!(a.abs_diff(b) <= 2, "{back:?}");
        }
    }

    #[test]
    fn rggb_order_follows_the_pattern() {
        let v = [1.0, 2.0, 3.0, 4.0]; // raster order of the pattern
        assert_eq!(rggb_of(CfaPattern::RGGB, v), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(rggb_of(CfaPattern::BGGR, v), [4.0, 2.0, 3.0, 1.0]);
        assert_eq!(rggb_of(CfaPattern::GRBG, v), [2.0, 1.0, 4.0, 3.0]);
    }
}
