//! Test-only Lossless L and Lossless M/S payloads.
//!
//! * [`LosslessL`] cuts the mosaic into tiles (512 x 512 by default), turns each tile into a
//!   lossless JPEG with four components at half the tile width and height, the way the Sony
//!   bodies do, and lets the test choose the predictor, restart interval and Huffman tables.
//! * [`LosslessMs`] starts from an unbalanced camera RGB image, applies white-balance gains,
//!   converts to YCbCr, subsamples the chroma and codes 3-component tiles, the inverse of
//!   `decode::ycbcr`.

use super::ljpeg_enc::{EncodeParams, HuffmanChoice, encode};
use super::{PayloadDescription, RawPayload};
use crate::decode::ycbcr::rgb_to_ycc;
use crate::layout::ChromaSubsampling;

/// Lossless L: tiled four-component lossless JPEG.
#[derive(Clone, Debug)]
pub struct LosslessL {
    /// Predictor 1 to 7.
    pub predictor: u8,
    /// Restart interval in MCU rows (0 = none).
    pub restart_rows: usize,
    /// Tile width and height in samples.
    pub tile: (u32, u32),
    /// Huffman table choice.
    pub huffman: HuffmanChoice,
    /// Sample precision in the stream.
    pub precision: u8,
}

impl Default for LosslessL {
    fn default() -> Self {
        Self {
            predictor: 1,
            restart_rows: 0,
            tile: (512, 512),
            huffman: HuffmanChoice::Optimal,
            precision: 14,
        }
    }
}

impl LosslessL {
    /// 512 x 512 tiles, predictor 1, no restarts, optimal Huffman tables, 14 bits.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the predictor.
    pub fn predictor(mut self, predictor: u8) -> Self {
        self.predictor = predictor;
        self
    }

    /// Sets the restart interval in MCU rows.
    pub fn restart_rows(mut self, rows: usize) -> Self {
        self.restart_rows = rows;
        self
    }

    /// Sets the tile size.
    pub fn tile(mut self, width: u32, height: u32) -> Self {
        self.tile = (width, height);
        self
    }

    /// Sets the Huffman table choice.
    pub fn huffman(mut self, huffman: HuffmanChoice) -> Self {
        self.huffman = huffman;
        self
    }
}

impl RawPayload for LosslessL {
    fn describe(&self) -> PayloadDescription {
        PayloadDescription {
            compression: 7,
            bits_per_sample: u16::from(self.precision),
            photometric: 32803,
            samples_per_pixel: 1,
            tile: Some(self.tile),
            sony_raw_file_type: Some(3),
            ycbcr_subsampling: None,
        }
    }

    fn encode(&self, mosaic: &[u16], width: u32, height: u32) -> Vec<Vec<u8>> {
        let (w, h) = (width as usize, height as usize);
        let (tw, tl) = (self.tile.0 as usize, self.tile.1 as usize);
        assert!(tw % 2 == 0 && tl % 2 == 0);
        let (hw, hl) = (tw / 2, tl / 2);
        let params = EncodeParams {
            precision: self.precision,
            predictor: self.predictor,
            point_transform: 0,
            restart_interval: self.restart_rows * hw,
            sony_420_quirk: false,
            huffman: self.huffman,
            width: hw,
            height: hl,
            sampling: vec![(1, 1); 4],
        };
        let mut tiles = Vec::new();
        for tile_row in 0..h.div_ceil(tl) {
            for tile_col in 0..w.div_ceil(tw) {
                let mut planes = vec![vec![0u16; hw * hl]; 4];
                for j in 0..hl {
                    for i in 0..hw {
                        for (k, plane) in planes.iter_mut().enumerate() {
                            let y = (tile_row * tl + 2 * j + (k >> 1)).min(h - 1);
                            let x = (tile_col * tw + 2 * i + (k & 1)).min(w - 1);
                            plane[j * hw + i] = mosaic[y * w + x];
                        }
                    }
                }
                tiles.push(encode(&planes, &params));
            }
        }
        tiles
    }
}

/// Lossless M or S: tiled three-component YCbCr lossless JPEG made from camera RGB.
#[derive(Clone, Debug)]
pub struct LosslessMs {
    /// Chroma subsampling.
    pub subsampling: ChromaSubsampling,
    /// Unbalanced camera RGB, three `u16` per pixel, row-major.
    pub rgb: Vec<u16>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// White-balance gains relative to green (red, green = 1, blue) the "camera" applies.
    pub gains: [f32; 3],
    /// The sensor black level; the pedestal of the RGB data is twice this.
    pub sensor_black: f32,
    /// Tile width and height.
    pub tile: (u32, u32),
    /// Huffman table choice.
    pub huffman: HuffmanChoice,
    /// Chroma siting: `false` (the default) stores the average of each subsampled group, the
    /// chroma sample then sits between its luma samples; `true` stores the top-left sample of the
    /// group, which is the siting the decoder's linear upsampling assumes (co-sited).
    pub co_sited: bool,
}

impl LosslessMs {
    /// A payload for `rgb` (three samples per pixel).
    pub fn new(
        subsampling: ChromaSubsampling,
        rgb: Vec<u16>,
        width: u32,
        height: u32,
        gains: [f32; 3],
    ) -> Self {
        assert_eq!(rgb.len(), width as usize * height as usize * 3);
        Self {
            subsampling,
            rgb,
            width,
            height,
            gains,
            sensor_black: 512.0,
            tile: (512, 512),
            huffman: HuffmanChoice::Optimal,
            co_sited: false,
        }
    }

    /// Stores co-sited chroma (the top-left sample of each group) instead of group averages.
    pub fn co_sited(mut self, yes: bool) -> Self {
        self.co_sited = yes;
        self
    }

    /// Sets the tile size.
    pub fn tile(mut self, width: u32, height: u32) -> Self {
        self.tile = (width, height);
        self
    }

    /// The SR2 white-balance levels (red, green, green, blue; green 1024) that match `gains`.
    pub fn sr2_wb(&self) -> [u16; 4] {
        [
            (self.gains[0] * 1024.0).round() as u16,
            1024,
            1024,
            (self.gains[2] * 1024.0).round() as u16,
        ]
    }

    fn factors(&self) -> (usize, usize) {
        match self.subsampling {
            ChromaSubsampling::Yuv444 => (1, 1),
            ChromaSubsampling::Yuv422 => (2, 1),
            ChromaSubsampling::Yuv420 => (2, 2),
        }
    }

    /// Full-resolution luma, and chroma averaged over each group, as the camera would store them.
    pub fn planes(&self) -> (Vec<u16>, Vec<u16>, Vec<u16>, usize, usize) {
        let (w, h) = (self.width as usize, self.height as usize);
        let (fh, fv) = self.factors();
        let pedestal = 2.0 * self.sensor_black;
        let mut y = vec![0u16; w * h];
        let mut cb_full = vec![0u16; w * h];
        let mut cr_full = vec![0u16; w * h];
        for i in 0..w * h {
            let mut balanced = [0u16; 3];
            for (c, slot) in balanced.iter_mut().enumerate() {
                let v = (f32::from(self.rgb[3 * i + c]) - pedestal) * self.gains[c] + pedestal;
                *slot = v.round().clamp(0.0, 65535.0) as u16;
            }
            let ycc = rgb_to_ycc(balanced);
            y[i] = ycc[0];
            cb_full[i] = ycc[1];
            cr_full[i] = ycc[2];
        }
        let (cw, ch) = (w.div_ceil(fh), h.div_ceil(fv));
        let mut cb = vec![0u16; cw * ch];
        let mut cr = vec![0u16; cw * ch];
        for cy in 0..ch {
            for cx in 0..cw {
                if self.co_sited {
                    let at = cy * fv * w + cx * fh;
                    cb[cy * cw + cx] = cb_full[at];
                    cr[cy * cw + cx] = cr_full[at];
                    continue;
                }
                let (mut sb, mut sr, mut n) = (0u32, 0u32, 0u32);
                for dy in 0..fv {
                    for dx in 0..fh {
                        let (x, yy) = (cx * fh + dx, cy * fv + dy);
                        if x < w && yy < h {
                            sb += u32::from(cb_full[yy * w + x]);
                            sr += u32::from(cr_full[yy * w + x]);
                            n += 1;
                        }
                    }
                }
                cb[cy * cw + cx] = ((sb + n / 2) / n) as u16;
                cr[cy * cw + cx] = ((sr + n / 2) / n) as u16;
            }
        }
        (y, cb, cr, cw, ch)
    }
}

impl RawPayload for LosslessMs {
    fn describe(&self) -> PayloadDescription {
        let (fh, fv) = self.factors();
        PayloadDescription {
            compression: 7,
            bits_per_sample: 14,
            photometric: 6,
            samples_per_pixel: 3,
            tile: Some(self.tile),
            sony_raw_file_type: Some(3),
            ycbcr_subsampling: Some((fh as u16, fv as u16)),
        }
    }

    fn encode(&self, _mosaic: &[u16], width: u32, height: u32) -> Vec<Vec<u8>> {
        assert_eq!((width, height), (self.width, self.height));
        let (w, h) = (width as usize, height as usize);
        let (tw, tl) = (self.tile.0 as usize, self.tile.1 as usize);
        let (fh, fv) = self.factors();
        let (y, cb, cr, cw, ch) = self.planes();
        let (ctw, ctl) = (tw / fh, tl / fv);
        let params = EncodeParams {
            precision: 16,
            predictor: 1,
            point_transform: 0,
            restart_interval: 0,
            sony_420_quirk: (fh, fv) == (2, 2),
            huffman: self.huffman,
            width: tw,
            height: tl,
            sampling: vec![(fh as u8, fv as u8), (1, 1), (1, 1)],
        };
        let mut tiles = Vec::new();
        for tile_row in 0..h.div_ceil(tl) {
            for tile_col in 0..w.div_ceil(tw) {
                let mut yt = vec![0u16; tw * tl];
                for j in 0..tl {
                    for i in 0..tw {
                        let yy = (tile_row * tl + j).min(h - 1);
                        let x = (tile_col * tw + i).min(w - 1);
                        yt[j * tw + i] = y[yy * w + x];
                    }
                }
                let mut cbt = vec![0u16; ctw * ctl];
                let mut crt = vec![0u16; ctw * ctl];
                for j in 0..ctl {
                    for i in 0..ctw {
                        let yy = (tile_row * ctl + j).min(ch - 1);
                        let x = (tile_col * ctw + i).min(cw - 1);
                        cbt[j * ctw + i] = cb[yy * cw + x];
                        crt[j * ctw + i] = cr[yy * cw + x];
                    }
                }
                tiles.push(encode(&[yt, cbt, crt], &params));
            }
        }
        tiles
    }
}
