//! BlazeFace, full-range model: the face finder (`docs/27` S-08, `docs/07`).
//!
//! The network is Google's `blaze_face_full_range.tflite` (Apache-2.0, model card of 19 January 2021), converted to f16
//! safetensors by `tools/mediapipe/convert.py` and run by [`crate::mediapipe::graph`]. Everything around it is our own code,
//! written from MediaPipe's documented behaviour (the graph `face_detection_full_range_common.pbtxt` and the model card):
//!
//! * **Input** (`prepare_tensor`): the picture, or a window of it, is letterboxed into 192 x 192 (aspect kept, centred, the
//!   bars black) and normalised to `[-1, 1]` (`x * 2 - 1`; a black bar is -1). [`Resample::MediaPipe`] samples bilinearly at
//!   `(dst - offset) / scale`, which is what MediaPipe's `ImageToTensor` does (OpenCV `warpAffine`, no filtering);
//!   [`Resample::Box`] does the same on a picture first reduced by an integer box filter until the rest of the reduction is
//!   at most 2 times, and is what [`FaceFinder`] uses by default, because a 24 MP photo reduced 30 times by plain bilinear
//!   sampling is aliased (and a picture reduced less than 2 times is sampled exactly as MediaPipe does).
//! * **Anchors**: 48 x 48 centres at stride 4, one anchor per cell, fixed size 1 x 1, offset 0.5 (2304 anchors).
//! * **Decoding**: 16 regressors per anchor, `x_center, y_center, width, height` and six keypoints (right eye, left eye,
//!   nose tip, mouth centre, right ear tragion, left ear tragion) as `(x, y)` pairs, each `/ 192` and offset by the anchor
//!   centre; the score is the sigmoid of the logit clipped to +-100.
//! * **Weighted NMS** ([`weighted_nms`]): the detections are sorted by score; the best one and every one overlapping it
//!   (IoU above the threshold) are replaced by their score-weighted average, which keeps the best one's score.
//! * **Tiling** ([`FaceFinder`]): the network sees faces between about 7 % and 50 % of its frame, so a large photo is run
//!   whole and on overlapping windows at several sizes (each four times the previous one, windows stepping by two thirds of
//!   their size so a face of a third of the window fits whole in one), the detections are merged with the same NMS, small
//!   ones are dropped, and at most [`FinderParams::max_faces`] are kept.
//!
//! # Determinism
//!
//! The same picture and parameters give the same faces on any number of threads: the windows are run in a fixed order
//! (in parallel, collected in order), the network is bit-stable ([`crate::mediapipe::graph`]) and the post-processing is
//! sequential.

use std::path::Path;

use rayon::prelude::*;

use crate::mediapipe::graph::{Graph, Tensor};
use crate::mediapipe::{expected_sha256, verify};
use crate::model::{MlError, Result};

/// The weights file in the model folder (`data/models/mediapipe/`).
pub const WEIGHTS_FILE: &str = "blaze_face_full_range.f16.safetensors";
/// The side of the network input, in pixels.
pub const INPUT_PX: usize = 192;
/// The number of anchors (48 x 48, one per cell).
pub const ANCHOR_COUNT: usize = GRID * GRID;
/// The number of regressors per anchor: a box and six keypoints.
pub const REGRESSORS: usize = 16;
/// The number of keypoints of a face.
pub const KEYPOINTS: usize = 6;
/// The names of the keypoints, in the order of [`Detection::keypoints`].
pub const KEYPOINT_NAMES: [&str; KEYPOINTS] = [
    "right eye",
    "left eye",
    "nose tip",
    "mouth centre",
    "right ear tragion",
    "left ear tragion",
];

const GRID: usize = 48;
/// The smallest face the network finds reliably, as a fraction of its frame (measured in S-08: 10 px of 192 start to be found,
/// 12 px are found with a score above 0.85).
const MIN_FRAME_FRACTION: f64 = 0.07;
/// The step of the windows, as a fraction of their size.
const WINDOW_STEP: f64 = 2.0 / 3.0;
/// Each window size is this many times the previous one (a face range of about 4.7 per size).
const WINDOW_GROWTH: f64 = 4.0;
/// Detections nearer than this fraction of the window to an inner window border are dropped (the face is cut).
const BORDER_MARGIN: f64 = 0.02;

/// Where the network's pixels come from. Values are in `0..=1`, display-referred (sRGB-like), as the network was trained.
pub trait PixelSource: Sync {
    /// Width in pixels.
    fn width(&self) -> usize;
    /// Height in pixels.
    fn height(&self) -> usize;
    /// The red, green and blue of the pixel at column `x`, row `y` (both inside the picture).
    fn rgb(&self, x: usize, y: usize) -> [f32; 3];
}

/// An 8-bit RGB picture, row-major, three bytes per pixel.
#[derive(Debug, Clone, Copy)]
pub struct Rgb8<'a> {
    width: usize,
    height: usize,
    data: &'a [u8],
}

impl<'a> Rgb8<'a> {
    /// Wraps `data` (`width * height * 3` bytes).
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when the size does not match or the picture is empty.
    pub fn new(width: usize, height: usize, data: &'a [u8]) -> Result<Self> {
        if width == 0 || height == 0 || data.len() != width * height * 3 {
            return Err(MlError::Shape(format!(
                "{} bytes for a {width} x {height} RGB picture",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

impl PixelSource for Rgb8<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn rgb(&self, x: usize, y: usize) -> [f32; 3] {
        let p = &self.data[(y * self.width + x) * 3..][..3];
        [
            f32::from(p[0]) / 255.0,
            f32::from(p[1]) / 255.0,
            f32::from(p[2]) / 255.0,
        ]
    }
}

/// A float RGB picture in `0..=1`, row-major, three values per pixel.
#[derive(Debug, Clone, Copy)]
pub struct RgbF32<'a> {
    width: usize,
    height: usize,
    data: &'a [f32],
}

impl<'a> RgbF32<'a> {
    /// Wraps `data` (`width * height * 3` values).
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when the size does not match or the picture is empty.
    pub fn new(width: usize, height: usize, data: &'a [f32]) -> Result<Self> {
        if width == 0 || height == 0 || data.len() != width * height * 3 {
            return Err(MlError::Shape(format!(
                "{} values for a {width} x {height} RGB picture",
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }
}

impl PixelSource for RgbF32<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn rgb(&self, x: usize, y: usize) -> [f32; 3] {
        let p = &self.data[(y * self.width + x) * 3..][..3];
        [p[0], p[1], p[2]]
    }
}

/// How a window of the picture is reduced to the network's 192 x 192.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Resample {
    /// MediaPipe's own sampling: bilinear at `(dst - offset) / scale`, no filtering, black outside the picture.
    MediaPipe,
    /// MediaPipe's sampling on the picture reduced by a box filter of `k` x `k` pixels, `k = floor(0.5 / scale)` (1, that
    /// is [`Resample::MediaPipe`] itself, unless the window is reduced more than 2 times): the rest of the reduction is
    /// then 1/4 to 1/2, close to what the network was trained on, and without the aliasing of sampling one pixel in 30.
    #[default]
    Box,
}

/// A rectangle of the picture, in pixels (may be fractional).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    /// Left edge.
    pub x_px: f64,
    /// Top edge.
    pub y_px: f64,
    /// Width.
    pub width_px: f64,
    /// Height.
    pub height_px: f64,
}

impl Window {
    /// The whole of a `width` x `height` picture.
    pub fn whole(width: usize, height: usize) -> Self {
        Self {
            x_px: 0.0,
            y_px: 0.0,
            width_px: width as f64,
            height_px: height as f64,
        }
    }
}

/// The placement of a window in the network input.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Letterbox {
    window: Window,
    /// Network pixels per picture pixel.
    scale: f64,
    /// Left and top offset of the content in the network input.
    offset_x: f64,
    offset_y: f64,
}

impl Letterbox {
    /// Fits `window` into the network input: aspect kept, centred.
    ///
    /// # Errors
    ///
    /// [`MlError::Input`] when the window is empty or not finite.
    pub fn new(window: Window) -> Result<Self> {
        let ok = window.width_px.is_finite()
            && window.height_px.is_finite()
            && window.x_px.is_finite()
            && window.y_px.is_finite()
            && window.width_px > 0.0
            && window.height_px > 0.0;
        if !ok {
            return Err(MlError::Input(format!("window {window:?}")));
        }
        let side = INPUT_PX as f64;
        let scale = (side / window.width_px).min(side / window.height_px);
        Ok(Self {
            window,
            scale,
            offset_x: (side - window.width_px * scale) / 2.0,
            offset_y: (side - window.height_px * scale) / 2.0,
        })
    }

    /// Network pixels per picture pixel.
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// A position of the network input (pixels) in the picture.
    fn to_picture(&self, nx: f64, ny: f64) -> (f64, f64) {
        (
            self.window.x_px + (nx - self.offset_x) / self.scale,
            self.window.y_px + (ny - self.offset_y) / self.scale,
        )
    }
}

/// The network input for a window: `[192][192][3]`, `x * 2 - 1`, black bars at -1.
pub fn prepare_tensor(src: &dyn PixelSource, lb: &Letterbox, resample: Resample) -> Vec<f32> {
    let k = match resample {
        Resample::MediaPipe => 1,
        Resample::Box => ((0.5 / lb.scale).floor() as usize).max(1),
    };
    let mut out = vec![0.0f32; INPUT_PX * INPUT_PX * 3];
    out.par_chunks_mut(INPUT_PX * 3)
        .enumerate()
        .for_each(|(dy, row)| {
            for (dx, px) in row.chunks_exact_mut(3).enumerate() {
                for (o, v) in px.iter_mut().zip(sample_bilinear(src, lb, k, dx, dy)) {
                    *o = v * 2.0 - 1.0;
                }
            }
        });
    out
}

/// OpenCV's `warpAffine` with linear interpolation and a constant black border, for the map `src = (dst - offset) / scale`;
/// with `k` > 1 each of the four taps is the mean of the `k` x `k` pixels around it (the ones inside the picture).
fn sample_bilinear(src: &dyn PixelSource, lb: &Letterbox, k: usize, dx: usize, dy: usize) -> [f32; 3] {
    let (sx, sy) = lb.to_picture(dx as f64, dy as f64);
    let w = lb.window;
    // Black more than one picture pixel outside the window: the letterbox bars of a window that is not the whole picture.
    if sx <= w.x_px - 1.0 || sx >= w.x_px + w.width_px || sy <= w.y_px - 1.0 || sy >= w.y_px + w.height_px {
        return [0.0; 3];
    }
    let (fx, fy) = (sx.floor(), sy.floor());
    let (tx, ty) = ((sx - fx) as f32, (sy - fy) as f32);
    let (x0, y0) = (fx as i64, fy as i64);
    let (pw, ph) = (src.width() as i64, src.height() as i64);
    let before = ((k - 1) / 2) as i64;
    let tap = |x: i64, y: i64| -> [f32; 3] {
        if x < 0 || y < 0 || x >= pw || y >= ph {
            return [0.0; 3];
        }
        if k == 1 {
            return src.rgb(x as usize, y as usize);
        }
        let (xa, xb) = ((x - before).max(0), (x - before + k as i64).min(pw));
        let (ya, yb) = ((y - before).max(0), (y - before + k as i64).min(ph));
        let mut acc = [0.0f32; 3];
        for yy in ya..yb {
            let mut row = [0.0f32; 3];
            for xx in xa..xb {
                let p = src.rgb(xx as usize, yy as usize);
                row[0] += p[0];
                row[1] += p[1];
                row[2] += p[2];
            }
            for c in 0..3 {
                acc[c] += row[c];
            }
        }
        let n = ((xb - xa) * (yb - ya)) as f32;
        [acc[0] / n, acc[1] / n, acc[2] / n]
    };
    let (a, b, c, d) = (tap(x0, y0), tap(x0 + 1, y0), tap(x0, y0 + 1), tap(x0 + 1, y0 + 1));
    let mut out = [0.0f32; 3];
    for i in 0..3 {
        let top = a[i] * (1.0 - tx) + b[i] * tx;
        let bottom = c[i] * (1.0 - tx) + d[i] * tx;
        out[i] = top * (1.0 - ty) + bottom * ty;
    }
    out
}

/// One detected face before it is turned into a [`DetectedFace`]: a box and keypoints in picture pixels and a score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Detection {
    /// Left, top, right and bottom edges, in picture pixels.
    pub box_px: [f32; 4],
    /// The keypoints `(x, y)` in picture pixels, in the order of [`KEYPOINT_NAMES`].
    pub keypoints_px: [[f32; 2]; KEYPOINTS],
    /// The network's confidence, 0 to 1.
    pub score: f32,
}

impl Detection {
    fn area(&self) -> f32 {
        let [x0, y0, x1, y1] = self.box_px;
        (x1 - x0).max(0.0) * (y1 - y0).max(0.0)
    }

    /// The intersection over union with another detection's box.
    pub fn iou(&self, other: &Detection) -> f32 {
        let (a, b) = (self.box_px, other.box_px);
        let iw = (a[2].min(b[2]) - a[0].max(b[0])).max(0.0);
        let ih = (a[3].min(b[3]) - a[1].max(b[1])).max(0.0);
        let inter = iw * ih;
        let union = self.area() + other.area() - inter;
        if union > 0.0 { inter / union } else { 0.0 }
    }

    /// The longer side of the box, in pixels.
    pub fn size_px(&self) -> f32 {
        (self.box_px[2] - self.box_px[0]).max(self.box_px[3] - self.box_px[1])
    }
}

/// Weighted non-maximum suppression: the detections sorted by score (ties by position); each time the best remaining one and
/// all the remaining ones whose IoU with it is above `iou_threshold` are merged into one, whose box and keypoints are the
/// score-weighted average of the group's and whose score is the best one's.
pub fn weighted_nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then(a.box_px[1].total_cmp(&b.box_px[1]))
            .then(a.box_px[0].total_cmp(&b.box_px[0]))
    });
    let mut out = Vec::new();
    let mut rest = detections;
    while !rest.is_empty() {
        let best = rest[0];
        let (group, remaining): (Vec<Detection>, Vec<Detection>) = rest
            .into_iter()
            .partition(|d| d.iou(&best) > iou_threshold || *d == best);
        rest = remaining;
        let total: f32 = group.iter().map(|d| d.score).sum();
        let mut merged = best;
        if group.len() > 1 && total > 0.0 {
            merged.box_px = [0.0; 4];
            merged.keypoints_px = [[0.0; 2]; KEYPOINTS];
            for d in &group {
                let w = d.score / total;
                for (m, v) in merged.box_px.iter_mut().zip(d.box_px) {
                    *m += v * w;
                }
                for (m, v) in merged.keypoints_px.iter_mut().zip(d.keypoints_px) {
                    m[0] += v[0] * w;
                    m[1] += v[1] * w;
                }
            }
        }
        out.push(merged);
    }
    out
}

/// The anchor centres, normalised to the network input: `((x + 0.5) / 48, (y + 0.5) / 48)` in row-major cell order.
pub fn anchors() -> Vec<[f32; 2]> {
    let step = 1.0 / GRID as f32;
    (0..ANCHOR_COUNT)
        .map(|i| {
            [
                ((i % GRID) as f32 + 0.5) * step,
                ((i / GRID) as f32 + 0.5) * step,
            ]
        })
        .collect()
}

/// The network's raw answer to one input.
#[derive(Debug, Clone, PartialEq)]
pub struct RawOutput {
    /// `[anchor][16]`: `x, y, width, height`, then six keypoints `(x, y)`, all in network pixels relative to the anchor.
    pub regressors: Vec<f32>,
    /// One logit per anchor.
    pub logits: Vec<f32>,
}

/// The probability of a logit, clipped to +-100 as MediaPipe does.
pub fn score_of(logit: f32) -> f32 {
    let l = logit.clamp(-100.0, 100.0);
    1.0 / (1.0 + libm::expf(-l))
}

/// The detector with its weights.
#[derive(Debug)]
pub struct BlazeFace {
    graph: Graph,
    anchors: Vec<[f32; 2]>,
    regressor_output: usize,
    sha256: Option<String>,
}

impl BlazeFace {
    /// Loads `dir/blaze_face_full_range.f16.safetensors`, checked against its line in `dir/SHA256SUMS` (the model folder
    /// `data/models/mediapipe/`).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when a file cannot be read, [`MlError::Checksum`] when the weights do not match `SHA256SUMS`, and the
    /// errors of [`BlazeFace::from_bytes`].
    pub fn load(dir: &Path) -> Result<Self> {
        let bytes = std::fs::read(dir.join(WEIGHTS_FILE))?;
        let expected = expected_sha256(dir, WEIGHTS_FILE)?;
        Self::from_bytes(&bytes, Some(&expected))
    }

    /// Loads the model of a data folder: `<data_root>/models/mediapipe/` ([`BlazeFace::load`]).
    ///
    /// # Errors
    ///
    /// As [`BlazeFace::load`].
    pub fn load_bundled(data_root: &Path) -> Result<Self> {
        Self::load(&data_root.join("models").join("mediapipe"))
    }

    /// Loads a weights file without a checksum (development and tests).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when the file cannot be read, and the errors of [`BlazeFace::from_bytes`].
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, None)
    }

    /// Builds the detector from the bytes of the converted weights, checked against `expected_sha256` when given.
    ///
    /// # Errors
    ///
    /// [`MlError::Checksum`] on a digest mismatch, [`MlError::Weights`] or [`MlError::Unsupported`] when the file is not the
    /// BlazeFace full-range graph (one 1 x 192 x 192 x 3 input; outputs of 2304 x 16 and 2304 x 1).
    pub fn from_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            verify("BlazeFace full-range weights", bytes, want)?;
        }
        let graph = Graph::from_bytes(bytes)?;
        let input_ok = graph.inputs().len() == 1 && graph.inputs()[0].shape == [1, INPUT_PX, INPUT_PX, 3];
        let regressor_output = graph
            .outputs()
            .iter()
            .position(|o| o.shape == [1, ANCHOR_COUNT, REGRESSORS]);
        let logit_output = graph
            .outputs()
            .iter()
            .position(|o| o.shape == [1, ANCHOR_COUNT, 1]);
        let (true, Some(regressor_output), Some(_)) = (input_ok, regressor_output, logit_output) else {
            return Err(MlError::Weights(
                "the graph is not BlazeFace full range (input 1x192x192x3, outputs 1x2304x16 and 1x2304x1)".into(),
            ));
        };
        Ok(Self {
            graph,
            anchors: anchors(),
            regressor_output,
            sha256: expected_sha256.map(str::to_owned),
        })
    }

    /// The SHA-256 the weights were checked against, when they were.
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }

    /// Runs the network on a prepared `[192][192][3]` input ([`prepare_tensor`]).
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] for an input of another size, [`MlError::Input`] for one holding a value that is not finite.
    pub fn forward(&self, input: &[f32]) -> Result<RawOutput> {
        let t = Tensor::new(vec![1, INPUT_PX, INPUT_PX, 3], input.to_vec())?;
        let mut outputs = self.graph.run(&[t])?;
        let (regressors, logits) = if self.regressor_output == 0 {
            let l = outputs.remove(1);
            (outputs.remove(0), l)
        } else {
            let r = outputs.remove(1);
            (r, outputs.remove(0))
        };
        Ok(RawOutput {
            regressors: regressors.data,
            logits: logits.data,
        })
    }

    /// Decodes a raw answer into detections in picture pixels, above `score_threshold`, before any suppression.
    pub fn decode(&self, raw: &RawOutput, lb: &Letterbox, score_threshold: f32) -> Vec<Detection> {
        let side = INPUT_PX as f32;
        let to_x = |n: f32| lb.to_picture(f64::from(n * side), 0.0).0 as f32;
        let to_y = |n: f32| lb.to_picture(0.0, f64::from(n * side)).1 as f32;
        let mut out = Vec::new();
        for (i, anchor) in self.anchors.iter().enumerate() {
            let score = score_of(raw.logits[i]);
            if score < score_threshold {
                continue;
            }
            let r = &raw.regressors[i * REGRESSORS..][..REGRESSORS];
            let (xc, yc) = (r[0] / side + anchor[0], r[1] / side + anchor[1]);
            let (w, h) = (r[2] / side, r[3] / side);
            let mut keypoints_px = [[0.0f32; 2]; KEYPOINTS];
            for (k, kp) in keypoints_px.iter_mut().enumerate() {
                *kp = [
                    to_x(r[4 + 2 * k] / side + anchor[0]),
                    to_y(r[5 + 2 * k] / side + anchor[1]),
                ];
            }
            out.push(Detection {
                box_px: [
                    to_x(xc - w / 2.0),
                    to_y(yc - h / 2.0),
                    to_x(xc + w / 2.0),
                    to_y(yc + h / 2.0),
                ],
                keypoints_px,
                score,
            });
        }
        out
    }

    /// One pass: the network on one window of the picture, decoded and suppressed (detections in picture pixels).
    ///
    /// # Errors
    ///
    /// [`MlError::Input`] for an empty window, and the errors of [`BlazeFace::forward`].
    pub fn detect_window(
        &self,
        src: &dyn PixelSource,
        window: Window,
        resample: Resample,
        score_threshold: f32,
        nms_iou: f32,
    ) -> Result<Vec<Detection>> {
        let lb = Letterbox::new(window)?;
        let raw = self.forward(&prepare_tensor(src, &lb, resample))?;
        Ok(weighted_nms(self.decode(&raw, &lb, score_threshold), nms_iou))
    }
}

/// The settings of the face finder.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FinderParams {
    /// The lowest score kept, 0 to 1. MediaPipe's own default is 0.5 (the full-range graph uses 0.6).
    pub score_threshold: f32,
    /// Detections with an IoU above this are merged.
    pub nms_iou: f32,
    /// The most faces returned (the best by score).
    pub max_faces: usize,
    /// The smallest face to look for, in pixels of the longer side of its box; `None` takes 1.5 % of the picture's longer
    /// side (at least 24 px). Smaller detections are dropped, and the windows are sized so that a face of this size is
    /// found.
    pub min_face_px: Option<u32>,
    /// How windows are reduced to the network input.
    pub resample: Resample,
}

impl Default for FinderParams {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            nms_iou: 0.3,
            max_faces: 32,
            min_face_px: None,
            resample: Resample::Box,
        }
    }
}

/// A rectangle in whole picture pixels, the same fields as `ym_retouch::Rect`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FaceRect {
    /// Left edge.
    pub x_px: i32,
    /// Top edge.
    pub y_px: i32,
    /// Width.
    pub width_px: u32,
    /// Height.
    pub height_px: u32,
}

/// A face the finder found, in the terms of `ym_retouch::Face` plus what the detector knows besides.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectedFace {
    /// The face box, rounded outwards and clipped to the picture.
    pub rect: FaceRect,
    /// The longer side of the box (before clipping) over the picture's longer side.
    pub scale: f32,
    /// The six keypoints in picture pixels, in the order of [`KEYPOINT_NAMES`].
    pub keypoints_px: [[f32; 2]; KEYPOINTS],
    /// The network's confidence, 0 to 1.
    pub score: f32,
}

/// The windows the finder runs for a picture of `width` x `height`: tiles of growing size, then the whole picture.
pub fn plan_windows(width: usize, height: usize, min_face_px: f64) -> Vec<Window> {
    let (w, h) = (width as f64, height as f64);
    let long = w.max(h);
    let mut windows = Vec::new();
    let mut side = min_face_px / MIN_FRAME_FRACTION;
    // A window of three quarters of the picture or more would repeat the whole-picture pass.
    while side < 0.75 * long {
        let (ww, wh) = (side.min(w), side.min(h));
        let axis = |len: f64, win: f64| -> Vec<f64> {
            if len <= win {
                return vec![0.0];
            }
            let n = ((len - win) / (win * WINDOW_STEP)).ceil() as usize + 1;
            (0..n)
                .map(|i| ((len - win) * i as f64 / (n - 1) as f64).round())
                .collect()
        };
        for y in axis(h, wh) {
            for x in axis(w, ww) {
                windows.push(Window {
                    x_px: x,
                    y_px: y,
                    width_px: ww,
                    height_px: wh,
                });
            }
        }
        side *= WINDOW_GROWTH;
    }
    windows.push(Window::whole(width, height));
    windows
}

/// The face finder: BlazeFace on a whole picture or on a large one in windows.
#[derive(Debug)]
pub struct FaceFinder {
    net: BlazeFace,
}

impl FaceFinder {
    /// A finder around a loaded detector.
    pub fn new(net: BlazeFace) -> Self {
        Self { net }
    }

    /// The detector.
    pub fn net(&self) -> &BlazeFace {
        &self.net
    }

    /// Finds the faces of a picture, best first.
    ///
    /// # Errors
    ///
    /// [`MlError::Input`] for a picture that is not at least 2 x 2 or for non-finite parameters, and the errors of
    /// [`BlazeFace::forward`].
    pub fn find(&self, src: &dyn PixelSource, params: &FinderParams) -> Result<Vec<DetectedFace>> {
        let (width, height) = (src.width(), src.height());
        if width < 2 || height < 2 {
            return Err(MlError::Input(format!("a {width} x {height} picture")));
        }
        if !params.score_threshold.is_finite() || !params.nms_iou.is_finite() {
            return Err(MlError::Input("non-finite finder parameters".into()));
        }
        let long = width.max(height) as f64;
        let min_face = params
            .min_face_px
            .map_or_else(|| (0.015 * long).max(24.0), f64::from);
        let windows = plan_windows(width, height, min_face);
        let passes: Vec<Vec<Detection>> = windows
            .par_iter()
            .map(|&window| {
                let mut found = self.net.detect_window(
                    src,
                    window,
                    params.resample,
                    params.score_threshold,
                    params.nms_iou,
                )?;
                // A face cut by an inner border of the window is found whole in a neighbouring window.
                let margin = (BORDER_MARGIN * window.width_px.max(window.height_px)) as f32;
                let inner_left = window.x_px > 0.0;
                let inner_top = window.y_px > 0.0;
                let inner_right = window.x_px + window.width_px < width as f64;
                let inner_bottom = window.y_px + window.height_px < height as f64;
                let (left, top) = (window.x_px as f32 + margin, window.y_px as f32 + margin);
                let right = (window.x_px + window.width_px) as f32 - margin;
                let bottom = (window.y_px + window.height_px) as f32 - margin;
                found.retain(|d| {
                    let [x0, y0, x1, y1] = d.box_px;
                    !(inner_left && x0 < left
                        || inner_top && y0 < top
                        || inner_right && x1 > right
                        || inner_bottom && y1 > bottom)
                });
                Ok(found)
            })
            .collect::<Result<_>>()?;
        let merged = weighted_nms(passes.into_iter().flatten().collect(), params.nms_iou);
        let mut faces: Vec<DetectedFace> = merged
            .into_iter()
            .filter(|d| f64::from(d.size_px()) >= min_face)
            .filter_map(|d| to_face(&d, width, height))
            .take(params.max_faces)
            .collect();
        faces.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(faces)
    }
}

fn to_face(d: &Detection, width: usize, height: usize) -> Option<DetectedFace> {
    let [x0, y0, x1, y1] = d.box_px;
    let clip = |v: f32, max: usize| (v as i64).clamp(0, max as i64);
    let (left, top) = (clip(x0.floor(), width), clip(y0.floor(), height));
    let (right, bottom) = (clip(x1.ceil(), width), clip(y1.ceil(), height));
    if right <= left || bottom <= top {
        return None;
    }
    Some(DetectedFace {
        rect: FaceRect {
            x_px: left as i32,
            y_px: top as i32,
            width_px: (right - left) as u32,
            height_px: (bottom - top) as u32,
        },
        scale: d.size_px() / width.max(height) as f32,
        keypoints_px: d.keypoints_px,
        score: d.score,
    })
}
