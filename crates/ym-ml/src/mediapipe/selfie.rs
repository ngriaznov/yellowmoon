//! Selfie Multiclass: six classes for every pixel of a 256 x 256 picture (`docs/27` S-10, `docs/07`).
//!
//! The network is Google's `selfie_multiclass_256x256.tflite` (Apache-2.0, model card of 10 May 2023), converted to f32
//! safetensors by `tools/mediapipe/convert.py` (Google ships it in f32; every bit is kept) and run by
//! [`crate::mediapipe::graph`]. It takes `1 x 256 x 256 x 3` values `0..1` and gives `1 x 256 x 256 x 6` scores (logits: the
//! TFLite graph ends before the softmax, which MediaPipe's segmenter task applies, and so do we): background, hair, body skin, face skin, clothes, others. The face-skin class includes the eyes,
//! brows, lips and ears and reaches the hairline; `FaceParser` of `ym-retouch` cuts those with the mesh.
//!
//! [`SelfieSegmenter::segment`] runs it on a square crop of the picture turned like the face ([`crate::mediapipe::roi`]):
//! the network was trained on whole selfies, so the crop shows the head with hair, ears and a little of the shoulders.
//!
//! # Determinism
//!
//! The network is bit-stable on any number of threads ([`crate::mediapipe::graph`]).

use std::path::Path;

use crate::mediapipe::blazeface::{PixelSource, Resample};
use crate::mediapipe::graph::{Graph, Tensor};
use crate::mediapipe::roi::{Roi, crop_tensor};
use crate::mediapipe::{expected_sha256, verify};
use crate::model::{MlError, Result};

/// The weights file in the model folder (`data/models/mediapipe/`).
pub const WEIGHTS_FILE: &str = "selfie_multiclass_256.f32.safetensors";
/// The side of the network input and output, in pixels.
pub const INPUT_PX: usize = 256;
/// The number of classes.
pub const CLASSES: usize = 6;

/// The classes, in the order of the network's channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Class {
    /// Everything that is none of the others.
    Background = 0,
    /// Hair.
    Hair = 1,
    /// Skin of the body (arms, hands, neck).
    BodySkin = 2,
    /// Skin of the face, with eyes, brows, lips and ears.
    FaceSkin = 3,
    /// Clothes.
    Clothes = 4,
    /// Accessories and other things worn (glasses, hats, jewellery).
    Others = 5,
}

/// The network with its weights.
#[derive(Debug)]
pub struct SelfieNet {
    graph: Graph,
    sha256: Option<String>,
}

impl SelfieNet {
    /// Loads `dir/selfie_multiclass_256.f32.safetensors`, checked against its line in `dir/SHA256SUMS`.
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when a file cannot be read, [`MlError::Checksum`] when the weights do not match `SHA256SUMS`, and the
    /// errors of [`SelfieNet::from_bytes`].
    pub fn load(dir: &Path) -> Result<Self> {
        let bytes = std::fs::read(dir.join(WEIGHTS_FILE))?;
        let expected = expected_sha256(dir, WEIGHTS_FILE)?;
        Self::from_bytes(&bytes, Some(&expected))
    }

    /// Loads a weights file without a checksum (development and tests).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when the file cannot be read, and the errors of [`SelfieNet::from_bytes`].
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, None)
    }

    /// Builds the network from the bytes of the converted weights, checked against `expected_sha256` when given.
    ///
    /// # Errors
    ///
    /// [`MlError::Checksum`] on a digest mismatch, [`MlError::Weights`] or [`MlError::Unsupported`] when the file is not the
    /// Selfie Multiclass graph (one 1 x 256 x 256 x 3 input, one 1 x 256 x 256 x 6 output).
    pub fn from_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            verify("Selfie Multiclass weights", bytes, want)?;
        }
        let graph = Graph::from_bytes(bytes)?;
        let ok = graph.inputs().len() == 1
            && graph.inputs()[0].shape == [1, INPUT_PX, INPUT_PX, 3]
            && graph.outputs().len() == 1
            && graph.outputs()[0].shape == [1, INPUT_PX, INPUT_PX, CLASSES];
        if !ok {
            return Err(MlError::Weights(
                "the graph is not Selfie Multiclass (input 1x256x256x3, output 1x256x256x6)".into(),
            ));
        }
        Ok(Self {
            graph,
            sha256: expected_sha256.map(str::to_owned),
        })
    }

    /// The SHA-256 the weights were checked against, when they were.
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }

    /// Runs the network on a prepared `[256][256][3]` input; returns the `[256][256][6]` scores (logits) of the graph.
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] for an input of another size, [`MlError::Input`] for one holding a value that is not finite.
    pub fn forward_logits(&self, input: &[f32]) -> Result<Vec<f32>> {
        let t = Tensor::new(vec![1, INPUT_PX, INPUT_PX, 3], input.to_vec())?;
        let mut outputs = self.graph.run(&[t])?;
        Ok(outputs.remove(0).data)
    }

    /// Runs the network and applies the softmax over the six classes of every pixel: `[256][256][6]` probabilities.
    ///
    /// # Errors
    ///
    /// As [`SelfieNet::forward_logits`].
    pub fn forward(&self, input: &[f32]) -> Result<Vec<f32>> {
        let mut data = self.forward_logits(input)?;
        for px in data.chunks_exact_mut(CLASSES) {
            let max = px.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for v in px.iter_mut() {
                *v = libm::expf(*v - max);
                sum += *v;
            }
            for v in px.iter_mut() {
                *v /= sum;
            }
        }
        Ok(data)
    }
}

/// The class probabilities of a crop.
#[derive(Debug, Clone, PartialEq)]
pub struct ClassMap {
    /// The crop of the picture the map covers.
    pub roi: Roi,
    /// `[256][256][6]` probabilities.
    pub probs: Vec<f32>,
}

impl ClassMap {
    /// The probability of `class` at the crop position `(u, v)` (each `0..1` over the crop), bilinear between the pixels,
    /// held to the edge outside. Crop pixel `i` was sampled at `u = i / 256` ([`crate::mediapipe::roi`]), so the pixel's own
    /// position is `u * 256`, with no half-pixel shift.
    pub fn prob(&self, class: Class, u: f64, v: f64) -> f32 {
        let n = INPUT_PX as f64;
        let (x, y) = (u * n, v * n);
        let (fx, fy) = (x.floor(), y.floor());
        let (tx, ty) = ((x - fx) as f32, (y - fy) as f32);
        let at = |xx: i64, yy: i64| {
            let xx = xx.clamp(0, INPUT_PX as i64 - 1) as usize;
            let yy = yy.clamp(0, INPUT_PX as i64 - 1) as usize;
            self.probs[(yy * INPUT_PX + xx) * CLASSES + class as usize]
        };
        let (x0, y0) = (fx as i64, fy as i64);
        let top = at(x0, y0) * (1.0 - tx) + at(x0 + 1, y0) * tx;
        let bottom = at(x0, y0 + 1) * (1.0 - tx) + at(x0 + 1, y0 + 1) * tx;
        top * (1.0 - ty) + bottom * ty
    }
}

/// The segmenter: the network and the crop.
#[derive(Debug)]
pub struct SelfieSegmenter {
    net: SelfieNet,
}

impl SelfieSegmenter {
    /// A segmenter around a loaded network.
    pub fn new(net: SelfieNet) -> Self {
        Self { net }
    }

    /// The network.
    pub fn net(&self) -> &SelfieNet {
        &self.net
    }

    /// The class probabilities of the square crop `roi` of the picture.
    ///
    /// # Errors
    ///
    /// [`MlError::Input`] for an empty crop, and the errors of [`SelfieNet::forward`].
    pub fn segment(
        &self,
        src: &dyn PixelSource,
        roi: &Roi,
        resample: Resample,
    ) -> Result<ClassMap> {
        let input = crop_tensor(src, roi, INPUT_PX, resample, (0.0, 1.0))?;
        Ok(ClassMap {
            roi: *roi,
            probs: self.net.forward(&input)?,
        })
    }
}
