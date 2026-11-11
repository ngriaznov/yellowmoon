//! Face Mesh V2: 478 landmarks of a face (`docs/27` S-09, `docs/07`).
//!
//! The network is Google's `face_landmarks_detector.tflite` (inside `face_landmarker.task`, Apache-2.0, model card of
//! 15 September 2022), converted to f16 safetensors by `tools/mediapipe/convert.py` and run by [`crate::mediapipe::graph`].
//! The crop around it is our own code, written from MediaPipe's documented behaviour (the graphs
//! `face_landmark_front` and `face_landmarks_to_roi`):
//!
//! * **Input**: a square crop of the face turned so that the eyes are level ([`crate::mediapipe::roi`]), 256 x 256, values
//!   `0..1` (`x / 255`).
//! * **Output**: 478 points `(x, y, z)` in pixels of the crop (z in the same scale, the head's depth relative to its centre),
//!   and a face-presence logit. Points 0 to 467 are the canonical mesh, 468 to 477 the irises (468 and 473 their centres).
//! * **First pass**: the crop is the detector's box turned by the line between its two eye keypoints, its long side times
//!   [`LandmarkerParams::detection_scale`] (1.5, MediaPipe's `RectTransformationCalculator`).
//! * **Second pass**: the crop is made again from the first pass's landmarks, as MediaPipe does between video frames: the box
//!   around all 478 points, turned by the line from point 33 to point 263 (the outer eye corners), its long side times
//!   [`LandmarkerParams::refine_scale`] (1.5). The second answer is the result; it is better centred and levelled than the
//!   detector's box can make it.
//!
//! # Determinism
//!
//! The network is bit-stable on any number of threads ([`crate::mediapipe::graph`]) and the crop is a pure function of the
//! picture.

use std::path::Path;

use crate::mediapipe::blazeface::{DetectedFace, PixelSource, Resample};
use crate::mediapipe::graph::{Graph, Tensor};
use crate::mediapipe::roi::{Roi, crop_tensor, level_rotation};
use crate::mediapipe::{expected_sha256, verify};
use crate::model::{MlError, Result};

/// The weights file in the model folder (`data/models/mediapipe/`).
pub const WEIGHTS_FILE: &str = "face_mesh_v2.f16.safetensors";
/// The side of the network input, in pixels.
pub const INPUT_PX: usize = 256;
/// The number of landmarks: 468 mesh points and 10 for the irises.
pub const LANDMARKS: usize = 478;
/// The number of points of the canonical mesh (the irises come after them).
pub const MESH_POINTS: usize = 468;
/// The landmark that is the outer corner of the eye on the left of the picture (the person's right eye).
pub const EYE_OUTER_LEFT: usize = 33;
/// The landmark that is the outer corner of the eye on the right of the picture (the person's left eye).
pub const EYE_OUTER_RIGHT: usize = 263;

/// The raw answer of the network for one crop.
#[derive(Debug, Clone, PartialEq)]
pub struct RawMesh {
    /// The 478 points `(x, y, z)` in pixels of the 256 x 256 crop.
    pub landmarks: Vec<[f32; 3]>,
    /// The face-presence logit (the graph's second output, before the sigmoid; see [`RawMesh::presence`]): about 15 to 30 for
    /// a face, negative for a crop without one.
    pub presence_logit: f32,
}

impl RawMesh {
    /// The face-presence score, `0..1`: the sigmoid of the logit.
    pub fn presence(&self) -> f32 {
        1.0 / (1.0 + libm::expf(-self.presence_logit.clamp(-100.0, 100.0)))
    }
}

/// The network with its weights.
#[derive(Debug)]
pub struct FaceMeshNet {
    graph: Graph,
    landmark_output: usize,
    presence_output: usize,
    sha256: Option<String>,
}

impl FaceMeshNet {
    /// Loads `dir/face_mesh_v2.f16.safetensors`, checked against its line in `dir/SHA256SUMS`.
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when a file cannot be read, [`MlError::Checksum`] when the weights do not match `SHA256SUMS`, and the
    /// errors of [`FaceMeshNet::from_bytes`].
    pub fn load(dir: &Path) -> Result<Self> {
        let bytes = std::fs::read(dir.join(WEIGHTS_FILE))?;
        let expected = expected_sha256(dir, WEIGHTS_FILE)?;
        Self::from_bytes(&bytes, Some(&expected))
    }

    /// Loads a weights file without a checksum (development and tests).
    ///
    /// # Errors
    ///
    /// [`MlError::Io`] when the file cannot be read, and the errors of [`FaceMeshNet::from_bytes`].
    pub fn from_file(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?, None)
    }

    /// Builds the network from the bytes of the converted weights, checked against `expected_sha256` when given.
    ///
    /// # Errors
    ///
    /// [`MlError::Checksum`] on a digest mismatch, [`MlError::Weights`] or [`MlError::Unsupported`] when the file is not the
    /// Face Mesh V2 graph (one 1 x 256 x 256 x 3 input; outputs of 1434 values and one value).
    pub fn from_bytes(bytes: &[u8], expected_sha256: Option<&str>) -> Result<Self> {
        if let Some(want) = expected_sha256 {
            verify("Face Mesh V2 weights", bytes, want)?;
        }
        let graph = Graph::from_bytes(bytes)?;
        let input_ok =
            graph.inputs().len() == 1 && graph.inputs()[0].shape == [1, INPUT_PX, INPUT_PX, 3];
        let count = |o: &crate::mediapipe::graph::Io| o.shape.iter().product::<usize>();
        let landmark_output = graph
            .outputs()
            .iter()
            .position(|o| count(o) == LANDMARKS * 3);
        let presence_output = graph
            .outputs()
            .iter()
            .position(|o| count(o) == 1 && o.shape.len() == 4);
        let (true, Some(landmark_output), Some(presence_output)) =
            (input_ok, landmark_output, presence_output)
        else {
            return Err(MlError::Weights(
                "the graph is not Face Mesh V2 (input 1x256x256x3, outputs of 1434 values and one value)".into(),
            ));
        };
        Ok(Self {
            graph,
            landmark_output,
            presence_output,
            sha256: expected_sha256.map(str::to_owned),
        })
    }

    /// The SHA-256 the weights were checked against, when they were.
    pub fn sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }

    /// Runs the network on a prepared `[256][256][3]` input.
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] for an input of another size, [`MlError::Input`] for one holding a value that is not finite.
    pub fn forward(&self, input: &[f32]) -> Result<RawMesh> {
        let t = Tensor::new(vec![1, INPUT_PX, INPUT_PX, 3], input.to_vec())?;
        let outputs = self.graph.run(&[t])?;
        let landmarks = outputs[self.landmark_output]
            .data
            .chunks_exact(3)
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        Ok(RawMesh {
            landmarks,
            presence_logit: outputs[self.presence_output].data[0],
        })
    }
}

/// The settings of the two-pass landmarker.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LandmarkerParams {
    /// The first crop's side over the long side of the detector's box.
    pub detection_scale: f64,
    /// The second crop's side over the long side of the box around the first pass's landmarks.
    pub refine_scale: f64,
    /// Whether to make the second pass. With `false` the first answer is the result.
    pub second_pass: bool,
    /// How the picture is reduced to the crop.
    pub resample: Resample,
    /// The lowest face-presence score of a result (`0..1`); below it [`FaceLandmarker::landmarks`] returns `None`.
    pub min_presence: f32,
}

impl Default for LandmarkerParams {
    fn default() -> Self {
        Self {
            detection_scale: 1.5,
            refine_scale: 1.5,
            second_pass: true,
            resample: Resample::Box,
            min_presence: 0.5,
        }
    }
}

/// The landmarks of one face in picture pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceLandmarks {
    /// The 478 points `(x, y, z)`: x and y in picture pixels, z in the same scale (negative toward the camera).
    pub points_px: Vec<[f32; 3]>,
    /// The face-presence score of the last pass, `0..1`.
    pub presence: f32,
    /// The crop of the last pass.
    pub roi: Roi,
}

/// The crop of the first pass: the detector's box, turned to level its two eye keypoints, its long side times `scale`.
pub fn roi_from_detection(face: &DetectedFace, scale: f64) -> Roi {
    let r = face.rect;
    let (w, h) = (f64::from(r.width_px), f64::from(r.height_px));
    let eyes = (face.keypoints_px[0], face.keypoints_px[1]);
    Roi {
        center_px: [f64::from(r.x_px) + w / 2.0, f64::from(r.y_px) + h / 2.0],
        side_px: w.max(h) * scale,
        rotation_rad: level_rotation(
            [f64::from(eyes.0[0]), f64::from(eyes.0[1])],
            [f64::from(eyes.1[0]), f64::from(eyes.1[1])],
        ),
    }
}

/// The crop of a first pass when only the box is known (no eye keypoints): the box's centre, its long side times `scale`,
/// not turned. The refinement passes that follow turn it ([`roi_from_landmarks`]).
pub fn roi_from_box(x_px: f64, y_px: f64, width_px: f64, height_px: f64, scale: f64) -> Roi {
    Roi {
        center_px: [x_px + width_px / 2.0, y_px + height_px / 2.0],
        side_px: width_px.max(height_px) * scale,
        rotation_rad: 0.0,
    }
}

/// The crop of the second pass: the box around all landmarks (picture pixels), turned to level the line from point 33 to
/// point 263, its long side times `scale`.
pub fn roi_from_landmarks(points_px: &[[f32; 3]], scale: f64) -> Option<Roi> {
    if points_px.len() < LANDMARKS {
        return None;
    }
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for p in points_px {
        b = [
            b[0].min(f64::from(p[0])),
            b[1].min(f64::from(p[1])),
            b[2].max(f64::from(p[0])),
            b[3].max(f64::from(p[1])),
        ];
    }
    let (a, c) = (points_px[EYE_OUTER_LEFT], points_px[EYE_OUTER_RIGHT]);
    Some(Roi {
        center_px: [(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0],
        side_px: (b[2] - b[0]).max(b[3] - b[1]) * scale,
        rotation_rad: level_rotation(
            [f64::from(a[0]), f64::from(a[1])],
            [f64::from(c[0]), f64::from(c[1])],
        ),
    })
}

/// A crop's landmarks mapped back to picture pixels.
pub fn to_picture(raw: &RawMesh, roi: &Roi) -> Vec<[f32; 3]> {
    let n = INPUT_PX as f64;
    raw.landmarks
        .iter()
        .map(|p| {
            let q = roi.to_picture(f64::from(p[0]) / n, f64::from(p[1]) / n);
            [
                q[0] as f32,
                q[1] as f32,
                (f64::from(p[2]) * roi.side_px / n) as f32,
            ]
        })
        .collect()
}

/// The landmarker: the network and the two-pass crop.
#[derive(Debug)]
pub struct FaceLandmarker {
    net: FaceMeshNet,
}

impl FaceLandmarker {
    /// A landmarker around a loaded network.
    pub fn new(net: FaceMeshNet) -> Self {
        Self { net }
    }

    /// The network.
    pub fn net(&self) -> &FaceMeshNet {
        &self.net
    }

    /// One pass: the landmarks (picture pixels) and presence for the crop `roi`.
    ///
    /// # Errors
    ///
    /// [`MlError::Input`] for an empty crop, and the errors of [`FaceMeshNet::forward`].
    pub fn pass(
        &self,
        src: &dyn PixelSource,
        roi: &Roi,
        resample: Resample,
    ) -> Result<FaceLandmarks> {
        let input = crop_tensor(src, roi, INPUT_PX, resample, (0.0, 1.0))?;
        let raw = self.net.forward(&input)?;
        Ok(FaceLandmarks {
            points_px: to_picture(&raw, roi),
            presence: raw.presence(),
            roi: *roi,
        })
    }

    /// The landmarks of a face given only as a box (`x`, `y`, `width`, `height` in picture pixels, no eye keypoints): a first
    /// pass on the unturned crop of the box, then `refinements` passes (at least 1) each on the crop made from the last
    /// answer, which also turns it to level the eyes. `None` when the final presence is under
    /// [`LandmarkerParams::min_presence`].
    ///
    /// # Errors
    ///
    /// The errors of [`FaceLandmarker::pass`].
    pub fn landmarks_from_box(
        &self,
        src: &dyn PixelSource,
        box_px: [f64; 4],
        refinements: usize,
        params: &LandmarkerParams,
    ) -> Result<Option<FaceLandmarks>> {
        let roi = roi_from_box(
            box_px[0],
            box_px[1],
            box_px[2],
            box_px[3],
            params.detection_scale,
        );
        let mut last = self.pass(src, &roi, params.resample)?;
        for _ in 0..refinements.max(1) {
            let Some(roi) = roi_from_landmarks(&last.points_px, params.refine_scale) else {
                break;
            };
            let next = self.pass(src, &roi, params.resample)?;
            if next.presence < params.min_presence {
                break;
            }
            last = next;
        }
        Ok((last.presence >= params.min_presence).then_some(last))
    }

    /// The landmarks of a face the finder found, in two passes ([`LandmarkerParams`]); `None` when the network does not see a
    /// face in the crop (presence under [`LandmarkerParams::min_presence`]).
    ///
    /// # Errors
    ///
    /// The errors of [`FaceLandmarker::pass`].
    pub fn landmarks(
        &self,
        src: &dyn PixelSource,
        face: &DetectedFace,
        params: &LandmarkerParams,
    ) -> Result<Option<FaceLandmarks>> {
        let first = self.pass(
            src,
            &roi_from_detection(face, params.detection_scale),
            params.resample,
        )?;
        if first.presence < params.min_presence {
            return Ok(None);
        }
        if !params.second_pass {
            return Ok(Some(first));
        }
        let Some(roi) = roi_from_landmarks(&first.points_px, params.refine_scale) else {
            return Ok(Some(first));
        };
        let second = self.pass(src, &roi, params.resample)?;
        Ok(Some(if second.presence >= params.min_presence {
            second
        } else {
            first
        }))
    }
}
