//! The scenes written by `training/synth/make_scenes.py`: a raw mosaic, an embedded JPEG and a JSON sidecar.
//!
//! The writer needs only `mosaic` and `writer`, which [`Sidecar`] parses eagerly. The ground truth (`truth`) is kept as
//! JSON and read on demand with [`Sidecar::truth`] into the typed [`Truth`] (the parts the colour, exposure and noise
//! checks of the golden runner use; the full format is in `training/synth/README.md`).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail, ensure};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use ym_raw::layout::CfaPattern;

/// What the sidecar says about the mosaic file.
#[derive(Clone, Debug, Deserialize)]
pub struct MosaicInfo {
    /// File name next to the sidecar.
    pub file: String,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// CFA phase: `RGGB`, `GRBG`, `GBRG` or `BGGR`.
    pub cfa: String,
    /// Bit depth of the samples: 14, or 12 for the 12-bit variants.
    pub bits: u16,
    /// Black level as decoded (512 at 14 bits, 128 at 12 bits).
    pub black: u16,
    /// White level as decoded (16383 at 14 bits, 4095 at 12 bits).
    pub white: u16,
    /// SHA-256 of the little-endian sample bytes.
    pub sha256: String,
}

/// The SR2 block as written to the file.
#[derive(Clone, Debug, Deserialize)]
pub struct Sr2Info {
    /// Black level per colour (R, G1, G2, B) on the 14-bit scale.
    pub black: [u16; 4],
    /// White level per colour (R, G, B) on the 14-bit scale.
    pub white: [u16; 3],
    /// As-shot white balance gains times 1024 (R, G1, G2, B).
    pub wb_rggb: [u16; 4],
}

/// The embedded JPEG preview.
#[derive(Clone, Debug, Deserialize)]
pub struct PreviewInfo {
    /// File name next to the sidecar.
    pub file: String,
    /// Width in pixels.
    pub width: u16,
    /// Height in pixels.
    pub height: u16,
}

/// How to write the ARW (`writer` in the sidecar).
#[derive(Clone, Debug, Deserialize)]
pub struct WriterInfo {
    /// EXIF `Make`.
    pub make: String,
    /// EXIF `Model`.
    pub model: String,
    /// Sony model ID (maker note 0xB001).
    pub model_id: Option<u16>,
    /// `LensModel`, none for a fixed lens.
    pub lens_model: Option<String>,
    /// Focal length in millimetres.
    pub focal_length_mm: f64,
    /// ISO.
    pub iso: u32,
    /// Exposure time as `[numerator, denominator]`.
    pub exposure_time: [u32; 2],
    /// F-number as `[numerator, denominator]`.
    pub f_number: [u32; 2],
    /// Exposure compensation in EV.
    pub exposure_comp_ev: f64,
    /// `YYYY:MM:DD HH:MM:SS`.
    pub capture_time: String,
    /// SR2 levels and white balance.
    pub sr2: Sr2Info,
    /// The preview JPEG.
    pub preview: PreviewInfo,
    /// The variants the generator recommends for this mosaic.
    pub variants: Vec<String>,
    /// The `Software` tag (says the file is synthetic).
    pub software: String,
}

/// The parts of a scene sidecar this crate reads.
#[derive(Clone, Debug, Deserialize)]
pub struct Sidecar {
    /// File stem of every file of the scene; also the stem of the ARW names.
    pub id: String,
    /// `synthetic` or `synthetic-eval` (the latter holds CIE data and stays local).
    pub tier: String,
    /// The mosaic.
    pub mosaic: MosaicInfo,
    /// How to write the file.
    pub writer: WriterInfo,
    /// The ground truth as written, see [`Sidecar::truth`].
    #[serde(default)]
    pub truth: serde_json::Value,
}

/// The true light of a scene.
#[derive(Clone, Debug, Deserialize)]
pub struct LightTruth {
    /// File-name token: `D6500`, `bb2850`, `F11`, `LEDB3`, `mixed`, ...
    pub token: String,
    /// SPD name.
    pub name: String,
    /// `daylight`, `tungsten`, `fluorescent`, `led` or `mixed`.
    pub class: String,
    /// Class probabilities (one class with 1.0, or two with 0.5 for the mixed scene).
    pub class_probabilities: BTreeMap<String, f64>,
    /// CCT used for ARRI's matrix lookup, kelvin.
    pub cct_k: f64,
    /// The camera RGB of the light, green = 1, in the camera's raw scale.
    pub w_true_camera_rgb: [f64; 3],
}

/// The as-shot white balance and how it was drawn.
#[derive(Clone, Debug, Deserialize)]
pub struct AsShotTruth {
    /// The true gains `[r, 1, b]`, i.e. `1 / w_true_camera_rgb`.
    pub gains_true: [f64; 3],
    /// Log-gain error of red and blue that was applied to get the SR2 levels.
    pub log_gain_error_rb: [f64; 2],
}

/// The Poisson-Gaussian noise truth: `var(DN) = a * (DN - black) + b + 1/12`.
#[derive(Clone, Debug, Deserialize)]
pub struct NoiseTruth {
    /// ISO.
    pub iso: u32,
    /// Bit depth of the samples.
    pub bits: u16,
    /// Shot-noise gain in DN per electron.
    pub a_dn_per_electron: f64,
    /// Read-noise variance in DN squared.
    pub b_dn2: f64,
    /// `a` in normalised units (`(DN - black) / (white - black)`).
    pub a_normalised: f64,
    /// `b` in normalised units squared.
    pub b_normalised: f64,
    /// `single gain`, `low gain` or `high gain`.
    pub gain_regime: String,
}

/// One reference colour: ALEXA-estimate or Sony-through-ARRI-matrix, reflectance-referred (a perfect white is 1).
#[derive(Clone, Debug, Deserialize)]
pub struct Reference {
    /// White-balanced native RGB before the ARRI matrix.
    pub native_balanced: [f64; 3],
    /// Linear AWG3.
    pub awg3: [f64; 3],
    /// Linear AWG4.
    pub awg4: [f64; 3],
    /// LogC4 of `awg4`.
    pub logc4: [f64; 3],
}

/// A place in the image with a known colour (a patch, or a sample on a gradient or ramp).
#[derive(Clone, Debug, Deserialize)]
pub struct Probe {
    /// Unique name: `rta-017`, `neutral-0.18`, `ladder-hue0-ev-3`, `lamp-white`, ...
    pub name: String,
    /// `chart`, `neutral`, `ladder`, `shadow`, `black`, `highlight`, `gradient` or `rolloff`.
    pub group: String,
    /// `grey_reference`, `white_reference` or empty.
    #[serde(default)]
    pub role: String,
    /// The whole patch: x, y, width, height in pixels.
    pub rect: [u32; 4],
    /// The part to average, away from edges and blur.
    pub interior: [u32; 4],
    /// Irradiance relative to the nominal light.
    pub irradiance: f64,
    /// Noise-free camera DN above black in this file's sample scale (unclipped).
    pub camera_dn_above_black: [f64; 3],
    /// Which channels the noise-free value pushes to or past the white level.
    pub clipped_channels: [bool; 3],
    /// The ALEXA-estimate reference.
    pub alexa: Reference,
    /// The sensor-map-free baseline: Sony camera RGB through the same ARRI matrix.
    pub sony_through_arri: Reference,
}

/// The typed ground truth of a scene (see [`Sidecar::truth`]).
#[derive(Clone, Debug, Deserialize)]
pub struct Truth {
    /// The true light.
    pub light: LightTruth,
    /// The as-shot white balance.
    pub as_shot: AsShotTruth,
    /// The noise model.
    pub noise: NoiseTruth,
    /// Every probe of the scene.
    pub probes: Vec<Probe>,
}

impl Sidecar {
    /// The ground truth, parsed from the sidecar's `truth` object.
    pub fn truth(&self) -> Result<Truth> {
        serde_json::from_value(self.truth.clone()).context("parsing the sidecar's truth")
    }
}

/// A scene loaded into memory: sidecar, samples and preview JPEG.
#[derive(Clone, Debug)]
pub struct Scene {
    /// The parsed sidecar.
    pub sidecar: Sidecar,
    /// `width * height` samples, row-major, black level included.
    pub mosaic: Vec<u16>,
    /// The embedded JPEG, as the generator wrote it.
    pub preview: Vec<u8>,
    /// The CFA phase.
    pub cfa: CfaPattern,
}

/// Parses a CFA name.
pub fn parse_cfa(name: &str) -> Result<CfaPattern> {
    Ok(match name {
        "RGGB" => CfaPattern::RGGB,
        "GRBG" => CfaPattern::GRBG,
        "GBRG" => CfaPattern::GBRG,
        "BGGR" => CfaPattern::BGGR,
        other => bail!("unknown CFA pattern {other:?}"),
    })
}

impl Scene {
    /// Loads the scene whose sidecar is `sidecar_path`, checking the mosaic's size and SHA-256.
    pub fn load(sidecar_path: &Path) -> Result<Self> {
        let dir = sidecar_path.parent().unwrap_or(Path::new("."));
        let text = fs::read_to_string(sidecar_path)
            .with_context(|| format!("reading {}", sidecar_path.display()))?;
        let sidecar: Sidecar = serde_json::from_str(&text)
            .with_context(|| format!("parsing {}", sidecar_path.display()))?;
        let m = &sidecar.mosaic;
        let raw =
            fs::read(dir.join(&m.file)).with_context(|| format!("reading mosaic {}", m.file))?;
        let expected = m.width as usize * m.height as usize * 2;
        ensure!(
            raw.len() == expected,
            "{}: {} bytes, expected {} for {} x {}",
            m.file,
            raw.len(),
            expected,
            m.width,
            m.height
        );
        let digest = hex::encode(Sha256::digest(&raw));
        ensure!(
            digest == m.sha256,
            "{}: SHA-256 {digest} differs from the sidecar's {} (regenerate the scenes)",
            m.file,
            m.sha256
        );
        let mosaic: Vec<u16> = raw
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        let preview_path = dir.join(&sidecar.writer.preview.file);
        let preview = fs::read(&preview_path)
            .with_context(|| format!("reading preview {}", preview_path.display()))?;
        let cfa = parse_cfa(&m.cfa)?;
        Ok(Self {
            sidecar,
            mosaic,
            preview,
            cfa,
        })
    }

    /// Mosaic width in pixels.
    pub fn width(&self) -> u32 {
        self.sidecar.mosaic.width
    }

    /// Mosaic height in pixels.
    pub fn height(&self) -> u32 {
        self.sidecar.mosaic.height
    }
}

/// The sidecars in a scenes folder (`*.json` except `index.json`), sorted by name.
pub fn list_sidecars(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let is_json = path.extension().is_some_and(|e| e == "json");
        let is_index = path.file_name().is_some_and(|n| n == "index.json");
        if is_json && !is_index {
            out.push(path);
        }
    }
    out.sort();
    Ok(out)
}
