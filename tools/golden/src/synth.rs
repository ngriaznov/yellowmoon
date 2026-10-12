//! `golden synth`: writes ARW files from the synthetic scenes and checks that they decode back to the mosaic.
//!
//! For every scene sidecar (`training/synth/make_scenes.py`) and every requested variant the writer builds an ARW with
//! `ym_raw::testgen` carrying realistic metadata (Make SONY, the body's model and model ID, ISO, exposure, lens, an
//! SR2 block with black, white and the as-shot white balance, the embedded JPEG preview), writes
//! `<stem>__<variant>.ARW`, and decodes the bytes again with `ym_raw::decode`:
//!
//! | Variant | Written as | Round-trip check |
//! |---|---|---|
//! | `uncompressed14` | 14-bit 16-bit words | every sample equal |
//! | `packed12` | packed 12-bit (ARW 2) from a 12-bit mosaic | every sample equal |
//! | `craw` | lossy cRAW, nearest rounding | `2 * |decoded - true| <= qstep` for every sample |
//! | `lossless-l` | tiled lossless JPEG, 512 x 512 tiles | every sample equal |
//! | `lossless-ms` | camera RGB (bilinear demosaic of the mosaic), YCbCr 4:2:0 | small error against that RGB |
//!
//! Lossless M and S are camera-processed RGB, not a mosaic, so the check there compares with the bilinear RGB the
//! writer started from (chroma subsampling and the white-balance round trip leave a few DN of error, more at sharp
//! colour edges).

use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail, ensure};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use ym_raw::decode::ycbcr::{CHROMA_CENTER, ycc_to_rgb};
use ym_raw::decode::{ChromaUpsampling, DecodeOptions, YcbcrOptions, decode_bytes};
use ym_raw::exif::CaptureTime;
use ym_raw::layout::ChromaSubsampling;
use ym_raw::metadata::probe_bytes;
use ym_raw::testgen::{ArwBuilder, CrawEncoder, LensTables, LosslessL, LosslessMs, Packed12};

use crate::demosaic::bilinear_rgb;
use crate::preview;
use crate::scene::{Scene, list_sidecars};

/// An ARW variant of the synthetic set.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Variant {
    /// 14-bit uncompressed (compression 1).
    Uncompressed14,
    /// Packed 12-bit (ARW 2).
    Packed12,
    /// Lossy cRAW.
    Craw,
    /// Lossless L (tiled lossless JPEG).
    LosslessL,
    /// Lossless M or S (camera YCbCr), 4:2:0.
    LosslessMs,
}

impl Variant {
    /// Every variant, in the order the CLI documents them.
    pub const ALL: [Variant; 5] = [
        Variant::Uncompressed14,
        Variant::Packed12,
        Variant::Craw,
        Variant::LosslessL,
        Variant::LosslessMs,
    ];

    /// The name used on the command line and in file names.
    pub fn name(self) -> &'static str {
        match self {
            Variant::Uncompressed14 => "uncompressed14",
            Variant::Packed12 => "packed12",
            Variant::Craw => "craw",
            Variant::LosslessL => "lossless-l",
            Variant::LosslessMs => "lossless-ms",
        }
    }

    /// The bit depth of the mosaic this variant is written from.
    pub fn bits(self) -> u16 {
        if self == Variant::Packed12 { 12 } else { 14 }
    }
}

impl fmt::Display for Variant {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Variant {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        Variant::ALL
            .into_iter()
            .find(|v| v.name() == s)
            .ok_or_else(|| {
                anyhow!(
                    "unknown variant {s:?}; expected one of {}",
                    Variant::ALL.map(Variant::name).join(", ")
                )
            })
    }
}

/// How a written file compared with its mosaic.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "check", rename_all = "snake_case")]
pub enum Verify {
    /// Every sample equal.
    Exact,
    /// cRAW: the largest `2 * |decoded - true| / qstep` over all samples (at most 1).
    WithinQstep {
        /// The largest ratio found.
        max_ratio: f64,
    },
    /// Lossless M and S: the planes decode exactly as the writer's YCbCr planes say, and with the gains undone they
    /// match the intended RGB in block means.
    Reduced {
        /// Median over 16 x 16 blocks of the absolute error of the block mean against the intended RGB, in DN.
        median_block_dn: f64,
    },
    /// Verification was switched off.
    NotChecked,
}

impl fmt::Display for Verify {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verify::Exact => f.write_str("exact"),
            Verify::WithinQstep { max_ratio } => {
                write!(f, "within qstep/2 (worst {:.2} of the bound)", max_ratio)
            }
            Verify::Reduced { median_block_dn } => {
                write!(
                    f,
                    "planes exact, block means within {median_block_dn:.1} DN of the intended RGB"
                )
            }
            Verify::NotChecked => f.write_str("not checked"),
        }
    }
}

/// Limit (DN) on the median over 16 x 16 blocks of the error of the block mean in a decode of a Lossless M or S file
/// (gains undone, chroma replicated) against the RGB the writer started from. Noise and chroma subsampling average
/// out over a block; what is left is the rounding of the YCbCr transform, of the gain and of its undo.
pub const MS_BLOCK_MEDIAN_LIMIT_DN: f64 = 2.0;

/// Options of [`run`].
#[derive(Clone, Debug, Default)]
pub struct SynthOptions {
    /// Write exactly these variants for every scene they fit; `None` takes each sidecar's recommendation.
    pub variants: Option<Vec<Variant>>,
    /// Only scenes whose id contains one of these strings (empty: all).
    pub only: Vec<String>,
    /// Skip the decode-and-compare step.
    pub no_verify: bool,
    /// Worker threads (0: up to four, or the number of CPUs when fewer).
    pub jobs: usize,
}

/// One written file.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Written {
    /// File name inside the output folder.
    pub file: String,
    /// The scene's id.
    pub scene: String,
    /// The variant.
    pub variant: Variant,
    /// File size in bytes.
    pub bytes: u64,
    /// SHA-256 of the file.
    pub sha256: String,
    /// The round-trip result.
    pub verify: Verify,
    /// SHA-256 of the little-endian samples our decoder returns (for Lossless M and S: with the white balance left in
    /// and replicated chroma, which is what LibRaw returns), `None` when verification was off. The LibRaw oracle
    /// check (`oracle_check.py`) compares its own pixels with this. In the report of a `--no-verify` run it is
    /// `None`; the index keeps the hash of an earlier verified file with the same bytes.
    pub decoded_sha256: Option<String>,
    /// Whether the file on disk already had exactly these bytes.
    pub unchanged: bool,
    /// Seconds spent on build, write and verification.
    pub seconds: f64,
}

/// What [`run`] did.
#[derive(Debug, Default)]
pub struct Report {
    /// The files written (or found identical), in file-name order.
    pub written: Vec<Written>,
    /// `(scene id, variant, reason)` for every requested file that does not fit its scene.
    pub skipped: Vec<(String, Variant, String)>,
    /// Failures: build or round-trip errors, `(file, message)`.
    pub failures: Vec<(String, String)>,
}

/// The ARW file name of a scene and variant: `<stem>__<variant>.ARW`.
pub fn arw_name(scene_id: &str, variant: Variant) -> String {
    format!("{scene_id}__{variant}.ARW")
}

/// Why `variant` cannot be written from `scene`, if it cannot.
pub fn incompatibility(scene: &Scene, variant: Variant) -> Option<String> {
    let m = &scene.sidecar.mosaic;
    if m.bits != variant.bits() {
        return Some(format!(
            "needs a {}-bit mosaic, this one is {}-bit",
            variant.bits(),
            m.bits
        ));
    }
    if variant == Variant::Craw && !m.width.is_multiple_of(32) {
        return Some(format!(
            "cRAW rows are whole 32-pixel groups, the width is {}",
            m.width
        ));
    }
    if !m.width.is_multiple_of(2) || !m.height.is_multiple_of(2) {
        return Some("width and height must be even".to_owned());
    }
    None
}

fn neutral_lens_tables() -> LensTables {
    let table = |knots: usize| -> Vec<i16> {
        let mut v = vec![16i16];
        v.extend(std::iter::repeat_n(0i16, knots));
        v
    };
    LensTables {
        vignetting: Some(table(16)),
        chromatic_aberration: Some(table(32)),
        distortion: Some(table(16)),
    }
}

/// The Lossless M/S payload of a scene: its bilinear RGB, the as-shot gains, 4:2:0.
fn ms_payload(scene: &Scene) -> LosslessMs {
    let m = &scene.sidecar.mosaic;
    let w = &scene.sidecar.writer;
    let rgb = bilinear_rgb(
        &scene.mosaic,
        m.width as usize,
        m.height as usize,
        scene.cfa,
        i32::from(m.black),
    );
    let gains = [
        f32::from(w.sr2.wb_rggb[0]) / 1024.0,
        1.0,
        f32::from(w.sr2.wb_rggb[3]) / 1024.0,
    ];
    LosslessMs::new(ChromaSubsampling::Yuv420, rgb, m.width, m.height, gains)
}

/// The RGB the writer's planes decode to with replicated chroma and the gains left in, computed without the decoder.
fn planes_rgb(payload: &LosslessMs) -> Vec<u16> {
    let (w, h) = (payload.width as usize, payload.height as usize);
    let (fh, fv) = (2usize, 2usize); // 4:2:0
    let (y, cb, cr, cw, ch) = payload.planes();
    let mut out = Vec::with_capacity(w * h * 3);
    for yy in 0..h {
        for x in 0..w {
            let c = (yy / fv).min(ch - 1) * cw + (x / fh).min(cw - 1);
            out.extend_from_slice(&ycc_to_rgb(
                y[yy * w + x],
                i32::from(cb[c]) - CHROMA_CENTER,
                i32::from(cr[c]) - CHROMA_CENTER,
            ));
        }
    }
    out
}

/// Builds the ARW bytes of `scene` as `variant`. Fails if the variant does not fit the scene
/// ([`incompatibility`]) or the preview cannot be embedded.
pub fn build_arw(scene: &Scene, variant: Variant) -> Result<Vec<u8>> {
    if let Some(reason) = incompatibility(scene, variant) {
        bail!(
            "{} cannot be written as {variant}: {reason}",
            scene.sidecar.id
        );
    }
    let w = &scene.sidecar.writer;
    let m = &scene.sidecar.mosaic;
    let (width, height) = (m.width, m.height);
    let (pw, ph) = (w.preview.width, w.preview.height);
    let min_bytes = preview::placeholder_min_bytes(&scene.preview, pw, ph);
    let comp = ((w.exposure_comp_ev * 10.0).round() as i32, 10);
    let mut builder = ArwBuilder::new(&w.model, width, height)
        .make(&w.make)
        .software(&w.software)
        .capture_time(&w.capture_time, "00", Some("+00:00"))
        .iso(w.iso)
        .exposure_time(w.exposure_time[0], w.exposure_time[1])
        .f_number(w.f_number[0], w.f_number[1])
        .exposure_comp(comp.0, comp.1)
        .focal_length((w.focal_length_mm * 10.0).round() as u32, 10)
        .lens_model(w.lens_model.as_deref())
        .black(w.sr2.black)
        .white(w.sr2.white)
        .raw_ifd_white_level(w.sr2.white[0])
        .wb(Some(w.sr2.wb_rggb))
        .cfa(scene.cfa)
        .lens_tables(neutral_lens_tables())
        .preview(Some((pw, ph)))
        .preview_min_bytes(min_bytes)
        // The probe shows the largest JPEG below 1024 px, so a preview of 160 px or less would lose to the builder's
        // 160 x 120 thumbnail: such tiny previews (tests only) go without one.
        .thumbnail(if pw.max(ph) > 160 {
            Some((160, 120))
        } else {
            None
        });
    if let Some(id) = w.model_id {
        builder = builder.model_id(id);
    }
    let mut bytes = match variant {
        Variant::Uncompressed14 => builder.uncompressed(14).build(&scene.mosaic),
        Variant::Packed12 => builder.payload(Packed12).build(&scene.mosaic),
        Variant::Craw => builder.craw(CrawEncoder::new()).build(&scene.mosaic),
        Variant::LosslessL => builder.payload(LosslessL::new()).build(&scene.mosaic),
        Variant::LosslessMs => builder.lossless_ms(ms_payload(scene)).build(&[]),
    };
    preview::embed(&mut bytes, &scene.preview, pw, ph, min_bytes)?;
    Ok(bytes)
}

/// The result of decoding a written file again.
#[derive(Clone, Debug, PartialEq)]
pub struct Checked {
    /// How the decoded samples compare with the scene.
    pub verify: Verify,
    /// SHA-256 of the decoded samples as little-endian bytes (see [`Written::decoded_sha256`]).
    pub decoded_sha256: String,
}

fn digest_samples(samples: &[u16]) -> String {
    let mut hasher = Sha256::new();
    for chunk in samples.chunks(1 << 16) {
        let bytes: Vec<u8> = chunk.iter().flat_map(|v| v.to_le_bytes()).collect();
        hasher.update(&bytes);
    }
    hex::encode(hasher.finalize())
}

/// Decodes `file` and compares it with the scene. Returns how it matched, or an error naming the first difference.
pub fn verify_round_trip(scene: &Scene, variant: Variant, file: &[u8]) -> Result<Checked> {
    let w = &scene.sidecar.writer;
    let m = &scene.sidecar.mosaic;
    let meta = probe_bytes(file).map_err(|e| anyhow!("probe: {e}"))?;
    ensure!(
        meta.make == w.make && meta.model == w.model,
        "make/model {} {}",
        meta.make,
        meta.model
    );
    ensure!(
        meta.model_id == w.model_id,
        "model id {:?}, expected {:?}",
        meta.model_id,
        w.model_id
    );
    ensure!(
        meta.iso == Some(w.iso),
        "ISO {:?}, expected {}",
        meta.iso,
        w.iso
    );
    ensure!(
        (meta.width, meta.height) == (m.width, m.height),
        "size {} x {}",
        meta.width,
        meta.height
    );
    ensure!(
        meta.cfa == scene.cfa,
        "CFA {:?}, expected {:?}",
        meta.cfa,
        scene.cfa
    );
    let near = |a: Option<f64>, b: f64| a.is_some_and(|a| (a - b).abs() <= 1e-6 * b.abs().max(1.0));
    ensure!(
        near(
            meta.exposure_time_s,
            f64::from(w.exposure_time[0]) / f64::from(w.exposure_time[1])
        ),
        "exposure time {:?}, expected {:?}",
        meta.exposure_time_s,
        w.exposure_time
    );
    ensure!(
        near(
            meta.f_number,
            f64::from(w.f_number[0]) / f64::from(w.f_number[1])
        ),
        "f-number {:?}, expected {:?}",
        meta.f_number,
        w.f_number
    );
    ensure!(
        near(meta.focal_length_mm, w.focal_length_mm),
        "focal length {:?}",
        meta.focal_length_mm
    );
    ensure!(
        near(
            meta.exposure_comp_ev,
            (w.exposure_comp_ev * 10.0).round() / 10.0
        ),
        "exposure compensation {:?}",
        meta.exposure_comp_ev
    );
    ensure!(
        meta.lens_model == w.lens_model,
        "lens {:?}, expected {:?}",
        meta.lens_model,
        w.lens_model
    );
    ensure!(
        meta.capture_time == CaptureTime::parse_exif(&w.capture_time, Some("00"), Some("+00:00")),
        "capture time {:?}, expected {}",
        meta.capture_time,
        w.capture_time
    );
    let wb = scene.cfa.rggb_to_positions(w.sr2.wb_rggb.map(f32::from));
    ensure!(
        meta.wb_as_shot == Some(wb),
        "white balance {:?}, expected {wb:?}",
        meta.wb_as_shot
    );
    let p = meta
        .preview
        .ok_or_else(|| anyhow!("no preview in the file"))?;
    let embedded = file
        .get(p.offset as usize..(p.offset + p.length) as usize)
        .ok_or_else(|| anyhow!("preview range outside the file"))?;
    ensure!(
        embedded == preview::pad_exact(&scene.preview, embedded.len())?.as_slice(),
        "the embedded preview is not the scene's JPEG"
    );

    let image =
        decode_bytes(file, &DecodeOptions::default()).map_err(|e| anyhow!("decode: {e}"))?;
    // Sony writes the levels on the 14-bit scale; the decoder shifts them right for 12-bit data.
    let shift = if m.bits == 12 { 2 } else { 0 };
    let levels = |v: u16| f32::from(v >> shift);
    match variant {
        Variant::LosslessMs => {
            ensure!(
                image.is_reduced(),
                "a Lossless M/S file must decode to camera RGB"
            );
            let (wi, hi) = (m.width as usize, m.height as usize);
            ensure!(
                image.pixel_width() == wi && image.pixel_height() == hi,
                "decoded size differs"
            );
            // (1) the tiled YCbCr stream decodes exactly to the writer's planes (gains still applied, chroma replicated)
            let raw_opts = DecodeOptions {
                ycbcr: YcbcrOptions {
                    chroma: ChromaUpsampling::Replicate,
                    undo_white_balance: false,
                },
            };
            let stored = decode_bytes(file, &raw_opts).map_err(|e| anyhow!("decode: {e}"))?;
            let payload = ms_payload(scene);
            let want = planes_rgb(&payload);
            ensure!(
                stored.data.data.len() == want.len(),
                "decoded sample count differs"
            );
            if let Some(i) = stored.data.data.iter().zip(&want).position(|(a, b)| a != b) {
                bail!(
                    "Lossless M/S sample {i} decoded as {}, the planes say {}",
                    stored.data.data[i],
                    want[i]
                );
            }
            // (2) with the gains divided out again and replicated chroma (the writer averages each 2 x 2 group, so
            // a replicated group is centred on it), block means match the intended RGB. Linear chroma upsampling
            // co-sites chroma with the upper left sample and so shifts it half a pixel, which a steep gradient shows.
            let expected = bilinear_rgb(&scene.mosaic, wi, hi, scene.cfa, i32::from(m.black));
            let balanced_back = decode_bytes(
                file,
                &DecodeOptions {
                    ycbcr: YcbcrOptions {
                        chroma: ChromaUpsampling::Replicate,
                        undo_white_balance: true,
                    },
                },
            )
            .map_err(|e| anyhow!("decode: {e}"))?;
            ensure!(
                balanced_back.data.data.len() == expected.len(),
                "decoded sample count differs"
            );
            let median = median_block_error(&balanced_back.data.data, &expected, wi, hi, 16);
            ensure!(
                median <= MS_BLOCK_MEDIAN_LIMIT_DN,
                "Lossless M/S block-mean error {median:.1} DN exceeds {MS_BLOCK_MEDIAN_LIMIT_DN} DN"
            );
            Ok(Checked {
                verify: Verify::Reduced {
                    median_block_dn: median,
                },
                decoded_sha256: digest_samples(&stored.data.data),
            })
        }
        _ => {
            ensure!(!image.is_reduced(), "a mosaic variant decoded to RGB");
            ensure!(
                image.data.data.len() == scene.mosaic.len(),
                "decoded sample count differs"
            );
            ensure!(
                image.black == [levels(w.sr2.black[0]); 4],
                "black {:?}",
                image.black
            );
            ensure!(
                image.white == [levels(w.sr2.white[0]); 4],
                "white {:?}",
                image.white
            );
            if variant == Variant::Craw {
                let qstep = image
                    .qstep
                    .as_ref()
                    .ok_or_else(|| anyhow!("cRAW without a quantisation map"))?;
                let mut worst = 0f64;
                for (i, (decoded, truth)) in image.data.data.iter().zip(&scene.mosaic).enumerate() {
                    let step = qstep.data[i].to_f32();
                    let err = 2.0 * (f32::from(*decoded) - f32::from(*truth)).abs();
                    ensure!(
                        err <= step,
                        "cRAW sample {i} (x {}, y {}): true {truth}, decoded {decoded}, qstep {step}",
                        i % m.width as usize,
                        i / m.width as usize
                    );
                    worst = worst.max(f64::from(err) / f64::from(step));
                }
                Ok(Checked {
                    verify: Verify::WithinQstep { max_ratio: worst },
                    decoded_sha256: digest_samples(&image.data.data),
                })
            } else {
                if let Some(i) = image
                    .data
                    .data
                    .iter()
                    .zip(&scene.mosaic)
                    .position(|(a, b)| a != b)
                {
                    bail!(
                        "{variant}: sample {i} (x {}, y {}) decoded as {}, the mosaic has {}",
                        i % m.width as usize,
                        i / m.width as usize,
                        image.data.data[i],
                        scene.mosaic[i]
                    );
                }
                Ok(Checked {
                    verify: Verify::Exact,
                    decoded_sha256: digest_samples(&image.data.data),
                })
            }
        }
    }
}

/// Median over `block` x `block` pixel blocks and channels of `|mean(got) - mean(want)|` for interleaved RGB.
fn median_block_error(got: &[u16], want: &[u16], width: usize, height: usize, block: usize) -> f64 {
    let mut errors = Vec::new();
    for by in (0..height).step_by(block) {
        for bx in (0..width).step_by(block) {
            let (y1, x1) = ((by + block).min(height), (bx + block).min(width));
            let n = ((y1 - by) * (x1 - bx)) as f64;
            for c in 0..3 {
                let (mut a, mut b) = (0u64, 0u64);
                for y in by..y1 {
                    for x in bx..x1 {
                        a += u64::from(got[(y * width + x) * 3 + c]);
                        b += u64::from(want[(y * width + x) * 3 + c]);
                    }
                }
                errors.push((a as f64 / n - b as f64 / n).abs());
            }
        }
    }
    errors.sort_by(f64::total_cmp);
    errors[errors.len() / 2]
}

fn write_if_changed(path: &Path, bytes: &[u8]) -> Result<bool> {
    if fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(true);
    }
    let tmp = path.with_extension("ARW.tmp");
    fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    fs::rename(&tmp, path).with_context(|| format!("renaming to {}", path.display()))?;
    Ok(false)
}

/// Writes one scene's files; the report holds the files written, what was skipped and what failed.
fn process_scene(sidecar: &Path, out: &Path, opts: &SynthOptions) -> Report {
    let mut report = Report::default();
    let scene = match Scene::load(sidecar) {
        Ok(s) => s,
        Err(e) => {
            report
                .failures
                .push((sidecar.display().to_string(), format!("{e:#}")));
            return report;
        }
    };
    let id = scene.sidecar.id.clone();
    let wanted: Vec<Variant> = match &opts.variants {
        Some(v) => v.clone(),
        None => {
            // A variant name this crate does not know is a failure, not a silent omission: the manifest lists
            // the file, so dropping it here would only show up later as a missing golden input.
            let mut known = Vec::new();
            for name in &scene.sidecar.writer.variants {
                match name.parse::<Variant>() {
                    Ok(v) => known.push(v),
                    Err(e) => report
                        .failures
                        .push((format!("{id}__{name}.ARW"), format!("{e:#}"))),
                }
            }
            known
        }
    };
    for variant in wanted {
        let file = arw_name(&id, variant);
        if let Some(reason) = incompatibility(&scene, variant) {
            report.skipped.push((id.clone(), variant, reason));
            continue;
        }
        let t0 = Instant::now();
        let result = (|| -> Result<Written> {
            let bytes = build_arw(&scene, variant)?;
            let (verify, decoded_sha256) = if opts.no_verify {
                (Verify::NotChecked, None)
            } else {
                let checked = verify_round_trip(&scene, variant, &bytes)?;
                (checked.verify, Some(checked.decoded_sha256))
            };
            let unchanged = write_if_changed(&out.join(&file), &bytes)?;
            Ok(Written {
                file: file.clone(),
                scene: id.clone(),
                variant,
                bytes: bytes.len() as u64,
                sha256: hex::encode(Sha256::digest(&bytes)),
                verify,
                decoded_sha256,
                unchanged,
                seconds: t0.elapsed().as_secs_f64(),
            })
        })();
        match result {
            Ok(w) => report.written.push(w),
            Err(e) => report.failures.push((file, format!("{e:#}"))),
        }
    }
    report
}

/// Writes the ARW files of every scene in `scenes` into `out` (created if missing) and an `index.json` next to them.
pub fn run(scenes: &Path, out: &Path, opts: &SynthOptions) -> Result<Report> {
    let mut sidecars = list_sidecars(scenes)?;
    if !opts.only.is_empty() {
        sidecars.retain(|p| {
            let name = p.file_stem().and_then(|s| s.to_str()).unwrap_or("");
            opts.only.iter().any(|t| name.contains(t.as_str()))
        });
    }
    ensure!(
        !sidecars.is_empty(),
        "no scene sidecars in {} (run training.synth.make_scenes first)",
        scenes.display()
    );
    fs::create_dir_all(out).with_context(|| format!("creating {}", out.display()))?;
    let jobs = if opts.jobs == 0 {
        std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(4)
    } else {
        opts.jobs
    };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(jobs).build()?;
    let results: Vec<_> = pool.install(|| {
        sidecars
            .par_iter()
            .map(|p| process_scene(p, out, opts))
            .collect()
    });
    let mut report = Report::default();
    for r in results {
        report.written.extend(r.written);
        report.skipped.extend(r.skipped);
        report.failures.extend(r.failures);
    }
    report.written.sort_by(|a, b| a.file.cmp(&b.file));
    update_index(out, &report.written)?;
    Ok(report)
}

/// Index of the written files: merged with what an earlier run listed, entries of vanished files dropped.
fn update_index(out: &Path, written: &[Written]) -> Result<()> {
    #[derive(Serialize, Deserialize)]
    struct Index {
        format: String,
        files: Vec<Written>,
    }
    let path: PathBuf = out.join("index.json");
    let old: Vec<Written> = fs::read_to_string(&path)
        .ok()
        .and_then(|t| serde_json::from_str::<Index>(&t).ok())
        .map(|i| i.files)
        .unwrap_or_default();
    let mut files: Vec<Written> = old
        .iter()
        .filter(|f| out.join(&f.file).exists() && !written.iter().any(|w| w.file == f.file))
        .cloned()
        .collect();
    for w in written {
        let mut w = w.clone();
        // A run with `--no-verify` decodes nothing. The same bytes as an entry that was verified decode the same,
        // so the index keeps that result instead of losing the hash the LibRaw oracle compares with.
        if w.decoded_sha256.is_none()
            && let Some(prev) = old
                .iter()
                .find(|f| f.file == w.file && f.sha256 == w.sha256 && f.decoded_sha256.is_some())
        {
            w.decoded_sha256.clone_from(&prev.decoded_sha256);
            w.verify = prev.verify.clone();
        }
        files.push(w);
    }
    files.sort_by(|a, b| a.file.cmp(&b.file));
    let index = Index {
        format: "ym-synth-arw-index".to_owned(),
        files,
    };
    fs::write(&path, serde_json::to_string_pretty(&index)? + "\n")?;
    Ok(())
}
