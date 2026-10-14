//! Adobe's reference renderer as a function: `dng_validate` from the DNG SDK (built by `tools/dng/build_dng_validate.sh`).
//!
//! The renderer is a per-pixel colour pipeline for a LinearRaw DNG (no spatial stage runs), so a DNG made of N colours
//! is N evaluations of it. [`Adobe::render`] writes such a DNG with the real `ym-export` writer (the same code path as
//! the product: scene-linear AWG4 pixels, `BaselineExposure` = `log2(x_white)` + the profile's offset, the profile's
//! tone curve and look table), renders it with `dng_validate -32 -cs1 -tif` (sRGB, 32-bit float, so nothing is
//! quantised) and reads the float TIFF back.
//!
//! `x_white` (the writer's scale, `docs/09`) is chosen by the writer from the 99.99th percentile of the pixels; a few
//! copies of the brightest colour are appended so that nothing is clipped by it, and the writer's report says so.

use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, bail, ensure};
use ym_export::dng::{BaselineExposure, DngProfile, DngReport};
use ym_export::{
    ColorSpace, DngInputs, ExportFormat, ExportHooks, ExportRequest, ImageRef, Metadata,
    PixelEncoding, Size, encode_to_vec,
};

/// Pixels per row of the DNGs this module writes.
const ROW: usize = 256;

/// A handle on `dng_validate`.
#[derive(Clone, Debug)]
pub struct Adobe {
    binary: PathBuf,
}

/// A rendered DNG file.
#[derive(Clone, Debug)]
pub struct FileRender {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// sRGB code values, 0 to 1, interleaved RGB.
    pub samples: Vec<f32>,
    /// `dng_validate`'s verbose output.
    pub log: String,
}

impl FileRender {
    /// The messages of the validation that count: warnings and errors (empty when the file is clean and the run
    /// completed).
    #[must_use]
    pub fn problems(&self) -> Vec<String> {
        problems_in(&self.log)
    }
}

/// The warnings and errors in the output of `dng_validate -v`, and a line when it did not say "Validation complete".
#[must_use]
pub fn problems_in(log: &str) -> Vec<String> {
    let mut v: Vec<String> = log
        .lines()
        .filter(|l| l.contains("Warning") || l.contains("Error"))
        .map(|l| l.trim().to_owned())
        .collect();
    if !log.contains("Validation complete") {
        v.push("dng_validate did not report \"Validation complete\"".to_owned());
    }
    v
}

/// What a render of N colours returned.
#[derive(Clone, Debug)]
pub struct Render {
    /// sRGB code values, 0 to 1, one per input colour, in order.
    pub srgb: Vec<[f32; 3]>,
    /// What the DNG writer measured (`x_white` and the clipped fractions).
    pub report: DngReport,
}

/// Where `dng_validate` is: `explicit`, else `$DNG_VALIDATE`, else `testdata/dng_sdk/dng_validate` of the workspace.
pub fn find_dng_validate(explicit: Option<&Path>) -> Result<PathBuf> {
    let candidates: Vec<PathBuf> = explicit
        .map(Path::to_path_buf)
        .into_iter()
        .chain(std::env::var_os("DNG_VALIDATE").map(PathBuf::from))
        .chain([golden::ymrun::workspace_root().join("testdata/dng_sdk/dng_validate")])
        .collect();
    candidates.iter().find(|p| p.is_file()).cloned().with_context(|| {
        format!(
            "dng_validate not found (looked for {}): run tools/dng/build_dng_validate.sh or give --dng-validate",
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    })
}

impl Adobe {
    /// A handle on the binary found by [`find_dng_validate`].
    pub fn new(explicit: Option<&Path>) -> Result<Self> {
        Ok(Self {
            binary: find_dng_validate(explicit)?,
        })
    }

    /// The binary.
    #[must_use]
    pub fn binary(&self) -> &Path {
        &self.binary
    }

    /// The version line `dng_validate` prints (`dng_validate, version 1.7.1 (2471) (64-bit)`), or `unknown`.
    #[must_use]
    pub fn version(&self) -> String {
        Command::new(&self.binary)
            .output()
            .ok()
            .and_then(|o| {
                // the usage text, with the version line first, goes to stderr
                format!(
                    "{}\n{}",
                    String::from_utf8_lossy(&o.stderr),
                    String::from_utf8_lossy(&o.stdout)
                )
                .lines()
                .map(str::trim)
                .find(|l| l.starts_with("dng_validate"))
                .map(str::to_owned)
            })
            .unwrap_or_else(|| "unknown".to_owned())
    }

    /// Renders a DNG file to a float sRGB TIFF and returns `(width, height, samples)` (interleaved RGB, 0 to 1).
    pub fn render_file(&self, dng: &Path) -> Result<(usize, usize, Vec<f32>)> {
        let r = self.render_file_verbose(dng)?;
        Ok((r.width, r.height, r.samples))
    }

    /// Runs `dng_validate -v` on a DNG without rendering it (a few milliseconds) and returns what it complains about:
    /// the warnings and errors of its validation of the file's structure, tags and digest.
    ///
    /// # Errors
    ///
    /// When the process cannot start or exits with a failure.
    pub fn validate_file(&self, dng: &Path) -> Result<Vec<String>> {
        let out = Command::new(&self.binary)
            .arg("-v")
            .arg(dng)
            .output()
            .with_context(|| format!("starting {}", self.binary.display()))?;
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if !out.status.success() {
            bail!("dng_validate failed on {}: {log}", dng.display());
        }
        Ok(problems_in(&log))
    }

    /// Like [`Adobe::render_file`], with `dng_validate`'s verbose output (the validation messages).
    pub fn render_file_verbose(&self, dng: &Path) -> Result<FileRender> {
        let scratch = tempfile::tempdir().context("creating a scratch folder")?;
        let base = scratch.path().join("render");
        let out = Command::new(&self.binary)
            .args(["-v", "-32", "-cs1", "-tif"])
            .arg(&base)
            .arg(dng)
            .output()
            .with_context(|| format!("starting {}", self.binary.display()))?;
        let log = format!(
            "{}{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        if !out.status.success() {
            bail!("dng_validate failed on {}: {log}", dng.display());
        }
        let (width, height, samples) = read_float_tiff(&base.with_extension("tif"))?;
        Ok(FileRender {
            width,
            height,
            samples,
            log,
        })
    }

    /// Renders `colors` (scene-linear AWG4, the pixels of the DNG after the look and curve) through `profile` with
    /// the product's `BaselineExposure` rule and the offset `fit_offset_ev`.
    ///
    /// # Errors
    ///
    /// When the DNG cannot be written (an invalid profile) or `dng_validate` fails.
    pub fn render(
        &self,
        colors: &[[f32; 3]],
        profile: &DngProfile,
        fit_offset_ev: f64,
    ) -> Result<Render> {
        ensure!(!colors.is_empty(), "nothing to render");
        let (bytes, report, width, height) = write_test_dng(colors, profile, fit_offset_ev)?;
        let scratch = tempfile::tempdir().context("creating a scratch folder")?;
        let dng = scratch.path().join("colors.dng");
        std::fs::write(&dng, bytes).with_context(|| format!("writing {}", dng.display()))?;
        let (w, h, samples) = self.render_file(&dng)?;
        ensure!(
            (w, h) == (width, height),
            "dng_validate returned {w} x {h}, expected {width} x {height}"
        );
        let srgb = samples
            .chunks_exact(3)
            .take(colors.len())
            .map(|c| [c[0], c[1], c[2]])
            .collect();
        Ok(Render { srgb, report })
    }
}

/// Writes the DNG of `colors` (one pixel each, `ROW` per row, the last row padded, a few copies of the brightest colour
/// appended so the writer's white level does not clip it). Returns the bytes, the writer's report and the size.
pub fn write_test_dng(
    colors: &[[f32; 3]],
    profile: &DngProfile,
    fit_offset_ev: f64,
) -> Result<(Vec<u8>, DngReport, usize, usize)> {
    ensure!(!colors.is_empty(), "nothing to write");
    let m = |c: &[f32; 3]| c[0].max(c[1]).max(c[2]);
    let brightest = colors
        .iter()
        .copied()
        .max_by(|a, b| m(a).total_cmp(&m(b)))
        .unwrap_or([0.0; 3]);
    // The product's rule (docs/09): x_white is at least 2^-offset, so that BaselineExposure (log2(x_white) + offset)
    // is never negative (`BaselineExposure::white_floor`; the writer applies it too). The white pixel below is kept
    // so that the fit and its numbers are reproduced exactly: with it x_white is the upper edge of the histogram bin
    // of the floor (16.008), without it the floor itself (16).
    let floor = (-fit_offset_ev).exp2() as f32;
    let brightest = if m(&brightest) < floor {
        [floor; 3]
    } else {
        brightest
    };
    // The writer's x_white is the 99.99th percentile of max(R, G, B): enough copies of the maximum put it at the maximum.
    let anchors = colors.len() / 2000 + 3;
    let mut pixels: Vec<[f32; 3]> = colors.to_vec();
    pixels.extend(std::iter::repeat_n(brightest, anchors));
    let rows = pixels.len().div_ceil(ROW);
    let width = ROW.min(pixels.len());
    pixels.resize(width * rows, *pixels.last().unwrap_or(&[0.0; 3]));
    let raw: Vec<f32> = pixels.iter().flatten().copied().collect();
    let (bytes, report) = write_dng_image(width, rows, &raw, profile, fit_offset_ev)?;
    Ok((bytes, report, width, rows))
}

/// Writes a DNG of `width` x `height` scene-linear AWG4 pixels (interleaved) with the product's writer: the profile's
/// tone curve and look table, `BaselineExposure` = `log2(x_white)` + `fit_offset_ev`, a flat preview, no EXIF or XMP.
pub fn write_dng_image(
    width: usize,
    height: usize,
    pixels: &[f32],
    profile: &DngProfile,
    fit_offset_ev: f64,
) -> Result<(Vec<u8>, DngReport)> {
    write_dng_image_with(
        width,
        height,
        pixels,
        profile,
        fit_offset_ev,
        &Metadata::default(),
    )
}

/// Like [`write_dng_image`], with EXIF and XMP content.
pub fn write_dng_image_with(
    width: usize,
    height: usize,
    pixels: &[f32],
    profile: &DngProfile,
    fit_offset_ev: f64,
    metadata: &Metadata,
) -> Result<(Vec<u8>, DngReport)> {
    write_dng_image_exposure(
        width,
        height,
        pixels,
        profile,
        // The fit is against the unplaced ACES 2 transform; the product adds the display placement
        // (`DISPLAY_PLACEMENT_EV`) to the written exposure on top of the fitted profile (docs/09).
        BaselineExposure::Auto {
            fit_offset_ev,
            placement_ev: 0.0,
        },
        1,
        metadata,
    )
}

/// Like [`write_dng_image_with`], with the `BaselineExposure` rule given instead of `Auto { fit_offset_ev }` (for a test
/// that needs the exposure the white floor of `Auto` rules out, such as a negative one: `Fixed`) and the TIFF `Orientation`
/// of the file (1 to 8; the pixels are not turned).
pub fn write_dng_image_exposure(
    width: usize,
    height: usize,
    pixels: &[f32],
    profile: &DngProfile,
    baseline_exposure: BaselineExposure,
    orientation: u16,
    metadata: &Metadata,
) -> Result<(Vec<u8>, DngReport)> {
    let preview = vec![0.2_f32; pixels.len()];
    let (w, h) = (width as u32, height as u32);
    let request = ExportRequest {
        pixels: ImageRef::new(w, h, pixels, PixelEncoding::SceneLinearAwg4)?,
        format: ExportFormat::Dng,
        color_space: ColorSpace::Srgb,
        size: Size::Full,
        metadata,
        remove_location: false,
        dng: Some(DngInputs {
            preview: ImageRef::new(w, h, &preview, PixelEncoding::DisplayLinear)?,
            orientation,
            crop: None,
            baseline_exposure,
            profile,
        }),
    };
    let (bytes, summary) = encode_to_vec(&request, &ExportHooks::none())?;
    let report = summary.dng.context("the writer made no DNG report")?;
    Ok((bytes, report))
}

/// Reads a float RGB TIFF (what `dng_validate -32` writes).
pub fn read_float_tiff(path: &Path) -> Result<(usize, usize, Vec<f32>)> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    // Real photographs are tens of megapixels: lift the decoder's default buffer limits.
    let mut decoder = tiff::decoder::Decoder::new(BufReader::new(file))
        .with_context(|| format!("reading {}", path.display()))?
        .with_limits(tiff::decoder::Limits::unlimited());
    let (w, h) = decoder.dimensions()?;
    let tiff::decoder::DecodingResult::F32(samples) = decoder.read_image()? else {
        bail!("{} is not a 32-bit float image", path.display());
    };
    ensure!(
        samples.len() == (w * h * 3) as usize,
        "{}: {} samples for {w} x {h} RGB",
        path.display(),
        samples.len()
    );
    Ok((w as usize, h as usize, samples))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_color::encode::srgb_encode;

    fn adobe() -> Option<Adobe> {
        match Adobe::new(None) {
            Ok(a) => Some(a),
            Err(e) => {
                assert!(
                    std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
                    "dng_validate is required: {e:#}"
                );
                eprintln!("SKIPPED: {e:#}");
                None
            }
        }
    }

    #[test]
    fn the_identity_profile_renders_the_srgb_of_the_color() {
        let Some(adobe) = adobe() else { return };
        let m = ym_color::gamut::matrices();
        let colors: Vec<[f32; 3]> = vec![
            [0.18; 3],
            [0.02; 3],
            [0.5, 0.2, 0.1],
            [0.1, 0.4, 0.2],
            [0.05, 0.1, 0.6],
        ];
        let r = adobe.render(&colors, &DngProfile::identity(), 0.0).unwrap();
        assert_eq!(r.srgb.len(), colors.len());
        assert_eq!(r.report.highlight_clipped_fraction, 0.0);
        for (c, got) in colors.iter().zip(&r.srgb) {
            let lin = ym_color::consts::linalg::mul_vec(&m.awg4_to_rec709, c.map(f64::from));
            for k in 0..3 {
                let want = srgb_encode(lin[k].max(0.0)).clamp(0.0, 1.0);
                assert!(
                    (f64::from(got[k]) - want).abs() < 1e-3,
                    "{c:?} channel {k}: {} vs {want}",
                    got[k]
                );
            }
        }
    }

    #[test]
    fn problems_are_read_from_the_log_of_dng_validate() {
        assert!(problems_in("Validating...\nValidation complete\n").is_empty());
        let p = problems_in(
            "*** Warning: the thing ***\n*** Error: the other ***\nValidation complete\n",
        );
        assert_eq!(p.len(), 2, "{p:?}");
        assert_eq!(problems_in("Validating...\n").len(), 1);
    }

    #[test]
    fn validation_without_a_render_passes_a_clean_file_and_not_a_broken_one() {
        let Some(adobe) = adobe() else { return };
        let (bytes, ..) =
            write_test_dng(&vec![[0.1_f32; 3]; 40], &DngProfile::identity(), 0.0).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let clean = dir.path().join("clean.dng");
        std::fs::write(&clean, &bytes).unwrap();
        assert_eq!(adobe.validate_file(&clean).unwrap(), Vec::<String>::new());
        // half a file: dng_validate either fails outright or says what is wrong, never "clean"
        let cut = dir.path().join("cut.dng");
        std::fs::write(&cut, &bytes[..bytes.len() / 2]).unwrap();
        assert!(adobe.validate_file(&cut).map_or(true, |p| !p.is_empty()));
    }

    #[test]
    fn a_bright_color_is_not_clipped_by_the_writers_white_level() {
        let Some(adobe) = adobe() else { return };
        // 40 grey pixels and one at 6: x_white must reach 6, and a fit offset of -3 EV scales it back by 8 for the renderer.
        let mut colors = vec![[0.1_f32; 3]; 40];
        colors.push([6.0; 3]);
        let r = adobe
            .render(&colors, &DngProfile::identity(), -3.0)
            .unwrap();
        assert!(r.report.x_white >= 6.0);
        assert_eq!(r.report.highlight_clipped_fraction, 0.0);
        let want = srgb_encode(6.0 / 8.0);
        assert!(
            (f64::from(r.srgb[40][0]) - want).abs() < 2e-3,
            "{} vs {want}",
            r.srgb[40][0]
        );
    }
}
