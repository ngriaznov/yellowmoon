//! Runs the noise estimator on ARW files and prints one JSON object per file.
//!
//! ```text
//! cargo run --release -p ym-noise --example estimate_arw -- [--points] [--priors DIR] FILE_OR_DIR...
//! ```
//!
//! Without `--priors` the estimator runs **without priors**, which is what
//! `training/noise/public_priors.py` needs to fit the public-data priors (`docs/06-noise.md`
//! section Priors from public files). With `--priors <data>/cameras/noise` it uses the prior the
//! file's body and ISO give. `--points` adds the variance points behind the fit.
//!
//! The example is also the reference for wiring `ym-raw` to `ym-noise`: the decoder's `black`
//! and `white` levels are per position of the readout, so they are rotated to the position of the
//! active area's origin; the cRAW `qstep` plane (output DN, `f16`) is normalised and read by
//! rectangle through the `PlaneSource` trait.
//!
//! `ym-raw` is a dev-dependency only: `ym-noise` itself takes a normalised mosaic.

use std::path::{Path, PathBuf};

use half::f16;
use serde_json::{Value, json};
use ym_noise::estimate::{EstimateInput, EstimatorConfig, PlaneSource, U16Mosaic, estimate};
use ym_noise::priors::{CameraContext, NoisePrior, PriorStore};
use ym_raw::decode::{RawImage, decode};
use ym_raw::layout::{DecodeKind, Layout, Plane};

/// The decoder's `f16` quantisation steps, normalised, as a rectangle source on the active area.
struct QstepSource<'a> {
    plane: &'a Plane<f16>,
    origin: (usize, usize),
    width: usize,
    height: usize,
    /// 1 / (white - black) per position of the active area's origin.
    scale: [f32; 4],
}

impl PlaneSource for QstepSource<'_> {
    fn width(&self) -> usize {
        self.width
    }
    fn height(&self) -> usize {
        self.height
    }
    fn read_rect(&self, x0: usize, y0: usize, w: usize, h: usize, out: &mut [f32]) {
        for r in 0..h {
            let src = &self.plane.data
                [(self.origin.1 + y0 + r) * self.plane.width + self.origin.0 + x0..][..w];
            for (c, (o, v)) in out[r * w..(r + 1) * w].iter_mut().zip(src).enumerate() {
                let p = (((y0 + r) & 1) << 1) | ((x0 + c) & 1);
                *o = v.to_f32() * self.scale[p];
            }
        }
    }
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
            .map(|d| d.filter_map(|e| e.ok().map(|e| e.path())).collect())
            .unwrap_or_default();
        entries.sort();
        for e in entries {
            collect(&e, out);
        }
    } else if path
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case("arw"))
    {
        out.push(path.to_owned());
    }
}

fn analyse(path: &Path, store: Option<&PriorStore>, with_points: bool) -> Result<Value, String> {
    let raw: RawImage = decode(path).map_err(|e| e.to_string())?;
    let Layout::Bayer { active_area, .. } = raw.layout else {
        return Err(
            "camera-processed YCbCr file (Lossless M or S): no mosaic to estimate from".into(),
        );
    };
    let full = (raw.data.width, raw.data.height);
    let (x0, y0, w, h) = match active_area {
        Some(r) => (
            r.x as usize,
            r.y as usize,
            r.width as usize,
            r.height as usize,
        ),
        None => (0, 0, full.0, full.1),
    };
    // Black and white are per raster position of the readout; rotate them to the active area's
    // origin (usually even, so this is the identity).
    let rotate = |v: [f32; 4]| -> [f32; 4] {
        std::array::from_fn(|p| v[(((y0 + (p >> 1)) & 1) << 1) | ((x0 + (p & 1)) & 1)])
    };
    let (black, white) = (rotate(raw.black), rotate(raw.white));
    let data = &raw.data.data[y0 * full.0 + x0..];
    let mosaic =
        U16Mosaic::with_stride(data, w, h, full.0, black, white).map_err(|e| e.to_string())?;
    let scale: [f32; 4] = std::array::from_fn(|p| 1.0 / (white[p] - black[p]));
    let qstep = raw.qstep.as_ref().map(|plane| QstepSource {
        plane,
        origin: (x0, y0),
        width: w,
        height: h,
        scale,
    });
    let q_norm = f64::from(scale.iter().copied().fold(f32::MAX, f32::min));
    let input = EstimateInput {
        mosaic: &mosaic,
        qstep: qstep.as_ref().map(|q| q as &dyn PlaneSource),
        q_norm,
    };
    let ctx = CameraContext {
        model_id: raw.meta.model_id.map(u32::from),
        model: Some(raw.meta.model.clone()),
        iso: raw.meta.iso,
        class: None,
    };
    let prior = store.map_or_else(NoisePrior::none, |s| s.prior_for(&ctx));
    let est = estimate(&input, &prior, &EstimatorConfig::default()).map_err(|e| e.to_string())?;
    let d = &est.diagnostics;
    let bit_depth = if raw.meta.decode == DecodeKind::Uncompressed12 {
        12
    } else {
        14
    };
    let mut v = json!({
        "file": path.file_name().and_then(|n| n.to_str()).unwrap_or_default(),
        "status": "ok",
        "make": raw.meta.make,
        "model": raw.meta.model,
        "model_id": raw.meta.model_id,
        "iso": raw.meta.iso,
        "decode": raw.meta.decode.id(),
        "bit_depth": bit_depth,
        "width": w,
        "height": h,
        "exposure_time_s": raw.meta.exposure_time_s,
        "model_fit": {
            "a": est.model.a, "b": est.model.b, "delta": est.model.delta,
            "sigma_row": est.model.sigma_row, "q": est.model.q,
        },
        "fitted": { "a": d.fitted.a, "b": d.fitted.b, "delta": d.fitted.delta },
        "coverage": d.coverage,
        "bins_surviving": d.bins_surviving,
        "texture_fallback": d.texture_fallback,
        "confidence": format!("{:?}", d.confidence),
        "residual_rms": d.residual_rms,
        "filtered": { "flagged": d.filtered.flagged, "ratio": d.filtered.ratio, "factor": d.filtered.factor, "evaluated": d.filtered.evaluated },
        "row_noise": { "measured": d.row_noise.measured, "sigma_row": d.row_noise.sigma_row },
        "prior_source": format!("{:?}", d.prior_source),
    });
    if with_points {
        v["points"] = serde_json::to_value(&est.points).map_err(|e| e.to_string())?;
    }
    Ok(v)
}

fn main() {
    let mut args = std::env::args().skip(1).peekable();
    let (mut with_points, mut priors_dir) = (false, None);
    let mut targets = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--points" => with_points = true,
            "--priors" => priors_dir = args.next(),
            _ => targets.push(PathBuf::from(a)),
        }
    }
    if targets.is_empty() {
        eprintln!("usage: estimate_arw [--points] [--priors DIR] FILE_OR_DIR...");
        std::process::exit(2);
    }
    let store = priors_dir.map(|d| match PriorStore::load_dir(Path::new(&d)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("cannot load priors: {e}");
            std::process::exit(2);
        }
    });
    let mut files = Vec::new();
    for t in &targets {
        collect(t, &mut files);
    }
    for f in files {
        let line = match analyse(&f, store.as_ref(), with_points) {
            Ok(v) => v,
            Err(e) => json!({
                "file": f.file_name().and_then(|n| n.to_str()).unwrap_or_default(),
                "status": "error",
                "error": e,
            }),
        };
        println!("{line}");
    }
}
