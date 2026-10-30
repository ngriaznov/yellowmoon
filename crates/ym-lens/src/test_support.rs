//! A complete synthetic lens folder in the format of `docs/21` section 6, for tests of this and other crates.
//!
//! Seven lenses with analytic spread functions (elliptical Gaussians with a wide halo that grow with the field
//! radius, a different centre per wavelength for the lateral colour), `T(rho) = 1 - f rho^2`, small distortion,
//! swirl and cat's-eye profiles, ghosts, a rim, glow and veil values, a spikes table, the attachment and a
//! correct `SHA256SUMS`. Nothing here is measured or computed from a lens design; the files say so with the
//! provenance statement the format requires, and the real data in `data/lenses/` replace them in shipped builds.
//!
//! The pieces are public so that tests can write a good folder and then damage one file
//! ([`write_sums`] re-hashes after the damage, to test the checks behind the digest).

use std::fmt::Write as _;
use std::io;
use std::path::Path;
use std::sync::OnceLock;

use safetensors::tensor::TensorView;
use safetensors::{Dtype, serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use crate::amounts::LensKind;
use crate::consts::{
    ATTACHMENT_PROVENANCE_STATEMENT, GRID_SAMPLES, GRID_START_NM, GRID_STEP_NM,
    PROVENANCE_STATEMENT, PSF_SIZE, PSF_WAVELENGTHS, PSF_WAVELENGTHS_NM, RADII, RIM_SAMPLES,
    SPIKE_SIZE,
};

/// How one synthetic lens differs from the others.
#[derive(Clone, Copy, Debug)]
pub struct Recipe {
    /// The lens.
    pub kind: LensKind,
    /// Half-width of the spread functions at the centre, in half-diagonals.
    pub w_centre: f64,
    /// Half-width at the edge.
    pub w_edge: f64,
    /// Radial and tangential Gaussian sigma at the edge, in table units (centre: 0.16).
    pub sigma_edge: (f64, f64),
    /// Lateral colour: the radial shift of the 420 and 700 nm tables at the edge, in table units.
    pub lateral: f64,
    /// `f` of `T = 1 - f rho^2`.
    pub falloff: f64,
    /// Distortion `(k1, k2)`.
    pub distortion: (f64, f64),
    /// Swirl at the edge.
    pub swirl: f64,
    /// Astigmatism at the edge, in half-diagonals (none of the synthetic lenses has any: a test that wants it
    /// edits `field.astigmatism`).
    pub astigmatism: f64,
    /// Cat's-eye at the edge.
    pub cats_eye: f64,
    /// Starting `(flares, edges)`.
    pub start: (f64, f64),
}

/// The recipe of each lens.
#[must_use]
pub fn recipe(kind: LensKind) -> Recipe {
    let r = |kind,
             w_centre,
             w_edge,
             sigma_edge,
             lateral,
             falloff,
             distortion,
             swirl,
             cats_eye,
             start| {
        Recipe {
            kind,
            w_centre,
            w_edge,
            sigma_edge,
            lateral,
            falloff,
            distortion,
            swirl,
            astigmatism: 0.0,
            cats_eye,
            start,
        }
    };
    match kind {
        LensKind::Default => r(
            kind,
            0.0020,
            0.0060,
            (0.24, 0.22),
            0.06,
            0.40,
            (-0.010, 0.004),
            0.0,
            0.0,
            (0.0, 0.0),
        ),
        LensKind::Classic => r(
            kind,
            0.0030,
            0.0070,
            (0.28, 0.24),
            0.10,
            0.30,
            (-0.015, 0.005),
            0.1,
            0.1,
            (25.0, 20.0),
        ),
        LensKind::Vintage => r(
            kind,
            0.0040,
            0.0100,
            (0.34, 0.26),
            0.14,
            0.45,
            (-0.030, 0.015),
            0.3,
            0.5,
            (35.0, 40.0),
        ),
        LensKind::Petzval => r(
            kind,
            0.0015,
            0.0120,
            (0.40, 0.22),
            0.12,
            0.55,
            (0.020, -0.010),
            0.8,
            0.7,
            (20.0, 60.0),
        ),
        LensKind::Triplet => r(
            kind,
            0.0020,
            0.0100,
            (0.36, 0.30),
            0.16,
            0.50,
            (-0.025, 0.010),
            0.2,
            0.3,
            (30.0, 50.0),
        ),
        LensKind::Sonnar => r(
            kind,
            0.0030,
            0.0070,
            (0.26, 0.24),
            0.08,
            0.35,
            (-0.012, 0.004),
            0.1,
            0.2,
            (25.0, 25.0),
        ),
        LensKind::Detuned => r(
            kind,
            0.0015,
            0.0120,
            (0.42, 0.34),
            0.18,
            0.50,
            (-0.020, 0.012),
            0.4,
            0.6,
            (40.0, 70.0),
        ),
    }
}

fn name_and_line(kind: LensKind) -> (&'static str, &'static str) {
    match kind {
        LensKind::Default => ("Default", "Clean and modern: even sharpness, faint glow."),
        LensKind::Classic => (
            "Classic",
            "Warm and gentle: soft glow, amber flares, smooth falloff.",
        ),
        LensKind::Vintage => (
            "Vintage",
            "Glows wide open, lower contrast, soft edges, a little swirl.",
        ),
        LensKind::Petzval => ("Petzval", "Very sharp centre, strong swirl, fast falloff."),
        LensKind::Triplet => ("Triplet", "Soft corners, busy flares, a crisp centre."),
        LensKind::Sonnar => ("Sonnar", "Smooth and glowing, with gentle edges."),
        LensKind::Detuned => ("Detuned", "Sharp only in the centre, with smeared flares."),
    }
}

/// The nine field radii.
#[must_use]
pub fn radii() -> Vec<f64> {
    (0..RADII).map(|i| i as f64 / (RADII - 1) as f64).collect()
}

fn gauss(x: f64, mean: f64, sigma: f64) -> f64 {
    let d = (x - mean) / sigma;
    libm::exp(-0.5 * d * d)
}

/// The spread functions of a recipe: `[9][8][33][33]` flattened, x radial, y tangential, cell-centred over
/// `+-1`, each slice summing to 1.
#[must_use]
pub fn psf_table(rc: &Recipe) -> Vec<f32> {
    let n = PSF_SIZE;
    let mut out = Vec::with_capacity(RADII * PSF_WAVELENGTHS * n * n);
    for rho in radii() {
        let rho2 = rho * rho;
        let sx = 0.16 + (rc.sigma_edge.0 - 0.16) * rho2;
        let sy = 0.16 + (rc.sigma_edge.1 - 0.16) * rho2;
        for nm in PSF_WAVELENGTHS_NM {
            // Lateral colour: the table's centre moves radially with wavelength, growing with the field radius.
            let shift = rc.lateral * rho * (nm - 560.0) / 140.0;
            let mut cells = vec![0.0f64; n * n];
            for (i, c) in cells.iter_mut().enumerate() {
                let tx = -1.0 + (2 * (i % n) + 1) as f64 / n as f64;
                let ty = -1.0 + (2 * (i / n) + 1) as f64 / n as f64;
                let core = gauss(tx, shift, sx) * gauss(ty, 0.0, sy);
                let halo = gauss(tx, shift, 2.2 * sx) * gauss(ty, 0.0, 2.2 * sy);
                *c = core + 0.12 * halo;
            }
            let sum: f64 = cells.iter().sum();
            out.extend(cells.iter().map(|c| (c / sum) as f32));
        }
    }
    out
}

/// The starburst tables: `[8][129][129]` flattened, each slice summing to 1.
#[must_use]
pub fn spikes_table(blades: u32) -> Vec<f32> {
    let n = SPIKE_SIZE;
    let mut out = Vec::with_capacity(PSF_WAVELENGTHS * n * n);
    for nm in PSF_WAVELENGTHS_NM {
        let reach = 0.25 * nm / 560.0;
        let mut cells = vec![0.0f64; n * n];
        for (i, c) in cells.iter_mut().enumerate() {
            let x = -1.0 + (2 * (i % n) + 1) as f64 / n as f64;
            let y = -1.0 + (2 * (i / n) + 1) as f64 / n as f64;
            let r = (x * x + y * y).sqrt();
            let theta = libm::atan2(y, x);
            let spoke = libm::cos(f64::from(blades.max(2)) * theta * 0.5);
            *c = libm::exp(-r / reach) * (spoke.powi(8) + 0.02);
        }
        let sum: f64 = cells.iter().sum();
        out.extend(cells.iter().map(|c| (c / sum) as f32));
    }
    out
}

fn grid() -> Vec<f64> {
    (0..GRID_SAMPLES)
        .map(|i| GRID_START_NM + GRID_STEP_NM * i as f64)
        .collect()
}

/// A smooth reflectance bump centred on `centre_nm`, between 0.15 and 1.
#[must_use]
pub fn coating(centre_nm: f64) -> Vec<f64> {
    grid()
        .into_iter()
        .map(|nm| 0.15 + 0.85 * gauss(nm, centre_nm, 60.0))
        .collect()
}

fn ghost(m: f64, n: f64, s: f64, clip: f64, fraction: f64, centre_nm: f64) -> Value {
    json!({ "m": m, "n": n, "s": s, "clip_radius": clip, "fraction": fraction, "soft": 0.1,
            "coating": coating(centre_nm) })
}

/// `lens.json` of a recipe.
#[must_use]
pub fn lens_json(rc: &Recipe) -> Value {
    let (name, line) = name_and_line(rc.kind);
    let radii = radii();
    let rho2: Vec<f64> = radii.iter().map(|r| r * r).collect();
    let rim: Vec<f64> = {
        let strength = 0.2 + 0.5 * rc.cats_eye;
        let raw: Vec<f64> = (0..RIM_SAMPLES)
            .map(|i| 1.0 + strength * (i as f64 / (RIM_SAMPLES - 1) as f64).powi(4) * 4.0)
            .collect();
        let mean = raw.iter().sum::<f64>() / RIM_SAMPLES as f64;
        raw.iter().map(|v| v / mean).collect()
    };
    let busy = matches!(rc.kind, LensKind::Triplet | LensKind::Detuned);
    json!({
        "id": rc.kind.as_str(),
        "version": 1,
        "name": name,
        "line": line,
        "start": { "flares": rc.start.0, "edges": rc.start.1 },
        "provenance": {
            "statement": PROVENANCE_STATEMENT,
            "design": { "type": "synthetic test lens", "source": "none (analytic test data)", "year": 1900,
                        "focal_mm": 50.0, "f_number": 2.0, "format": "36 x 24 mm" },
            "coatings": "synthetic test coatings",
            "generated": "2026-10-24",
            "method": "Analytic test data for unit tests; not a lens design."
        },
        "field": {
            "radii": radii,
            "psf_half_width": rho2.iter().map(|r2| rc.w_centre + (rc.w_edge - rc.w_centre) * r2).collect::<Vec<_>>(),
            "relative_illumination": rho2.iter().map(|r2| 1.0 - rc.falloff * r2).collect::<Vec<_>>(),
            "distortion": { "k1": rc.distortion.0, "k2": rc.distortion.1 },
            "swirl": rho2.iter().map(|r2| rc.swirl * r2).collect::<Vec<_>>(),
            "astigmatism": rho2.iter().map(|r2| rc.astigmatism * r2).collect::<Vec<_>>(),
            "cats_eye": rho2.iter().map(|r2| rc.cats_eye * r2).collect::<Vec<_>>(),
        },
        "bokeh": { "rim": rim },
        "flare": {
            "glow": { "fraction": 0.04, "core_radius": 0.0015, "r0": 0.01, "power": 2.0, "reach": 1.0,
                      "chroma": [1.3, 1.2, 1.1, 1.0, 0.95, 0.9, 0.85, 0.8] },
            "veil": { "fraction": 0.02, "radius": 1.0 },
            "aperture": { "blades": if busy { 6 } else { 9 }, "rotation_deg": 10.0, "roundness": 0.8 },
            "ghosts": [
                ghost(-0.5, 0.3, 0.04, 0.08, 0.010, 450.0),
                ghost(0.7, -0.2, 0.06, 0.10, 0.008, 600.0),
                ghost(1.4, 0.0, 0.03, 0.05, 0.006, 520.0),
            ],
            "starburst": { "fraction": 0.004, "radius": 0.06, "f_number_full": 11.0 },
        }
    })
}

/// `anamorphic.json`.
#[must_use]
pub fn attachment_json() -> Value {
    json!({
        "id": "anamorphic",
        "version": 1,
        "provenance": { "statement": ATTACHMENT_PROVENANCE_STATEMENT },
        "q_max": 2.0,
        // The synthetic bend is the plain two-term one: the sides stretched (`a1`), horizontal lines bowing (`b1`).
        "bend": { "frame_share": 1.0, "x": [0.08, 0.0, 0.0, 0.0, 0.0], "y": [-0.06, 0.0, 0.0, 0.0, 0.0] },
        "field": { "kappa": 0.15, "side_softness": 0.8, "fringe": 1.0 },
        "ghost_stretch": 0.6,
        "streak": { "fraction": 0.05, "reach": 1.0, "core": 0.002, "power": 1.5, "tint": coating(470.0) },
        "ghosts": [
            ghost(-0.4, 0.2, 0.05, 0.09, 0.006, 470.0),
            ghost(0.9, 0.1, 0.04, 0.07, 0.004, 560.0),
        ],
    })
}

/// `index.json`.
#[must_use]
pub fn index_json() -> Value {
    json!({
        "version": 1,
        "lenses": LensKind::ALL.iter().map(|k| k.as_str()).collect::<Vec<_>>(),
        "attachment": "anamorphic.json",
    })
}

fn io_err(e: impl std::fmt::Display) -> io::Error {
    io::Error::other(e.to_string())
}

/// The bytes of a `tables.safetensors` file with the given `psf` and `spikes` data.
///
/// # Errors
///
/// A shape that does not fit the data.
pub fn tables_bytes(psf: &[f32], spikes: &[f32]) -> io::Result<Vec<u8>> {
    let bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let (psf_bytes, spike_bytes) = (bytes(psf), bytes(spikes));
    let psf_view = TensorView::new(
        Dtype::F32,
        vec![RADII, PSF_WAVELENGTHS, PSF_SIZE, PSF_SIZE],
        &psf_bytes,
    )
    .map_err(io_err)?;
    let spike_view = TensorView::new(
        Dtype::F32,
        vec![PSF_WAVELENGTHS, SPIKE_SIZE, SPIKE_SIZE],
        &spike_bytes,
    )
    .map_err(io_err)?;
    serialize([("psf", psf_view), ("spikes", spike_view)], None).map_err(io_err)
}

/// Writes `tables.safetensors` with the given `psf` and `spikes` data.
///
/// # Errors
///
/// Any I/O error, or a shape that does not fit the data.
pub fn write_tables(path: &Path, psf: &[f32], spikes: &[f32]) -> io::Result<()> {
    std::fs::write(path, tables_bytes(psf, spikes)?)
}

/// The tables file of each lens, computed once per process.
fn cached_tables(kind: LensKind) -> io::Result<&'static [u8]> {
    static CACHE: OnceLock<Vec<Vec<u8>>> = OnceLock::new();
    let all = CACHE.get_or_init(|| {
        LensKind::ALL
            .iter()
            .map(|k| {
                let blades = if matches!(k, LensKind::Triplet | LensKind::Detuned) {
                    6
                } else {
                    9
                };
                tables_bytes(&psf_table(&recipe(*k)), &spikes_table(blades)).unwrap_or_default()
            })
            .collect()
    });
    let i = LensKind::ALL.iter().position(|k| *k == kind).unwrap_or(0);
    if all[i].is_empty() {
        return Err(io::Error::other("cannot build the synthetic tables"));
    }
    Ok(&all[i])
}

fn write_json(path: &Path, v: &Value) -> io::Result<()> {
    let mut text = serde_json::to_string_pretty(v).map_err(io_err)?;
    text.push('\n');
    std::fs::write(path, text)
}

/// Writes a complete synthetic lenses folder into `dir` (created if missing), with a correct `SHA256SUMS`.
///
/// # Errors
///
/// Any I/O error.
pub fn write_synthetic_lenses(dir: &Path) -> io::Result<()> {
    write_synthetic_lenses_with(dir, &|_| {}, &|_| {})
}

/// [`write_synthetic_lenses`] with `lens_edit` applied to the JSON of every lens and `att_edit` to the
/// attachment's before they are written.
///
/// # Errors
///
/// Any I/O error.
pub fn write_synthetic_lenses_with(
    dir: &Path,
    lens_edit: &dyn Fn(&mut Value),
    att_edit: &dyn Fn(&mut Value),
) -> io::Result<()> {
    std::fs::create_dir_all(dir)?;
    write_json(&dir.join("index.json"), &index_json())?;
    let mut att = attachment_json();
    att_edit(&mut att);
    write_json(&dir.join("anamorphic.json"), &att)?;
    std::fs::write(
        dir.join("README.md"),
        "Synthetic lens files for tests. Not lens data.\n",
    )?;
    for kind in LensKind::ALL {
        let rc = recipe(kind);
        let sub = dir.join(kind.as_str());
        std::fs::create_dir_all(&sub)?;
        let mut lens = lens_json(&rc);
        lens_edit(&mut lens);
        write_json(&sub.join("lens.json"), &lens)?;
        std::fs::write(sub.join("tables.safetensors"), cached_tables(kind)?)?;
    }
    write_sums(dir)
}

/// A library whose lenses have only ghosts, big and soft ones as the shipped lenses have (apothems up to a whole
/// half-diagonal, some cut by their clipping circle), with fractions of light `fraction` times 1 to 3, and no
/// glow, veil, starburst or streak: the flare terms of a plan are the ghosts' alone. For tests of the drawn items.
///
/// # Errors
///
/// Any I/O error writing and reading back the folder, which is made under the system's temporary folder and
/// removed again.
pub fn ghost_library(fraction: f64) -> io::Result<crate::LensLibrary> {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNT: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "ym-ghost-lenses-{}-{}",
        std::process::id(),
        COUNT.fetch_add(1, Ordering::SeqCst)
    ));
    let ghost = |m: f64, n: f64, s: f64, clip: f64, f: f64, soft: f64, nm: f64| {
        json!({ "m": m, "n": n, "s": s, "clip_radius": clip, "fraction": f, "soft": soft,
                "coating": coating(nm) })
    };
    let written = write_synthetic_lenses_with(
        &dir,
        &|lens| {
            lens["flare"]["glow"]["fraction"] = json!(0.0);
            lens["flare"]["veil"]["fraction"] = json!(0.0);
            lens["flare"]["starburst"]["fraction"] = json!(0.0);
            lens["flare"]["aperture"] =
                json!({ "blades": 7, "rotation_deg": 10.0, "roundness": 0.5 });
            lens["flare"]["ghosts"] = json!([
                ghost(-0.9, 0.0, 0.8, 4.0, fraction, 0.30, 450.0),
                ghost(0.7, 0.6, 0.45, 0.9, 2.0 * fraction, 0.12, 600.0),
                ghost(1.6, 0.0, 0.9, 4.0, 3.0 * fraction, 0.05, 520.0),
                ghost(0.3, 0.2, 0.15, 4.0, fraction, 0.03, 500.0),
            ]);
        },
        &|att| {
            att["streak"]["fraction"] = json!(0.0);
            att["ghosts"] = json!([ghost(-0.5, 0.0, 0.7, 4.0, fraction, 0.2, 470.0)]);
        },
    );
    let library = written.and_then(|()| {
        crate::LensLibrary::from_dir(&dir).map_err(|e| io::Error::other(e.to_string()))
    });
    let _ = std::fs::remove_dir_all(&dir);
    library
}

/// Rewrites `SHA256SUMS` to list every file of the folder (except itself) with its current digest.
///
/// # Errors
///
/// Any I/O error.
pub fn write_sums(dir: &Path) -> io::Result<()> {
    fn walk(dir: &Path, rel: &str, out: &mut Vec<String>) -> io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let name = entry.file_name().to_string_lossy().into_owned();
            let path = if rel.is_empty() {
                name.clone()
            } else {
                format!("{rel}/{name}")
            };
            if entry.file_type()?.is_dir() {
                walk(&entry.path(), &path, out)?;
            } else if path != "SHA256SUMS" {
                out.push(path);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(dir, "", &mut files)?;
    files.sort();
    let mut text = String::new();
    for f in files {
        let digest = hex::encode(Sha256::digest(std::fs::read(dir.join(&f))?));
        let _ = writeln!(text, "{digest}  {f}");
    }
    std::fs::write(dir.join("SHA256SUMS"), text)
}
