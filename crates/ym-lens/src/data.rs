//! The lens data of `data/lenses/` (`docs/21` section 6): reading, checking and the types the passes use.
//!
//! The folder holds `index.json`, `anamorphic.json`, `SHA256SUMS`, and per lens `<id>/lens.json` and
//! `<id>/tables.safetensors`. Everything is checked when it is loaded: shapes, finite numbers, slices that
//! sum to 1, monotone radii, `T(0) = 1`, the provenance statement and the SHA-256 of every file. A damaged
//! folder is a [`LensError`], never a panic.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use safetensors::{Dtype, SafeTensors};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

use crate::amounts::LensKind;
use crate::consts::{
    GRID_SAMPLES, PROVENANCE_STATEMENT, PSF_SIZE, PSF_WAVELENGTHS, RADII, RIM_SAMPLES, SPIKE_SIZE,
    STARBURST_MAX_RADIUS, UNIT_SUM_TOLERANCE,
};
use crate::error::LensError;

/// What the menu shows about a lens (`ym lenses list`, the `lens_list` command).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LensInfo {
    /// The lens.
    pub kind: LensKind,
    /// The name shown in the menu.
    pub name: String,
    /// The one-line description shown under the name.
    pub line: String,
    /// Flares (0 to 100) the card moves to when this lens is chosen.
    pub start_flares: f64,
    /// Edges (0 to 100) the card moves to when this lens is chosen.
    pub start_edges: f64,
    /// The data version, raised whenever the lens's files change.
    pub version: u32,
    /// The design the lens was computed from (`provenance.design.source`).
    pub design_source: String,
}

/// The design a lens was computed from (`provenance.design`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Design {
    /// The kind of design in generic words ("double Gauss").
    #[serde(rename = "type")]
    pub kind: String,
    /// Patent number or publication.
    pub source: String,
    /// Year of the source.
    pub year: i32,
    /// Focal length of the design in millimetres.
    pub focal_mm: f64,
    /// f-number of the design.
    pub f_number: f64,
    /// The frame the design covers ("36 x 24 mm").
    pub format: String,
}

/// A lens's provenance block (`lens.json`, `provenance`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct LensProvenance {
    /// Exactly [`PROVENANCE_STATEMENT`].
    pub statement: String,
    /// The design.
    pub design: Design,
    /// The coatings, as text (a list of sentences is joined with "; ").
    #[serde(deserialize_with = "text_or_list")]
    pub coatings: String,
    /// The date the data were generated.
    pub generated: String,
    /// One sentence on how the data were computed.
    pub method: String,
}

/// The field curves of a lens (`lens.json`, `field`), each at the nine radii of [`FieldData::radii`].
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct FieldData {
    /// The nine field radii in half-diagonals, 0 to 1, strictly increasing.
    pub radii: Vec<f64>,
    /// `w(rho)`: the spread function tables span `+-w` half-diagonals.
    pub psf_half_width: Vec<f64>,
    /// `T(rho)`: relative illumination, `T(0) = 1`, read linearly between radii.
    pub relative_illumination: Vec<f64>,
    /// The lens's radial distortion.
    pub distortion: Distortion,
    /// `sigma(rho)`: the swirl profile (0 at the centre).
    pub swirl: Vec<f64>,
    /// `c(rho)`: the cat's-eye offset in ball radii (0: none).
    pub cats_eye: Vec<f64>,
}

/// The radial distortion `r' = r (1 + e (k1 r^2 + k2 r^4))` of a lens.
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Distortion {
    /// Coefficient of `r^2` (the spec's `delta_1`).
    pub k1: f64,
    /// Coefficient of `r^4` (the spec's `delta_2`).
    pub k2: f64,
}

/// The glow of a lens (`flare.glow`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Glow {
    /// Fraction of the light at Flares 100.
    pub fraction: f64,
    /// Radius of the core disc, in half-diagonals.
    pub core_radius: f64,
    /// Scale of the tail, in half-diagonals.
    pub r0: f64,
    /// The tail falls as `(1 + r / r0)^-power`.
    pub power: f64,
    /// The tail ends at this radius, in half-diagonals.
    pub reach: f64,
    /// Per PSF wavelength, the factor on `r0`.
    pub chroma: Vec<f64>,
}

/// The veiling glare of a lens (`flare.veil`).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Veil {
    /// Fraction of the light at Flares 100.
    pub fraction: f64,
    /// Radius of the nearly flat kernel, in half-diagonals.
    pub radius: f64,
}

/// The aperture shape of a lens (`flare.aperture`).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Aperture {
    /// 0 for a round aperture, else the blade count.
    pub blades: u32,
    /// Rotation of the blades in degrees.
    pub rotation_deg: f64,
    /// 0 (straight blades) to 1 (round).
    pub roundness: f64,
}

/// One ghost (`flare.ghosts`, and the attachment's oval ghosts).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct GhostData {
    /// Where it lies on the line from the source through the centre (negative: the other side).
    pub m: f64,
    /// Centre of the clipping circle on that line, as a multiple of the source's offset.
    pub n: f64,
    /// Radius in half-diagonals.
    pub s: f64,
    /// Radius of the clipping circle in half-diagonals.
    pub clip_radius: f64,
    /// Fraction of the light at Flares 100.
    pub fraction: f64,
    /// Edge width in ghost radii.
    pub soft: f64,
    /// The coating's reflectance on the 31-sample grid.
    pub coating: Vec<f64>,
}

/// The starburst of a lens (`flare.starburst`).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Starburst {
    /// Fraction of the light at Flares 100.
    pub fraction: f64,
    /// Radius of the spike tables in half-diagonals (at most 0.08).
    pub radius: f64,
    /// The f-number at which it reaches its full strength.
    pub f_number_full: f64,
}

/// The flare part of a lens (`lens.json`, `flare`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct FlareData {
    /// Glow.
    pub glow: Glow,
    /// Veiling glare.
    pub veil: Veil,
    /// The aperture.
    pub aperture: Aperture,
    /// The ghosts.
    pub ghosts: Vec<GhostData>,
    /// The starburst.
    pub starburst: Starburst,
}

#[derive(Clone, Copy, Debug, Deserialize)]
struct Start {
    flares: f64,
    edges: f64,
}

#[derive(Debug, Deserialize)]
struct Bokeh {
    // `docs/21` calls the table `bokeh.rim` in the file format and `bokeh.edge` in the stage text; both read.
    #[serde(alias = "edge")]
    rim: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct RawLens {
    id: String,
    version: u32,
    name: String,
    line: String,
    start: Start,
    provenance: LensProvenance,
    field: FieldData,
    flare: FlareData,
    bokeh: Bokeh,
}

/// One lens: its description, curves and tables, all checked.
#[derive(Clone, Debug)]
pub struct LensData {
    /// The menu entry.
    pub info: LensInfo,
    /// Where the data come from.
    pub provenance: LensProvenance,
    /// The field curves.
    pub field: FieldData,
    /// A ball's brightness against its normalised radius, 16 samples, mean 1 (`bokeh.rim`).
    pub rim: Vec<f64>,
    /// The flare parameters.
    pub flare: FlareData,
    psf: Vec<f32>,
    spikes: Vec<f32>,
}

impl LensData {
    /// The spread function at field radius index `radius` (0 to 8) and PSF wavelength index `wavelength`
    /// (0 to 7): 33 x 33 samples, row-major, x radial (outwards) and y tangential (anticlockwise). It sums
    /// to 1 and spans `+-w(rho)` (`docs/21` section 6, "Grid of the tables").
    ///
    /// # Panics
    ///
    /// Panics when an index is out of range.
    #[must_use]
    pub fn psf(&self, radius: usize, wavelength: usize) -> &[f32] {
        assert!(radius < RADII && wavelength < PSF_WAVELENGTHS);
        let n = PSF_SIZE * PSF_SIZE;
        let at = (radius * PSF_WAVELENGTHS + wavelength) * n;
        &self.psf[at..at + n]
    }

    /// The starburst table at PSF wavelength index `wavelength`: 129 x 129 samples over
    /// `+-flare.starburst.radius`, summing to 1.
    ///
    /// # Panics
    ///
    /// Panics when the index is out of range.
    #[must_use]
    pub fn spikes(&self, wavelength: usize) -> &[f32] {
        assert!(wavelength < PSF_WAVELENGTHS);
        let n = SPIKE_SIZE * SPIKE_SIZE;
        &self.spikes[wavelength * n..(wavelength + 1) * n]
    }

    /// The largest swirl `sigma(rho)`.
    #[must_use]
    pub fn swirl_max(&self) -> f64 {
        self.field.swirl.iter().copied().fold(0.0, f64::max)
    }

    /// The largest cat's-eye offset `c(rho)`.
    #[must_use]
    pub fn cats_eye_max(&self) -> f64 {
        self.field.cats_eye.iter().copied().fold(0.0, f64::max)
    }

    /// The largest `w(rho)`.
    #[must_use]
    pub fn psf_half_width_max(&self) -> f64 {
        self.field
            .psf_half_width
            .iter()
            .copied()
            .fold(0.0, f64::max)
    }

    /// The fractions of light at Flares 100: glow, veil, the ghosts and the starburst.
    #[must_use]
    pub fn flare_fraction(&self) -> f64 {
        self.flare.glow.fraction
            + self.flare.veil.fraction
            + self.flare.ghosts.iter().map(|g| g.fraction).sum::<f64>()
            + self.flare.starburst.fraction
    }
}

/// The attachment's provenance (only the statement is required).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct AttachmentProvenance {
    /// Exactly [`PROVENANCE_STATEMENT`].
    pub statement: String,
}

/// The attachment's bend constants (`bend`).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct Bend {
    /// `mu_0`: the edge stretch `x' = x (1 + mu x^2)` at Anamorphic 100 is `mu_0 (q_max - 1)`.
    pub mu0: f64,
    /// `beta_0`: the bowing `y' = y (1 - beta x^2)` at Anamorphic 100 is `beta_0 (q_max - 1)`.
    pub beta0: f64,
}

/// The attachment's field terms (`field`).
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
pub struct AttachmentField {
    /// `kappa`: the oval field radius stretches the axis by `1 + kappa (q - 1)`.
    pub kappa: f64,
    /// `a_s`: side softness `1 + a_s (q - 1) x^2`.
    pub side_softness: f64,
    /// `h`: the axis part of the channels' centroid offsets is multiplied by `1 + h (q - 1)`.
    pub fringe: f64,
}

/// The attachment's streak (`streak`).
#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct Streak {
    /// Fraction of the light at Flares 100 and Anamorphic 100.
    pub fraction: f64,
    /// Length of the tail at Anamorphic 100, in half-diagonals.
    pub reach: f64,
    /// Half-height-independent width of the sharp core, in half-diagonals.
    pub core: f64,
    /// The tail's power-law exponent.
    pub power: f64,
    /// The streak's tint as a reflectance on the 31-sample grid.
    pub tint: Vec<f64>,
}

#[derive(Debug, Deserialize)]
struct RawAttachment {
    id: String,
    version: u32,
    provenance: AttachmentProvenance,
    q_max: f64,
    bend: Bend,
    field: AttachmentField,
    ghost_stretch: f64,
    streak: Streak,
    ghosts: Vec<GhostData>,
}

/// The anamorphic attachment (`anamorphic.json`), checked.
#[derive(Clone, Debug)]
pub struct AttachmentData {
    /// The data version.
    pub version: u32,
    /// Provenance.
    pub provenance: AttachmentProvenance,
    /// `Q_max`: the bokeh is `q_max` times taller than wide at Anamorphic 100.
    pub q_max: f64,
    /// The bend.
    pub bend: Bend,
    /// The field terms.
    pub field: AttachmentField,
    /// `kappa_g`: ghosts widen along the axis by `1 + kappa_g (q - 1)`.
    pub ghost_stretch: f64,
    /// The streak.
    pub streak: Streak,
    /// The attachment's own oval ghosts.
    pub ghosts: Vec<GhostData>,
}

impl AttachmentData {
    /// The attachment's fractions of light at Flares 100 and Anamorphic 100: the streak and its ghosts.
    #[must_use]
    pub fn flare_fraction(&self) -> f64 {
        self.streak.fraction + self.ghosts.iter().map(|g| g.fraction).sum::<f64>()
    }
}

#[derive(Debug, Deserialize)]
struct RawIndex {
    version: u32,
    lenses: Vec<String>,
    attachment: String,
}

/// The seven lenses and the attachment, loaded and checked once.
#[derive(Clone, Debug)]
pub struct LensLibrary {
    infos: Vec<LensInfo>,
    lenses: Vec<Arc<LensData>>,
    attachment: Arc<AttachmentData>,
}

impl LensLibrary {
    /// Loads `<data_root>/lenses/` (see [`LensLibrary::from_dir`]).
    ///
    /// # Errors
    ///
    /// The errors of [`LensLibrary::from_dir`].
    pub fn load(data_root: &Path) -> Result<Self, LensError> {
        Self::from_dir(&data_root.join("lenses"))
    }

    /// Loads the lenses folder itself: `index.json` and everything it names, checked against `SHA256SUMS`.
    ///
    /// # Errors
    ///
    /// [`LensError`] when a file is missing, damaged, does not match `SHA256SUMS`, or breaks a rule of
    /// `docs/21` section 6.
    pub fn from_dir(dir: &Path) -> Result<Self, LensError> {
        let mut files = Files::open(dir)?;
        let index: RawIndex = files.json("index.json")?;
        if index.version != 1 {
            return Err(LensError::invalid(
                "index.json",
                format!("version is {}, this reader reads 1", index.version),
            ));
        }
        let mut kinds = Vec::new();
        for id in &index.lenses {
            let kind: LensKind = id.parse()?;
            if kinds.contains(&kind) {
                return Err(LensError::invalid(
                    "index.json",
                    format!("lens {id:?} is listed twice"),
                ));
            }
            kinds.push(kind);
        }
        if let Some(missing) = LensKind::ALL.iter().find(|k| !kinds.contains(k)) {
            return Err(LensError::invalid(
                "index.json",
                format!("lens {:?} is missing; all seven are required", missing.as_str()),
            ));
        }
        if !is_plain_relative(&index.attachment) {
            return Err(LensError::invalid(
                "index.json",
                format!("attachment {:?} is not a relative file name", index.attachment),
            ));
        }

        let mut lenses = Vec::with_capacity(kinds.len());
        for kind in &kinds {
            lenses.push(Arc::new(load_lens(&mut files, *kind)?));
        }
        let attachment = load_attachment(&mut files, &index.attachment)?;

        for lens in &lenses {
            let total = lens.flare_fraction() + attachment.flare_fraction();
            if total >= 1.0 {
                return Err(LensError::invalid(
                    &format!("{}/lens.json", lens.info.kind),
                    format!(
                        "the fractions of light at Flares 100 and Anamorphic 100 add up to {total}, they must stay below 1"
                    ),
                ));
            }
        }
        files.check_unread()?;

        Ok(LensLibrary {
            infos: lenses.iter().map(|l| l.info.clone()).collect(),
            lenses,
            attachment: Arc::new(attachment),
        })
    }

    /// The lenses in index order (the order of the menu).
    #[must_use]
    pub fn lenses(&self) -> &[LensInfo] {
        &self.infos
    }

    /// One lens's data.
    #[must_use]
    pub fn get(&self, kind: LensKind) -> Option<&LensData> {
        self.lenses
            .iter()
            .find(|l| l.info.kind == kind)
            .map(Arc::as_ref)
    }

    /// The anamorphic attachment.
    #[must_use]
    pub fn attachment(&self) -> &AttachmentData {
        &self.attachment
    }

    /// A shared handle to one lens, for plans that outlive a borrow of the library.
    pub(crate) fn get_shared(&self, kind: LensKind) -> Option<Arc<LensData>> {
        self.lenses.iter().find(|l| l.info.kind == kind).cloned()
    }

    /// A shared handle to the attachment.
    pub(crate) fn attachment_shared(&self) -> Arc<AttachmentData> {
        Arc::clone(&self.attachment)
    }
}

// ---------------------------------------------------------------------------------------------------------
// Files and SHA256SUMS

/// The lenses folder with its `SHA256SUMS`: every read goes through the digest check.
struct Files {
    dir: PathBuf,
    sums: BTreeMap<String, String>,
    read: BTreeSet<String>,
}

impl Files {
    fn open(dir: &Path) -> Result<Self, LensError> {
        let path = dir.join("SHA256SUMS");
        let text = std::fs::read_to_string(&path).map_err(|source| LensError::Io { path, source })?;
        let mut sums = BTreeMap::new();
        for (n, line) in text.lines().enumerate() {
            let line = line.trim_end();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let bad = |why: &str| LensError::invalid("SHA256SUMS", format!("line {}: {why}", n + 1));
            let (hash, rest) = line
                .split_once(' ')
                .ok_or_else(|| bad("expected \"<sha256>  <file>\""))?;
            // `sha256sum` writes "<hash>  <file>" (text) or "<hash> *<file>" (binary).
            let name = rest.strip_prefix(' ').or_else(|| rest.strip_prefix('*')).unwrap_or(rest);
            if hash.len() != 64 || !hash.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(bad("the digest is not 64 hexadecimal digits"));
            }
            if !is_plain_relative(name) {
                return Err(bad("the file name is not a relative path inside the folder"));
            }
            if sums
                .insert(name.to_owned(), hash.to_ascii_lowercase())
                .is_some()
            {
                return Err(bad("a file is listed twice"));
            }
        }
        Ok(Files {
            dir: dir.to_path_buf(),
            sums,
            read: BTreeSet::new(),
        })
    }

    /// Reads `rel` and checks it against its line in `SHA256SUMS`.
    fn bytes(&mut self, rel: &str) -> Result<Vec<u8>, LensError> {
        let expected = self
            .sums
            .get(rel)
            .ok_or_else(|| LensError::MissingChecksum {
                file: rel.to_owned(),
            })?
            .clone();
        let path = self.dir.join(rel);
        let bytes = std::fs::read(&path).map_err(|source| LensError::Io { path, source })?;
        let actual = hex::encode(Sha256::digest(&bytes));
        if actual != expected {
            return Err(LensError::Checksum {
                file: rel.to_owned(),
                expected,
                actual,
            });
        }
        self.read.insert(rel.to_owned());
        Ok(bytes)
    }

    fn json<T: DeserializeOwned>(&mut self, rel: &str) -> Result<T, LensError> {
        let bytes = self.bytes(rel)?;
        serde_json::from_slice(&bytes).map_err(|source| LensError::Json {
            file: rel.to_owned(),
            source,
        })
    }

    /// Checks the files `SHA256SUMS` lists that the loader did not need (`README.md`).
    fn check_unread(&mut self) -> Result<(), LensError> {
        let rest: Vec<String> = self
            .sums
            .keys()
            .filter(|k| !self.read.contains(*k))
            .cloned()
            .collect();
        for rel in rest {
            self.bytes(&rel)?;
        }
        Ok(())
    }
}

/// True for a non-empty `/`-separated path of normal components (no `..`, no absolute path, no `\`).
fn is_plain_relative(name: &str) -> bool {
    !name.is_empty()
        && !name.contains('\\')
        && name
            .split('/')
            .all(|c| !c.is_empty() && c != "." && c != "..")
}

// ---------------------------------------------------------------------------------------------------------
// Checks

fn text_or_list<'de, D: Deserializer<'de>>(d: D) -> Result<String, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Text {
        One(String),
        Many(Vec<String>),
    }
    Ok(match Text::deserialize(d)? {
        Text::One(s) => s,
        Text::Many(v) => v.join("; "),
    })
}

/// Number checks that name the file and the field.
struct Check<'a> {
    file: &'a str,
}

impl Check<'_> {
    fn err(&self, what: &str, reason: impl std::fmt::Display) -> LensError {
        LensError::invalid(self.file, format!("{what}: {reason}"))
    }

    fn finite(&self, what: &str, v: f64) -> Result<f64, LensError> {
        if v.is_finite() {
            Ok(v)
        } else {
            Err(self.err(what, "must be a finite number"))
        }
    }

    fn at_least(&self, what: &str, v: f64, min: f64) -> Result<f64, LensError> {
        self.finite(what, v)?;
        if v < min {
            return Err(self.err(what, format!("must be at least {min}, found {v}")));
        }
        Ok(v)
    }

    fn greater(&self, what: &str, v: f64, min: f64) -> Result<f64, LensError> {
        self.finite(what, v)?;
        if v <= min {
            return Err(self.err(what, format!("must be greater than {min}, found {v}")));
        }
        Ok(v)
    }

    fn range(&self, what: &str, v: f64, lo: f64, hi: f64) -> Result<f64, LensError> {
        self.finite(what, v)?;
        if !(lo..=hi).contains(&v) {
            return Err(self.err(what, format!("must be from {lo} to {hi}, found {v}")));
        }
        Ok(v)
    }

    fn list(&self, what: &str, v: &[f64], len: usize) -> Result<(), LensError> {
        if v.len() != len {
            return Err(self.err(what, format!("needs {len} values, found {}", v.len())));
        }
        for (i, x) in v.iter().enumerate() {
            self.finite(&format!("{what}[{i}]"), *x)?;
        }
        Ok(())
    }

    fn non_negative_list(&self, what: &str, v: &[f64], len: usize) -> Result<(), LensError> {
        self.list(what, v, len)?;
        for (i, x) in v.iter().enumerate() {
            if *x < 0.0 {
                return Err(self.err(&format!("{what}[{i}]"), format!("must not be negative, found {x}")));
            }
        }
        Ok(())
    }

    fn statement(&self, s: &str) -> Result<(), LensError> {
        if s == PROVENANCE_STATEMENT {
            Ok(())
        } else {
            Err(self.err(
                "provenance.statement",
                format!("must be exactly {PROVENANCE_STATEMENT:?}, found {s:?}"),
            ))
        }
    }

    fn ghost(&self, what: &str, g: &GhostData) -> Result<(), LensError> {
        self.finite(&format!("{what}.m"), g.m)?;
        self.finite(&format!("{what}.n"), g.n)?;
        self.greater(&format!("{what}.s"), g.s, 0.0)?;
        self.greater(&format!("{what}.clip_radius"), g.clip_radius, 0.0)?;
        self.range(&format!("{what}.fraction"), g.fraction, 0.0, 1.0)?;
        self.at_least(&format!("{what}.soft"), g.soft, 0.0)?;
        self.non_negative_list(&format!("{what}.coating"), &g.coating, GRID_SAMPLES)
    }
}

fn load_lens(files: &mut Files, kind: LensKind) -> Result<LensData, LensError> {
    let json_name = format!("{kind}/lens.json");
    let raw: RawLens = files.json(&json_name)?;
    let c = Check { file: &json_name };

    if raw.id != kind.as_str() {
        return Err(c.err("id", format!("is {:?}, the folder is {:?}", raw.id, kind.as_str())));
    }
    if raw.version == 0 {
        return Err(c.err("version", "must be an integer from 1"));
    }
    if raw.name.trim().is_empty() || raw.line.trim().is_empty() {
        return Err(c.err("name/line", "must not be empty"));
    }
    c.range("start.flares", raw.start.flares, 0.0, 100.0)?;
    c.range("start.edges", raw.start.edges, 0.0, 100.0)?;

    c.statement(&raw.provenance.statement)?;
    let d = &raw.provenance.design;
    if d.kind.trim().is_empty() || d.source.trim().is_empty() || d.format.trim().is_empty() {
        return Err(c.err("provenance.design", "type, source and format must not be empty"));
    }
    c.greater("provenance.design.focal_mm", d.focal_mm, 0.0)?;
    c.greater("provenance.design.f_number", d.f_number, 0.0)?;

    // Field curves.
    let f = &raw.field;
    c.list("field.radii", &f.radii, RADII)?;
    if f.radii[0] != 0.0 || f.radii[RADII - 1] != 1.0 {
        return Err(c.err("field.radii", "must run from 0 to 1"));
    }
    if f.radii.windows(2).any(|w| w[1] <= w[0]) {
        return Err(c.err("field.radii", "must be strictly increasing"));
    }
    c.list("field.psf_half_width", &f.psf_half_width, RADII)?;
    if let Some(w) = f.psf_half_width.iter().find(|w| **w <= 0.0) {
        return Err(c.err("field.psf_half_width", format!("must be positive, found {w}")));
    }
    c.list("field.relative_illumination", &f.relative_illumination, RADII)?;
    if f.relative_illumination[0] != 1.0 {
        return Err(c.err("field.relative_illumination", "T(0) must be 1"));
    }
    for (i, t) in f.relative_illumination.iter().enumerate() {
        if *t <= 0.0 || *t > 1.0 {
            return Err(c.err(
                &format!("field.relative_illumination[{i}]"),
                format!("must be above 0 and at most 1, found {t}"),
            ));
        }
    }
    if f.relative_illumination.windows(2).any(|w| w[1] > w[0]) {
        return Err(c.err("field.relative_illumination", "must not rise towards the edge"));
    }
    c.finite("field.distortion.k1", f.distortion.k1)?;
    c.finite("field.distortion.k2", f.distortion.k2)?;
    c.non_negative_list("field.swirl", &f.swirl, RADII)?;
    if f.swirl[0] != 0.0 {
        return Err(c.err("field.swirl", "sigma(0) must be 0"));
    }
    c.non_negative_list("field.cats_eye", &f.cats_eye, RADII)?;

    // Bokeh rim.
    c.non_negative_list("bokeh.rim", &raw.bokeh.rim, RIM_SAMPLES)?;
    let mean = raw.bokeh.rim.iter().sum::<f64>() / RIM_SAMPLES as f64;
    if (mean - 1.0).abs() > UNIT_SUM_TOLERANCE {
        return Err(c.err("bokeh.rim", format!("must have mean 1, found {mean}")));
    }

    // Flare.
    let fl = &raw.flare;
    c.range("flare.glow.fraction", fl.glow.fraction, 0.0, 1.0)?;
    c.at_least("flare.glow.core_radius", fl.glow.core_radius, 0.0)?;
    c.greater("flare.glow.r0", fl.glow.r0, 0.0)?;
    c.greater("flare.glow.power", fl.glow.power, 0.0)?;
    c.greater("flare.glow.reach", fl.glow.reach, 0.0)?;
    c.list("flare.glow.chroma", &fl.glow.chroma, PSF_WAVELENGTHS)?;
    if let Some(x) = fl.glow.chroma.iter().find(|x| **x <= 0.0) {
        return Err(c.err("flare.glow.chroma", format!("must be positive, found {x}")));
    }
    c.range("flare.veil.fraction", fl.veil.fraction, 0.0, 1.0)?;
    c.greater("flare.veil.radius", fl.veil.radius, 0.0)?;
    if fl.aperture.blades == 1 || fl.aperture.blades == 2 {
        return Err(c.err("flare.aperture.blades", "must be 0 (round) or at least 3"));
    }
    c.finite("flare.aperture.rotation_deg", fl.aperture.rotation_deg)?;
    c.range("flare.aperture.roundness", fl.aperture.roundness, 0.0, 1.0)?;
    for (i, g) in fl.ghosts.iter().enumerate() {
        c.ghost(&format!("flare.ghosts[{i}]"), g)?;
    }
    c.range("flare.starburst.fraction", fl.starburst.fraction, 0.0, 1.0)?;
    c.range(
        "flare.starburst.radius",
        fl.starburst.radius,
        f64::MIN_POSITIVE,
        STARBURST_MAX_RADIUS,
    )?;
    c.greater("flare.starburst.f_number_full", fl.starburst.f_number_full, 0.0)?;

    // Tables.
    let tables_name = format!("{kind}/tables.safetensors");
    let bytes = files.bytes(&tables_name)?;
    let (psf, spikes) = read_tables(&tables_name, &bytes)?;

    let info = LensInfo {
        kind,
        name: raw.name,
        line: raw.line,
        start_flares: raw.start.flares,
        start_edges: raw.start.edges,
        version: raw.version,
        design_source: raw.provenance.design.source.clone(),
    };
    Ok(LensData {
        info,
        provenance: raw.provenance,
        field: raw.field,
        rim: raw.bokeh.rim,
        flare: raw.flare,
        psf,
        spikes,
    })
}

fn load_attachment(files: &mut Files, name: &str) -> Result<AttachmentData, LensError> {
    let raw: RawAttachment = files.json(name)?;
    let c = Check { file: name };
    if raw.id != "anamorphic" {
        return Err(c.err("id", format!("is {:?}, expected \"anamorphic\"", raw.id)));
    }
    if raw.version == 0 {
        return Err(c.err("version", "must be an integer from 1"));
    }
    c.statement(&raw.provenance.statement)?;
    c.range("q_max", raw.q_max, 1.0 + f64::EPSILON, 4.0)?;
    c.range("bend.mu0", raw.bend.mu0, -0.5, 0.5)?;
    c.range("bend.beta0", raw.bend.beta0, -0.5, 0.5)?;
    c.at_least("field.kappa", raw.field.kappa, 0.0)?;
    if raw.field.kappa * (raw.q_max - 1.0) > 1.0 {
        return Err(c.err("field.kappa", "kappa (q_max - 1) must be at most 1"));
    }
    c.at_least("field.side_softness", raw.field.side_softness, 0.0)?;
    c.at_least("field.fringe", raw.field.fringe, 0.0)?;
    c.at_least("ghost_stretch", raw.ghost_stretch, 0.0)?;
    c.range("streak.fraction", raw.streak.fraction, 0.0, 1.0)?;
    c.greater("streak.reach", raw.streak.reach, 0.0)?;
    c.greater("streak.core", raw.streak.core, 0.0)?;
    c.greater("streak.power", raw.streak.power, 0.0)?;
    c.non_negative_list("streak.tint", &raw.streak.tint, GRID_SAMPLES)?;
    for (i, g) in raw.ghosts.iter().enumerate() {
        c.ghost(&format!("ghosts[{i}]"), g)?;
    }
    Ok(AttachmentData {
        version: raw.version,
        provenance: raw.provenance,
        q_max: raw.q_max,
        bend: raw.bend,
        field: raw.field,
        ghost_stretch: raw.ghost_stretch,
        streak: raw.streak,
        ghosts: raw.ghosts,
    })
}

/// Reads `psf` and `spikes` from a `tables.safetensors` file and checks them.
fn read_tables(file: &str, bytes: &[u8]) -> Result<(Vec<f32>, Vec<f32>), LensError> {
    let st = SafeTensors::deserialize(bytes).map_err(|e| LensError::Safetensors {
        file: file.to_owned(),
        message: e.to_string(),
    })?;
    let psf = read_tensor(
        file,
        &st,
        "psf",
        &[RADII, PSF_WAVELENGTHS, PSF_SIZE, PSF_SIZE],
    )?;
    let spikes = read_tensor(file, &st, "spikes", &[PSF_WAVELENGTHS, SPIKE_SIZE, SPIKE_SIZE])?;
    Ok((psf, spikes))
}

/// One f32 tensor of the given shape whose slices over the last two axes are non-negative and sum to 1.
fn read_tensor(
    file: &str,
    st: &SafeTensors<'_>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>, LensError> {
    let view = st.tensor(name).map_err(|e| LensError::Safetensors {
        file: file.to_owned(),
        message: format!("tensor {name:?}: {e}"),
    })?;
    if view.dtype() != Dtype::F32 {
        return Err(LensError::invalid(
            file,
            format!("tensor {name:?} is {:?}, expected F32", view.dtype()),
        ));
    }
    if view.shape() != shape {
        return Err(LensError::invalid(
            file,
            format!("tensor {name:?} has shape {:?}, expected {shape:?}", view.shape()),
        ));
    }
    let data: Vec<f32> = view
        .data()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    let slice = shape[shape.len() - 1] * shape[shape.len() - 2];
    if data.len() != shape.iter().product::<usize>() {
        return Err(LensError::invalid(file, format!("tensor {name:?} is truncated")));
    }
    for (i, s) in data.chunks_exact(slice).enumerate() {
        let mut sum = 0.0f64;
        for v in s {
            if !v.is_finite() || *v < 0.0 {
                return Err(LensError::invalid(
                    file,
                    format!("tensor {name:?} slice {i} holds a negative or non-finite sample"),
                ));
            }
            sum += f64::from(*v);
        }
        if (sum - 1.0).abs() > UNIT_SUM_TOLERANCE {
            return Err(LensError::invalid(
                file,
                format!("tensor {name:?} slice {i} sums to {sum}, it must sum to 1"),
            ));
        }
    }
    Ok(data)
}
