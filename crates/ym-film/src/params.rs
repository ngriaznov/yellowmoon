//! The node's settings, and our own parameter files in `data/film/params/`.
//!
//! The parameter files (`couplers.json`, `enlarger.json`, `kg3.json`) are Yellowmoon's own work, Apache-2.0.
//! The numbers are chosen from the literature and documented in the files; none is taken from another
//! film-simulation program (`docs/24` section 3.1).

use std::path::Path;

use serde::Deserialize;
use ym_color::display::OutputTarget;

use crate::error::FilmError;
use crate::math::{falling_edge, rising_edge};
use crate::{N_LAMBDA, Spectrum, lambda_nm};

/// The output the print is scanned into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Output {
    /// Rec.709 primaries (an sRGB export).
    Srgb,
    /// Display P3 primaries (the viewport and a P3 export).
    DisplayP3,
    /// Rec.2020 primaries (a Rec.2020 export): the print is scanned and returned in its own gamut.
    Rec2020,
}

impl Output {
    /// The cine rendering's target for this output.
    #[must_use]
    pub fn target(self) -> OutputTarget {
        match self {
            Self::Srgb => OutputTarget::Srgb,
            Self::DisplayP3 => OutputTarget::DisplayP3,
            Self::Rec2020 => OutputTarget::Rec2020,
        }
    }
}

/// The settings of a film chain. [`FilmParams::default`] is the node's defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct FilmParams {
    /// Film exposure, in stops (EV). 0 is the Input's exposure.
    pub exposure_ev: f64,
    /// How much of the DIR couplers acts: 1 is the profile's curves on a neutral ramp, 0 turns them off.
    pub coupler_amount: f64,
    /// The print's exposure, in stops. 0 is the placement of the brightness match.
    pub print_exposure_stops: f64,
    /// Enlarger yellow filter, a shift from neutral in Kodak CC units (100 units = density 1.0).
    pub light_yellow_cc: f64,
    /// Enlarger magenta filter shift, in CC units.
    pub light_magenta_cc: f64,
    /// Enlarger cyan filter shift, in CC units.
    pub light_cyan_cc: f64,
    /// The output primaries.
    pub output: Output,
    /// Whether the profile's adaptation window is applied (it is, by default).
    pub use_window: bool,
}

impl Default for FilmParams {
    fn default() -> Self {
        Self {
            exposure_ev: 0.0,
            coupler_amount: 1.0,
            print_exposure_stops: 0.0,
            light_yellow_cc: 0.0,
            light_magenta_cc: 0.0,
            light_cyan_cc: 0.0,
            output: Output::DisplayP3,
            use_window: true,
        }
    }
}

#[derive(Deserialize)]
struct CouplersFile {
    matrix: [[f64; 3]; 3],
}

#[derive(Deserialize)]
struct EdgeFilter {
    /// Where the filter stops: the centre of its edge, in nm.
    #[serde(default)]
    edge_nm: f64,
    #[serde(default)]
    width_nm: f64,
    /// For the band-stop (magenta): the long-wavelength edge.
    #[serde(default)]
    edge2_nm: f64,
    #[serde(default)]
    width2_nm: f64,
}

#[derive(Deserialize)]
struct FiltersFile {
    lamp_k: f64,
    yellow: EdgeFilter,
    magenta: EdgeFilter,
    cyan: EdgeFilter,
}

#[derive(Deserialize)]
struct Kg3File {
    #[serde(default)]
    points: Vec<[f64; 2]>,
    #[serde(default)]
    peak: f64,
    #[serde(default)]
    edge_nm: f64,
    #[serde(default)]
    width_nm: f64,
}

/// The enlarger's optical parts: lamp temperature, glass and the three filters' unit densities.
#[derive(Debug, Clone)]
pub struct EnlargerParts {
    /// The lamp's colour temperature, in kelvin.
    pub lamp_k: f64,
    /// Transmittance of the heat-absorbing glass per sample.
    pub glass: Spectrum,
    /// Density of the yellow, magenta and cyan filters at 100 CC units (1.0), per sample.
    pub filter_density: [Spectrum; 3],
}

fn read_json<T: for<'de> Deserialize<'de>>(dir: &Path, name: &str) -> Result<T, FilmError> {
    let path = dir.join(name);
    let bytes = std::fs::read(&path).map_err(|source| FilmError::Io {
        path: path.clone(),
        source,
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|e| FilmError::data(format!("params/{name}"), format!("it does not parse ({e})")))
}

/// Reads the coupler matrix of `<params>/couplers.json`: row `i` is the layer inhibited, column `j` the layer
/// whose density inhibits it, in log10 exposure per unit of density.
///
/// # Errors
///
/// [`FilmError::Io`] / [`FilmError::Data`]; a negative or non-finite entry is refused.
pub fn load_couplers(params: &Path) -> Result<[[f64; 3]; 3], FilmError> {
    let f: CouplersFile = read_json(params, "couplers.json")?;
    for r in &f.matrix {
        for v in r {
            if !v.is_finite() || !(0.0..=1.0).contains(v) {
                return Err(FilmError::data(
                    "params/couplers.json",
                    format!("the matrix entry {v} is not between 0 and 1"),
                ));
            }
        }
    }
    Ok(f.matrix)
}

/// The enlarger as numbers: the lamp's temperature, the heat-absorbing glass's low-pass edge and the three filters'
/// edges. The shipped `enlarger.json` plus `kg3.json` are read into [`EnlargerParts`] directly; this form exists for
/// the fit of `tests/oracle_fit.rs` (F-06b), which found the hand-chosen numbers already as good as a fit to the oracle.
#[derive(Debug, Clone, Deserialize)]
pub struct EnlargerModel {
    /// The lamp's colour temperature, in kelvin.
    pub lamp_k: f64,
    /// The glass: transmittance `peak * 1/2 erfc((l - edge_nm) / width_nm)`.
    pub glass: GlassEdge,
    /// Yellow: dense below its edge.
    pub yellow: BandEdge,
    /// Magenta: dense between its two edges.
    pub magenta: BandEdge,
    /// Cyan: dense above its edge.
    pub cyan: BandEdge,
}

/// The glass's low-pass edge.
#[derive(Debug, Clone, Copy, Deserialize)]
pub struct GlassEdge {
    /// Transmittance in the pass band.
    pub peak: f64,
    /// Centre of the edge, in nm.
    pub edge_nm: f64,
    /// Width of the edge, in nm.
    pub width_nm: f64,
}

/// One filter's edges, in nm.
#[derive(Debug, Clone, Copy, Default, Deserialize)]
pub struct BandEdge {
    /// Centre of the (first) edge.
    #[serde(default)]
    pub edge_nm: f64,
    /// Width of the (first) edge.
    #[serde(default)]
    pub width_nm: f64,
    /// For the band-stop (magenta): the long-wavelength edge.
    #[serde(default)]
    pub edge2_nm: f64,
    /// Width of the second edge.
    #[serde(default)]
    pub width2_nm: f64,
}

impl EnlargerModel {
    /// The optical parts as spectra.
    ///
    /// # Errors
    ///
    /// [`FilmError::Data`] when the lamp temperature or an edge is not a physical number.
    pub fn parts(&self, file: &str) -> Result<EnlargerParts, FilmError> {
        let bad = |what: String| FilmError::data(file, what);
        if !(1000.0..=10000.0).contains(&self.lamp_k) {
            return Err(bad(format!(
                "lamp_k = {} is not a lamp temperature",
                self.lamp_k
            )));
        }
        let g = self.glass;
        if g.peak <= 0.0 || g.width_nm <= 0.0 {
            return Err(bad("the glass has no peak or no edge width".into()));
        }
        for (name, e) in [
            ("yellow", self.yellow),
            ("magenta", self.magenta),
            ("cyan", self.cyan),
        ] {
            if e.width_nm <= 0.0 || (name == "magenta" && e.width2_nm <= 0.0) {
                return Err(bad(format!(
                    "the {name} filter has an edge width that is not positive"
                )));
            }
        }
        let mut glass = [0.0; N_LAMBDA];
        let mut dens = [[0.0; N_LAMBDA]; 3];
        for i in 0..N_LAMBDA {
            let l = lambda_nm(i);
            glass[i] = g.peak * falling_edge(l, g.edge_nm, g.width_nm);
            // Yellow stops blue: dense below its edge. Magenta stops green: dense between its two edges. Cyan stops
            // red: dense above its edge.
            dens[0][i] = falling_edge(l, self.yellow.edge_nm, self.yellow.width_nm);
            dens[1][i] = rising_edge(l, self.magenta.edge_nm, self.magenta.width_nm)
                * falling_edge(l, self.magenta.edge2_nm, self.magenta.width2_nm);
            dens[2][i] = rising_edge(l, self.cyan.edge_nm, self.cyan.width_nm);
        }
        Ok(EnlargerParts {
            lamp_k: self.lamp_k,
            glass,
            filter_density: dens,
        })
    }
}

/// Reads the lamp, the glass and the filters (`enlarger.json` and `kg3.json`).
///
/// # Errors
///
/// [`FilmError::Io`] / [`FilmError::Data`].
pub fn load_enlarger_parts(params: &Path) -> Result<EnlargerParts, FilmError> {
    let f: FiltersFile = read_json(params, "enlarger.json")?;
    let g: Kg3File = read_json(params, "kg3.json")?;
    if !(1000.0..=10000.0).contains(&f.lamp_k) {
        return Err(FilmError::data(
            "params/enlarger.json",
            format!("lamp_k = {} is not a lamp temperature", f.lamp_k),
        ));
    }
    let mut glass = [0.0; N_LAMBDA];
    for (i, t) in glass.iter_mut().enumerate() {
        let l = lambda_nm(i);
        *t = if g.points.len() >= 2 {
            interp(&g.points, l)
        } else {
            if g.peak <= 0.0 || g.width_nm <= 0.0 {
                return Err(FilmError::data(
                    "params/kg3.json",
                    "it has neither points nor an edge (peak, edge_nm, width_nm)",
                ));
            }
            g.peak * falling_edge(l, g.edge_nm, g.width_nm)
        };
    }
    let mut dens = [[0.0; N_LAMBDA]; 3];
    for i in 0..N_LAMBDA {
        let l = lambda_nm(i);
        dens[0][i] = falling_edge(l, f.yellow.edge_nm, f.yellow.width_nm);
        dens[1][i] = rising_edge(l, f.magenta.edge_nm, f.magenta.width_nm)
            * falling_edge(l, f.magenta.edge2_nm, f.magenta.width2_nm);
        dens[2][i] = rising_edge(l, f.cyan.edge_nm, f.cyan.width_nm);
    }
    Ok(EnlargerParts {
        lamp_k: f.lamp_k,
        glass,
        filter_density: dens,
    })
}

fn interp(points: &[[f64; 2]], x: f64) -> f64 {
    if x <= points[0][0] {
        return points[0][1];
    }
    for w in points.windows(2) {
        if x <= w[1][0] {
            let t = (x - w[0][0]) / (w[1][0] - w[0][0]);
            return w[0][1] + t * (w[1][1] - w[0][1]);
        }
    }
    points[points.len() - 1][1]
}
