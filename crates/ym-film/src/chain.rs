//! The whole chain: stages 1 to 10, the neutral solve and the brightness match.
//!
//! Written from the references of the stage modules, and from Hunt, *The Reproduction of Colour* (colour printing
//! balance: a grey is printed neutral by the filters; chapter 18 on the exposure of prints).
//!
//! **Calibration at build time.** For a negative on a paper:
//! * the *neutral solve* finds the yellow and magenta filter values (cyan fixed at 0) that make an 18 % grey print
//!   with CIELAB (D50) `a* = b* = 0` after the scan; the node's lights are shifts from there, so 0 is neutral;
//! * the exposure is normalised so that the grey's red layer lands at `shift` on the paper's axis, and the
//!   *brightness match* finds `shift` so that scene 0.18 comes out at the cine rendering's grey (display 0.111);
//! * the two are solved together (alternating a few times).
//!
//! `docs/24` section 3.6 asks for a per-paper constant; the brightness depends a little on the film too (its
//! mask and dyes), so the constant is solved per film and paper pair, at build time, and is a number of the
//! chain, never of a photo. For a slide film there is no print: a per-film gain on the exposure does the match.
//! A coupler amount other than the default is part of the calibrated chain, so changing it keeps the grey where
//! it was.

use std::path::Path;
use std::sync::Arc;

use ym_color::display::cine::CineRendering;
use ym_color::gamut::LinearAwg4;
use ym_color::logc::{logc4_decode_rgb, logc4_encode_rgb};

use crate::Vec3;
use crate::cie::Cie;
use crate::develop::Develop;
use crate::enlarger::Enlarger;
use crate::error::FilmError;
use crate::exposure::{ExposureStage, reference_spectrum};
use crate::params::{FilmParams, load_couplers, load_enlarger_parts};
use crate::profile::{Names, Profile, StockKind};
use crate::scan::{Scan, WhiteKind};
use crate::scene::{SceneLight, SceneStage};
use crate::spectra::Spectra;

const LOG10_2: f64 = std::f64::consts::LOG10_2;

/// The paper for a negative.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PaperChoice {
    /// The stock's own `target_print` (a still paper stands in for the cine print films).
    Matching,
    /// A paper by id.
    Id(String),
}

/// The paper that stands in where a negative's matching print is a cine print film, which stays hidden until its
/// projector lamp is sourced (`docs/24` section 6.4).
pub const FALLBACK_PAPER: &str = "kodak_portra_endura";

/// The values after stages 2, 4, 5, 7, 8 and 9.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Taps {
    /// After stage 2: `log10` raw exposure of the three layers.
    pub log_e_film: Vec3,
    /// After stage 4: density from the profile's curves.
    pub density_curves: Vec3,
    /// After stage 5: density with the couplers acting.
    pub density_film: Vec3,
    /// After stage 7: the paper's `log10` exposure (zero for a slide).
    pub log_e_print: Vec3,
    /// After stage 8: the paper's dye densities (zero for a slide).
    pub density_print: Vec3,
    /// After stage 9: the display colour, linear RGB in the output's primaries.
    pub display: Vec3,
}

struct Knobs {
    exposure_ev: f64,
    cc: Vec3,
    print_stops: f64,
}

/// A film chain: build once, call [`FilmChain::run`] per pixel.
pub struct FilmChain {
    film: Profile,
    paper: Option<Profile>,
    params: FilmParams,
    scene: SceneStage,
    exposure: ExposureStage,
    develop: Develop,
    enlarger: Option<Enlarger>,
    scan: Scan,
    cine: CineRendering,
    /// Neutral filtering (yellow, magenta) in CC units.
    neutral_cc: [f64; 2],
    /// The red layer's raw exposure from the grey at the neutral filtering.
    p_ref_red: f64,
    /// The brightness match: the grey's red layer lands here on the paper's axis (log10).
    print_shift: f64,
    /// Slides: the brightness match as a gain on the exposure (log10).
    film_gain: f64,
}

impl std::fmt::Debug for FilmChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FilmChain")
            .field("film", &self.film.id)
            .field("paper", &self.paper.as_ref().map(|p| p.id.as_str()))
            .field("neutral_cc", &self.neutral_cc)
            .field("print_shift", &self.print_shift)
            .field("film_gain", &self.film_gain)
            .finish_non_exhaustive()
    }
}

impl FilmChain {
    /// Builds the chain for `film_id` on `paper` with `params`, reading `<data_dir>/film/`.
    ///
    /// # Errors
    ///
    /// Any [`FilmError`]: a file that is missing or damaged, an unknown or wrong-kind stock, curves the couplers
    /// cannot undo, or a calibration that does not settle.
    pub fn new(
        data_dir: &Path,
        film_id: &str,
        paper: &PaperChoice,
        params: FilmParams,
        spectra: Arc<dyn Spectra>,
    ) -> Result<Self, FilmError> {
        let root = data_dir.join("film");
        let names = Names::load(&root.join("params"))?;
        let profiles = root.join("spektrafilm").join("profiles");
        let cie = Cie::load(&root.join("cie"))?;
        let film_name = names
            .get(film_id)
            .ok_or_else(|| FilmError::UnknownStock(film_id.to_string()))?;
        let is_slide = match film_name.kind {
            StockKind::ColourNegative | StockKind::CineNegative => false,
            StockKind::Slide => true,
            _ => {
                return Err(FilmError::WrongKind {
                    id: film_id.to_string(),
                    wanted: "a film",
                });
            }
        };
        let film = Profile::load(&profiles, film_id)?;
        let paper = if is_slide {
            None
        } else {
            let id = match paper {
                PaperChoice::Id(id) => id.clone(),
                PaperChoice::Matching => match film.target_print.as_deref() {
                    Some(t) if names.get(t).is_some_and(|s| s.kind == StockKind::Paper) => {
                        t.to_string()
                    }
                    _ => FALLBACK_PAPER.to_string(),
                },
            };
            let kind = names
                .get(&id)
                .ok_or_else(|| FilmError::UnknownStock(id.clone()))?
                .kind;
            if kind != StockKind::Paper && kind != StockKind::PrintFilm {
                return Err(FilmError::WrongKind {
                    id,
                    wanted: "a paper",
                });
            }
            let p = Profile::load(&profiles, &id)?;
            if p.viewing_illuminant != "D50" {
                return Err(FilmError::ViewingLight {
                    id,
                    light: p.viewing_illuminant.clone(),
                });
            }
            Some(p)
        };

        let reference = reference_spectrum(&film, &cie)?;
        let scene = SceneStage::new(&cie, &reference);
        let exposure = ExposureStage::new(&film, &cie, spectra, params.use_window)?;
        let develop = Develop::new(&film, load_couplers(&root.join("params"))?)?;
        let (enlarger, scan) = match &paper {
            Some(p) => {
                let parts = load_enlarger_parts(&root.join("params"))?;
                (
                    Some(Enlarger::new(&film, p, &parts)),
                    Scan::new(p, &cie, params.output, WhiteKind::PaperBase),
                )
            }
            None => (
                None,
                Scan::new(&film, &cie, params.output, WhiteKind::Light),
            ),
        };
        let cine = CineRendering::new(params.output.target());
        let mut chain = Self {
            film,
            paper,
            params,
            scene,
            exposure,
            develop,
            enlarger,
            scan,
            cine,
            neutral_cc: [0.0, 0.0],
            p_ref_red: 1.0,
            print_shift: 0.0,
            film_gain: 0.0,
        };
        chain.calibrate()?;
        Ok(chain)
    }

    /// The film's profile.
    #[must_use]
    pub fn film(&self) -> &Profile {
        &self.film
    }

    /// The paper's profile (none for a slide).
    #[must_use]
    pub fn paper(&self) -> Option<&Profile> {
        self.paper.as_ref()
    }

    /// The settings the chain was built with.
    #[must_use]
    pub fn params(&self) -> &FilmParams {
        &self.params
    }

    /// Stage 1.
    #[must_use]
    pub fn scene_stage(&self) -> &SceneStage {
        &self.scene
    }

    /// Stage 2.
    #[must_use]
    pub fn exposure_stage(&self) -> &ExposureStage {
        &self.exposure
    }

    /// Stages 4 and 5.
    #[must_use]
    pub fn develop_stage(&self) -> &Develop {
        &self.develop
    }

    /// Stages 7 and 8 (none for a slide).
    #[must_use]
    pub fn enlarger_stage(&self) -> Option<&Enlarger> {
        self.enlarger.as_ref()
    }

    /// Stage 9.
    #[must_use]
    pub fn scan_stage(&self) -> &Scan {
        &self.scan
    }

    /// The neutral filtering the solve found: yellow and magenta, in CC units (cyan is 0).
    #[must_use]
    pub fn neutral_filters_cc(&self) -> [f64; 2] {
        self.neutral_cc
    }

    /// The brightness match: where the grey's red layer lands on the paper's log exposure axis (0 for a slide).
    #[must_use]
    pub fn print_shift(&self) -> f64 {
        self.print_shift
    }

    /// The display grey that scene 0.18 must come out at: the cine rendering's own rendering of 0.18.
    #[must_use]
    pub fn target_grey(&self) -> f64 {
        self.cine.forward(LinearAwg4([0.18; 3]))[0]
    }

    fn knobs(&self) -> Knobs {
        Knobs {
            exposure_ev: self.params.exposure_ev,
            cc: [
                self.params.light_yellow_cc,
                self.params.light_magenta_cc,
                self.params.light_cyan_cc,
            ],
            print_stops: self.params.print_exposure_stops,
        }
    }

    /// Stages 1 to 9 with explicit knobs.
    fn forward(&self, logc4: Vec3, k: &Knobs) -> Taps {
        let light: SceneLight = self.scene.run(logc4);
        let gain = k.exposure_ev * LOG10_2 + self.film_gain;
        let log_e_film = self.exposure.log_exposure(&light, gain);
        let density_curves = self.develop.curves(log_e_film);
        let density_film = self.develop.couple(log_e_film, self.params.coupler_amount);
        match &self.enlarger {
            Some(enl) => {
                let cc = [
                    self.neutral_cc[0] + k.cc[0],
                    self.neutral_cc[1] + k.cc[1],
                    k.cc[2],
                ];
                let p = enl.light_on_paper(density_film, cc);
                let shift = self.print_shift + k.print_stops * LOG10_2;
                let mut log_e_print = [0.0; 3];
                for c in 0..3 {
                    log_e_print[c] = (p[c].max(1e-300) / self.p_ref_red).log10() + shift;
                }
                let density_print = enl.paper_density(log_e_print);
                let display = self.scan.display(density_print);
                Taps {
                    log_e_film,
                    density_curves,
                    density_film,
                    log_e_print,
                    density_print,
                    display,
                }
            }
            None => Taps {
                log_e_film,
                density_curves,
                density_film,
                log_e_print: [0.0; 3],
                density_print: [0.0; 3],
                display: self.scan.display(density_film),
            },
        }
    }

    /// Stage 10: a display colour back to LogC4 / AWG4 through the inverse of the cine rendering, the continuous
    /// one (`CineRendering::inverse_continuous`, `docs/24` section 3.8a): the pre-image is a continuous function
    /// of the display colour, which is what T3b stores.
    #[must_use]
    pub fn return_stage(&self, display: Vec3) -> Vec3 {
        logc4_encode_rgb(self.cine.inverse_continuous(display).scene.0)
    }

    /// The whole chain on one LogC4 / AWG4 pixel.
    #[must_use]
    pub fn run(&self, logc4: Vec3) -> Vec3 {
        self.return_stage(self.forward(logc4, &self.knobs()).display)
    }

    /// The whole chain, with the values after stages 2, 4, 5, 7, 8 and 9.
    #[must_use]
    pub fn run_tapped(&self, logc4: Vec3) -> (Vec3, Taps) {
        let taps = self.forward(logc4, &self.knobs());
        (self.return_stage(taps.display), taps)
    }

    /// Stages 1 to 9 only (the print's display colour).
    #[must_use]
    pub fn display_of(&self, logc4: Vec3) -> Vec3 {
        self.forward(logc4, &self.knobs()).display
    }

    fn calibration_error(&self, what: impl Into<String>) -> FilmError {
        FilmError::Calibration {
            film: self.film.id.clone(),
            paper: self
                .paper
                .as_ref()
                .map_or_else(|| "(none)".into(), |p| p.id.clone()),
            what: what.into(),
        }
    }

    fn grey_logc4() -> Vec3 {
        logc4_encode_rgb([0.18; 3])
    }

    /// The grey's display colour with the neutral filtering `(y, m)` and the shift `shift`.
    fn grey_display(&mut self, y: f64, m: f64, shift: f64) -> Vec3 {
        let zero = Knobs {
            exposure_ev: 0.0,
            cc: [0.0; 3],
            print_stops: 0.0,
        };
        self.neutral_cc = [y, m];
        self.print_shift = shift;
        // The reference: the red layer's exposure from the grey at this filtering.
        if let Some(enl) = &self.enlarger {
            let light = self.scene.run(Self::grey_logc4());
            let log_e = self.exposure.log_exposure(&light, 0.0);
            let d = self.develop.couple(log_e, self.params.coupler_amount);
            self.p_ref_red = enl.light_on_paper(d, [y, m, 0.0])[0];
        }
        self.forward(Self::grey_logc4(), &zero).display
    }

    fn grey_lum(&mut self, y: f64, m: f64, shift: f64) -> f64 {
        let d = self.grey_display(y, m, shift);
        self.scan.luminance(d)
    }

    fn calibrate(&mut self) -> Result<(), FilmError> {
        let target = self.target_grey();
        if self.enlarger.is_none() {
            // Slide: a gain on the exposure so that the grey lands on the target.
            let (mut lo, mut hi) = (-4.0_f64, 4.0_f64);
            let f = |s: &mut Self, g: f64| {
                s.film_gain = g;
                let d = s.grey_display(0.0, 0.0, 0.0);
                s.scan.luminance(d) - target
            };
            if f(self, lo) > 0.0 || f(self, hi) < 0.0 {
                return Err(self.calibration_error(
                    "the slide cannot reach the grey with an exposure gain of 4 stops either way",
                ));
            }
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if f(self, mid) < 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            self.film_gain = 0.5 * (lo + hi);
            let _ = self.grey_display(0.0, 0.0, 0.0);
            return Ok(());
        }
        let (mut y, mut m, mut shift) = (0.0_f64, 0.0_f64, 0.0_f64);
        for _outer in 0..8 {
            // Brightness: bisection on the shift (a larger shift darkens the print).
            let (mut lo, mut hi) = (-3.0_f64, 3.0_f64);
            let flo = self.grey_lum(y, m, lo) - target;
            let fhi = self.grey_lum(y, m, hi) - target;
            if flo < 0.0 || fhi > 0.0 {
                return Err(self.calibration_error(
                    "the grey cannot reach the target brightness within 3 stops",
                ));
            }
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                if self.grey_lum(y, m, mid) > target {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            shift = 0.5 * (lo + hi);

            // Neutral: damped Newton on (a*, b*) over (y, m).
            for _ in 0..40 {
                let lab = |s: &mut Self, y: f64, m: f64| {
                    let d = s.grey_display(y, m, shift);
                    let l = s.scan.lab_d50(d);
                    [l[1], l[2]]
                };
                let r = lab(self, y, m);
                let norm = r[0].abs() + r[1].abs();
                if norm < 1e-10 {
                    break;
                }
                let h = 0.5;
                let ry = lab(self, y + h, m);
                let rm = lab(self, y, m + h);
                let jac = [
                    [(ry[0] - r[0]) / h, (rm[0] - r[0]) / h],
                    [(ry[1] - r[1]) / h, (rm[1] - r[1]) / h],
                ];
                let Some(step) = crate::math::solve2(jac, [-r[0], -r[1]]) else {
                    return Err(
                        self.calibration_error("the filters do not change the grey's colour")
                    );
                };
                let len = step[0].hypot(step[1]);
                let cap = if len > 30.0 { 30.0 / len } else { 1.0 };
                let mut t = cap;
                let mut moved = false;
                for _ in 0..20 {
                    let (cy, cm) = (y + t * step[0], m + t * step[1]);
                    let rc = lab(self, cy, cm);
                    if rc[0].abs() + rc[1].abs() < norm {
                        y = cy;
                        m = cm;
                        moved = true;
                        break;
                    }
                    t *= 0.5;
                }
                if !moved {
                    break;
                }
            }
        }
        // Leave the chain at the final values and check them.
        let d = self.grey_display(y, m, shift);
        let lab = self.scan.lab_d50(d);
        let lum = self.scan.luminance(d);
        if lab[1].abs() > 1e-3 || lab[2].abs() > 1e-3 || (lum - target).abs() > 1e-6 {
            return Err(self.calibration_error(format!(
                "the grey ended at a* {:.4}, b* {:.4}, brightness {:.6} against {:.6}",
                lab[1], lab[2], lum, target
            )));
        }
        Ok(())
    }
}

/// Decodes a LogC4 triple to linear AWG4 (a convenience for callers and tests).
#[must_use]
pub fn logc4_to_linear(logc4: Vec3) -> Vec3 {
    logc4_decode_rgb(logc4)
}
