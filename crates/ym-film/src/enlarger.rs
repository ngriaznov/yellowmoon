//! Stages 7 and 8: the enlarger and the paper's curves.
//!
//! Written from Hunt, *The Reproduction of Colour*, 6th ed. (colour printing: a negative printed by a lamp
//! through colour-compensating filters on a three-layer paper), Giorgianni and Madden, *Digital Color
//! Management*, 2nd ed. (spectral transmission of a negative from its dye densities), Planck's law, the Kodak
//! colour-compensation filter convention (CC units; 100 units = density 1.0) and the paper data sheets the
//! profiles are read from.
//!
//! **Light on the paper.** The negative's spectral transmission is
//! `T(l) = 10^-(base(l) + sum_c D_c cd_c(l))` (`cd` the dyes' density per unit density). The paper's raw exposure
//! in layer `k` is `P_k = sum_l lamp(l) glass(l) 10^-(filters(l)) T(l) S_k(l)`, with the lamp a black body and
//! the filters' densities `cc/100 * shape(l)` (`data/film/params/enlarger.json`, `kg3.json`).
//!
//! **Exposure.** The caller normalises: the paper's log exposure is `log10(P_k / P_ref) + shift`, where `P_ref` is
//! the red layer's exposure from a mid-grey negative at the neutral filtering (so the red layer of grey lands at
//! `shift` on the paper's axis) and `shift` carries the brightness match and the print exposure. See
//! [`crate::chain`].
//!
//! **Paper curves.** The paper's `density_curves`, as monotone cubics, no couplers.

use crate::develop::MonotoneCubic;
use crate::math::{blackbody, pow10};
use crate::params::EnlargerParts;
use crate::profile::Profile;
use crate::{N_LAMBDA, Spectrum, Vec3};

/// Stages 7 and 8 for one film and paper.
#[derive(Debug, Clone)]
pub struct Enlarger {
    lamp: Spectrum,
    filter_density: [Spectrum; 3],
    paper_sens: [Spectrum; 3],
    neg_base: Spectrum,
    neg_dye: Vec<Vec3>,
    curves: [MonotoneCubic; 3],
}

impl Enlarger {
    /// Builds the enlarger for `negative` printed on `paper`.
    #[must_use]
    pub fn new(negative: &Profile, paper: &Profile, parts: &EnlargerParts) -> Self {
        let bb = blackbody(parts.lamp_k);
        let mut lamp = [0.0; N_LAMBDA];
        for i in 0..N_LAMBDA {
            lamp[i] = bb[i] * parts.glass[i];
        }
        let sens = paper.sensitivity();
        let mut paper_sens = [[0.0; N_LAMBDA]; 3];
        for i in 0..N_LAMBDA {
            for c in 0..3 {
                paper_sens[c][i] = sens[i][c];
            }
        }
        let mk = |c: usize| {
            let (x, y) = paper.curve(c);
            MonotoneCubic::new(x, y)
        };
        Self {
            lamp,
            filter_density: parts.filter_density,
            paper_sens,
            neg_base: negative.base_density,
            neg_dye: negative.channel_density.clone(),
            curves: [mk(0), mk(1), mk(2)],
        }
    }

    /// The paper's raw exposure `P_k` (linear, arbitrary scale) from the negative's dye densities `d_neg`
    /// and the filters `cc` (yellow, magenta, cyan) in CC units.
    #[must_use]
    pub fn light_on_paper(&self, d_neg: Vec3, cc: Vec3) -> Vec3 {
        let mut p = [0.0; 3];
        for i in 0..N_LAMBDA {
            let dye = &self.neg_dye[i];
            let mut dens =
                self.neg_base[i] + d_neg[0] * dye[0] + d_neg[1] * dye[1] + d_neg[2] * dye[2];
            for (f, c) in cc.iter().enumerate() {
                dens += c * 0.01 * self.filter_density[f][i];
            }
            let l = self.lamp[i] * pow10(-dens);
            for k in 0..3 {
                p[k] += l * self.paper_sens[k][i];
            }
        }
        p
    }

    /// Stage 8: the paper's dye densities at the paper's log exposures.
    #[must_use]
    pub fn paper_density(&self, log_e: Vec3) -> Vec3 {
        [
            self.curves[0].eval(log_e[0]),
            self.curves[1].eval(log_e[1]),
            self.curves[2].eval(log_e[2]),
        ]
    }
}
