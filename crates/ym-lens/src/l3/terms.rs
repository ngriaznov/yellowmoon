//! The light budget of L3 (`docs/21` section 4, L3).
//!
//! `out = (1 - a) Z + F (a_g K_g * R + a_v K_v * R + sum a_i G_i[H] + a_* B[H] + a_s (q - 1) / (Q_max - 1) S[H])`
//!
//! with `F = flares / 100` and `a = F * (the fractions in use)`. The fractions come from the lens data at
//! Flares 100: glow `a_g`, veiling `a_v`, each ghost `a_i`, the starburst `a_*` and, scaled by the Anamorphic
//! factor `(q - 1) / (Q_max - 1)`, the attachment's streak `a_s` and its own ghosts. The starburst's fraction is
//! in use in proportion to its strength, `clamp((N - 2) / (f_full - 2), 0, 1)` for the photo's f-number `N`: the
//! light a weak starburst does not take stays in the picture. At full strength `a` equals
//! [`LensPlan::light_fraction`].

use crate::consts::{STARBURST_DEFAULT_F_NUMBER, STARBURST_MIN_F_NUMBER};
use crate::plan::LensPlan;

/// The amounts and fractions of one plan, as L3 uses them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Terms {
    /// `F = flares / 100`.
    pub(crate) f: f64,
    /// The fraction of the picture's light the flare terms take: `F` times the fractions in use.
    pub(crate) a: f64,
    /// The Anamorphic factor `(q - 1) / (Q_max - 1)`.
    pub(crate) anam: f64,
    /// The starburst's fraction in use: `a_*` times its strength.
    pub(crate) burst: f64,
    /// The streak's fraction in use: `a_s` times the Anamorphic factor.
    pub(crate) streak: f64,
    /// The widening of ghosts along the axis, `1 + kappa_g (q - 1)`.
    pub(crate) stretch: f64,
}

impl Terms {
    /// The terms of `plan`.
    pub(crate) fn new(plan: &LensPlan) -> Self {
        let lens = plan.lens();
        let att = plan.attachment();
        let fl = &lens.flare;
        let f = plan.flares();
        let q = plan.q();
        let anam = ((q - 1.0) / (att.q_max - 1.0)).clamp(0.0, 1.0);
        let n = plan.frame().f_number.unwrap_or(STARBURST_DEFAULT_F_NUMBER);
        let full = fl.starburst.f_number_full;
        let strength = if full <= STARBURST_MIN_F_NUMBER {
            f64::from(u8::from(n >= full))
        } else {
            ((n - STARBURST_MIN_F_NUMBER) / (full - STARBURST_MIN_F_NUMBER)).clamp(0.0, 1.0)
        };
        let burst = fl.starburst.fraction * strength;
        let streak = att.streak.fraction * anam;
        let att_ghosts: f64 = att.ghosts.iter().map(|g| g.fraction).sum::<f64>() * anam;
        let lens_ghosts: f64 = fl.ghosts.iter().map(|g| g.fraction).sum();
        let a =
            f * (fl.glow.fraction + fl.veil.fraction + lens_ghosts + burst + streak + att_ghosts);
        Terms {
            f,
            a,
            anam,
            burst,
            streak,
            stretch: 1.0 + att.ghost_stretch * (q - 1.0),
        }
    }
}
