//! Parameters of the ACES 2 output transform for the GPU: the constant matrices and the uniform
//! block built from `ym_color::aces2::Aces2Params`.

use ym_color::consts::Primaries;
use ym_color::consts::linalg::{self, Mat3};

/// ACES AP1 (the reach primaries) as the colour crate's `Primaries`.
fn ap1() -> Primaries {
    let p = ym_color::aces2::Primaries::AP1;
    Primaries {
        red: p.red,
        green: p.green,
        blue: p.blue,
        white: p.white,
    }
}

/// AP0 to AP1 (column-vector form; both have the ACES white, so no adaptation). The reference's
/// `clamp_AP0_to_AP1` applies this, clamps in AP1 and returns to AP0.
pub(crate) fn ap0_to_ap1() -> Mat3 {
    let m = ym_color::gamut::matrices();
    linalg::mul(&ap1().xyz_to_rgb(), &m.ap0_to_xyz)
}

/// AP1 to AP0 (column-vector form).
pub(crate) fn ap1_to_ap0() -> Mat3 {
    let m = ym_color::gamut::matrices();
    linalg::mul(&m.xyz_to_ap0, &ap1().rgb_to_xyz())
}
