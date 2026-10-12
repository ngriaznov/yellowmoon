//! AWG3 and AWG4 primaries, matrices from primaries (NPM), CAT02, AWG to AP0 (`docs/02-colour-pipeline.md` section Gamuts).
//!
//! All matrices are computed in f64 from primaries and white points (see [`crate::consts`]) the
//! first time [`matrices`] is called. Nothing is a rounded table: AP0 in particular is derived
//! from its own primaries, so every row of an AWG to AP0 matrix sums to 1 within 1e-15.
//!
//! Colour spaces are part of the type names ([`LinearAwg3`], [`LinearAwg4`], [`LinearAp0`],
//! [`LinearRec709`], [`LinearDisplayP3`], [`LinearRec2020`], [`Xyz`]) so linear triples of
//! different gamuts cannot be mixed silently. Each is a `pub` newtype over [`Vec3`].

use std::sync::LazyLock;

use crate::consts::{
    ACES_WHITE_XY, AP0, AWG3, AWG4, D65_XY, DISPLAY_P3, Mat3, REC709, REC2020, Vec3,
    cat02_adaptation, linalg, xy_to_xyz,
};

/// Every derived gamut matrix, computed once in f64 from primaries and whites.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GamutMatrices {
    /// AWG3 to XYZ (D65 white).
    pub awg3_to_xyz: Mat3,
    /// AWG4 to XYZ (D65 white).
    pub awg4_to_xyz: Mat3,
    /// AP0 to XYZ (ACES white).
    pub ap0_to_xyz: Mat3,
    /// Rec.709 / sRGB to XYZ (D65 white).
    pub rec709_to_xyz: Mat3,
    /// Display P3 to XYZ (D65 white).
    pub p3_to_xyz: Mat3,
    /// Rec.2020 to XYZ (D65 white).
    pub rec2020_to_xyz: Mat3,
    /// XYZ to AWG3.
    pub xyz_to_awg3: Mat3,
    /// XYZ to AWG4.
    pub xyz_to_awg4: Mat3,
    /// XYZ to AP0.
    pub xyz_to_ap0: Mat3,
    /// XYZ to Rec.709 / sRGB.
    pub xyz_to_rec709: Mat3,
    /// XYZ to Display P3.
    pub xyz_to_p3: Mat3,
    /// XYZ to Rec.2020.
    pub xyz_to_rec2020: Mat3,
    /// CAT02 von Kries adaptation from the D65 white to the ACES white.
    pub cat02_d65_to_aces: Mat3,
    /// CAT02 von Kries adaptation from the ACES white to the D65 white.
    pub cat02_aces_to_d65: Mat3,
    /// AWG3 to AWG4 (same white, no adaptation).
    pub awg3_to_awg4: Mat3,
    /// AWG4 to AWG3.
    pub awg4_to_awg3: Mat3,
    /// AWG4 to linear Display P3.
    pub awg4_to_p3: Mat3,
    /// AWG4 to linear Rec.709 / sRGB.
    pub awg4_to_rec709: Mat3,
    /// AWG4 to linear Rec.2020.
    pub awg4_to_rec2020: Mat3,
    /// AWG4 to ACES AP0 (CAT02 D65 to the ACES white).
    pub awg4_to_ap0: Mat3,
    /// ACES AP0 to AWG4.
    pub ap0_to_awg4: Mat3,
    /// AWG3 to ACES AP0 (CAT02 D65 to the ACES white).
    pub awg3_to_ap0: Mat3,
    /// ACES AP0 to AWG3.
    pub ap0_to_awg3: Mat3,
    /// Linear Rec.709 to Display P3.
    pub rec709_to_p3: Mat3,
    /// Linear Rec.709 to Rec.2020.
    pub rec709_to_rec2020: Mat3,
}

impl GamutMatrices {
    fn compute() -> Self {
        let awg3_to_xyz = AWG3.rgb_to_xyz();
        let awg4_to_xyz = AWG4.rgb_to_xyz();
        let ap0_to_xyz = AP0.rgb_to_xyz();
        let rec709_to_xyz = REC709.rgb_to_xyz();
        let p3_to_xyz = DISPLAY_P3.rgb_to_xyz();
        let rec2020_to_xyz = REC2020.rgb_to_xyz();
        let xyz_to_awg3 = AWG3.xyz_to_rgb();
        let xyz_to_awg4 = AWG4.xyz_to_rgb();
        let xyz_to_ap0 = AP0.xyz_to_rgb();
        let xyz_to_rec709 = REC709.xyz_to_rgb();
        let xyz_to_p3 = DISPLAY_P3.xyz_to_rgb();
        let xyz_to_rec2020 = REC2020.xyz_to_rgb();
        let d65 = xy_to_xyz(D65_XY);
        let aces = xy_to_xyz(ACES_WHITE_XY);
        let cat_fwd = cat02_adaptation(d65, aces);
        let cat_back = cat02_adaptation(aces, d65);
        let to_ap0 = |to_xyz: &Mat3| linalg::mul(&xyz_to_ap0, &linalg::mul(&cat_fwd, to_xyz));
        let awg3_to_ap0 = to_ap0(&awg3_to_xyz);
        let awg4_to_ap0 = to_ap0(&awg4_to_xyz);
        let from_ap0 = |xyz_to: &Mat3| linalg::mul(xyz_to, &linalg::mul(&cat_back, &ap0_to_xyz));
        Self {
            awg3_to_xyz,
            awg4_to_xyz,
            ap0_to_xyz,
            rec709_to_xyz,
            p3_to_xyz,
            rec2020_to_xyz,
            xyz_to_awg3,
            xyz_to_awg4,
            xyz_to_ap0,
            xyz_to_rec709,
            xyz_to_p3,
            xyz_to_rec2020,
            cat02_d65_to_aces: cat_fwd,
            cat02_aces_to_d65: cat_back,
            awg3_to_awg4: linalg::mul(&xyz_to_awg4, &awg3_to_xyz),
            awg4_to_awg3: linalg::mul(&xyz_to_awg3, &awg4_to_xyz),
            awg4_to_p3: linalg::mul(&xyz_to_p3, &awg4_to_xyz),
            awg4_to_rec709: linalg::mul(&xyz_to_rec709, &awg4_to_xyz),
            awg4_to_rec2020: linalg::mul(&xyz_to_rec2020, &awg4_to_xyz),
            awg4_to_ap0,
            ap0_to_awg4: from_ap0(&xyz_to_awg4),
            awg3_to_ap0,
            ap0_to_awg3: from_ap0(&xyz_to_awg3),
            rec709_to_p3: linalg::mul(&xyz_to_p3, &rec709_to_xyz),
            rec709_to_rec2020: linalg::mul(&xyz_to_rec2020, &rec709_to_xyz),
        }
    }
}

static MATRICES: LazyLock<GamutMatrices> = LazyLock::new(GamutMatrices::compute);

/// All derived gamut matrices (computed once, then cached).
#[must_use]
pub fn matrices() -> &'static GamutMatrices {
    &MATRICES
}

/// The AWG4 luminance row: the `Y` row of AWG4 to XYZ, `[0.2545, 0.7815, -0.0360]`.
#[must_use]
pub fn awg4_luminance_row() -> Vec3 {
    matrices().awg4_to_xyz[1]
}

/// Luminance `Y` of a linear AWG4 triple (the AWG4 `Y` row).
#[must_use]
pub fn awg4_luminance(rgb: Vec3) -> f64 {
    let y = matrices().awg4_to_xyz[1];
    y[0] * rgb[0] + y[1] * rgb[1] + y[2] * rgb[2]
}

macro_rules! linear_space {
    ($(#[$doc:meta])* $name:ident) => {
        $(#[$doc])*
        #[derive(Debug, Clone, Copy, PartialEq, Default)]
        pub struct $name(pub Vec3);

        impl $name {
            /// The three channels as an array.
            #[must_use]
            pub fn into_array(self) -> Vec3 {
                self.0
            }
        }

        impl From<Vec3> for $name {
            fn from(v: Vec3) -> Self {
                Self(v)
            }
        }

        impl From<$name> for Vec3 {
            fn from(v: $name) -> Vec3 {
                v.0
            }
        }
    };
}

linear_space!(
    /// Linear (scene-referred) ARRI Wide Gamut 3 RGB.
    LinearAwg3
);
linear_space!(
    /// Linear (scene-referred) ARRI Wide Gamut 4 RGB, the working space of the pipeline.
    LinearAwg4
);
linear_space!(
    /// Linear ACES AP0 RGB.
    LinearAp0
);
linear_space!(
    /// Linear Rec.709 / sRGB primaries RGB (display or scene linear, D65).
    LinearRec709
);
linear_space!(
    /// Linear Display P3 (P3-D65) RGB.
    LinearDisplayP3
);
linear_space!(
    /// Linear Rec.2020 RGB.
    LinearRec2020
);
linear_space!(
    /// CIE 1931 XYZ.
    Xyz
);

impl LinearAwg3 {
    /// Converts to linear AWG4 (exact; same white).
    #[must_use]
    pub fn to_awg4(self) -> LinearAwg4 {
        LinearAwg4(linalg::mul_vec(&matrices().awg3_to_awg4, self.0))
    }

    /// Converts to linear ACES AP0 (CAT02 D65 to the ACES white).
    #[must_use]
    pub fn to_ap0(self) -> LinearAp0 {
        LinearAp0(linalg::mul_vec(&matrices().awg3_to_ap0, self.0))
    }

    /// Converts to XYZ.
    #[must_use]
    pub fn to_xyz(self) -> Xyz {
        Xyz(linalg::mul_vec(&matrices().awg3_to_xyz, self.0))
    }
}

impl LinearAwg4 {
    /// Converts to linear AWG3 (exact; same white).
    #[must_use]
    pub fn to_awg3(self) -> LinearAwg3 {
        LinearAwg3(linalg::mul_vec(&matrices().awg4_to_awg3, self.0))
    }

    /// Converts to linear ACES AP0 (CAT02 D65 to the ACES white).
    #[must_use]
    pub fn to_ap0(self) -> LinearAp0 {
        LinearAp0(linalg::mul_vec(&matrices().awg4_to_ap0, self.0))
    }

    /// Converts to linear Display P3.
    #[must_use]
    pub fn to_display_p3(self) -> LinearDisplayP3 {
        LinearDisplayP3(linalg::mul_vec(&matrices().awg4_to_p3, self.0))
    }

    /// Converts to linear Rec.709 / sRGB primaries.
    #[must_use]
    pub fn to_rec709(self) -> LinearRec709 {
        LinearRec709(linalg::mul_vec(&matrices().awg4_to_rec709, self.0))
    }

    /// Converts to linear Rec.2020.
    #[must_use]
    pub fn to_rec2020(self) -> LinearRec2020 {
        LinearRec2020(linalg::mul_vec(&matrices().awg4_to_rec2020, self.0))
    }

    /// Converts to XYZ.
    #[must_use]
    pub fn to_xyz(self) -> Xyz {
        Xyz(linalg::mul_vec(&matrices().awg4_to_xyz, self.0))
    }

    /// Luminance `Y` (the AWG4 `Y` row).
    #[must_use]
    pub fn luminance(self) -> f64 {
        awg4_luminance(self.0)
    }
}

impl LinearAp0 {
    /// Converts to linear AWG4.
    #[must_use]
    pub fn to_awg4(self) -> LinearAwg4 {
        LinearAwg4(linalg::mul_vec(&matrices().ap0_to_awg4, self.0))
    }

    /// Converts to linear AWG3.
    #[must_use]
    pub fn to_awg3(self) -> LinearAwg3 {
        LinearAwg3(linalg::mul_vec(&matrices().ap0_to_awg3, self.0))
    }

    /// Converts to XYZ (ACES white).
    #[must_use]
    pub fn to_xyz(self) -> Xyz {
        Xyz(linalg::mul_vec(&matrices().ap0_to_xyz, self.0))
    }
}

impl LinearRec709 {
    /// Converts to linear Display P3.
    #[must_use]
    pub fn to_display_p3(self) -> LinearDisplayP3 {
        LinearDisplayP3(linalg::mul_vec(&matrices().rec709_to_p3, self.0))
    }

    /// Converts to linear Rec.2020.
    #[must_use]
    pub fn to_rec2020(self) -> LinearRec2020 {
        LinearRec2020(linalg::mul_vec(&matrices().rec709_to_rec2020, self.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn awg3_to_ap0_rows_sum_to_one() {
        for m in [matrices().awg3_to_ap0, matrices().awg4_to_ap0] {
            for s in linalg::row_sums(&m) {
                assert!((s - 1.0).abs() < 1e-14, "{s}");
            }
        }
    }

    #[test]
    fn typed_conversions_use_the_same_matrices() {
        let v = LinearAwg4([0.18, 0.18, 0.18]);
        let p3 = v.to_display_p3().0;
        for c in p3 {
            assert!((c - 0.18).abs() < 1e-12);
        }
        assert!((v.luminance() - 0.18).abs() < 1e-12);
    }
}
