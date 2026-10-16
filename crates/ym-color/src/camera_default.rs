//! The camera default rendering behind the Before view (`docs/02-colour-pipeline.md`, section The
//! Before view).
//!
//! The Before view shows the photo the way the camera would render it by default, so that the person
//! can see what Yellowmoon changes. It takes the camera RGB that enters the colour chain (the
//! demosaiced mosaic times the initial gains `g0`, raw stages R1 to R9 as for the final picture) and
//! applies only constants and per-photo facts of the file:
//!
//! 1. the camera's **as-shot white balance** `gs = [R/G, 1, B/G]` (the gains stored in the file, not our
//!    estimate), replacing `g0`;
//! 2. the body's published **D65 colour matrix** `CM` (XYZ to camera), its rows scaled so that D65 white
//!    maps to camera `[1, 1, 1]`, inverted, then XYZ to linear Rec.709;
//! 3. the camera gain [`CAMERA_GAIN_EV`];
//! 4. the **camera tone curve** ([`CameraTone`]) per channel in linear Rec.709.
//!
//! Steps 1 to 3 fold into one 3x3 ([`CameraDefault::matrix`]):
//!
//! ```text
//! M_before = M_xyz_to_rec709 . inv(CM_n) . diag(gs / g0) . 2^CAMERA_GAIN_EV
//! ```
//!
//! The chain then converts display-linear Rec.709 to the output primaries and applies C13, as for the
//! final picture. There is no sensor map, look, grade, curve, display transform, white lift or output
//! sharpening, and no user setting reaches this path.
//!
//! The curve is a normalised Naka-Rushton hyperbola, fitted to the embedded JPEGs of nine Standard-style
//! photos from nine Sony bodies and checked on 12 photos from 10 bodies
//! (`data/cameras/camera-default-tone.json` holds the constants, the golden table and the fit summary; a
//! test pins the file to the constants here). The body matrices are
//! numbers from the bodies' published D65 colour matrices (`data/cameras/color-matrices.json`,
//! `docs/14-licensing.md` rule 8).

use crate::consts::linalg::{self, Mat3, Vec3};
use crate::consts::{D65_XY, xy_to_xyz};
use crate::exposure::CAMERA_GAIN_EV;
use crate::gamut::matrices;

/// Exponent `n` of the camera tone curve: `t = x^n`.
pub const TONE_N: f64 = 1.6558;
/// Half-saturation `sigma` of the camera tone curve; `s = sigma^n`.
pub const TONE_SIGMA: f64 = 0.4569;
/// The scene-linear level at which the curve reaches 1 and stays there.
pub const TONE_WHITE: f64 = 2.5;
/// `s = TONE_SIGMA^TONE_N`, written out so that the CPU and the GPU start from the same number on every
/// platform (a test checks it against the formula).
pub const TONE_S: f64 = 0.273_358_053_617_412_83;
/// `k = (white^n + s) / white^n`, the gain that lets the curve reach exactly 1 at [`TONE_WHITE`].
pub const TONE_K: f64 = 1.059_954_641_044_243;

/// Why a camera default could not be made.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CameraDefaultError {
    /// A gain is not a finite positive number.
    #[error("the white balance gains {0:?} are not finite positive numbers")]
    BadGains(Vec3),
    /// The colour matrix has a value that is not finite, a row that D65 white does not map to a
    /// positive number, or no inverse.
    #[error("the camera's colour matrix cannot be used: {0}")]
    BadMatrix(&'static str),
    /// The tone parameters are not finite and positive.
    #[error("the camera tone parameters are not finite and positive")]
    BadTone,
}

/// The camera tone curve: `y = min(1, k * t / (t + s))`, `t = max(x, 0)^n`, one channel at a time.
///
/// The input is scene-linear light in Rec.709 primaries after the as-shot white balance, the body's
/// colour matrix and the camera gain (0.18 is middle grey); the output is display-linear on the same
/// primaries. Monotone, 0 at 0, exactly 1 from `white` up.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraTone {
    n: f64,
    s: f64,
    k: f64,
    white: f64,
}

impl CameraTone {
    /// The fitted curve: [`TONE_N`], [`TONE_SIGMA`], [`TONE_WHITE`].
    pub const FITTED: CameraTone = CameraTone {
        n: TONE_N,
        s: TONE_S,
        k: TONE_K,
        white: TONE_WHITE,
    };

    /// A curve from its exponent, half-saturation and white level.
    ///
    /// # Errors
    ///
    /// [`CameraDefaultError::BadTone`] when a value is not finite and positive.
    pub fn new(n: f64, sigma: f64, white: f64) -> Result<Self, CameraDefaultError> {
        if ![n, sigma, white].iter().all(|v| v.is_finite() && *v > 0.0) {
            return Err(CameraDefaultError::BadTone);
        }
        let s = sigma.powf(n);
        let wn = white.powf(n);
        let k = (wn + s) / wn;
        if !(s.is_finite() && k.is_finite()) {
            return Err(CameraDefaultError::BadTone);
        }
        Ok(Self { n, s, k, white })
    }

    /// The exponent `n`.
    #[must_use]
    pub fn n(&self) -> f64 {
        self.n
    }

    /// `s = sigma^n`.
    #[must_use]
    pub fn s(&self) -> f64 {
        self.s
    }

    /// The gain `k`.
    #[must_use]
    pub fn k(&self) -> f64 {
        self.k
    }

    /// The level at which the curve reaches 1.
    #[must_use]
    pub fn white(&self) -> f64 {
        self.white
    }

    /// One channel: scene-linear to display-linear in `[0, 1]`. NaN and negative values give 0; values at
    /// or above `white` give exactly 1.
    #[must_use]
    pub fn apply(&self, x: f64) -> f64 {
        if x.is_nan() || x <= 0.0 {
            return 0.0;
        }
        if x >= self.white {
            return 1.0;
        }
        let t = x.powf(self.n);
        (self.k * t / (t + self.s)).min(1.0)
    }

    /// The three channels of a triple.
    #[must_use]
    pub fn apply_rgb(&self, v: Vec3) -> Vec3 {
        v.map(|x| self.apply(x))
    }
}

/// XYZ of D65 white at `Y = 1`: the white that every camera matrix row and the Rec.709 matrix agree on.
fn d65_white() -> Vec3 {
    xy_to_xyz(D65_XY)
}

/// Scales every row of a body's D65 colour matrix (XYZ to camera RGB) so that D65 white maps to camera
/// `[1, 1, 1]`, as the camera's own white balance would have it.
///
/// # Errors
///
/// [`CameraDefaultError::BadMatrix`] for a value that is not finite or a row whose response to D65 white
/// is not positive.
pub fn normalise_color_matrix(xyz_d65_to_camera: &Mat3) -> Result<Mat3, CameraDefaultError> {
    if xyz_d65_to_camera.iter().flatten().any(|v| !v.is_finite()) {
        return Err(CameraDefaultError::BadMatrix("a value is not finite"));
    }
    let white = d65_white();
    let mut out = *xyz_d65_to_camera;
    for row in &mut out {
        let r = row[0] * white[0] + row[1] * white[1] + row[2] * white[2];
        if !(r.is_finite() && r > 0.0) {
            return Err(CameraDefaultError::BadMatrix(
                "a row does not respond to D65 white",
            ));
        }
        for v in row.iter_mut() {
            *v /= r;
        }
    }
    Ok(out)
}

/// The Before view's per-photo parameters: one 3x3 from the camera RGB that enters the colour chain to
/// scene-linear Rec.709 (as-shot white balance, body matrix and camera gain folded in), and the tone
/// curve. See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraDefault {
    matrix: Mat3,
    tone: CameraTone,
}

impl CameraDefault {
    /// The camera default of one photo.
    ///
    /// * `xyz_d65_to_camera`: the body's D65 colour matrix as published (XYZ to camera RGB); the rows
    ///   are normalised here ([`normalise_color_matrix`]);
    /// * `as_shot_gains`: the file's white balance as gains `[R/G, 1, B/G]` (a white surface records
    ///   `1 / gains` in the raw data), on the body's own raw scale;
    /// * `initial_gains`: the gains `g0` the raw stages already applied to the camera RGB that will be
    ///   fed to the chain. Pass the same numbers as `as_shot_gains` when the file has no white balance:
    ///   the Before view then keeps our estimate.
    ///
    /// # Errors
    ///
    /// [`CameraDefaultError::BadGains`] for a gain that is not finite and positive,
    /// [`CameraDefaultError::BadMatrix`] for a matrix that cannot be normalised or inverted.
    pub fn new(
        xyz_d65_to_camera: &Mat3,
        as_shot_gains: Vec3,
        initial_gains: Vec3,
    ) -> Result<Self, CameraDefaultError> {
        for g in [as_shot_gains, initial_gains] {
            if !g.iter().all(|v| v.is_finite() && *v > 0.0) {
                return Err(CameraDefaultError::BadGains(g));
            }
        }
        let normalised = normalise_color_matrix(xyz_d65_to_camera)?;
        let camera_to_xyz = linalg::inv(&normalised)
            .ok_or(CameraDefaultError::BadMatrix("the matrix has no inverse"))?;
        let balance = linalg::diag([
            as_shot_gains[0] / initial_gains[0],
            as_shot_gains[1] / initial_gains[1],
            as_shot_gains[2] / initial_gains[2],
        ]);
        let gain = CAMERA_GAIN_EV.exp2();
        let to_rec709 = linalg::mul(&matrices().xyz_to_rec709, &camera_to_xyz);
        let matrix = linalg::mul(&to_rec709, &balance).map(|row| row.map(|v| v * gain));
        if matrix.iter().flatten().any(|v| !v.is_finite()) {
            return Err(CameraDefaultError::BadMatrix(
                "the product is not a finite matrix",
            ));
        }
        Ok(Self {
            matrix,
            tone: CameraTone::FITTED,
        })
    }

    /// A camera default for a caller that knows nothing of the body: the camera RGB is taken as linear
    /// Rec.709 light that is already white-balanced, with only the camera gain and the tone curve. The
    /// app always builds the real one ([`CameraDefault::new`]); this is the placeholder of
    /// `ChainParams::new`.
    #[must_use]
    pub fn neutral() -> Self {
        let gain = CAMERA_GAIN_EV.exp2();
        Self {
            matrix: linalg::IDENTITY.map(|row| row.map(|v| v * gain)),
            tone: CameraTone::FITTED,
        }
    }

    /// The same matrix with another tone curve (tests and experiments).
    #[must_use]
    pub fn with_tone(mut self, tone: CameraTone) -> Self {
        self.tone = tone;
        self
    }

    /// The fused 3x3 from the camera RGB entering the chain to scene-linear Rec.709.
    #[must_use]
    pub fn matrix(&self) -> &Mat3 {
        &self.matrix
    }

    /// The tone curve.
    #[must_use]
    pub fn tone(&self) -> CameraTone {
        self.tone
    }

    /// Camera RGB (as it enters the chain) to scene-linear Rec.709, before the tone curve.
    #[must_use]
    pub fn scene_linear_rec709(&self, camera_rgb: Vec3) -> Vec3 {
        linalg::mul_vec(&self.matrix, camera_rgb)
    }

    /// Camera RGB (as it enters the chain) to display-linear Rec.709 in `[0, 1]`: the matrix, then the
    /// tone curve per channel. The caller converts to the output primaries and encodes.
    #[must_use]
    pub fn display_linear_rec709(&self, camera_rgb: Vec3) -> Vec3 {
        self.tone
            .apply_rgb(linalg::mul_vec(&self.matrix, camera_rgb))
    }
}

impl Default for CameraDefault {
    fn default() -> Self {
        Self::neutral()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode::srgb_encode;

    #[test]
    fn the_written_out_constants_are_the_formula() {
        let t = CameraTone::new(TONE_N, TONE_SIGMA, TONE_WHITE).unwrap();
        assert!((t.s() - TONE_S).abs() < 1e-15, "{} vs {TONE_S}", t.s());
        assert!((t.k() - TONE_K).abs() < 1e-14, "{} vs {TONE_K}", t.k());
        assert_eq!(CameraTone::FITTED.n(), TONE_N);
        assert_eq!(CameraTone::FITTED.white(), TONE_WHITE);
    }

    #[test]
    fn the_curve_lands_on_the_fitted_anchors() {
        let t = CameraTone::FITTED;
        // Middle grey: display-linear 0.18675, sRGB code 119.7 of 255.
        let y = t.apply(0.18);
        assert!((y - 0.186_751).abs() < 1e-5, "{y}");
        assert!((srgb_encode(y) * 255.0 - 119.7).abs() < 0.05);
        // The half-stop table of the fit (stops over 0.18, display-linear y).
        for (stops, want) in [
            (-6.0, 0.000_232),
            (-3.0, 0.007_197),
            (-1.0, 0.067_371),
            (1.0, 0.426_729),
            (2.0, 0.720_599),
            (3.0, 0.922_135),
        ] {
            let got = t.apply(0.18 * f64::powf(2.0, stops));
            assert!((got - want).abs() < 2e-6, "{stops} stops: {got} vs {want}");
        }
        assert_eq!(t.apply(TONE_WHITE), 1.0);
        assert_eq!(t.apply(16.0), 1.0);
        assert_eq!(t.apply(1e30), 1.0);
    }

    #[test]
    fn the_curve_is_monotone_bounded_and_defined_for_bad_input() {
        let t = CameraTone::FITTED;
        let mut last = 0.0;
        for i in 0..=4000 {
            let x = f64::from(i) * 0.001;
            let y = t.apply(x);
            assert!((0.0..=1.0).contains(&y), "{x}: {y}");
            assert!(y >= last, "not monotone at {x}");
            last = y;
        }
        assert_eq!(t.apply(0.0), 0.0);
        assert_eq!(t.apply(-0.5), 0.0);
        assert_eq!(t.apply(f64::NAN), 0.0);
        assert_eq!(t.apply(f64::NEG_INFINITY), 0.0);
        assert_eq!(t.apply(f64::INFINITY), 1.0);
    }

    #[test]
    fn the_tone_rejects_bad_numbers() {
        assert_eq!(
            CameraTone::new(0.0, 0.4, 2.5),
            Err(CameraDefaultError::BadTone)
        );
        assert_eq!(
            CameraTone::new(1.6, f64::NAN, 2.5),
            Err(CameraDefaultError::BadTone)
        );
        assert_eq!(
            CameraTone::new(1.6, 0.4, -1.0),
            Err(CameraDefaultError::BadTone)
        );
    }

    /// A made-up body matrix with the shape of a published one (strong red-green overlap).
    fn body() -> Mat3 {
        [
            [0.7657, -0.2847, -0.0607],
            [-0.4083, 1.1966, 0.2389],
            [-0.0684, 0.1418, 0.5844],
        ]
    }

    #[test]
    fn normalising_maps_d65_white_to_camera_one() {
        let n = normalise_color_matrix(&body()).unwrap();
        let cam = linalg::mul_vec(&n, d65_white());
        for c in cam {
            assert!((c - 1.0).abs() < 1e-13, "{cam:?}");
        }
    }

    #[test]
    fn a_bad_matrix_is_refused() {
        let mut m = body();
        m[1][1] = f64::NAN;
        assert!(matches!(
            normalise_color_matrix(&m),
            Err(CameraDefaultError::BadMatrix(_))
        ));
        assert!(matches!(
            normalise_color_matrix(&[[0.0; 3]; 3]),
            Err(CameraDefaultError::BadMatrix(_))
        ));
        // Rows that are all the same: D65 maps fine, but there is no inverse.
        let same = [[0.5, 0.4, 0.3]; 3];
        assert!(matches!(
            CameraDefault::new(&same, [2.0, 1.0, 1.5], [2.0, 1.0, 1.5]),
            Err(CameraDefaultError::BadMatrix(_))
        ));
    }

    #[test]
    fn bad_gains_are_refused() {
        // NaN is not equal to itself, so the error is matched, not compared.
        for bad in [[0.0, 1.0, 1.0], [f64::NAN, 1.0, 1.0], [1.0, -1.0, 1.0]] {
            assert!(matches!(
                CameraDefault::new(&body(), bad, [1.0; 3]),
                Err(CameraDefaultError::BadGains(_))
            ));
            assert!(matches!(
                CameraDefault::new(&body(), [1.0; 3], bad),
                Err(CameraDefaultError::BadGains(_))
            ));
        }
    }

    #[test]
    fn the_as_shot_white_comes_out_neutral_whatever_the_initial_gains() {
        let gs = [2.1, 1.0, 1.45];
        for g0 in [[2.1, 1.0, 1.45], [1.7, 1.0, 2.2], [1.0, 1.0, 1.0]] {
            let cd = CameraDefault::new(&body(), gs, g0).unwrap();
            // A white surface records 1 / gs in the raw data; the raw stages multiply by g0.
            let cam_in = [g0[0] / gs[0], g0[1] / gs[1], g0[2] / gs[2]];
            let lin = cd.scene_linear_rec709(cam_in);
            let gain = CAMERA_GAIN_EV.exp2();
            for c in lin {
                assert!((c - gain).abs() < 1e-12, "{lin:?}");
            }
            let out = cd.display_linear_rec709(cam_in);
            assert!((out[0] - out[1]).abs() < 1e-12 && (out[1] - out[2]).abs() < 1e-12);
        }
    }

    #[test]
    fn the_matrix_is_independent_of_the_initial_gains_given_the_raw_signal() {
        // raw * g0 through M_before(g0) is raw * gs through M_before(gs = g0): g0 cancels.
        let gs = [2.1, 1.0, 1.45];
        let raw = [0.31, 0.22, 0.17];
        let reference = {
            let cd = CameraDefault::new(&body(), gs, gs).unwrap();
            cd.display_linear_rec709([raw[0] * gs[0], raw[1] * gs[1], raw[2] * gs[2]])
        };
        for g0 in [[1.7, 1.0, 2.2], [1.0, 1.0, 1.0], [3.0, 1.0, 1.1]] {
            let cd = CameraDefault::new(&body(), gs, g0).unwrap();
            let got = cd.display_linear_rec709([raw[0] * g0[0], raw[1] * g0[1], raw[2] * g0[2]]);
            for c in 0..3 {
                assert!(
                    (got[c] - reference[c]).abs() < 1e-12,
                    "{got:?} {reference:?}"
                );
            }
        }
    }

    #[test]
    fn mid_grey_after_the_camera_gain_lands_on_code_119_7() {
        let gs = [2.0, 1.0, 1.5];
        let cd = CameraDefault::new(&body(), gs, gs).unwrap();
        // A grey card at 0.18 scene-linear, before the camera gain: the raw white is 1 / gs.
        let white = [1.0 / gs[0], 1.0, 1.0 / gs[2]];
        let grey = white.map(|w| w * 0.18 / CAMERA_GAIN_EV.exp2());
        // The raw signal times g0 = gs.
        let cam_in = [grey[0] * gs[0], grey[1] * gs[1], grey[2] * gs[2]];
        let out = cd.display_linear_rec709(cam_in);
        for y in out {
            let code = srgb_encode(y) * 255.0;
            assert!((code - 119.7).abs() < 0.5, "{code}");
        }
    }

    #[test]
    fn the_neutral_placeholder_is_a_grey_ramp() {
        let cd = CameraDefault::default();
        let out = cd.display_linear_rec709([0.09, 0.09, 0.09]);
        assert_eq!(out[0], out[1]);
        assert_eq!(out[1], out[2]);
        assert!(out[0] > 0.0 && out[0] < 1.0);
    }
}
