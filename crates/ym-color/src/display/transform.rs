//! Stage C12: the display transform, ACES 2 or an imported ARRI display LUT (`docs/02-colour-pipeline.md`
//! sections C12 and Imported ARRI display LUT).
//!
//! C12 takes the LogC4 output of C11 and returns **display-linear** RGB in the primaries of the
//! chosen [`OutputTarget`](super::OutputTarget), with 1.0 the 100 nit SDR white (the value C13 then
//! encodes, and the value an exporter resizes in).
//!
//! # ACES 2
//!
//! LogC4 to linear AWG4 to AP0 (`gamut::matrices().awg4_to_ap0`: AWG4 to XYZ, CAT02 from D65 to the
//! ACES white, XYZ to AP0), then [`Aces2Transform::transform_ap0_linear`] for the output's limiting
//! and encoding primaries.
//!
//! # Imported ARRI display LUT
//!
//! * A LogC4 display LUT takes the LogC4 output of C11 directly. A LogC3 one takes
//!   LogC4 to linear AWG4 to AWG3 to LogC3 EI 800 first ([`LogC4::to_logc3_ei800`]).
//! * The LUT outputs display **code values**. They are decoded with the LUT's output transfer
//!   ([`DisplayLutTransfer`], gamma 2.4 for Rec.709) to display-linear in the LUT's primaries
//!   ([`DisplayLutPrimaries`]), then converted by a 3x3 matrix to the output's primaries. For the
//!   viewport that means Display P3, which C13 encodes with the sRGB transfer; for an export it is
//!   the chosen colour space. A colour outside the output gamut after the matrix is clipped by
//!   C13's clamp, not gamut-mapped.
//! * The LUT's output is described by [`DisplayLutOutput`], normally parsed from the target slug
//!   the importer records (`rec709-g24`, `p3-d65-g26`, `rec2020-g24`; [`DisplayLutOutput::from_target`]).
//!
//! # Decisions where the spec is silent
//!
//! * **Default transfer when the target names none:** gamma 2.4 for Rec.709 and Rec.2020 (the
//!   BT.1886 convention the spec uses for both), gamma 2.6 for P3-D65 (cinema practice). An explicit
//!   `g22`, `g24`, `g26`, `gamma 2.8` or `srgb` in the target wins. Check these against the real
//!   packages once their file names are fingerprinted (`docs/08`, M3-H2).
//! * **Not supported:** HLG and PQ targets (HDR is V2) and any DCI, D50, D55, D60 or D75 white
//!   point; all are refused with [`DisplayError::UnsupportedTarget`] rather than rendered with
//!   the wrong white or range.
//! * **Negative LUT output** decodes to 0.

use std::fmt;
use std::sync::Arc;

use super::{DisplayError, OutputTarget};
use crate::aces2::Aces2Transform;
use crate::consts::Vec3;
use crate::consts::linalg::{self, Mat3};
use crate::encode::{OutputSpace, srgb_decode};
use crate::gamut::matrices;
use crate::logc::{LogC4, logc4_decode_rgb};
use crate::lut::Lut;

/// The encoding an ARRI display LUT expects as input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayLutInput {
    /// LogC4 / AWG4: applied to the C11 output directly.
    LogC4,
    /// LogC3 EI 800 / AWG3: applied after LogC4 to linear AWG4 to AWG3 to LogC3 EI 800.
    LogC3Ei800,
}

/// The primaries (all with a D65 white) of a display LUT's output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DisplayLutPrimaries {
    /// Rec.709 / sRGB primaries.
    Rec709,
    /// P3-D65 (Display P3) primaries.
    P3D65,
    /// Rec.2020 primaries.
    Rec2020,
}

/// The transfer function a display LUT's output is encoded with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DisplayLutTransfer {
    /// A pure power law `linear = code^gamma` (gamma 2.4 for Rec.709 / BT.1886).
    Gamma(f64),
    /// The sRGB piecewise transfer.
    Srgb,
}

/// What a display LUT outputs: primaries and transfer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayLutOutput {
    /// Primaries of the output code values.
    pub primaries: DisplayLutPrimaries,
    /// Transfer function of the output code values.
    pub transfer: DisplayLutTransfer,
}

impl DisplayLutOutput {
    /// Rec.709 primaries, gamma 2.4: ARRI's classic Rec.709 display rendering.
    pub const REC709_G24: Self = Self {
        primaries: DisplayLutPrimaries::Rec709,
        transfer: DisplayLutTransfer::Gamma(2.4),
    };

    /// Parses a target slug such as `rec709-g24`, `p3-d65-g26`, `rec2020` or `srgb`.
    ///
    /// The text is case-insensitive; `-`, `_`, spaces and dots are separators (so `Rec.709 Gamma 2.4`
    /// works). Primaries come from `709`, `srgb`, `p3` or `2020`; the transfer from `gNN` or
    /// `gamma NN` (`24` is 2.4) or `srgb`, with the defaults in the module documentation.
    ///
    /// # Errors
    ///
    /// [`DisplayError::UnsupportedTarget`] for HLG, PQ, a DCI, D50, D55, D60 or D75 white point, or text that names no
    /// primaries or an unreadable gamma.
    pub fn from_target(target: &str) -> Result<Self, DisplayError> {
        let unsupported = |reason: &'static str| DisplayError::UnsupportedTarget {
            target: target.to_string(),
            reason,
        };
        let lower = target.to_ascii_lowercase().replace('.', "");
        let tokens: Vec<&str> = lower
            .split(|c: char| !c.is_ascii_alphanumeric())
            .filter(|t| !t.is_empty())
            .collect();
        let folded: String = tokens.concat();
        // Substring, not whole-token: names such as `Rec2020PQ1000` or `hlg1000nits` have no separator.
        if ["hlg", "pq", "st2084", "hdr"]
            .iter()
            .any(|word| folded.contains(word))
        {
            return Err(unsupported("HDR outputs (HLG, PQ) are not supported in V1"));
        }
        if folded.contains("dci")
            || tokens
                .iter()
                .any(|t| matches!(*t, "d50" | "d55" | "d60" | "d75"))
        {
            return Err(unsupported(
                "a DCI or other non-D65 white point is not supported; only D65 displays are",
            ));
        }
        let primaries = if folded.contains("2020") {
            DisplayLutPrimaries::Rec2020
        } else if folded.contains("p3") {
            DisplayLutPrimaries::P3D65
        } else if folded.contains("709") || folded.contains("srgb") {
            DisplayLutPrimaries::Rec709
        } else {
            return Err(unsupported("the target names no known primaries"));
        };
        let transfer = parse_transfer(&tokens).map_err(unsupported)?;
        let transfer = transfer.unwrap_or(match primaries {
            DisplayLutPrimaries::P3D65 => DisplayLutTransfer::Gamma(2.6),
            DisplayLutPrimaries::Rec709 | DisplayLutPrimaries::Rec2020 => {
                DisplayLutTransfer::Gamma(2.4)
            }
        });
        Ok(Self {
            primaries,
            transfer,
        })
    }

    /// Decodes one output code value to display-linear in the LUT's primaries (negative input gives 0).
    #[must_use]
    pub fn decode_channel(&self, code: f64) -> f64 {
        if code.is_nan() || code <= 0.0 {
            return 0.0;
        }
        match self.transfer {
            DisplayLutTransfer::Gamma(gamma) => code.powf(gamma),
            DisplayLutTransfer::Srgb => srgb_decode(code),
        }
    }

    /// Decodes an output triple to display-linear in the LUT's primaries.
    #[must_use]
    pub fn decode(&self, codes: Vec3) -> Vec3 {
        [
            self.decode_channel(codes[0]),
            self.decode_channel(codes[1]),
            self.decode_channel(codes[2]),
        ]
    }
}

/// The transfer named in the tokens of a target, if any: `g24`, `gamma24`, `gamma 24` or `srgb`.
fn parse_transfer(tokens: &[&str]) -> Result<Option<DisplayLutTransfer>, &'static str> {
    for (i, token) in tokens.iter().enumerate() {
        let digits = if let Some(rest) = token.strip_prefix("gamma") {
            if rest.is_empty() {
                tokens.get(i + 1).copied().unwrap_or("")
            } else {
                rest
            }
        } else if let Some(rest) = token.strip_prefix('g') {
            rest
        } else {
            continue;
        };
        if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
            // A token such as "grade" starts with g but is not a gamma.
            if token.starts_with("gamma") {
                return Err("the gamma is not a number");
            }
            continue;
        }
        let value: f64 = digits.parse().map_err(|_| "the gamma is not a number")?;
        let gamma = match digits.len() {
            1 => value,
            2 => value / 10.0,
            3 => value / 100.0,
            _ => return Err("the gamma has too many digits"),
        };
        if !(1.0..=4.0).contains(&gamma) {
            return Err("the gamma is outside 1 to 4");
        }
        return Ok(Some(DisplayLutTransfer::Gamma(gamma)));
    }
    if tokens.contains(&"srgb") {
        return Ok(Some(DisplayLutTransfer::Srgb));
    }
    Ok(None)
}

/// An imported ARRI display LUT with the description of what it takes and gives.
///
/// The LUT is shared (`Arc`): the looks library hands out `Arc<Lut>` and a chain per photo must not
/// copy a 65-point cube. `Debug` summarises the LUT (title and size) instead of printing every entry.
#[derive(Clone, PartialEq)]
pub struct ArriDisplayLut {
    lut: Arc<Lut>,
    input: DisplayLutInput,
    output: DisplayLutOutput,
}

impl fmt::Debug for ArriDisplayLut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ArriDisplayLut")
            .field("title", &self.lut.title)
            .field("cube_size", &self.lut.cube.size())
            .field("has_shaper", &self.lut.shaper.is_some())
            .field("input", &self.input)
            .field("output", &self.output)
            .finish()
    }
}

impl ArriDisplayLut {
    /// Describes an imported display LUT.
    ///
    /// # Errors
    ///
    /// [`DisplayError::BadGamma`] when the output transfer is a power law whose gamma is not a
    /// number between 1 and 4.
    pub fn new(
        lut: Arc<Lut>,
        input: DisplayLutInput,
        output: DisplayLutOutput,
    ) -> Result<Self, DisplayError> {
        if let DisplayLutTransfer::Gamma(g) = output.transfer
            && !(1.0..=4.0).contains(&g)
        {
            return Err(DisplayError::BadGamma(g));
        }
        Ok(Self { lut, input, output })
    }

    /// The input encoding.
    #[must_use]
    pub fn input(&self) -> DisplayLutInput {
        self.input
    }

    /// The output description.
    #[must_use]
    pub fn output(&self) -> DisplayLutOutput {
        self.output
    }

    /// The LUT.
    #[must_use]
    pub fn lut(&self) -> &Arc<Lut> {
        &self.lut
    }

    /// The LUT's output **code values** for a LogC4 pixel (the C11 output).
    #[must_use]
    pub fn eval_codes(&self, v: LogC4) -> Vec3 {
        let input = match self.input {
            DisplayLutInput::LogC4 => v.0,
            DisplayLutInput::LogC3Ei800 => v.to_logc3_ei800().0,
        };
        self.lut.sample(input)
    }

    /// Display-linear RGB in the LUT's own primaries for a LogC4 pixel.
    #[must_use]
    pub fn eval_display_linear(&self, v: LogC4) -> Vec3 {
        self.output.decode(self.eval_codes(v))
    }
}

/// The display transform of stage C12.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum DisplayTransform {
    /// ACES 2 output transform, 100 nit SDR (the default).
    #[default]
    Aces2,
    /// An imported ARRI display LUT.
    ArriDisplayLut(ArriDisplayLut),
}

impl DisplayTransform {
    /// Wraps an imported ARRI display LUT.
    #[must_use]
    pub fn arri(lut: ArriDisplayLut) -> Self {
        Self::ArriDisplayLut(lut)
    }
}

/// Linear RGB from the LUT's primaries to the output colour space's primaries, `None` when equal.
fn primaries_matrix(from: DisplayLutPrimaries, to: OutputSpace) -> Option<Mat3> {
    let m = matrices();
    let (from_xyz, same) = match from {
        DisplayLutPrimaries::Rec709 => (m.rec709_to_xyz, to == OutputSpace::Srgb),
        DisplayLutPrimaries::P3D65 => (m.p3_to_xyz, to == OutputSpace::DisplayP3),
        DisplayLutPrimaries::Rec2020 => (m.rec2020_to_xyz, to == OutputSpace::Rec2020),
    };
    if same {
        return None;
    }
    let xyz_to = match to {
        OutputSpace::Srgb => m.xyz_to_rec709,
        OutputSpace::DisplayP3 => m.xyz_to_p3,
        OutputSpace::Rec2020 => m.xyz_to_rec2020,
    };
    Some(linalg::mul(&xyz_to, &from_xyz))
}

/// C12 prepared for one output: tables built, matrices multiplied out.
#[derive(Debug, Clone)]
pub(crate) struct PreparedDisplay {
    kind: Kind,
}

#[derive(Debug, Clone)]
enum Kind {
    Aces2 {
        transform: &'static Aces2Transform,
        awg4_to_ap0: Mat3,
    },
    Arri {
        lut: ArriDisplayLut,
        to_output: Option<Mat3>,
    },
}

impl PreparedDisplay {
    /// Prepares `transform` for `output`. Builds (or fetches) the ACES 2 tables, which takes
    /// 10 to 35 ms the first time per limiting gamut.
    pub(crate) fn new(transform: &DisplayTransform, output: OutputTarget) -> Self {
        let kind = match transform {
            DisplayTransform::Aces2 => Kind::Aces2 {
                transform: Aces2Transform::shared(output.aces2()),
                awg4_to_ap0: matrices().awg4_to_ap0,
            },
            DisplayTransform::ArriDisplayLut(lut) => Kind::Arri {
                to_output: primaries_matrix(lut.output.primaries, output.space()),
                lut: lut.clone(),
            },
        };
        Self { kind }
    }

    /// C12: LogC4 to display-linear RGB in the output's primaries (1.0 is 100 nit white).
    pub(crate) fn display_linear(&self, v: LogC4) -> Vec3 {
        match &self.kind {
            Kind::Aces2 {
                transform,
                awg4_to_ap0,
            } => {
                let ap0 = linalg::mul_vec(awg4_to_ap0, logc4_decode_rgb(v.0));
                transform.transform_ap0_linear(ap0)
            }
            Kind::Arri { lut, to_output } => {
                let linear = lut.eval_display_linear(v);
                match to_output {
                    Some(m) => linalg::mul_vec(m, linear),
                    None => linear,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(target: &str) -> DisplayLutOutput {
        DisplayLutOutput::from_target(target).unwrap()
    }

    #[test]
    fn target_slugs_parse() {
        assert_eq!(parse("rec709-g24"), DisplayLutOutput::REC709_G24);
        assert_eq!(parse("Rec.709 Gamma 2.4"), DisplayLutOutput::REC709_G24);
        assert_eq!(parse("rec709"), DisplayLutOutput::REC709_G24);
        assert_eq!(parse("Gamma24_Rec709"), DisplayLutOutput::REC709_G24);
        assert_eq!(
            parse("rec709-g22"),
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::Rec709,
                transfer: DisplayLutTransfer::Gamma(2.2)
            }
        );
        assert_eq!(
            parse("p3-d65"),
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::P3D65,
                transfer: DisplayLutTransfer::Gamma(2.6)
            }
        );
        assert_eq!(
            parse("p3-d65-g24"),
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::P3D65,
                transfer: DisplayLutTransfer::Gamma(2.4)
            }
        );
        assert_eq!(
            parse("rec2020"),
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::Rec2020,
                transfer: DisplayLutTransfer::Gamma(2.4)
            }
        );
        assert_eq!(
            parse("srgb"),
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::Rec709,
                transfer: DisplayLutTransfer::Srgb
            }
        );
        assert_eq!(
            parse("p3_gamma28"),
            DisplayLutOutput {
                primaries: DisplayLutPrimaries::P3D65,
                transfer: DisplayLutTransfer::Gamma(2.8)
            }
        );
    }

    #[test]
    fn unsupported_targets_are_refused_with_a_reason() {
        for t in [
            "hlg",
            "rec2020-pq",
            "Rec2020 HLG",
            "p3-dci",
            "DCIP3",
            "p3-d60",
            "P3_D60_g26",
            "rec709-d50",
            "rec709-g99",
            "film",
        ] {
            assert!(
                matches!(
                    DisplayLutOutput::from_target(t),
                    Err(DisplayError::UnsupportedTarget { .. })
                ),
                "{t}"
            );
        }
    }

    #[test]
    fn decode_handles_negative_and_nan() {
        let o = DisplayLutOutput::REC709_G24;
        assert_eq!(o.decode_channel(-0.5), 0.0);
        assert_eq!(o.decode_channel(f64::NAN), 0.0);
        assert!((o.decode_channel(0.5) - 0.5_f64.powf(2.4)).abs() < 1e-15);
        let s = DisplayLutOutput {
            primaries: DisplayLutPrimaries::Rec709,
            transfer: DisplayLutTransfer::Srgb,
        };
        assert!((s.decode_channel(0.5) - srgb_decode(0.5)).abs() < 1e-15);
    }

    #[test]
    fn primaries_matrices_are_identity_only_when_equal() {
        assert!(primaries_matrix(DisplayLutPrimaries::Rec709, OutputSpace::Srgb).is_none());
        assert!(primaries_matrix(DisplayLutPrimaries::P3D65, OutputSpace::DisplayP3).is_none());
        assert!(primaries_matrix(DisplayLutPrimaries::Rec2020, OutputSpace::Rec2020).is_none());
        let m = primaries_matrix(DisplayLutPrimaries::Rec709, OutputSpace::DisplayP3).unwrap();
        assert_eq!(m, matrices().rec709_to_p3);
        // White stays white in every conversion.
        for from in [
            DisplayLutPrimaries::Rec709,
            DisplayLutPrimaries::P3D65,
            DisplayLutPrimaries::Rec2020,
        ] {
            for to in [
                OutputSpace::Srgb,
                OutputSpace::DisplayP3,
                OutputSpace::Rec2020,
            ] {
                if let Some(m) = primaries_matrix(from, to) {
                    for s in linalg::row_sums(&m) {
                        assert!((s - 1.0).abs() < 1e-13, "{from:?} -> {to:?}: {s}");
                    }
                }
            }
        }
    }

    #[test]
    fn bad_gamma_is_rejected() {
        let lut = Arc::new(Lut::from_cube(crate::lut::Lut3d::identity(2).unwrap()));
        let bad = DisplayLutOutput {
            primaries: DisplayLutPrimaries::Rec709,
            transfer: DisplayLutTransfer::Gamma(f64::NAN),
        };
        assert!(matches!(
            ArriDisplayLut::new(lut.clone(), DisplayLutInput::LogC4, bad),
            Err(DisplayError::BadGamma(_))
        ));
        let high = DisplayLutOutput {
            transfer: DisplayLutTransfer::Gamma(5.0),
            ..bad
        };
        assert!(ArriDisplayLut::new(lut, DisplayLutInput::LogC4, high).is_err());
    }
}
