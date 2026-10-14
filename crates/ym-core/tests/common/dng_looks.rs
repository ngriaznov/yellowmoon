//! The looks of the DNG tests, all of our own (never an ARRI file): a look with a LogC3 to Rec.709
//! rendering baked in, a log-to-log look, and a display LUT.
#![allow(dead_code)]

use std::sync::Arc;

use ym_color::cdl::Cdl;
use ym_color::consts::linalg;
use ym_color::display::{ArriDisplayLut, DisplayLutInput, DisplayLutOutput, DisplayTransform};
use ym_color::logc::logc3_ei800_decode_rgb;
use ym_color::look::{LookInput, LookTransform};
use ym_color::lut::{Lut, Lut3d};
use ym_core::pipeline::{LookProblem, LookProvider, NeutralOnly, ResolvedLook};
use ym_looks::library::LookId;

/// A smooth tone map from LogC3 code values to Rec.709 gamma 2.4 code values, with a small colour
/// twist: a stand-in of our own for a look with ARRI's rendering baked in (no ARRI data).
pub fn logc3_to_rec709_lut() -> Lut {
    logc3_to_rec709_lut_with(0.55)
}

/// [`logc3_to_rec709_lut`] with the scene brightened by `gain` before the tone curve. At 0.55 mid grey is at 0.12 in
/// display light and only the very brightest scene values reach white; at 2.2 the highlights of an ordinary
/// photograph do, which is where a DNG of such a look is least like its picture.
pub fn logc3_to_rec709_lut_with(gain: f64) -> Lut {
    // A filmic curve (Narkowicz's fit of a reference tone map), darkened to put mid grey at 0.12 in
    // display light, then gamma 2.4 as a Rec.709 display LUT has it.
    let tone = |x: f64| {
        let x = (x * gain).max(0.0);
        ((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)).clamp(0.0, 1.0)
    };
    let m = ym_color::gamut::matrices();
    let lut = Lut3d::bake(33, [0.0; 3], [1.0; 3], |code| {
        let scene = logc3_ei800_decode_rgb(code);
        let rec709 = linalg::mul_vec(&linalg::mul(&m.xyz_to_rec709, &m.awg3_to_xyz), scene);
        let shown = rec709.map(|c| tone(c).powf(1.0 / 2.4));
        // The twist: a little warmth in the mids, a little density in the blues.
        [
            (shown[0] * 1.03).min(1.0),
            shown[1],
            (shown[2] * 0.96).min(1.0),
        ]
    })
    .expect("a LUT");
    Lut::from_cube(lut)
}

/// A display LUT of our own (LogC4 in, Rec.709 gamma 2.4 out) for the Preferences choice.
pub fn display_lut() -> ArriDisplayLut {
    let m = ym_color::gamut::matrices();
    let lut = Lut3d::bake(33, [0.0; 3], [1.0; 3], |code| {
        let scene = ym_color::logc::logc4_decode_rgb(code);
        let rec709 = linalg::mul_vec(&m.awg4_to_rec709, scene);
        rec709.map(|c| {
            let x = (c * 0.7).max(0.0);
            (x / (x + 0.18)).clamp(0.0, 1.0).powf(1.0 / 2.4)
        })
    })
    .expect("a LUT");
    ArriDisplayLut::new(
        Arc::new(Lut::from_cube(lut)),
        DisplayLutInput::LogC4,
        DisplayLutOutput::REC709_G24,
    )
    .expect("a display LUT")
}

/// Neutral, a warm look, the LogC3 to Rec.709 look, and a display LUT.
pub struct Looks;

impl LookProvider for Looks {
    fn look(&self, id: &LookId) -> Result<ResolvedLook, LookProblem> {
        let transform = match id.as_str() {
            "builtin:neutral" => return NeutralOnly.look(id),
            "user:warm" => LookTransform::new(
                LookInput::LogC4,
                Some(Cdl::new([1.25, 1.0, 0.8], [0.0; 3], [1.0; 3], 1.2).expect("cdl")),
                None,
            ),
            "user:rec709" => {
                LookTransform::new(LookInput::LogC3ToRec709, None, Some(logc3_to_rec709_lut()))
            }
            // The same look with the highlights reaching display white (`logc3_to_rec709_lut_with`).
            "user:rec709bright" => LookTransform::new(
                LookInput::LogC3ToRec709,
                None,
                Some(logc3_to_rec709_lut_with(2.2)),
            ),
            _ => return Err(LookProblem("File missing or changed. Import again.".into())),
        }
        .expect("a look");
        Ok(ResolvedLook {
            transform: Arc::new(transform),
            fingerprint: id.to_string(),
        })
    }

    fn display_transform(&self, id: &LookId) -> Result<(DisplayTransform, String), LookProblem> {
        if id.as_str() == "arri-display:logc4:rec709-g24" {
            Ok((DisplayTransform::arri(display_lut()), "test".to_owned()))
        } else {
            Err(LookProblem(format!(
                "The display LUT {id} is not in the library."
            )))
        }
    }
}

/// The id of the display LUT [`Looks`] knows.
pub const DISPLAY_LUT_ID: &str = "arri-display:logc4:rec709-g24";
