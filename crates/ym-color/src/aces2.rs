// SPDX-License-Identifier: Apache-2.0
// Copyright Contributors to the ACES Project (the reference this module ports) and Yellowmoon contributors.

//! ACES 2.0 output transform, CPU reference for the WGSL port (stage C12 of
//! `docs/02-colour-pipeline.md`).
//!
//! # Attribution
//!
//! This module is a Rust port of the ACES 2.0 Output Transform.
//!
//! - Source: the CTL reference of the Academy Software Foundation's ACES project, library files
//!   `lib/Lib.Academy.OutputTransform.ctl`, `Lib.Academy.Tonescale.ctl`,
//!   `Lib.Academy.DisplayEncoding.ctl`, `Lib.Academy.Utilities.ctl` and `Lib.Academy.ColorSpaces.ctl`
//!   from <https://github.com/aces-aswf/aces-core> at commit
//!   `069b0bc3e1f6c62820f19fdae2fecec3f4fc0f80`, and the output presets in
//!   <https://github.com/aces-aswf/aces-output> at commit `6d8f9071a67b044bac0fbcb3d51ad0543f065e66`
//!   (`Output.Academy.P3-D65_100nit_in_P3-D65_sRGB-Piecewise.ctl` and its Rec.709 siblings).
//! - Licence: Apache-2.0. Copyright Contributors to the ACES Project. The licence text is in the
//!   repository's `LICENSE` file and the notice is repeated in `NOTICE`.
//! - Cross-checked against OpenColorIO's C++ implementation (BSD-3-Clause,
//!   `src/OpenColorIO/ops/fixedfunction/ACES2/`), which generates the test vectors in
//!   `tests/data/aces2/`.
//!
//! This is a derivative work: the maths, constants and table generation are the reference's; the
//! structure, naming and f64 arithmetic are ours.
//!
//! # What it does
//!
//! Linear ACES2065-1 (AP0) in, display-encoded RGB in [0, 1] out, for 100 nit SDR:
//!
//! | [`Aces2Output`] | Limiting primaries | Encoding primaries | Transfer |
//! |---|---|---|---|
//! | `Viewport` | P3-D65 | P3-D65 (Display P3) | sRGB |
//! | `Srgb` | Rec.709 | Rec.709 | sRGB |
//! | `DisplayP3` | P3-D65 | P3-D65 | sRGB |
//! | `Rec2020` | P3-D65 | Rec.2020 | BT.1886 (gamma 2.4) |
//!
//! Steps: clamp to the AP1 range the model accepts, convert to JMh (lightness, colourfulness,
//! hue) in the Hellwig 2022 appearance model, apply the tone scale to J, compress M (in-gamut chroma
//! compression), compress the gamut towards the limiting primaries, convert back to RGB of the
//! limiting primaries, clamp to [0, peak], convert to the encoding primaries and apply the inverse
//! EOTF.
//!
//! # Numerics
//!
//! The CTL runs in 32-bit `float`, and so does OpenColorIO. This port runs in f64 (the CPU
//! reference convention) with the reference's algorithms and iteration tolerances unchanged, so
//! tables built by bisection match the reference to its own tolerance (reach M 1e-2, upper hull
//! gamma 1e-5, cusp search 1e-7, reach corners 1e-3). The f32 reference carries about 1e-6
//! relative noise; the gate against it is 1 code value at 10 bits (about 1e-3).
//!
//! Deviations from the CTL, all below the noise of the f32 reference unless noted:
//!
//! - Hue interpolation weights are normalised by the hue interval (`(h - h_lo) / (h_hi - h_lo)`),
//!   for the cusp and the upper hull gamma alike, as OpenColorIO does. The CTL's
//!   `interpolation_weight` returns `h - h_lo` for the gamma and only agrees where the hue spacing
//!   is exactly 1 degree.
//! - The hue tables have a third wrap entry (as OpenColorIO's), so a hue that rounds to exactly
//!   360.0 is looked up without a modulo, and the hue interval search is windowed on the 360-entry
//!   nominal grid instead of the CTL's 362-entry one. The interval found is the same.
//! - The cusp bisection always starts at t = 0, as the CTL does (OpenColorIO restarts from the
//!   previous hue's result, within the 1e-7 tolerance).
//! - NaN input is treated as 0 and infinities are limited.
//! - The peak luminance is limited to [`MIN_PEAK_LUMINANCE_NITS`]..=[`MAX_PEAK_LUMINANCE_NITS`]
//!   (48 to 10 000 nits; OpenColorIO accepts 1 to 10 000, but below 46.4 nits the tone scale roof is
//!   negative). The peak-dependent maths is checked against OpenColorIO at five peaks in
//!   `tests/aces2_hdr_vectors.rs`; the four outputs are all 100 nit.
//! - NaN guards in the gamut compression. Where the tone-scaled J lands on the display's maximum J
//!   (scene values of about 128 times white, `r_hit`) the compression line's axis intersection can
//!   round a hair above the maximum, and the reference takes `pow` of a tiny negative number. In
//!   float32 that is NaN, which OpenColorIO turns into a black pixel in a thin band of scene
//!   luminance (for example AP0 `[109.496, 135.590, 239.389]`). Here the ratio is clamped at 0 and
//!   a NaN or non-positive boundary M means "no colour left", so the pixel becomes white as the
//!   limit demands. Nothing else changes. A float32 GPU port needs the same guards.
//!
//! # Agreement with the reference
//!
//! Against OpenColorIO 2.6 (float32): at most 0.03 code values at 10 bits over the committed
//! vectors and 0.05 over random sweeps of 1.9 million samples per output (gate: 1.0), and the
//! pre-computed tables agree with the reference's float32 tables to 1e-5 relative (hue, cusp,
//! reach) and 8e-5 (upper hull gamma, whose bisection tolerance is 1e-5 absolute).
//!
//! # Cost
//!
//! Building the tables takes 10 to 35 ms. [`Aces2Transform::new`] caches the two SDR parameter sets
//! (Rec.709 and P3-D65 limited) for the process, so repeated construction is cheap and always
//! returns identical tables. A pixel costs about 1 microsecond on one thread of the development
//! machine (`tests/aces2_perf.rs`).

mod cam;
mod compress;
mod display;
mod matrix;
mod params;
mod tables;
mod tonescale;

use std::sync::{Arc, OnceLock};

pub use crate::consts::Primaries;
pub use cam::{JMhParams, REF_LUMINANCE};
pub use display::DisplayTransfer;
pub use params::{
    Aces2Params, BASE_INDEX, MAX_PEAK_LUMINANCE_NITS, MIN_PEAK_LUMINANCE_NITS, TABLE_SIZE,
    TOTAL_TABLE_SIZE, UPPER_WRAP_INDEX,
};
pub use tonescale::ToneScaleParams;

use matrix::{IDENTITY, M33, ctl_rgb_to_xyz, ctl_xyz_to_rgb, mul_mat_mat, mul_vec_mat};

/// Peak luminance of every Yellowmoon SDR output, in nits.
pub const SDR_PEAK_LUMINANCE_NITS: f64 = 100.0;

/// One of the four display outputs of `docs/02-colour-pipeline.md` section C12.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Aces2Output {
    /// The viewport: P3-D65 limited, Display P3 primaries, sRGB transfer.
    Viewport,
    /// sRGB export: Rec.709 limited, Rec.709 primaries, sRGB transfer.
    Srgb,
    /// Display P3 export: P3-D65 limited, Display P3 primaries, sRGB transfer (same as the viewport).
    DisplayP3,
    /// Rec.2020 export: P3-D65 limited, Rec.2020 primaries, BT.1886 gamma 2.4.
    Rec2020,
}

impl Aces2Output {
    /// All four outputs, in the order of the table in the spec.
    pub const ALL: [Aces2Output; 4] = [
        Aces2Output::Viewport,
        Aces2Output::Srgb,
        Aces2Output::DisplayP3,
        Aces2Output::Rec2020,
    ];

    /// Stable lower-case name (matches the file names under `tests/data/aces2/`).
    pub fn name(self) -> &'static str {
        match self {
            Aces2Output::Viewport => "viewport",
            Aces2Output::Srgb => "srgb",
            Aces2Output::DisplayP3 => "display_p3",
            Aces2Output::Rec2020 => "rec2020",
        }
    }

    /// Primaries the tone and gamut mapping limits the image to.
    pub fn limiting_primaries(self) -> Primaries {
        match self {
            Aces2Output::Srgb => Primaries::REC709,
            Aces2Output::Viewport | Aces2Output::DisplayP3 | Aces2Output::Rec2020 => {
                Primaries::P3_D65
            }
        }
    }

    /// Primaries the output is encoded in.
    pub fn encoding_primaries(self) -> Primaries {
        match self {
            Aces2Output::Srgb => Primaries::REC709,
            Aces2Output::Viewport | Aces2Output::DisplayP3 => Primaries::P3_D65,
            Aces2Output::Rec2020 => Primaries::REC2020,
        }
    }

    /// Transfer function the output is encoded with.
    pub fn transfer(self) -> DisplayTransfer {
        match self {
            Aces2Output::Rec2020 => DisplayTransfer::Bt1886,
            _ => DisplayTransfer::Srgb,
        }
    }
}

/// The ACES 2 output transform for one [`Aces2Output`], with its tables built.
#[derive(Clone, Debug)]
pub struct Aces2Transform {
    output: Aces2Output,
    params: Arc<Aces2Params>,
    /// Limiting primaries RGB to encoding primaries RGB (row-vector convention).
    limit_to_display: M33,
    /// Largest display-linear value: peak luminance over the 100 nit reference.
    peak_linear: f64,
}

impl Aces2Transform {
    /// Builds (or reuses the cached tables for) the transform of `output`.
    ///
    /// Deterministic: two calls return bit-identical tables.
    pub fn new(output: Aces2Output) -> Self {
        let limiting = output.limiting_primaries();
        let encoding = output.encoding_primaries();
        let params = Aces2Params::cached_sdr(limiting);
        let limit_to_display = if limiting == encoding {
            IDENTITY
        } else {
            mul_mat_mat(
                &ctl_rgb_to_xyz(&limiting, 1.0),
                &ctl_xyz_to_rgb(&encoding, 1.0),
            )
        };
        Aces2Transform {
            output,
            peak_linear: params.peak_luminance_nits / REF_LUMINANCE,
            params,
            limit_to_display,
        }
    }

    /// Process-wide shared transform for `output`, built on first use.
    pub fn shared(output: Aces2Output) -> &'static Aces2Transform {
        static SHARED: [OnceLock<Aces2Transform>; 4] = [
            OnceLock::new(),
            OnceLock::new(),
            OnceLock::new(),
            OnceLock::new(),
        ];
        let slot = match output {
            Aces2Output::Viewport => &SHARED[0],
            Aces2Output::Srgb => &SHARED[1],
            Aces2Output::DisplayP3 => &SHARED[2],
            Aces2Output::Rec2020 => &SHARED[3],
        };
        slot.get_or_init(|| Aces2Transform::new(output))
    }

    /// The output this transform renders.
    pub fn output(&self) -> Aces2Output {
        self.output
    }

    /// The pre-computed tables and constants (for GPU ports and tests).
    pub fn params(&self) -> &Aces2Params {
        &self.params
    }

    /// Linear ACES2065-1 to display-linear RGB of the *limiting* primaries, clamped to
    /// [0, peak / 100 nit] (1.0 is 100 nits).
    pub fn transform_ap0_limiting_linear(&self, ap0: [f64; 3]) -> [f64; 3] {
        let rgb = self.params.output_transform_fwd(ap0);
        [
            rgb[0].clamp(0.0, self.peak_linear),
            rgb[1].clamp(0.0, self.peak_linear),
            rgb[2].clamp(0.0, self.peak_linear),
        ]
    }

    /// Reference behaviour of the step between the limiting and the encoding primaries: converts the
    /// clamped limiting RGB to the encoding primaries and clamps below at 0 only.
    ///
    /// The reference does not clamp above. Only the Rec.2020 output can exceed `peak / 100 nit`,
    /// by up to about 1.2e-3 (saturated cyan), because the P3 red primary lies marginally outside
    /// Rec.2020 and so the matrix gives a small negative contribution to blue.
    fn encoding_linear_unclamped(&self, ap0: [f64; 3]) -> [f64; 3] {
        let limited = self.transform_ap0_limiting_linear(ap0);
        let rgb = if self.output.limiting_primaries() == self.output.encoding_primaries() {
            limited
        } else {
            mul_vec_mat(limited, &self.limit_to_display)
        };
        [rgb[0].max(0.0), rgb[1].max(0.0), rgb[2].max(0.0)]
    }

    /// Linear ACES2065-1 to display-linear RGB in the output's *encoding* primaries (what the display
    /// emits, 1.0 = 100 nits), before the transfer function, clamped to [0, peak / 100 nit]. This
    /// is the value the ARRI display LUT blend and the DNG fit work with.
    pub fn transform_ap0_linear(&self, ap0: [f64; 3]) -> [f64; 3] {
        let rgb = self.encoding_linear_unclamped(ap0);
        [
            rgb[0].min(self.peak_linear),
            rgb[1].min(self.peak_linear),
            rgb[2].min(self.peak_linear),
        ]
    }

    /// Linear ACES2065-1 to display-encoded RGB in [0, 1] (primaries and transfer of the output).
    pub fn transform_ap0(&self, ap0: [f64; 3]) -> [f64; 3] {
        self.encode_display_linear(self.transform_ap0_linear(ap0))
    }

    /// Same as [`Aces2Transform::transform_ap0`] without the final clamp to 1.0, which is what the
    /// reference computes. Differs from `transform_ap0` only for saturated cyan in the Rec.2020
    /// output, by up to about 5e-4 (1.00121 in linear light is 1.000504 encoded: half a 10-bit
    /// code value). Used to compare against the reference without hiding that overshoot.
    pub fn transform_ap0_unclamped(&self, ap0: [f64; 3]) -> [f64; 3] {
        let t = self.output.transfer();
        let lin = self.encoding_linear_unclamped(ap0);
        [t.encode(lin[0]), t.encode(lin[1]), t.encode(lin[2])]
    }

    /// Applies the output's transfer function to display-linear RGB (1.0 = 100 nits). The result is
    /// clamped to [0, 1].
    pub fn encode_display_linear(&self, linear: [f64; 3]) -> [f64; 3] {
        let t = self.output.transfer();
        [
            t.encode(linear[0]).clamp(0.0, 1.0),
            t.encode(linear[1]).clamp(0.0, 1.0),
            t.encode(linear[2]).clamp(0.0, 1.0),
        ]
    }

    /// Inverse of [`Aces2Transform::encode_display_linear`]: display-encoded RGB to display-linear.
    pub fn decode_to_display_linear(&self, encoded: [f64; 3]) -> [f64; 3] {
        let t = self.output.transfer();
        [
            t.decode(encoded[0]),
            t.decode(encoded[1]),
            t.decode(encoded[2]),
        ]
    }
}
