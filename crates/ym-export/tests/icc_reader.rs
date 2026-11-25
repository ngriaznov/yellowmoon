//! The reader of ICC matrix/TRC profiles (`ym_color::icc`, `docs/34-reference-grade.md` section 6.3, RG-05) against profiles of our
//! own making, and its conversion to sRGB against an independent colour-management crate.
//!
//! **Two oracles.**
//!
//! 1. `moxcms` 0.8 (BSD-3-Clause OR Apache-2.0), a dev-dependency of this crate only: it reads the same bytes with its own parser,
//!    curve tables and matrices, and converts to its own sRGB profile (relative colorimetric, no black point compensation). It
//!    works on 16-bit curve tables, so its own resolution is about a code in the shadows (its sRGB-to-sRGB conversion of the sRGB
//!    profile is itself 0.96 codes off the identity): the bound [`BOUND_VS_MOXCMS`] is for what it can tell, an independent
//!    parser and a different arithmetic agreeing to a code.
//! 2. **The closed form.** Every fixture is made from primaries and a curve whose formulas are known, so the expected linear sRGB
//!    of a code is computed without a profile at all: the curve's formula, the primaries' normalised matrix (D65) and sRGB's
//!    inverse. The Bradford adaptation to D50 and back cancels, so what is left is the reader's parsing, the s15Fixed16
//!    rounding of the file and the matrix algebra: the bound is [`BOUND_VS_FORMULAS`], a few hundredths of a code.
//!
//! Both are compared on a grid of 8-bit codes (all 17 x 17 x 17 combinations of 0, 16, ..., 255 and 256 greys), in **sRGB display
//! codes of the 0 to 255 scale**, after the reader's clamp to the sRGB gamut. The measured worst differences are printed
//! (`cargo test -p ym-export --test icc_reader -- --nocapture`) and written in `docs/34` section 19.
//!
//! The fixtures: the sRGB, Display P3 and Rec.2020 profiles of the exports (`icc::profile`), a `para` variant of each function type,
//! a `curv` gamma and a `curv` table, and a profile made to the structure of the "Display" profile that macOS embeds in screenshots
//! (a version 2.4 header, `desc` as a `textDescription`, P3-like colorants, a `para` sRGB curve); it is generated here, never
//! Apple's file.

use moxcms::{ColorProfile, Layout, TransformOptions};
use ym_color::consts::{Primaries, Vec3};
use ym_color::encode::srgb_encode;
use ym_color::icc::{Curve, IccError, RgbProfile, ToLinearSrgb};
use ym_export::ColorSpace;
use ym_export::icc::{FixtureCurve, FixtureLayout, colorants_d50, fixture_profile, profile};

/// The bound of the conversion against `moxcms`, in sRGB display codes (0 to 255): measured worst 1.25, bound 1.5.
const BOUND_VS_MOXCMS: f64 = 1.5;

/// The bound against the closed form, in sRGB display codes.
const BOUND_VS_FORMULAS: f64 = 0.15;

fn srgb_curve() -> FixtureCurve {
    FixtureCurve::Para(
        3,
        vec![2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45],
    )
}

fn p3_colorants() -> [Vec3; 3] {
    colorants_d50(ColorSpace::DisplayP3).0
}

/// The profile of a "Display" like the one macOS puts in screenshots: our own, to its structure.
fn display_profile() -> Vec<u8> {
    fixture_profile(
        "Display",
        FixtureLayout::V2,
        p3_colorants(),
        [srgb_curve(), srgb_curve(), srgb_curve()],
        &[],
    )
}

/// An old-style display profile: a version 2 header, one gamma 2.2 `curv` for each channel, the Adobe RGB (1998) primaries
/// adapted to D50 (the colorants the standard tabulates).
fn adobe_like_profile() -> Vec<u8> {
    fixture_profile(
        "Wide gamut, gamma 2.2",
        FixtureLayout::V2,
        [
            [0.609_74, 0.311_11, 0.019_47],
            [0.205_28, 0.625_66, 0.060_87],
            [0.149_19, 0.063_22, 0.744_57],
        ],
        [
            FixtureCurve::Gamma(2.2),
            FixtureCurve::Gamma(2.2),
            FixtureCurve::Gamma(2.2),
        ],
        &[],
    )
}

/// A `curv` table of 1024 entries that follows the sRGB curve.
fn table_profile() -> Vec<u8> {
    let table: Vec<u16> = (0..1024)
        .map(|i| (ym_color::encode::srgb_decode(f64::from(i) / 1023.0) * 65535.0).round() as u16)
        .collect();
    fixture_profile(
        "Display, table",
        FixtureLayout::V2,
        p3_colorants(),
        [
            FixtureCurve::Table(table.clone()),
            FixtureCurve::Table(table.clone()),
            FixtureCurve::Table(table),
        ],
        &[],
    )
}

/// One profile per `para` function type, with the same colorants (Display P3's) and curves that are not the sRGB one.
fn para_profiles() -> Vec<(String, Vec<u8>)> {
    let curves = [
        FixtureCurve::Para(0, vec![2.2]),
        FixtureCurve::Para(1, vec![2.2, 0.95, 0.05]),
        FixtureCurve::Para(2, vec![2.2, 0.95, 0.05, 0.01]),
        srgb_curve(),
        FixtureCurve::Para(
            4,
            vec![
                2.4,
                1.0 / 1.055,
                0.055 / 1.055,
                1.0 / 12.92,
                0.040_45,
                0.0,
                0.0,
            ],
        ),
        FixtureCurve::Para(4, vec![2.2, 0.9, 0.1, 0.06, 0.08, 0.02, 0.0]),
    ];
    curves
        .into_iter()
        .enumerate()
        .map(|(i, c)| {
            (
                format!("para variant {i}"),
                fixture_profile(
                    "para",
                    FixtureLayout::V4,
                    p3_colorants(),
                    [c.clone(), c.clone(), c],
                    &[],
                ),
            )
        })
        .collect()
}

fn all_profiles() -> Vec<(String, Vec<u8>)> {
    let mut v = vec![
        ("sRGB".to_owned(), profile(ColorSpace::Srgb).to_vec()),
        (
            "Display P3".to_owned(),
            profile(ColorSpace::DisplayP3).to_vec(),
        ),
        ("Rec.2020".to_owned(), profile(ColorSpace::Rec2020).to_vec()),
        ("screenshot Display".to_owned(), display_profile()),
        ("gamma 2.2, wide".to_owned(), adobe_like_profile()),
        ("curv table".to_owned(), table_profile()),
    ];
    v.extend(para_profiles());
    v
}

/// The codes compared: every 17 x 17 x 17 combination of 0, 16, ..., 255 and a ramp of 256 greys.
fn grid() -> Vec<[u8; 3]> {
    let mut g = Vec::new();
    let steps: Vec<u8> = (0..=16u32).map(|i| (i * 255 / 16) as u8).collect();
    for &r in &steps {
        for &gr in &steps {
            for &b in &steps {
                g.push([r, gr, b]);
            }
        }
    }
    for v in 0..=255u8 {
        g.push([v, v, v]);
    }
    g
}

/// The reader's sRGB display codes (0 to 255) for `codes`.
fn ours(profile: &[u8], codes: &[[u8; 3]]) -> Vec<[f64; 3]> {
    let t = RgbProfile::parse(profile)
        .expect("the reader takes it")
        .to_linear_srgb();
    let tables = [t.table(0, 256), t.table(1, 256), t.table(2, 256)];
    codes
        .iter()
        .map(|c| {
            let lin = t.convert([
                tables[0][usize::from(c[0])],
                tables[1][usize::from(c[1])],
                tables[2][usize::from(c[2])],
            ]);
            [
                srgb_encode(lin[0]) * 255.0,
                srgb_encode(lin[1]) * 255.0,
                srgb_encode(lin[2]) * 255.0,
            ]
        })
        .collect()
}

/// The oracle's sRGB display codes (0 to 255), clamped to the gamut like ours.
fn oracle(profile: &[u8], codes: &[[u8; 3]]) -> Vec<[f64; 3]> {
    let src = ColorProfile::new_from_slice(profile).expect("the oracle takes it");
    let dst = ColorProfile::new_srgb();
    let transform = src
        .create_transform_f64(Layout::Rgb, &dst, Layout::Rgb, TransformOptions::default())
        .expect("the oracle makes the transform");
    let input: Vec<f64> = codes
        .iter()
        .flat_map(|c| c.iter().map(|&v| f64::from(v) / 255.0))
        .collect();
    let mut output = vec![0.0f64; input.len()];
    transform
        .transform(&input, &mut output)
        .expect("the oracle converts");
    output
        .chunks_exact(3)
        .map(|p| {
            [
                p[0].clamp(0.0, 1.0) * 255.0,
                p[1].clamp(0.0, 1.0) * 255.0,
                p[2].clamp(0.0, 1.0) * 255.0,
            ]
        })
        .collect()
}

#[test]
fn the_conversion_agrees_with_the_oracle_on_every_fixture() {
    let codes = grid();
    let mut worst_overall = 0.0f64;
    for (name, bytes) in all_profiles() {
        let a = ours(&bytes, &codes);
        let b = oracle(&bytes, &codes);
        let mut worst = 0.0f64;
        let mut at = [0u8; 3];
        for ((x, y), code) in a.iter().zip(&b).zip(&codes) {
            for c in 0..3 {
                let d = (x[c] - y[c]).abs();
                if d > worst {
                    worst = d;
                    at = *code;
                }
            }
        }
        println!("{name:<20} worst difference {worst:.5} codes at {at:?}");
        assert!(
            worst <= BOUND_VS_MOXCMS,
            "{name}: {worst} codes at {at:?} (bound {BOUND_VS_MOXCMS})"
        );
        worst_overall = worst_overall.max(worst);
    }
    println!(
        "worst against moxcms over all fixtures: {worst_overall:.5} sRGB codes (bound {BOUND_VS_MOXCMS})"
    );
}

#[test]
fn the_srgb_export_profile_is_the_identity_in_srgb() {
    let t = RgbProfile::parse(profile(ColorSpace::Srgb))
        .expect("reads")
        .to_linear_srgb();
    let m = t.matrix().expect("a profile has a matrix");
    for (r, row) in m.iter().enumerate() {
        for (c, v) in row.iter().enumerate() {
            let want = if r == c { 1.0 } else { 0.0 };
            // The colorants are stored to 1.5e-5.
            assert!((v - want).abs() < 2e-4, "{r}{c}: {v}");
        }
    }
    // A colour of Display P3 is moved: its pure red is outside sRGB and clamps to the edge.
    let p3 = RgbProfile::parse(profile(ColorSpace::DisplayP3))
        .expect("reads")
        .to_linear_srgb();
    let red = p3.convert([1.0, 0.0, 0.0]);
    assert_eq!(red[0], 1.0);
    assert_eq!(red[1], 0.0);
    // A grey is a grey.
    for g in [0.0, 0.05, 0.5, 1.0] {
        let out = p3.convert([g; 3]);
        assert!(out.iter().all(|c| (c - g).abs() < 2e-4), "{g}: {out:?}");
    }
}

#[test]
fn the_primaries_read_back_from_a_profile_are_the_ones_it_was_made_from() {
    // A profile of Rec.2020 colorants, read and converted, maps the Rec.2020 primaries to what the matrix of the primaries says.
    let want = ym_color::icc::RgbProfile::from_primaries(&Primaries::REC2020, Curve::Gamma(2.4))
        .to_linear_srgb();
    let got = RgbProfile::parse(profile(ColorSpace::Rec2020))
        .expect("reads")
        .to_linear_srgb();
    let (a, b) = (want.matrix().expect("m"), got.matrix().expect("m"));
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (a[r][c] - b[r][c]).abs() < 2e-3,
                "{r}{c}: {} {}",
                a[r][c],
                b[r][c]
            );
        }
    }
}

#[test]
fn a_profile_with_a_lookup_table_or_damage_is_refused() {
    let lut = fixture_profile(
        "with a table",
        FixtureLayout::V4,
        p3_colorants(),
        [srgb_curve(), srgb_curve(), srgb_curve()],
        &[(*b"A2B0", vec![0; 64])],
    );
    assert_eq!(RgbProfile::parse(&lut), Err(IccError::HasLookupTable));
    let mut cut = display_profile();
    cut.truncate(cut.len() / 2);
    assert!(RgbProfile::parse(&cut).is_err());
    let mut bad_sig = display_profile();
    bad_sig[36] = b'x';
    assert_eq!(RgbProfile::parse(&bad_sig), Err(IccError::NotAProfile));
    let mut grey = display_profile();
    grey[16..20].copy_from_slice(b"GRAY");
    assert_eq!(RgbProfile::parse(&grey), Err(IccError::NotRgbXyz));
    // Cut at every length: the reader answers, it never panics.
    let whole = profile(ColorSpace::DisplayP3);
    for n in 0..whole.len() {
        let _ = RgbProfile::parse(&whole[..n]);
    }
    // And with each byte changed in turn (the tag table, the colorants and the curves).
    let mut d = whole.to_vec();
    for i in 0..d.len() {
        let was = d[i];
        d[i] = !was;
        let _ = RgbProfile::parse(&d);
        d[i] = was;
    }
}

#[test]
fn the_conversion_is_the_same_bits_every_time() {
    let codes = grid();
    for (_, bytes) in all_profiles() {
        let a = ours(&bytes, &codes);
        let b = ours(&bytes, &codes);
        assert_eq!(a, b);
    }
    // A cICP picture of the exports' code points converts like the exports' own profile does.
    for (space, cicp) in [
        (ColorSpace::DisplayP3, [12, 13, 0, 1]),
        (ColorSpace::Rec2020, [9, 1, 0, 1]),
    ] {
        let from_profile = RgbProfile::parse(profile(space))
            .expect("reads")
            .to_linear_srgb();
        let from_cicp = ToLinearSrgb::from_cicp(cicp).expect("handled");
        let (a, b) = (
            from_profile.matrix().expect("m"),
            from_cicp.matrix().expect("m"),
        );
        for r in 0..3 {
            for c in 0..3 {
                assert!((a[r][c] - b[r][c]).abs() < 2e-3, "{space:?} {r}{c}");
            }
        }
        for code in [0u8, 40, 128, 255] {
            let x = f64::from(code) / 255.0;
            let (ta, tb) = (from_profile.table(0, 256), from_cicp.table(0, 256));
            assert!(
                (ta[usize::from(code)] - tb[usize::from(code)]).abs() < 3e-5,
                "{space:?} {x}"
            );
        }
    }
}

/// The linear value of a `para` curve of `kind` with `p` at `x`, written out from ICC.1 section 10.18 (not through the reader).
fn para_formula(kind: u16, p: &[f64], x: f64) -> f64 {
    let g = p[0];
    match kind {
        0 => x.powf(g),
        1 => {
            let (a, b) = (p[1], p[2]);
            if x >= -b / a {
                (a * x + b).powf(g)
            } else {
                0.0
            }
        }
        2 => {
            let (a, b, c) = (p[1], p[2], p[3]);
            if x >= -b / a {
                (a * x + b).powf(g) + c
            } else {
                c
            }
        }
        3 => {
            let (a, b, c, d) = (p[1], p[2], p[3], p[4]);
            if x >= d { (a * x + b).powf(g) } else { c * x }
        }
        _ => {
            let (a, b, c, d, e, f) = (p[1], p[2], p[3], p[4], p[5], p[6]);
            if x >= d {
                (a * x + b).powf(g) + e
            } else {
                c * x + f
            }
        }
    }
}

/// The expected sRGB display codes of `codes` for a fixture made of `primaries` (white D65) and the curve function `curve`: closed
/// form, no profile.
fn formulas(primaries: &Primaries, curve: &dyn Fn(f64) -> f64, codes: &[[u8; 3]]) -> Vec<[f64; 3]> {
    let m = ym_color::consts::linalg::mul(&Primaries::REC709.xyz_to_rgb(), &primaries.rgb_to_xyz());
    codes
        .iter()
        .map(|c| {
            let lin = [
                curve(f64::from(c[0]) / 255.0),
                curve(f64::from(c[1]) / 255.0),
                curve(f64::from(c[2]) / 255.0),
            ];
            let v = ym_color::consts::linalg::mul_vec(&m, lin);
            [
                srgb_encode(v[0].clamp(0.0, 1.0)) * 255.0,
                srgb_encode(v[1].clamp(0.0, 1.0)) * 255.0,
                srgb_encode(v[2].clamp(0.0, 1.0)) * 255.0,
            ]
        })
        .collect()
}

#[test]
fn the_conversion_agrees_with_the_closed_form_to_a_few_hundredths_of_a_code() {
    let codes = grid();
    let adobe = Primaries {
        red: [0.64, 0.33],
        green: [0.21, 0.71],
        blue: [0.15, 0.06],
        white: ym_color::consts::D65_XY,
    };
    let srgb_fn = |x: f64| ym_color::encode::srgb_decode(x);
    /// A fixture: its name, its bytes, the primaries it was made from and the formula of its curve.
    type Case = (String, Vec<u8>, Primaries, Box<dyn Fn(f64) -> f64>);
    let mut cases: Vec<Case> = vec![
        (
            "sRGB".into(),
            profile(ColorSpace::Srgb).to_vec(),
            Primaries::REC709,
            Box::new(srgb_fn),
        ),
        (
            "Display P3".into(),
            profile(ColorSpace::DisplayP3).to_vec(),
            Primaries::P3_D65,
            Box::new(srgb_fn),
        ),
        (
            "Rec.2020".into(),
            profile(ColorSpace::Rec2020).to_vec(),
            Primaries::REC2020,
            Box::new(|x: f64| x.powf(2.4)),
        ),
        (
            "screenshot Display".into(),
            display_profile(),
            Primaries::P3_D65,
            Box::new(srgb_fn),
        ),
        (
            "gamma 2.2, wide".into(),
            adobe_like_profile(),
            adobe,
            // A `curv` gamma is u8Fixed8: 2.2 is stored as 563 / 256.
            Box::new(|x: f64| x.powf(563.0 / 256.0)),
        ),
        (
            "curv table".into(),
            table_profile(),
            Primaries::P3_D65,
            Box::new(srgb_fn),
        ),
    ];
    let kinds: Vec<(u16, Vec<f64>)> = vec![
        (0, vec![2.2]),
        (1, vec![2.2, 0.95, 0.05]),
        (2, vec![2.2, 0.95, 0.05, 0.01]),
        (
            3,
            vec![2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.040_45],
        ),
        (
            4,
            vec![
                2.4,
                1.0 / 1.055,
                0.055 / 1.055,
                1.0 / 12.92,
                0.040_45,
                0.0,
                0.0,
            ],
        ),
        (4, vec![2.2, 0.9, 0.1, 0.06, 0.08, 0.02, 0.0]),
    ];
    for ((name, bytes), (kind, params)) in para_profiles().into_iter().zip(kinds) {
        cases.push((
            name,
            bytes,
            Primaries::P3_D65,
            Box::new(move |x: f64| para_formula(kind, &params, x).clamp(0.0, 1.0)),
        ));
    }
    let mut worst_overall = 0.0f64;
    for (name, bytes, primaries, curve) in &cases {
        let a = ours(bytes, &codes);
        let b = formulas(primaries, curve.as_ref(), &codes);
        let mut worst = 0.0f64;
        let mut at = [0u8; 3];
        for ((x, y), code) in a.iter().zip(&b).zip(&codes) {
            for c in 0..3 {
                let d = (x[c] - y[c]).abs();
                if d > worst {
                    worst = d;
                    at = *code;
                }
            }
        }
        println!("{name:<20} against the closed form: {worst:.5} codes at {at:?}");
        assert!(
            worst <= BOUND_VS_FORMULAS,
            "{name}: {worst} codes at {at:?}"
        );
        worst_overall = worst_overall.max(worst);
    }
    println!(
        "worst against the closed form over all fixtures: {worst_overall:.5} sRGB codes (bound {BOUND_VS_FORMULAS})"
    );
}
