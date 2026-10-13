//! The model of Adobe's renderer (`src/model.rs`) against `dng_validate`, on random colours with a random look table and
//! a non-trivial tone curve. This is what the analytic parts of the fit stand on: the table's HSV maths, the interpolation,
//! the value scale in the encoded domain (found by this test: a linear value scale was off by up to 0.12 in code values),
//! the RGB tone stage and the colour matrices.
//!
//! Needs `dng_validate`; says `SKIPPED` and passes without it unless `YM_REQUIRE_EXTERNAL_TOOLS` is set.

use dngfit::adobe::Adobe;
use dngfit::fit::{TONE_GRID, measure_tone};
use dngfit::model::{Model, ToneTable};
use dngfit::samples::Rng;
use ym_export::dng::{DngProfile, LookTable};

fn adobe() -> Option<Adobe> {
    match Adobe::new(None) {
        Ok(a) => Some(a),
        Err(e) => {
            assert!(
                std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
                "dng_validate is required: {e:#}"
            );
            eprintln!("SKIPPED: {e:#}");
            None
        }
    }
}

fn random_table(rng: &mut Rng, dims: [u32; 3]) -> LookTable {
    let mut data = Vec::new();
    for _v in 0..dims[2] {
        for _h in 0..dims[0] {
            for s in 0..dims[1] {
                if s == 0 {
                    data.extend_from_slice(&[0.0, 1.0, 1.0]);
                } else {
                    data.extend_from_slice(&[
                        (20.0 * (rng.uniform() - 0.5)) as f32,
                        (0.8 + 0.4 * rng.uniform()) as f32,
                        (0.8 + 0.4 * rng.uniform()) as f32,
                    ]);
                }
            }
        }
    }
    LookTable {
        dims,
        data,
        encoding: 1,
    }
}

#[test]
fn the_model_reproduces_adobes_renderer() {
    let Some(adobe) = adobe() else { return };
    let offset = -4.0_f64;
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let profile = DngProfile {
        tone_curve: vec![
            [0.0, 0.0],
            [0.05, 0.2],
            [0.2, 0.55],
            [0.5, 0.82],
            [1.0, 1.0],
        ],
        look_table: random_table(&mut rng, [12, 6, 5]),
    };
    profile.validate().unwrap();
    // the tone table as the renderer has it (neutrals, identity look), then the model with the random look table
    let tone: ToneTable = measure_tone(&adobe, &profile.tone_curve, 16.0).unwrap();
    assert_eq!(tone.samples.len(), TONE_GRID + 1);
    assert!(
        tone.samples.windows(2).all(|w| w[1] >= w[0] - 1e-9),
        "the curve is monotone"
    );
    let model = Model::new(tone, profile.look_table.clone(), offset);

    let mut colors = Vec::new();
    for _ in 0..3000 {
        let m = 16.0 * rng.uniform().powi(2);
        colors.push([
            (m * rng.uniform()) as f32,
            (m * rng.uniform()) as f32,
            (m * rng.uniform()) as f32,
        ]);
    }
    let r = adobe.render(&colors, &profile, offset).unwrap();
    assert_eq!(r.report.highlight_clipped_fraction, 0.0);
    let mut diffs: Vec<f64> = colors
        .iter()
        .zip(&r.srgb)
        .map(|(c, got)| {
            let want = model.render(c.map(f64::from));
            (0..3)
                .map(|k| (want[k] - f64::from(got[k])).abs())
                .fold(0.0, f64::max)
        })
        .collect();
    diffs.sort_by(f64::total_cmp);
    let q = |f: f64| diffs[((diffs.len() - 1) as f64 * f) as usize];
    // encoded values: 1/255 is 0.0039
    assert!(q(0.5) < 5e-4, "median {}", q(0.5));
    assert!(q(0.99) < 3e-3, "p99 {}", q(0.99));
    assert!(q(1.0) < 8e-3, "max {}", q(1.0));
}

#[test]
fn a_negative_baseline_exposure_makes_the_renderer_shoulder_the_highlights() {
    // The reason the fit needs x_white >= 2^-offset: with a negative BaselineExposure Adobe's renderer is not a gain. The
    // writer's `Auto` rule floors x_white so that this never happens, so the DNG here is written with a fixed
    // BaselineExposure of -3 (x_white = 1, the stored values are the scene values).
    let Some(adobe) = adobe() else { return };
    let values = [0.05_f32, 0.1, 0.2, 0.4, 0.6, 0.84, 1.0];
    let px: Vec<f32> = values.iter().flat_map(|&v| [v; 3]).collect();
    let (bytes, report) = dngfit::adobe::write_dng_image_exposure(
        values.len(),
        1,
        &px,
        &DngProfile::identity(),
        ym_export::dng::BaselineExposure::Fixed(-3.0),
        1,
        &ym_export::Metadata::default(),
    )
    .unwrap();
    assert!(
        (report.baseline_exposure_ev + 3.0).abs() < 0.01,
        "{report:?}"
    );
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("shoulder.dng");
    std::fs::write(&path, bytes).unwrap();
    let (_, _, out) = adobe.render_file(&path).unwrap();
    let linear = |i: usize| ym_color::encode::srgb_decode(f64::from(out[i * 3 + 1]));
    // a gain of 1/8 would give v / 8; the shoulder is linear below a quarter of the range and then rises above it
    for (i, &v) in values.iter().enumerate() {
        let plain = f64::from(v) / 8.0;
        if v <= 0.2 {
            assert!(
                (linear(i) - plain).abs() < 2e-3 * plain.max(0.01),
                "{v}: {} vs {plain}",
                linear(i)
            );
        }
    }
    assert!(
        linear(3) > 1.5 * (0.4 / 8.0),
        "0.4 renders at {} instead of 0.05",
        linear(3)
    );
    assert!(
        linear(5) > 5.0 * (0.84 / 8.0),
        "0.84 renders at {} instead of 0.105",
        linear(5)
    );
    assert!(linear(6) > 0.95, "white stays white: {}", linear(6));
}
