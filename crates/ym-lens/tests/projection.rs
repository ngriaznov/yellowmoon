//! The spectral projection of the lens data on a camera's channels (`docs/21` section 6, "Colour").

mod common;

use ym_color::spectral::SpectralCurves;
use ym_lens::{CameraResponse, LensError};

fn response(file: &str) -> CameraResponse {
    let curves = SpectralCurves::from_path(common::curves_path(file)).expect("curves");
    CameraResponse::from_curves(&curves).expect("projection")
}

#[test]
fn weights_sum_to_one_per_channel() {
    for file in [
        "Sony_ILCE-7RM4_380_780_5.json",
        "Sony_ILCE-6400_380_780_5.json",
        "Sony_DSC-RX100M4_380_780_5.json",
    ] {
        let r = response(file);
        for (c, row) in r.weights().iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-12, "{file} channel {c}: {sum}");
            assert!(row.iter().all(|w| *w >= 0.0 && w.is_finite()));
        }
    }
}

#[test]
fn a_flat_spectrum_projects_neutral() {
    for file in [
        "Sony_ILCE-7RM4_380_780_5.json",
        "Sony_ILCE-6400_380_780_5.json",
    ] {
        let r = response(file);
        for level in [0.0, 0.25, 1.0] {
            let t = r.tint(&[level; 31]);
            for v in t {
                assert!((v - level).abs() < 1e-12, "{file}: {t:?}");
            }
        }
        // The same quantity at every PSF wavelength projects to itself.
        let p = r.project(&[3.5; 8]);
        for v in p {
            assert!((v - 3.5).abs() < 1e-12);
        }
    }
    let n = CameraResponse::neutral();
    assert_eq!(n.tint(&[0.5; 31]), [0.5; 3]);
    for row in n.weights() {
        assert!(row.iter().all(|w| (*w - 0.125).abs() < 1e-15));
    }
}

#[test]
fn channels_differ_and_follow_the_curves() {
    let r = response("Sony_ILCE-7RM4_380_780_5.json");
    // A reflectance that is 1 only at the red end tints R far more than B, and the reverse at the blue end.
    let mut red_end = [0.0; 31];
    red_end[24..].fill(1.0); // 640 nm and above
    let mut blue_end = [0.0; 31];
    blue_end[..7].fill(1.0); // 460 nm and below
    let tr = r.tint(&red_end);
    let tb = r.tint(&blue_end);
    assert!(tr[0] > 4.0 * tr[2], "{tr:?}");
    assert!(tb[2] > 4.0 * tb[0], "{tb:?}");
    // The weights put the red channel's mass at long wavelengths and the blue channel's at short ones.
    let w = r.weights();
    let centre = |row: &[f64; 8]| {
        row.iter()
            .enumerate()
            .map(|(j, v)| v * (420.0 + 40.0 * j as f64))
            .sum::<f64>()
    };
    assert!(centre(&w[0]) > centre(&w[1]) && centre(&w[1]) > centre(&w[2]));
}

#[test]
fn the_key_identifies_the_projection() {
    let a = response("Sony_ILCE-7RM4_380_780_5.json");
    let b = response("Sony_ILCE-7RM4_380_780_5.json");
    let c = response("Sony_ILCE-6400_380_780_5.json");
    assert_eq!(a.key(), b.key());
    assert_ne!(a.key(), c.key());
    assert_ne!(a.key(), CameraResponse::neutral().key());
    assert_eq!(a, b);
}

#[test]
fn curves_that_cannot_be_projected_are_refused() {
    let text = |lo: u32, hi: u32, channel_g: f64| {
        let rows: Vec<String> = (lo..=hi)
            .step_by(5)
            .map(|nm| format!("\"{nm}\": [1.0, {channel_g}, 1.0]"))
            .collect();
        format!(
            r#"{{"header": {{"manufacturer": "T", "model": "M"}}, "spectral_data": {{"index": {{"main": ["R","G","B"]}}, "data": {{"main": {{{}}}}}}}}}"#,
            rows.join(", ")
        )
    };
    // Does not reach 700 nm.
    let short = SpectralCurves::parse(&text(380, 650, 1.0)).unwrap();
    assert!(matches!(
        CameraResponse::from_curves(&short),
        Err(LensError::Camera(_))
    ));
    // A channel with no sensitivity.
    let dead = SpectralCurves::parse(&text(380, 780, 0.0)).unwrap();
    assert!(matches!(
        CameraResponse::from_curves(&dead),
        Err(LensError::Camera(_))
    ));
    // Negative noise is clipped, not an error.
    let noisy = SpectralCurves::parse(&text(380, 780, -0.5).replace("-0.5", "0.5")).unwrap();
    assert!(CameraResponse::from_curves(&noisy).is_ok());
}
