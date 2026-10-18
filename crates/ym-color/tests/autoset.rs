//! AS-03, the automatic settings (`docs/18-auto-settings.md`): the feature vector (§3), the model file and the
//! ensemble
//! (§4.2 and §4.3), the shrinkage (§4.4) and the preferred white (§5).
//!
//! The model here is hand-made: two small members whose outputs are computed by hand in the test, and refusals made by
//! editing a valid file. The trained model and its vectors come with AS-04 and AS-09.

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use ym_color::autoset::{
    AutosetError, AutosetModel, FEATURE_COUNT, FEATURE_NAMES, FeatureInputs, Features, Prediction,
    PreferredWhite, apply, features, preferred_white, rule_only,
};
use ym_color::consts::linalg::{IDENTITY, Mat3, diag};
use ym_color::exposure::{CAMERA_GAIN_EV, EV_LIMIT, ExposureError, PreviewRgb};
use ym_color::gamut::awg4_luminance;
use ym_color::spectral::ClassProbabilities;
use ym_color::wb::SceneLight;

/// The clip level used by most tests: `Y = 0.36`, which is one stop (`log2(0.36 / 0.18)`) above middle grey.
const CLIP_Y: f64 = 0.36;

/// Probabilities of the key-relative quantiles of features 2 to 18.
const PROBS_S: [f64; 17] = [
    0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99,
    0.995, 0.999,
];

/// Probabilities of the key-relative quantiles of features 19 to 23.
const PROBS_T: [f64; 5] = [0.50, 0.90, 0.98, 0.995, 0.999];

/// SplitMix64, the seeded generator of the tests.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// A scene light with the given CCT and Duv, confidence 0.9 and class probabilities 0.1, 0.6, 0.2, 0.1.
fn light(cct_k: f64, duv: f64) -> SceneLight {
    let classes = ClassProbabilities::new(0.1, 0.6, 0.2, 0.1).unwrap();
    SceneLight::from_parts([1.0, 1.0, 1.0], classes, 0.9, cct_k, duv).unwrap()
}

/// The inputs of a test photo, with ISO 400 and no flash.
fn inputs<'a>(
    preview: &'a PreviewRgb<'a>,
    matrix: &'a Mat3,
    light: &'a SceneLight,
    mean_luminance: f64,
    bv: Option<f64>,
) -> FeatureInputs<'a> {
    FeatureInputs {
        preview,
        matrix_ev0: matrix,
        mean_luminance,
        clip_y_ev0: CLIP_Y,
        bv_settings: bv,
        light,
        iso: Some(400.0),
        flash_fired: false,
    }
}

/// The 64 x 48 preview of the hand checks: row `y` is grey at `0.18 * 2^s_y` with `s_y = (y - 24) / 4 + 1/32`, and
/// the top-left 8 x 6 block is clipped. The 1/32 keeps every threshold of the checks away from a row.
const ROWS_W: usize = 64;
const ROWS_H: usize = 48;

fn row_stops(y: usize) -> f64 {
    (y as f64 - 24.0) / 4.0 + 1.0 / 32.0
}

fn rows_preview() -> (Vec<f32>, Vec<u8>) {
    let mut rgb = Vec::with_capacity(3 * ROWS_W * ROWS_H);
    let mut mask = Vec::with_capacity(ROWS_W * ROWS_H);
    for y in 0..ROWS_H {
        for x in 0..ROWS_W {
            let v = (0.18 * row_stops(y).exp2()) as f32;
            rgb.extend([v, v, v]);
            mask.push(u8::from(x < 8 && y < 6));
        }
    }
    (rgb, mask)
}

/// The linearly interpolated quantile of a sorted list, written out here as the reference.
fn reference_quantile(sorted: &[f64], p: f64) -> f64 {
    let pos = p * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

fn assert_close(got: f64, want: f64, tol: f64, what: &str) {
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, want {want} (tolerance {tol})"
    );
}

#[test]
fn feature_vector_has_the_spec_layout() {
    assert_eq!(FEATURE_COUNT, 39);
    assert_eq!(FEATURE_NAMES.len(), FEATURE_COUNT);
    assert_eq!(FEATURE_NAMES[0], "key_bv");
    assert_eq!(FEATURE_NAMES[1], "bv_known");
    assert_eq!(FEATURE_NAMES[2], "q01");
    assert_eq!(FEATURE_NAMES[18], "q999");
    assert_eq!(FEATURE_NAMES[19], "mq50");
    assert_eq!(FEATURE_NAMES[23], "mq999");
    assert_eq!(FEATURE_NAMES[24], "clip_headroom");
    assert_eq!(FEATURE_NAMES[26], "centre_minus_key");
    assert_eq!(FEATURE_NAMES[33], "p_led");
    assert_eq!(FEATURE_NAMES[34], "mired");
    assert_eq!(FEATURE_NAMES[38], "flash");
}

#[test]
fn features_of_the_row_preview_match_the_hand_values() {
    let (rgb, mask) = rows_preview();
    let preview = PreviewRgb {
        width: ROWS_W,
        height: ROWS_H,
        rgb: &rgb,
        clip_mask: Some(&mask),
    };
    let scene = light(4000.0, 0.003);
    let got = features(&inputs(&preview, &IDENTITY, &scene, 0.18, Some(5.0))).unwrap();
    let v = got.values;

    // The key: the mean luminance is middle grey.
    assert_eq!(got.key, 0.0);
    // Brightness value: the settings' Bv plus the key (zero).
    assert_eq!(v[0], 5.0);
    assert_eq!(v[1], 1.0);
    // The quantiles of the stops over grey: each row is 64 equal pixels, so the sorted list is the rows in order.
    let sorted: Vec<f64> = (0..ROWS_H)
        .flat_map(|y| std::iter::repeat_n(row_stops(y), ROWS_W))
        .collect();
    for (k, p) in PROBS_S.iter().enumerate() {
        assert_close(
            v[2 + k],
            reference_quantile(&sorted, *p),
            1e-6,
            FEATURE_NAMES[2 + k],
        );
    }
    // The grey pixels have M = Y, so the channel-maximum quantiles are the same numbers.
    for (k, p) in PROBS_T.iter().enumerate() {
        assert_close(
            v[19 + k],
            reference_quantile(&sorted, *p),
            1e-6,
            FEATURE_NAMES[19 + k],
        );
    }
    // Clip headroom: the clipped neutral is one stop over grey, the key is zero.
    assert_close(v[24], 1.0, 1e-12, "clip_headroom");
    // 48 of 3072 pixels are clipped.
    assert_close(v[25], 48.0 / 3072.0, 1e-15, "clipped_share");
    // Centre (rows 16 to 31, columns 21 to 42): the mean row is 23.5.
    assert_close(
        v[26],
        (23.5 - 24.0) / 4.0 + 1.0 / 32.0,
        1e-6,
        "centre_minus_key",
    );
    // Top (rows 0 to 15) minus bottom (rows 32 to 47): (7.5 - 39.5) / 4 exactly.
    assert_close(v[27], -8.0, 1e-6, "top_minus_bottom");
    // Bright: rows 32 to 47 (16 of 48 rows), above two stops. Dark: rows 0 to 11, below minus three stops.
    assert_close(v[28], 16.0 / 48.0, 1e-15, "bright_share");
    assert_close(v[29], 12.0 / 48.0, 1e-15, "dark_share");
    // Light: the class probabilities in profile order, the mired and Duv scaled as in the spec.
    for (got, want) in v[30..34].iter().zip([0.1, 0.6, 0.2, 0.1]) {
        assert_close(*got, want, 1e-15, "class probability");
    }
    assert_close(v[34], 2.5, 1e-12, "mired");
    assert_close(v[35], 0.3, 1e-12, "duv");
    assert_close(v[36], 0.9, 0.0, "light_confidence");
    assert_close(v[37], 2.0, 1e-12, "log_iso");
    assert_eq!(v[38], 0.0);
}

#[test]
fn unknown_settings_read_as_zero_and_flash_reads_one() {
    let (rgb, mask) = rows_preview();
    let preview = PreviewRgb {
        width: ROWS_W,
        height: ROWS_H,
        rgb: &rgb,
        clip_mask: Some(&mask),
    };
    let scene = light(4000.0, 0.0);
    let mut i = inputs(&preview, &IDENTITY, &scene, 0.18, None);
    i.iso = None;
    i.flash_fired = true;
    let v = features(&i).unwrap().values;
    assert_eq!(v[0], 0.0, "key_bv without the settings");
    assert_eq!(v[1], 0.0, "bv_known");
    assert_eq!(v[37], 0.0, "log_iso without ISO");
    assert_eq!(v[38], 1.0, "flash");
    // A Bv that is not finite is unknown too.
    let mut j = inputs(&preview, &IDENTITY, &scene, 0.18, Some(f64::NAN));
    j.iso = Some(0.0);
    let w = features(&j).unwrap().values;
    assert_eq!((w[0], w[1], w[37]), (0.0, 0.0, 0.0));
}

#[test]
fn a_uniform_colour_gives_the_awg4_luminance_and_the_channel_maximum_of_the_awg4_pixel() {
    // The camera RGB goes through the matrix first: AWG4 = (1.0, 0.25, 0.0625) for the camera (0.5, 0.25, 0.125).
    // The values are exact in f32, so the preview holds them as they are.
    let matrix = diag([2.0, 1.0, 0.5]);
    let rgb = [0.5_f32, 0.25, 0.125].repeat(16);
    let preview = PreviewRgb {
        width: 4,
        height: 4,
        rgb: &rgb,
        clip_mask: None,
    };
    let scene = light(4000.0, 0.0);
    let v = features(&inputs(&preview, &matrix, &scene, 0.18, None))
        .unwrap()
        .values;
    let y = awg4_luminance([1.0, 0.25, 0.0625]);
    let s = (y / 0.18).log2();
    let m = (1.0_f64 / 0.18).log2();
    for value in &v[2..19] {
        assert_close(*value, s, 1e-12, "a quantile of a uniform colour");
    }
    for value in &v[19..24] {
        assert_close(
            *value,
            m,
            1e-12,
            "a channel-maximum quantile of a uniform colour",
        );
    }
    assert_close(v[27], 0.0, 1e-12, "top minus bottom of a uniform frame");
    assert_close(v[26], s, 1e-12, "centre of a uniform frame");
}

#[test]
fn equivariance_under_exposure_changes_follows_the_full_rescue_rule() {
    // A colourful photo with no clipped pixels. Exposure k moves the raw data by 2^k and the settings' Bv by -k,
    // so the scene is the same (docs/18 §3). For k = +-1 the f32 preview is scaled exactly; for k = 0.5 the same
    // scaling goes into the matrix, since 2^0.5 times an f32 is not exact.
    let (w, h) = (64, 48);
    let mut rng = Rng(11);
    let rgb: Vec<f32> = (0..3 * w * h)
        .map(|_| (0.02 + 0.8 * rng.next_f64()) as f32)
        .collect();
    let mask = vec![0_u8; w * h];
    let matrix: Mat3 = [[0.9, 0.1, 0.0], [0.05, 0.85, 0.1], [0.0, 0.2, 0.8]];
    let scene = light(5000.0, 0.002);
    let base_mean = 0.3;
    let base_bv = 4.0;
    let base_preview = PreviewRgb {
        width: w,
        height: h,
        rgb: &rgb,
        clip_mask: Some(&mask),
    };
    let base = features(&inputs(
        &base_preview,
        &matrix,
        &scene,
        base_mean,
        Some(base_bv),
    ))
    .unwrap()
    .values;

    for k in [-1.0_f64, 0.5, 1.0] {
        let factor = k.exp2();
        let (scaled_rgb, scaled_matrix): (Vec<f32>, Mat3) = if k.fract() == 0.0 {
            (rgb.iter().map(|v| v * factor as f32).collect(), matrix)
        } else {
            let mut m = matrix;
            for row in &mut m {
                for c in row {
                    *c *= factor;
                }
            }
            (rgb.clone(), m)
        };
        let scaled_preview = PreviewRgb {
            width: w,
            height: h,
            rgb: &scaled_rgb,
            clip_mask: Some(&mask),
        };
        let got = features(&inputs(
            &scaled_preview,
            &scaled_matrix,
            &scene,
            base_mean * factor,
            Some(base_bv - k),
        ))
        .unwrap()
        .values;
        let unchanged =
            (0..FEATURE_COUNT).filter(|i| *i == 0 || (2..=23).contains(i) || (26..=33).contains(i));
        for i in unchanged {
            assert_close(
                got[i],
                base[i],
                1e-9,
                &format!("k = {k}: {}", FEATURE_NAMES[i]),
            );
        }
        assert_close(
            got[24] - base[24],
            -k,
            1e-9,
            &format!("k = {k}: clip_headroom"),
        );
        for i in [1, 25, 34, 35, 36, 37, 38] {
            assert_eq!(
                got[i], base[i],
                "k = {k}: {} is not an exposure feature",
                FEATURE_NAMES[i]
            );
        }
    }
}

#[test]
fn the_same_inputs_give_the_same_bits_twice() {
    let (rgb, mask) = rows_preview();
    let preview = PreviewRgb {
        width: ROWS_W,
        height: ROWS_H,
        rgb: &rgb,
        clip_mask: Some(&mask),
    };
    let scene = light(3200.0, -0.004);
    let run = || features(&inputs(&preview, &IDENTITY, &scene, 0.21, Some(3.0))).unwrap();
    let (a, b): (Features, Features) = (run(), run());
    let bits = |f: &Features| -> Vec<u64> {
        f.values
            .iter()
            .map(|v| v.to_bits())
            .chain([f.key.to_bits()])
            .collect()
    };
    assert_eq!(bits(&a), bits(&b));
}

#[test]
fn non_finite_pixels_are_skipped_and_masked_ones_still_count_as_clipped() {
    // Grey at 0.25 everywhere (exact in f32) and the key of that grey: every usable pixel gives zero, so the
    // quantiles and the spatial means are zero.
    let (w, h) = (12, 9);
    let mut rgb = [0.25_f32; 3].repeat(w * h);
    let mut mask = vec![0_u8; w * h];
    for &i in &[0_usize, 7, 40, 100] {
        rgb[3 * i] = f32::NAN;
    }
    rgb[3 * 50 + 2] = f32::INFINITY;
    rgb[3 * 60 + 1] = f32::NEG_INFINITY;
    mask[0] = 1;
    mask[50] = 1;
    let preview = PreviewRgb {
        width: w,
        height: h,
        rgb: &rgb,
        clip_mask: Some(&mask),
    };
    let scene = light(4000.0, 0.0);
    let v = features(&inputs(&preview, &IDENTITY, &scene, 0.25, None))
        .unwrap()
        .values;
    for i in (2..=23).chain(26..=27) {
        assert_close(v[i], 0.0, 1e-12, FEATURE_NAMES[i]);
    }
    // Six pixels are not usable; none of the others is above or below the key, so the shares are zero.
    assert_eq!(v[28], 0.0);
    assert_eq!(v[29], 0.0);
    // Clipped share counts the mask over every pixel, the non-finite ones included.
    assert_close(v[25], 2.0 / 108.0, 1e-15, "clipped_share");
}

#[test]
fn a_preview_with_no_finite_pixel_is_an_error() {
    let rgb = [f32::NAN; 12];
    let preview = PreviewRgb {
        width: 2,
        height: 2,
        rgb: &rgb,
        clip_mask: None,
    };
    let scene = light(4000.0, 0.0);
    assert_eq!(
        features(&inputs(&preview, &IDENTITY, &scene, 0.18, None)).unwrap_err(),
        AutosetError::NoUsablePixel
    );
}

#[test]
fn bad_buffers_and_readings_are_errors() {
    let scene = light(4000.0, 0.0);
    let rgb = vec![0.2_f32; 12];
    let short = PreviewRgb {
        width: 2,
        height: 3,
        rgb: &rgb,
        clip_mask: None,
    };
    assert!(matches!(
        features(&inputs(&short, &IDENTITY, &scene, 0.18, None)),
        Err(AutosetError::Preview(ExposureError::BufferSize { .. }))
    ));
    let mask = vec![0_u8; 3];
    let wrong_mask = PreviewRgb {
        width: 2,
        height: 2,
        rgb: &rgb,
        clip_mask: Some(&mask),
    };
    assert!(matches!(
        features(&inputs(&wrong_mask, &IDENTITY, &scene, 0.18, None)),
        Err(AutosetError::Preview(ExposureError::MaskSize { .. }))
    ));
    let good = PreviewRgb {
        width: 2,
        height: 2,
        rgb: &rgb,
        clip_mask: None,
    };
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(
                features(&inputs(&good, &IDENTITY, &scene, bad, None)),
                Err(AutosetError::MeanLuminance(_))
            ),
            "mean luminance {bad}"
        );
    }
    for bad in [0.0, -0.36, f64::NAN] {
        let mut i = inputs(&good, &IDENTITY, &scene, 0.18, None);
        i.clip_y_ev0 = bad;
        assert!(
            matches!(features(&i), Err(AutosetError::ClipLevel(_))),
            "clip level {bad}"
        );
    }
    // A scene light whose CCT is zero (built by hand: `SceneLight::from_parts` would refuse it).
    let broken = SceneLight {
        w_scene: [1.0, 1.0, 1.0],
        classes: ClassProbabilities::new(0.1, 0.6, 0.2, 0.1).unwrap(),
        confidence: 0.9,
        cct_k: 0.0,
        duv: 0.0,
    };
    assert_eq!(
        features(&inputs(&good, &IDENTITY, &broken, 0.18, None)).unwrap_err(),
        AutosetError::NonFinite("mired")
    );
}

#[test]
fn hostile_pixels_never_give_a_non_finite_feature() {
    let specials = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        0.0,
        1e-30,
        1.0,
        3.0e38,
        f32::MAX,
    ];
    let mut rng = Rng(5);
    let scene = light(2500.0, 0.01);
    for (w, h) in [(1, 1), (1, 9), (2, 2), (3, 3), (17, 5), (64, 48)] {
        for round in 0..5 {
            let rgb: Vec<f32> = (0..3 * w * h)
                .map(|_| {
                    if round == 0 || rng.next_f64() < 0.7 {
                        (rng.next_f64() * 2.0) as f32
                    } else {
                        specials[(rng.next_f64() * specials.len() as f64) as usize % specials.len()]
                    }
                })
                .collect();
            let mask: Vec<u8> = (0..w * h).map(|_| u8::from(rng.next_f64() < 0.2)).collect();
            let preview = PreviewRgb {
                width: w,
                height: h,
                rgb: &rgb,
                clip_mask: Some(&mask),
            };
            match features(&inputs(&preview, &IDENTITY, &scene, 0.1, Some(6.0))) {
                Ok(f) => {
                    assert!(
                        f.values.iter().all(|v| v.is_finite()),
                        "{w}x{h} round {round}"
                    );
                    assert!(f.key.is_finite());
                }
                Err(e) => assert!(
                    matches!(e, AutosetError::NoUsablePixel),
                    "{w}x{h} round {round}: {e}"
                ),
            }
        }
    }
}

#[test]
#[ignore = "timing: cargo test --release -p ym-color --test autoset -- --ignored --nocapture"]
fn features_timing_on_a_1188_by_792_preview() {
    // Timing only; the budget of docs/18 section 2 is 40 ms for the features of a 1/8 preview.
    let (w, h) = (1188, 792);
    let mut rng = Rng(3);
    let rgb: Vec<f32> = (0..3 * w * h)
        .map(|_| (0.001 + 0.9 * rng.next_f64() * rng.next_f64()) as f32)
        .collect();
    let mask: Vec<u8> = (0..w * h)
        .map(|_| u8::from(rng.next_f64() < 0.01))
        .collect();
    let matrix: Mat3 = [[0.9, 0.1, 0.0], [0.05, 0.85, 0.1], [0.0, 0.2, 0.8]];
    let scene = light(5000.0, 0.002);
    let preview = PreviewRgb {
        width: w,
        height: h,
        rgb: &rgb,
        clip_mask: Some(&mask),
    };
    let mut times = Vec::new();
    for _ in 0..9 {
        let start = std::time::Instant::now();
        let f = features(&inputs(&preview, &matrix, &scene, 0.2, Some(7.0))).unwrap();
        times.push(start.elapsed().as_secs_f64() * 1e3);
        std::hint::black_box(f);
    }
    times.sort_by(f64::total_cmp);
    eprintln!(
        "autoset features on {w} x {h}: median {:.1} ms, min {:.1} ms, max {:.1} ms",
        times[times.len() / 2],
        times[0],
        times[times.len() - 1]
    );
}

// ---------------------------------------------------------------------------------------------------------------------
// The model file.
// ---------------------------------------------------------------------------------------------------------------------

/// One layer of a test model: `w` is `rows x cols` row-major.
#[derive(Clone)]
struct TestLayer {
    rows: usize,
    cols: usize,
    w: Vec<f32>,
    b: Vec<f32>,
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// The tensor object of the model file, and the little-endian bytes the digest covers.
fn tensor(shape: &[usize], values: &[f32]) -> (Value, Vec<u8>) {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let object = json!({
        "shape": shape,
        "data_b64": STANDARD.encode(&bytes),
        "dtype": "f32le",
    });
    (object, bytes)
}

/// A complete model file for the given members, with a correct digest. The header is the v1 header, with
/// `feature_names` the names of §3, `input_mean` 0.5 at feature 0 and 0 elsewhere, `input_std` 2 at feature 0 and 1
/// elsewhere, and the priors of §4.4.
fn model_file(members: &[Vec<TestLayer>]) -> Value {
    let mut hasher = Sha256::new();
    let mut members_json = Vec::new();
    for layers in members {
        let mut layers_json = Vec::new();
        for layer in layers {
            let (w, w_bytes) = tensor(&[layer.rows, layer.cols], &layer.w);
            let (b, b_bytes) = tensor(&[layer.rows], &layer.b);
            hasher.update(&w_bytes);
            hasher.update(&b_bytes);
            layers_json.push(json!({"w": w, "b": b}));
        }
        members_json.push(json!({"layers": layers_json}));
    }
    let mut input_mean = vec![0.0; FEATURE_COUNT];
    input_mean[0] = 0.5;
    let mut input_std = vec![1.0; FEATURE_COUNT];
    input_std[0] = 2.0;
    json!({
        "format": "yellowmoon-autoset",
        "format_version": 1,
        "model_version": "0.0.0-test",
        "feature_names": FEATURE_NAMES.as_slice(),
        "input_mean": input_mean,
        "input_std": input_std,
        "activation": "tanh",
        "outputs": ["key_target", "key_logvar", "contrast", "contrast_logvar"],
        "prior_sigma": {"key_target": 0.75, "contrast": 10.0},
        "contrast_limit": 30.0,
        "members": members_json,
        "sha256": hex(&hasher.finalize()),
    })
}

/// The first hand-made member, 39 -> 2 -> 2 -> 4, with weights small enough to compute by hand.
fn member_a() -> Vec<TestLayer> {
    let mut w0 = vec![0.0_f32; 2 * FEATURE_COUNT];
    w0[0] = 0.5; // row 0, input 0
    w0[FEATURE_COUNT + 1] = -0.25; // row 1, input 1
    vec![
        TestLayer {
            rows: 2,
            cols: FEATURE_COUNT,
            w: w0,
            b: vec![0.125, 0.0],
        },
        TestLayer {
            rows: 2,
            cols: 2,
            w: vec![1.0, -1.0, 0.5, 0.5],
            b: vec![0.0, 0.25],
        },
        TestLayer {
            rows: 4,
            cols: 2,
            w: vec![0.25, 0.0, 0.0, 0.125, 1.0, 1.0, -0.5, 0.5],
            b: vec![0.0625, 0.0, -0.125, 0.25],
        },
    ]
}

/// The second member, which reads inputs 0, 1 and 2.
fn member_b() -> Vec<TestLayer> {
    let mut w0 = vec![0.0_f32; 2 * FEATURE_COUNT];
    w0[0] = -0.5;
    w0[2] = 0.25;
    w0[FEATURE_COUNT + 1] = 0.625;
    vec![
        TestLayer {
            rows: 2,
            cols: FEATURE_COUNT,
            w: w0,
            b: vec![0.0, -0.125],
        },
        TestLayer {
            rows: 2,
            cols: 2,
            w: vec![0.25, 0.75, -0.75, 0.125],
            b: vec![0.125, 0.0],
        },
        TestLayer {
            rows: 4,
            cols: 2,
            w: vec![0.5, -0.5, 0.25, 0.75, -1.0, 0.0, 0.0, 2.0],
            b: vec![0.0, -0.25, 0.375, -0.25],
        },
    ]
}

fn hand_model_json() -> Value {
    model_file(&[member_a(), member_b()])
}

/// The features of the hand test: standardised, they are `[0.25, 2.0, -1.0, 0, 0, ...]`.
fn hand_input() -> [f64; FEATURE_COUNT] {
    let mut x = [0.0; FEATURE_COUNT];
    x[0] = 1.0; // (1.0 - 0.5) / 2.0 = 0.25
    x[1] = 2.0;
    x[2] = -1.0;
    x
}

/// Member A by hand: `tanh` on the hidden layers, the output linear.
fn hand_member_a() -> [f64; 4] {
    let (x0, x1) = (0.25_f64, 2.0_f64);
    let h0 = (0.5 * x0 + 0.125).tanh();
    let h1 = (-0.25 * x1).tanh();
    let g0 = (h0 - h1).tanh();
    let g1 = (0.5 * h0 + 0.5 * h1 + 0.25).tanh();
    [
        0.25 * g0 + 0.0625,
        0.125 * g1,
        g0 + g1 - 0.125,
        -0.5 * g0 + 0.5 * g1 + 0.25,
    ]
}

/// Member B by hand.
fn hand_member_b() -> [f64; 4] {
    let (x0, x1, x2) = (0.25_f64, 2.0_f64, -1.0_f64);
    let h0 = (-0.5 * x0 + 0.25 * x2).tanh();
    let h1 = (0.625 * x1 - 0.125).tanh();
    let g0 = (0.25 * h0 + 0.75 * h1 + 0.125).tanh();
    let g1 = (-0.75 * h0 + 0.125 * h1).tanh();
    [
        0.5 * g0 - 0.5 * g1,
        0.25 * g0 + 0.75 * g1 - 0.25,
        -g0 + 0.375,
        2.0 * g1 - 0.25,
    ]
}

#[test]
fn member_outputs_match_the_hand_computation() {
    let model = AutosetModel::from_json(&hand_model_json().to_string()).unwrap();
    assert_eq!(model.member_count(), 2);
    assert_eq!(model.version(), "0.0.0-test");
    assert_eq!(model.feature_names(), &FEATURE_NAMES[..]);
    assert_eq!(model.prior_sigma().key_target, 0.75);
    assert_eq!(model.prior_sigma().contrast, 10.0);
    assert_eq!(model.contrast_limit(), 30.0);
    assert_eq!(model.sha256().len(), 64);
    let outputs = model.member_outputs(&hand_input());
    assert_eq!(outputs.len(), 2);
    for (got, want) in outputs[0].iter().zip(hand_member_a()) {
        assert_close(*got, want, 1e-12, "member A");
    }
    for (got, want) in outputs[1].iter().zip(hand_member_b()) {
        assert_close(*got, want, 1e-12, "member B");
    }
}

#[test]
fn the_ensemble_is_the_mixture_of_the_members() {
    let model = AutosetModel::from_json(&hand_model_json().to_string()).unwrap();
    let p: Prediction = model.predict(&hand_input());
    let (a, b) = (hand_member_a(), hand_member_b());
    // Key: the mean of the means; the variance is the mean of the variances plus the variance of the means.
    let key_mean = (a[0] + b[0]) / 2.0;
    let key_var = (a[1].exp() + b[1].exp()) / 2.0
        + ((a[0] - key_mean).powi(2) + (b[0] - key_mean).powi(2)) / 2.0;
    let contrast_mean = (a[2] + b[2]) / 2.0;
    let contrast_var = (a[3].exp() + b[3].exp()) / 2.0
        + ((a[2] - contrast_mean).powi(2) + (b[2] - contrast_mean).powi(2)) / 2.0;
    assert_close(p.key_target, key_mean, 1e-12, "key_target");
    assert_close(p.key_var, key_var, 1e-12, "key_var");
    assert_close(p.contrast, contrast_mean, 1e-12, "contrast");
    assert_close(p.contrast_var, contrast_var, 1e-12, "contrast_var");
}

#[test]
fn the_log_variance_is_clamped_before_the_exponential() {
    // One member whose key log-variance is 50 (clamped to 20) and contrast log-variance -50 (clamped to -20).
    let mut layers = member_a();
    let last = layers.len() - 1;
    layers[last].b[1] = 50.0;
    layers[last].b[3] = -50.0;
    let model = AutosetModel::from_json(&model_file(&[layers]).to_string()).unwrap();
    let p = model.predict(&hand_input());
    assert_close(
        p.key_var,
        20.0_f64.exp(),
        1e-3 * 20.0_f64.exp(),
        "key_var at the top clamp",
    );
    assert_close(
        p.contrast_var,
        (-20.0_f64).exp(),
        1e-15,
        "contrast_var at the bottom clamp",
    );
}

#[test]
fn the_reader_takes_the_hidden_widths_from_the_file() {
    // Hidden widths 5 and 3 (not 32): the shapes come from the file, the input and output are checked.
    let layers = vec![
        TestLayer {
            rows: 5,
            cols: FEATURE_COUNT,
            w: (0..5 * FEATURE_COUNT)
                .map(|i| ((i % 7) as f32 - 3.0) * 0.01)
                .collect(),
            b: vec![0.0; 5],
        },
        TestLayer {
            rows: 3,
            cols: 5,
            w: (0..15).map(|i| (i as f32 - 7.0) * 0.05).collect(),
            b: vec![0.1, -0.1, 0.0],
        },
        TestLayer {
            rows: 4,
            cols: 3,
            w: (0..12).map(|i| ((i % 5) as f32 - 2.0) * 0.2).collect(),
            b: vec![0.0; 4],
        },
    ];
    let model = AutosetModel::from_json(&model_file(&[layers]).to_string()).unwrap();
    assert_eq!(model.member_count(), 1);
    assert!(model.predict(&hand_input()).key_var > 0.0);
}

/// Every refusal: the mutation of a valid file and the variant it must give.
#[test]
fn bad_model_files_are_refused_with_the_right_error() {
    let good = hand_model_json();
    let refuse = |f: &dyn Fn(&mut Value)| {
        let mut v = good.clone();
        f(&mut v);
        AutosetModel::from_json(&v.to_string()).unwrap_err()
    };
    assert!(AutosetModel::from_json(&good.to_string()).is_ok());
    assert!(matches!(
        AutosetModel::from_json("not json"),
        Err(AutosetError::Json(_))
    ));
    assert!(matches!(
        AutosetModel::from_json("{}"),
        Err(AutosetError::Json(_))
    ));
    let invalid = |e: AutosetError| matches!(e, AutosetError::InvalidModel(_));
    assert!(invalid(refuse(&|v| v["format"] = "other".into())));
    assert!(invalid(refuse(&|v| v["format_version"] = 2.into())));
    assert!(invalid(refuse(&|v| v["model_version"] = "".into())));
    assert!(invalid(refuse(&|v| v["feature_names"][0] = "q01".into())));
    assert!(invalid(refuse(&|v| {
        v["feature_names"].as_array_mut().unwrap().pop();
    })));
    assert!(invalid(refuse(&|v| v["activation"] = "relu".into())));
    assert!(invalid(refuse(&|v| v["outputs"][1] = "key_sigma".into())));
    assert!(invalid(refuse(&|v| {
        v["input_mean"].as_array_mut().unwrap().pop();
    })));
    assert!(invalid(refuse(&|v| v["input_std"][5] = 0.0.into())));
    assert!(invalid(refuse(&|v| v["input_std"][5] = (-1.0).into())));
    assert!(invalid(refuse(
        &|v| v["prior_sigma"]["key_target"] = 0.0.into()
    )));
    assert!(invalid(refuse(
        &|v| v["prior_sigma"]["contrast"] = (-10.0).into()
    )));
    assert!(invalid(refuse(&|v| v["contrast_limit"] = 0.0.into())));
    assert!(invalid(refuse(&|v| v["members"] = json!([]))));
    assert!(invalid(refuse(&|v| {
        v["members"][0]["layers"]
            .as_array_mut()
            .unwrap()
            .truncate(1);
    })));
    // A first layer that reads 38 inputs, and a second layer that reads 3 where the first gives 2.
    assert!(invalid(refuse(&|v| {
        v["members"][0]["layers"][0]["w"]["shape"] = json!([2, 38]);
    })));
    assert!(invalid(refuse(&|v| {
        v["members"][0]["layers"][1]["w"]["shape"] = json!([2, 3]);
    })));
    // A tensor whose bias does not match its rows, bad base64, a wrong byte count and a wrong dtype.
    assert!(invalid(refuse(&|v| {
        v["members"][0]["layers"][1]["b"]["shape"] = json!([3]);
    })));
    assert!(invalid(refuse(&|v| {
        v["members"][0]["layers"][1]["b"]["data_b64"] = "@@@@".into();
    })));
    assert!(invalid(refuse(&|v| {
        v["members"][0]["layers"][1]["b"]["data_b64"] = "AAAA".into();
    })));
    assert!(invalid(refuse(&|v| {
        v["members"][1]["layers"][0]["w"]["dtype"] = "f64le".into();
    })));
    // A value that is not finite: a NaN bit pattern in the bias of member A's hidden layer, with the right shape.
    let nan_bias = STANDARD.encode([f32::NAN.to_le_bytes(), 0.0_f32.to_le_bytes()].concat());
    let not_finite = refuse(&|v| {
        v["members"][0]["layers"][1]["b"]["data_b64"] = nan_bias.clone().into();
    });
    assert!(
        matches!(&not_finite, AutosetError::InvalidModel(reason) if reason.contains("not finite")),
        "{not_finite}"
    );
    // The last layer must give the four outputs: three rows (with matching data) is refused.
    let mut layers = member_a();
    layers[2] = TestLayer {
        rows: 3,
        cols: 2,
        w: vec![0.1; 6],
        b: vec![0.0; 3],
    };
    assert!(invalid(
        AutosetModel::from_json(&model_file(&[layers]).to_string()).unwrap_err()
    ));
    // The digest: a wrong one, and one byte of a tensor changed (valid base64, different bytes).
    assert!(matches!(
        refuse(&|v| v["sha256"] = "00".into()),
        AutosetError::HashMismatch { .. }
    ));
    assert!(matches!(
        refuse(&|v| {
            let data = v["members"][0]["layers"][0]["w"]["data_b64"]
                .as_str()
                .unwrap()
                .to_string();
            let mut bytes = data.into_bytes();
            bytes[8] = if bytes[8] == b'A' { b'B' } else { b'A' };
            v["members"][0]["layers"][0]["w"]["data_b64"] =
                String::from_utf8(bytes).unwrap().into();
        }),
        AutosetError::HashMismatch { .. }
    ));
    assert!(matches!(
        AutosetModel::load(std::path::Path::new("/nonexistent/autoset.json")),
        Err(AutosetError::Io { .. })
    ));
}

#[test]
fn the_digest_covers_every_tensor_in_order() {
    // The digest of the file is the digest of the tensors' bytes in order: members, layers, `w` before `b`.
    let members = [member_a(), member_b()];
    let mut hasher = Sha256::new();
    for layers in &members {
        for layer in layers {
            hasher.update(
                layer
                    .w
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
            hasher.update(
                layer
                    .b
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>(),
            );
        }
    }
    let model = AutosetModel::from_json(&hand_model_json().to_string()).unwrap();
    assert_eq!(model.sha256(), hex(&hasher.finalize()));
}

// ---------------------------------------------------------------------------------------------------------------------
// The shrinkage.
// ---------------------------------------------------------------------------------------------------------------------

fn prediction(key_target: f64, key_var: f64, contrast: f64, contrast_var: f64) -> Prediction {
    Prediction {
        key_target,
        key_var,
        contrast,
        contrast_var,
    }
}

#[test]
fn the_rule_alone_is_the_camera_gain_with_no_contrast() {
    let r = rule_only();
    assert_eq!(r.ev_before_guard, CAMERA_GAIN_EV);
    assert_eq!(r.ev_weight, 0.0);
    assert_eq!(r.contrast, 0.0);
    assert_eq!(r.contrast_weight, 0.0);
}

#[test]
fn the_weights_shrink_the_prediction_by_its_variance() {
    let model = AutosetModel::from_json(&hand_model_json().to_string()).unwrap();
    // sigma0 = 0.75 for the key and 10 for the contrast. Variance equal to sigma0 squared: weight one half.
    let p = prediction(0.0 + 2.0, 0.75 * 0.75, 25.0, 100.0);
    let a = apply(&p, 0.0, &model);
    assert_close(a.ev_weight, 0.5, 1e-15, "ev weight");
    // Full rescue: the exposure is the model's in full whatever its weight, which is only reported.
    assert_close(a.ev_before_guard, 2.0, 1e-12, "ev");
    assert_close(a.contrast_weight, 0.5, 1e-15, "contrast weight");
    assert_eq!(a.contrast, 12.5);
    // Variance three times sigma0 squared: weight a quarter.
    let q = apply(
        &prediction(0.0 + 2.0, 3.0 * 0.5625, 0.0, 300.0),
        0.0,
        &model,
    );
    assert_close(q.ev_weight, 0.25, 1e-15, "a quarter weight");
    assert_eq!(q.contrast_weight, 0.25);
    // A certain prediction applies in full: the exposure the model asks for, the contrast bounded.
    let certain = apply(&prediction(0.0 + 10.0, 0.0, 1000.0, 0.0), 0.0, &model);
    assert_eq!(certain.ev_weight, 1.0);
    assert_eq!(
        certain.ev_before_guard, EV_LIMIT,
        "clamped to the exposure limit"
    );
    assert_eq!(certain.contrast, 30.0, "bounded to the contrast limit");
    assert_eq!(certain.contrast_weight, 1.0);
}

#[test]
fn the_exposure_is_clamped_and_the_contrast_rounded() {
    let model = AutosetModel::from_json(&hand_model_json().to_string()).unwrap();
    let low = apply(&prediction(-10.0, 0.0, -1000.0, 0.0), 0.0, &model);
    assert_eq!(low.ev_before_guard, -EV_LIMIT);
    assert_eq!(low.contrast, -30.0);
    // Contrast rounds to a tenth, half away from zero.
    for (c, want) in [
        (0.25, 0.3),
        (-0.25, -0.3),
        (0.04, 0.0),
        (-0.04, 0.0),
        (7.46, 7.5),
        (-7.46, -7.5),
    ] {
        let got = apply(&prediction(0.0, 0.0, c, 0.0), 0.0, &model).contrast;
        assert_eq!(got, want, "contrast {c}");
        assert_eq!(
            got.to_bits(),
            (want + 0.0_f64).to_bits(),
            "no negative zero for {c}"
        );
    }
}

#[test]
fn a_doubtful_contrast_fades_and_a_broken_prediction_gives_the_rule() {
    let model = AutosetModel::from_json(&hand_model_json().to_string()).unwrap();
    // A doubtful exposure still applies (full rescue, owner's decision of 2026-10-18): only its confidence is low.
    let doubtful = apply(&prediction(2.5, 1e6, 20.0, 1e6), 0.0, &model);
    assert!(doubtful.ev_weight < 1e-5);
    assert_eq!(doubtful.ev_before_guard, 2.5);
    assert!(doubtful.contrast.abs() < 1e-3);
    let nan = apply(&prediction(f64::NAN, 0.1, f64::NAN, 0.1), 0.0, &model);
    assert_eq!(nan.ev_weight, 0.0);
    assert_eq!(nan.ev_before_guard, CAMERA_GAIN_EV);
    assert_eq!(nan.contrast_weight, 0.0);
    assert_eq!(nan.contrast, 0.0);
    // A key that is not finite cannot move the exposure either.
    let no_key = apply(&prediction(3.0, 0.1, 0.0, 0.1), f64::NAN, &model);
    assert_eq!(no_key.ev_before_guard, CAMERA_GAIN_EV);
}

// ---------------------------------------------------------------------------------------------------------------------
// The preferred white.
// ---------------------------------------------------------------------------------------------------------------------

fn mired(cct_k: f64) -> f64 {
    1e6 / cct_k
}

#[test]
fn the_preferred_white_of_docs_18_section_5() {
    let p = PreferredWhite::default();
    assert_eq!(p.r_warm, 0.30);
    assert_eq!(p.w_span_mired, 150.0);
    assert_eq!(p.r_dim, 0.15);
    assert_eq!(p.b_dim, 2.0);
    assert_eq!(p.b_span, 4.0);
    assert_eq!(p.d_min, 0.5);
    assert_eq!(p.mired_ref, 1e6 / 6000.0);
    // 3000 K in a bright scene: D = 0.70, the user white is 283.3 mired (about 3530 K).
    let (cct, duv) = preferred_white(3000.0, 0.004, Some(8.0), 1.0, &p);
    assert_close(mired(cct), 283.333_333, 1e-4, "mired_u");
    assert_close(cct, 3529.41, 0.01, "cct_u");
    assert_eq!(duv, 0.004, "the Duv is kept");
}

#[test]
fn the_preferred_white_is_the_input_when_it_should_be() {
    let p = PreferredWhite::default();
    let input = (3000.0, 0.003);
    // Strength zero: the old behaviour, the white of the scene light.
    assert_eq!(preferred_white(3000.0, 0.003, Some(8.0), 0.0, &p), input);
    assert_eq!(
        preferred_white(3000.0, 0.003, Some(8.0), f64::NAN, &p),
        input
    );
    // A cool light (6000 K or bluer) and a light at the reference.
    assert_eq!(
        preferred_white(6500.0, 0.003, Some(-3.0), 1.0, &p),
        (6500.0, 0.003)
    );
    assert_eq!(
        preferred_white(6000.0, 0.003, None, 1.0, &p),
        (6000.0, 0.003)
    );
    // An invalid CCT.
    for bad in [0.0, -5.0, f64::NAN, f64::INFINITY] {
        let (c, d) = preferred_white(bad, 0.003, None, 1.0, &p);
        assert!(c.to_bits() == bad.to_bits() && d == 0.003, "cct {bad}");
    }
    // Strength above one is clamped to one.
    let full = preferred_white(3000.0, 0.0, Some(8.0), 1.0, &p).0;
    let over = preferred_white(3000.0, 0.0, Some(8.0), 7.0, &p).0;
    assert_eq!(full.to_bits(), over.to_bits(), "strength is clamped to 1");
}

#[test]
fn the_strength_scales_one_minus_d() {
    let p = PreferredWhite::default();
    // Half strength: D' = 1 - 0.5 * 0.3 = 0.85, mired_u = ref + 0.85 (s - ref).
    let (cct, _) = preferred_white(3000.0, 0.0, Some(8.0), 0.5, &p);
    let want = 1e6 / (p.mired_ref + 0.85 * (mired(3000.0) - p.mired_ref));
    assert_close(cct, want, 1e-9, "half strength");
}

#[test]
fn a_dim_scene_keeps_more_warmth_and_the_clamp_holds() {
    let p = PreferredWhite::default();
    let bright = preferred_white(3000.0, 0.0, Some(8.0), 1.0, &p).0;
    let dim = preferred_white(3000.0, 0.0, Some(-2.0), 1.0, &p).0;
    // The dim scene has D smaller: its white is cooler, so the photo keeps more warmth.
    assert!(dim > bright, "dim {dim} vs bright {bright}");
    // D_low at Bv -2 is 1 - 0.15 = 0.85, so D = 0.7 * 0.85 = 0.595.
    let d = 0.7 * 0.85;
    let want = 1e6 / (p.mired_ref + d * (mired(3000.0) - p.mired_ref));
    assert_close(dim, want, 1e-9, "dim scene");
    // A floor above the rule's D raises D to the floor.
    let floored = PreferredWhite { d_min: 0.8, ..p };
    let got = preferred_white(3000.0, 0.0, Some(8.0), 1.0, &floored).0;
    let want = 1e6 / (p.mired_ref + 0.8 * (mired(3000.0) - p.mired_ref));
    assert_close(got, want, 1e-9, "the floor");
    // An unknown Bv is the bright case, and a NaN Bv counts as unknown.
    assert_eq!(
        preferred_white(3000.0, 0.0, None, 1.0, &p),
        preferred_white(3000.0, 0.0, Some(8.0), 1.0, &p)
    );
    assert_eq!(
        preferred_white(3000.0, 0.0, Some(f64::NAN), 1.0, &p),
        preferred_white(3000.0, 0.0, None, 1.0, &p)
    );
}

#[test]
fn the_user_white_is_monotonic_in_the_scene_mired() {
    // A warmer scene light (higher mired) gives a warmer (higher mired, or equal) user white, for every brightness.
    let p = PreferredWhite::default();
    for key_bv in [None, Some(-2.0), Some(2.0), Some(8.0)] {
        for strength in [0.3, 1.0] {
            let mut last = f64::NEG_INFINITY;
            // From 10000 K down to 2000 K: the scene's mired rises, and so must the user white's.
            for k in 0..=160 {
                let cct = 10000.0 - 50.0 * f64::from(k);
                let (out, _) = preferred_white(cct, 0.0, key_bv, strength, &p);
                let m = mired(out);
                assert!(
                    m >= last - 1e-9,
                    "non-monotonic at {cct} K ({key_bv:?}, {strength}): {m} after {last}"
                );
                last = m;
            }
        }
    }
}
