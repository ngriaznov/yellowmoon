//! M1-08, auto exposure (`docs/02-colour-pipeline.md`, section "Auto exposure"): the interim rule (the
//! camera's own exposure, a constant gain, with the highlight guard) steps 1 to 5, burst consistency
//! (step 6), and the comparison with the camera's embedded JPEG that gates it.
//!
//! The gate itself ("within 0.5 EV of the embedded JPEG for at least 80 % of the golden set") needs real
//! ARW files with their embedded previews, and the environment has none; what is tested here is every
//! piece of the harness on synthetic data: the algorithm against an independent implementation, the
//! JPEG side on two small JPEG files decoded with `zune-jpeg`, and the gate bookkeeping on a synthetic
//! set with planted offsets.

use ym_color::consts::linalg::{IDENTITY, Mat3};
use ym_color::exposure::{
    self, AutoExposure, BURST_WINDOW_S, BurstFrame, CAMERA_GAIN_EV, CENTRE_SIGMA, ExposureError,
    ExposureSetting, GATE_REQUIRED_FRACTION, GATE_TOLERANCE_EV, GateEntry, LUMINANCE_FLOOR,
    PreviewRgb, ShotSettings, TARGET_GREY, auto_exposure, burst_groups, evaluate_gate,
    exposure_error_ev, jpeg_median_luminance, preview_median_luminance, share_burst_ev,
    srgb_to_linear,
};
use ym_color::gamut::awg4_luminance;

/// Two JPEG files made with Pillow (quality 100, 4:4:4), 16 x 16 pixels in 2 x 2 blocks of 8 x 8:
/// grey codes 30, 90, 150, 210 and the colours (200, 40, 40), (40, 200, 40), (40, 40, 200), (128, 128, 128).
/// ```python
/// img = np.zeros((16, 16, 3), np.uint8); img[y*8:(y+1)*8, x*8:(x+1)*8] = code  # block (x, y) in reading order
/// Image.fromarray(img, "RGB").save(buf, "JPEG", quality=100, subsampling=0)
/// ```
const GREY_JPEG_HEX: &str = concat!(
    "ffd8ffe000104a46494600010100000100010000ffdb0043000101010101010101010101010101010101010101010101",
    "0101010101010101010101010101010101010101010101010101010101010101010101010101010101ffdb0043010101",
    "010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101",
    "0101010101010101010101010101ffc00011080010001003011100021101031101ffc4001f0000010501010101010100",
    "000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411051221",
    "31410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a",
    "434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a9293949596979899",
    "9aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1",
    "f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100",
    "020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f015",
    "6272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a63646566676869",
    "6a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4",
    "c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00fe",
    "3be803f782803f782803f782803fffd9"
);
const COLOR_JPEG_HEX: &str = concat!(
    "ffd8ffe000104a46494600010100000100010000ffdb0043000101010101010101010101010101010101010101010101",
    "0101010101010101010101010101010101010101010101010101010101010101010101010101010101ffdb0043010101",
    "010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101",
    "0101010101010101010101010101ffc00011080010001003011100021101031101ffc4001f0000010501010101010100",
    "000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411051221",
    "31410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a",
    "434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a9293949596979899",
    "9aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1",
    "f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100",
    "020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f015",
    "6272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a63646566676869",
    "6a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4",
    "c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00fc",
    "bfafe273fea00fd70afe2f3ff3673f99faff00d0a0ff006c0fe8c2bfe5fcfda0ffd9"
);

fn from_hex(hex: &str) -> Vec<u8> {
    (0..hex.len() / 2)
        .map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap())
        .collect()
}

/// Decodes a JPEG to 8-bit RGB with zune-jpeg.
fn decode_jpeg(bytes: &[u8]) -> (usize, usize, Vec<u8>) {
    let mut decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(bytes.to_vec()));
    let pixels = decoder.decode().unwrap();
    let info = decoder.info().unwrap();
    (usize::from(info.width), usize::from(info.height), pixels)
}

/// Deterministic SplitMix64 in [0, 1).
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

/// A preview made of grey pixels (R = G = B = Y, so Y is the value through the identity matrix).
struct Image {
    width: usize,
    height: usize,
    rgb: Vec<f32>,
    mask: Vec<u8>,
}

impl Image {
    fn new(width: usize, height: usize, f: impl Fn(usize, usize) -> f64) -> Self {
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let v = f(x, y) as f32;
                rgb.extend([v, v, v]);
            }
        }
        Self {
            width,
            height,
            rgb,
            mask: vec![0; width * height],
        }
    }

    fn flat(width: usize, height: usize, v: f64) -> Self {
        Self::new(width, height, |_, _| v)
    }

    fn preview(&self) -> PreviewRgb<'_> {
        PreviewRgb {
            width: self.width,
            height: self.height,
            rgb: &self.rgb,
            clip_mask: Some(&self.mask),
        }
    }

    fn meter(&self) -> AutoExposure {
        auto_exposure(&self.preview(), &IDENTITY).unwrap()
    }
}

/// `Y` of a grey `v` through the AWG4 Y row (1 to 1e-15: the row sums to the D65 white's Y).
fn y_of_grey(v: f64) -> f64 {
    awg4_luminance([v; 3])
}

#[test]
fn the_exposure_is_the_cameras_gain_for_scenes_of_any_key() {
    // Dark, middle, bright, flat and a lens-cap frame: the scene does not move the exposure. None of these
    // ends above 8 after exposure (1.5 * 2^1.11 = 3.2), so the guard stays out of it.
    for v in [0.0, 1e-5, 0.01, 0.05, 0.18, 0.5, 1.5] {
        let img = Image::flat(40, 30, v);
        let r = img.meter();
        let y = y_of_grey(f64::from(v as f32)).max(LUMINANCE_FLOOR);
        assert!(r.metered);
        assert_eq!(r.ev_auto, CAMERA_GAIN_EV, "flat {v}");
        assert_eq!(r.ev_before_guard, CAMERA_GAIN_EV);
        assert_eq!(r.guard_reduction_ev, 0.0);
        assert_eq!(r.unmasked_pixels, 40 * 30);
        // The key statistic is still reported, and does not move the exposure.
        assert!(
            (r.mean_luminance - y).abs() < 1e-9 * y,
            "{v}: {}",
            r.mean_luminance
        );
    }
    // A gradient, a high-key and a low-key scene, and a spotlight: all the same exposure.
    let (w, h) = (60, 40);
    let scenes = [
        Image::new(w, h, |x, _| 0.02 + 0.6 * x as f64 / w as f64),
        Image::new(w, h, |x, y| 0.9 + 0.02 * ((x + y) % 5) as f64),
        Image::new(w, h, |x, y| 0.004 + 0.001 * ((x * 3 + y) % 7) as f64),
        Image::new(w, h, |x, y| {
            if (x as f64 - 30.0).abs() < 8.0 && (y as f64 - 20.0).abs() < 6.0 {
                1.2
            } else {
                0.03
            }
        }),
    ];
    for (i, img) in scenes.iter().enumerate() {
        let r = img.meter();
        assert_eq!(r.ev_auto, CAMERA_GAIN_EV, "scene {i}");
        assert_eq!(r.guard_reduction_ev, 0.0);
        // What the old metering would have chosen differs scene by scene by many stops; it is not used.
        let old = (TARGET_GREY / r.mean_luminance).log2();
        assert!((old - CAMERA_GAIN_EV).abs() > 0.05, "scene {i}: {old}");
    }
    // A key spread of more than 6 stops (0.004 to 1.2) gives exactly the same exposure.
    assert_eq!(scenes[1].meter().ev_auto, scenes[2].meter().ev_auto);
}

#[test]
fn the_camera_gain_is_the_documented_median_of_the_camera_engines() {
    // docs/02: +1.11 EV, the median of the 12 local files' camera JPEGs (std 0.13, range +0.87 to +1.33); it
    // is within the clamp, and not the +0.79 of an ideal ISO 12232 camera.
    assert_eq!(CAMERA_GAIN_EV, 1.11);
    assert!(CAMERA_GAIN_EV.abs() < exposure::EV_LIMIT);
    assert!((0.87..=1.33).contains(&CAMERA_GAIN_EV));
}

#[test]
fn the_exposure_compensation_reaches_the_picture_in_full() {
    // The raw of a shot taken with EC stops of compensation holds the scene 2^EC brighter. The exposure is the
    // same constant for both, so the picture (raw * 2^EV) is exactly 2^EC brighter: the whole compensation,
    // as in the camera's JPEG, and no part of it metered away. Scenes away from the highlight guard.
    let mut rng = Rng(0xEC_0001);
    for case in 0..200 {
        let (w, h) = (24 + case % 17, 18 + case % 11);
        let base = 0.004 + 0.05 * rng.next_f64();
        let tilt = 3.0 * rng.next_f64();
        let ec = 6.0 * rng.next_f64() - 3.0;
        let scene = Image::new(w, h, |x, y| {
            base * (1.0 + tilt * x as f64 / w as f64) * (0.6 + y as f64 / h as f64)
        });
        let scale = ec.exp2();
        let shot = Image {
            rgb: scene.rgb.iter().map(|v| v * scale as f32).collect(),
            ..Image::new(w, h, |_, _| 0.0)
        };
        let (a, b) = (scene.meter(), shot.meter());
        // Neither is near the guard (the brightest pixel stays below 8 / 2^1.11 = 3.7).
        let brightest = shot.rgb.iter().fold(0.0_f32, |m, v| m.max(*v));
        assert!(f64::from(brightest) < 3.0, "case {case}");
        assert_eq!(a.ev_auto, CAMERA_GAIN_EV);
        assert_eq!(a.ev_auto, b.ev_auto, "case {case}, EC {ec}");
        // The pictures: raw * 2^EV, brighter by exactly 2^EC (f32 scaling of the raw: 1e-6).
        for (x, y) in scene.rgb.iter().zip(&shot.rgb) {
            let ours = f64::from(*x) * a.ev_auto.exp2();
            let theirs = f64::from(*y) * b.ev_auto.exp2();
            assert!(
                (theirs / ours / scale - 1.0).abs() < 1e-6,
                "case {case}, EC {ec}: {theirs} / {ours}"
            );
        }
        // The metering statistic, which does not move the exposure, follows the scene by 2^EC.
        assert!((b.mean_luminance / a.mean_luminance / scale - 1.0).abs() < 1e-5);
    }
}

#[test]
fn the_ev_limit_stays_and_the_guard_can_only_lower_the_gain() {
    // The clamp of step 4 is unchanged: the gain is inside it, whatever the preview.
    assert_eq!(exposure::EV_LIMIT, 3.0);
    for img in [
        Image::flat(20, 20, 1e-5),
        Image::flat(20, 20, 0.0),
        Image::flat(20, 20, -0.01),
        Image::flat(20, 20, 100.0),
    ] {
        let r = img.meter();
        assert_eq!(r.ev_before_guard, CAMERA_GAIN_EV);
        assert!(r.ev_auto <= r.ev_before_guard);
    }
    // 100 after the gain is 216, above 8 everywhere: the guard would put 100 at 8 (-3.6 EV), which is more
    // than a stop, so it takes the whole stop it may.
    let r = Image::flat(20, 20, 100.0).meter();
    assert_eq!(r.guard_reduction_ev, 1.0);
    assert!(
        (r.ev_auto - (CAMERA_GAIN_EV - 1.0)).abs() < 1e-12,
        "{}",
        r.ev_auto
    );
    let very = Image::flat(20, 20, 4000.0).meter();
    assert_eq!(very.ev_auto, CAMERA_GAIN_EV - 1.0);
    assert_eq!(very.guard_reduction_ev, 1.0);
    // The lens-cap frame keeps the gain: no metering pushes it to the +3 limit any more.
    assert_eq!(Image::flat(37, 29, 0.0).meter().ev_auto, CAMERA_GAIN_EV);
}

/// An independent, deliberately plain implementation of steps 1 to 5 for the test scenes: the centre-weighted
/// geometric mean `m` (reported only), and `EV_auto`, the camera gain lowered by a bisection for the highlight guard.
fn reference_ev(img: &Image, matrix: &Mat3) -> (f64, f64) {
    let y_row = ym_color::gamut::awg4_luminance_row();
    let (w, h) = (img.width as f64, img.height as f64);
    let mut ys: Vec<(f64, f64)> = Vec::new(); // (Y, weight)
    let mut raw_ys: Vec<f64> = Vec::new(); // every unmasked pixel, before the floor
    for py in 0..img.height {
        for px in 0..img.width {
            let i = py * img.width + px;
            if img.mask[i] != 0 {
                continue;
            }
            let rgb = [
                f64::from(img.rgb[3 * i]),
                f64::from(img.rgb[3 * i + 1]),
                f64::from(img.rgb[3 * i + 2]),
            ];
            let lin = [
                matrix[0][0] * rgb[0] + matrix[0][1] * rgb[1] + matrix[0][2] * rgb[2],
                matrix[1][0] * rgb[0] + matrix[1][1] * rgb[1] + matrix[1][2] * rgb[2],
                matrix[2][0] * rgb[0] + matrix[2][1] * rgb[1] + matrix[2][2] * rgb[2],
            ];
            let y =
                (y_row[0] * lin[0] + y_row[1] * lin[1] + y_row[2] * lin[2]).max(LUMINANCE_FLOOR);
            raw_ys.push(y);
            let dx = px as f64 + 0.5 - w / 2.0;
            let dy = py as f64 + 0.5 - h / 2.0;
            let r = (dx * dx + dy * dy).sqrt() / (0.5 * (w * w + h * h).sqrt());
            ys.push((y, (-(r * r) / (2.0 * CENTRE_SIGMA * CENTRE_SIGMA)).exp()));
        }
    }
    let mut sorted: Vec<f64> = ys.iter().map(|p| p.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |p: f64| {
        let pos = p * (sorted.len() - 1) as f64;
        let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
        sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
    };
    let (lo, hi) = (q(0.01), q(0.99));
    let (mut sw, mut swl) = (0.0, 0.0);
    for &(y, wt) in &ys {
        if y >= lo && y <= hi {
            sw += wt;
            swl += wt * y.ln();
        }
    }
    let m = (swl / sw).exp();
    // The guard by bisection: the largest EV in [gain - 1, gain] with at most floor(0.005 n) pixels above 8.
    let allowed = (0.005 * raw_ys.len() as f64).floor() as usize;
    let over = |ev: f64| raw_ys.iter().filter(|&&y| y * ev.exp2() > 8.0).count();
    let ev = if over(CAMERA_GAIN_EV) <= allowed {
        CAMERA_GAIN_EV
    } else if over(CAMERA_GAIN_EV - 1.0) > allowed {
        CAMERA_GAIN_EV - 1.0
    } else {
        let (mut ok, mut bad) = (CAMERA_GAIN_EV - 1.0, CAMERA_GAIN_EV);
        for _ in 0..80 {
            let mid = 0.5 * (ok + bad);
            if over(mid) <= allowed {
                ok = mid;
            } else {
                bad = mid;
            }
        }
        ok
    };
    (m, ev)
}

#[test]
fn metering_agrees_with_a_plain_reimplementation_on_random_scenes() {
    let mut rng = Rng(0xE5_0C_5E);
    // A matrix that is not the identity: a mild colour matrix plus a gain.
    let matrix: Mat3 = [[1.3, -0.2, -0.1], [-0.15, 1.2, -0.05], [0.02, -0.3, 1.28]];
    let mut guarded = 0;
    for scene in 0..24 {
        let (w, h) = (24 + scene * 5, 18 + scene * 3);
        let base = 0.02 + 0.5 * rng.next_f64();
        let tilt = rng.next_f64() * 2.0;
        let mut img = Image::new(w, h, |x, y| {
            base * (1.0 + tilt * x as f64 / w as f64) * (0.5 + y as f64 / h as f64)
        });
        // Colour noise, hot pixels, black pixels, a clipped patch in the mask.
        for i in 0..w * h {
            for c in 0..3 {
                img.rgb[3 * i + c] *= (0.8 + 0.4 * rng.next_f64()) as f32;
            }
            let r = rng.next_f64();
            if r < 0.004 {
                img.rgb[3 * i..3 * i + 3].copy_from_slice(&[30.0, 25.0, 28.0]);
            } else if r < 0.008 {
                img.rgb[3 * i..3 * i + 3].copy_from_slice(&[0.0, 0.0, 0.0]);
            }
        }
        // Every other scene gets a bright patch that is not clipped, for the guard.
        if scene % 2 == 1 {
            for i in 0..w * h / 40 {
                img.rgb[3 * i..3 * i + 3].copy_from_slice(&[5.0, 4.5, 4.8]);
            }
        }
        for y in 0..h / 4 {
            for x in 0..w / 3 {
                img.mask[y * w + x] = 1;
                img.rgb[3 * (y * w + x)..3 * (y * w + x) + 3].copy_from_slice(&[50.0, 50.0, 50.0]);
            }
        }
        let got = auto_exposure(&img.preview(), &matrix).unwrap();
        let (m, ev) = reference_ev(&img, &matrix);
        assert!(
            (got.mean_luminance - m).abs() < 1e-9 * m,
            "scene {scene}: {} vs {m}",
            got.mean_luminance
        );
        // The exposure is the gain, lowered by the guard only.
        assert_eq!(got.ev_before_guard, CAMERA_GAIN_EV, "scene {scene}");
        assert!(
            (got.ev_auto - ev).abs() < 1e-9,
            "scene {scene}: {} vs {ev}",
            got.ev_auto
        );
        // The guard can only lower the exposure, by at most a stop.
        assert!(got.ev_auto <= got.ev_before_guard && got.ev_auto >= got.ev_before_guard - 1.0);
        assert_eq!(got.guard_reduction_ev, got.ev_before_guard - got.ev_auto);
        guarded += usize::from(got.guard_reduction_ev > 0.0);
    }
    assert!(guarded >= 4, "the sweep must exercise the guard: {guarded}");
}

#[test]
fn clipped_pixels_do_not_count() {
    // The left half is blown out and in the mask; the right half is middle grey.
    let mut img = Image::new(40, 20, |x, _| if x < 20 { 100.0 } else { 0.18 });
    for y in 0..20 {
        for x in 0..20 {
            img.mask[y * 40 + x] = 1;
        }
    }
    let r = img.meter();
    // The masked half is outside the statistic and outside the guard: the camera gain, untouched.
    assert_eq!(r.ev_auto, CAMERA_GAIN_EV);
    assert_eq!(r.guard_reduction_ev, 0.0);
    let grey = y_of_grey(f64::from(0.18_f32));
    assert!(
        (r.mean_luminance - grey).abs() < 1e-9,
        "{}",
        r.mean_luminance
    );
    assert_eq!(r.unmasked_pixels, 400);
    // Without the mask the same pixels are half the picture above 8 after the gain: the guard takes its stop,
    // and they dominate the reported statistic.
    let without = auto_exposure(
        &PreviewRgb {
            clip_mask: None,
            ..img.preview()
        },
        &IDENTITY,
    )
    .unwrap();
    assert_eq!(without.ev_auto, CAMERA_GAIN_EV - 1.0);
    assert!(without.mean_luminance > 1.0, "{}", without.mean_luminance);
    // Everything masked: nothing to report, and the exposure is the same camera gain.
    let all = Image {
        mask: vec![1; 40 * 20],
        ..Image::flat(40, 20, 100.0)
    };
    let r = all.meter();
    assert!(!r.metered);
    assert_eq!(r.ev_auto, CAMERA_GAIN_EV);
    assert_eq!(r.ev_before_guard, CAMERA_GAIN_EV);
    assert_eq!(r.guard_reduction_ev, 0.0);
    assert_eq!(r.metered_pixels, 0);
    // An empty preview likewise.
    let empty = auto_exposure(
        &PreviewRgb {
            width: 0,
            height: 0,
            rgb: &[],
            clip_mask: None,
        },
        &IDENTITY,
    )
    .unwrap();
    assert!(!empty.metered);
    assert_eq!(empty.ev_auto, CAMERA_GAIN_EV);
}

#[test]
fn pixels_that_are_not_finite_are_skipped() {
    let mut img = Image::flat(30, 30, 0.18);
    img.rgb[3 * 5] = f32::NAN;
    img.rgb[3 * 17 + 1] = f32::INFINITY;
    img.rgb[3 * 40 + 2] = f32::NEG_INFINITY;
    let r = img.meter();
    assert_eq!(r.unmasked_pixels, 900 - 3);
    assert_eq!(r.ev_auto, CAMERA_GAIN_EV);
    assert!((r.mean_luminance - y_of_grey(f64::from(0.18_f32))).abs() < 1e-9);
}

#[test]
fn flat_frames_of_any_size_keep_every_pixel_in_the_band() {
    // Regression: the percentile used `(1 - f) a + f b`, which is off by an ulp for some equal neighbours.
    // On a flat frame (a lens-cap frame is all at the luminance floor) the 1st percentile then landed above
    // every pixel, the band was empty and the EV was NaN, e.g. for 37 x 29 pixels.
    let mut checked = 0;
    for (w, h) in [
        (20, 20),
        (37, 29),
        (40, 30),
        (33, 33),
        (50, 50),
        (64, 48),
        (100, 100),
        (101, 103),
        (7, 11),
        (1188, 792),
    ] {
        for v in [0.0, 1e-5, 0.007, 0.07, 0.123_456_79, 0.18, 0.3, 1.0, 7.77] {
            let img = Image::flat(w, h, v);
            let r = img.meter();
            assert!(r.ev_auto.is_finite(), "{w} x {h} at {v}: {r:?}");
            assert_eq!(
                r.metered_pixels,
                w * h,
                "{w} x {h} at {v} lost pixels from the band"
            );
            let y = y_of_grey(f64::from(v as f32)).max(LUMINANCE_FLOOR);
            assert!(
                (r.mean_luminance - y).abs() < 1e-9 * y,
                "{w} x {h} at {v}: {}",
                r.mean_luminance
            );
            // The camera gain, except that the guard puts a frame brighter than 8 after the gain at 8 (by at
            // most the stop it may take): 7.77 here.
            let expect = if y * CAMERA_GAIN_EV.exp2() > 8.0 {
                (8.0 / y).log2().max(CAMERA_GAIN_EV - 1.0)
            } else {
                CAMERA_GAIN_EV
            };
            assert_eq!(r.ev_before_guard, CAMERA_GAIN_EV, "{w} x {h} at {v}");
            assert!(
                (r.ev_auto - expect).abs() < 1e-12,
                "{w} x {h} at {v}: {} vs {expect}",
                r.ev_auto
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 90);
    // The lens-cap frame of the documented size keeps the camera gain.
    assert_eq!(Image::flat(37, 29, 0.0).meter().ev_auto, CAMERA_GAIN_EV);
}

#[test]
fn a_preview_of_two_different_pixels_is_metered_not_nan() {
    // With only two distinct values neither lies between the 1st and 99th percentile; the band is empty
    // and every usable pixel is metered instead.
    let rgb = [0.1_f32, 0.1, 0.1, 0.3, 0.3, 0.3];
    let r = auto_exposure(
        &PreviewRgb {
            width: 2,
            height: 1,
            rgb: &rgb,
            clip_mask: None,
        },
        &IDENTITY,
    )
    .unwrap();
    assert!(r.metered && r.ev_auto.is_finite(), "{r:?}");
    assert_eq!(r.metered_pixels, 2);
    let y = [0.1_f32, 0.3].map(|v| y_of_grey(f64::from(v)));
    // Both pixels are the same distance from the centre, so the weights are equal: the geometric mean.
    assert!((r.mean_luminance - (y[0] * y[1]).sqrt()).abs() < 1e-12);
    // Other tiny previews stay finite too.
    for n in 1..=6 {
        let rgb: Vec<f32> = (0..n)
            .flat_map(|i| {
                let v = 0.05 * (i + 1) as f32;
                [v, v, v]
            })
            .collect();
        let r = auto_exposure(
            &PreviewRgb {
                width: n,
                height: 1,
                rgb: &rgb,
                clip_mask: None,
            },
            &IDENTITY,
        )
        .unwrap();
        assert!(
            r.ev_auto.is_finite() && r.metered_pixels > 0,
            "n = {n}: {r:?}"
        );
    }
}

#[test]
fn a_size_that_overflows_is_an_error_not_a_panic() {
    let rgb = [0.1_f32; 3];
    for (width, height) in [(usize::MAX, 2), (1 << 40, 1 << 40), (usize::MAX / 2, 3)] {
        let preview = PreviewRgb {
            width,
            height,
            rgb: &rgb,
            clip_mask: None,
        };
        assert!(matches!(
            auto_exposure(&preview, &IDENTITY),
            Err(ExposureError::BufferSize { .. })
        ));
        assert!(preview_median_luminance(&preview, &IDENTITY, 0.0).is_err());
    }
}

#[test]
fn the_first_and_last_percent_are_ignored() {
    // 0.4 % of the pixels near black and 0.4 % very bright (both inside 1 %): the metered mean is the
    // grey of the rest, and the guard (0.5 % above 8) does not act on 0.4 %.
    let (w, h) = (100, 100);
    let img = Image::new(w, h, |x, y| match y * w + x {
        i if i < 40 => 1e-3,
        i if i >= w * h - 40 => 50.0,
        _ => 0.18,
    });
    let r = img.meter();
    assert!(
        (r.mean_luminance - 0.18).abs() < 1e-6,
        "{}",
        r.mean_luminance
    );
    assert_eq!(r.ev_auto, CAMERA_GAIN_EV);
    assert_eq!(r.guard_reduction_ev, 0.0);
    assert!(r.metered_pixels < w * h, "the outliers were dropped");
    // Two percent each side: the 1st and 99th percentiles fall inside the outlier blocks, nothing is
    // dropped, and the mean moves (+0.8 % here); it is still the plain reimplementation's number.
    let more = Image::new(w, h, |x, y| match y * w + x {
        i if i < 200 => 1e-3,
        i if i >= w * h - 200 => 50.0,
        _ => 0.18,
    });
    let r = more.meter();
    assert_eq!(r.metered_pixels, w * h);
    assert!(
        (r.mean_luminance / 0.18 - 1.0).abs() > 1e-3,
        "{}",
        r.mean_luminance
    );
    let (m, _) = reference_ev(&more, &IDENTITY);
    assert!((r.mean_luminance - m).abs() < 1e-9 * m);
}

#[test]
fn the_centre_counts_more_than_the_edges() {
    // A bright centre on a dark frame meters brighter than a plain geometric mean (the weight is 1 at the
    // centre and exp(-1/(2 * 0.35^2)) = 0.017 at a corner).
    let (w, h) = (60, 40);
    let inside =
        |x: usize, y: usize| (x as f64 - 30.0).abs() < 12.0 && (y as f64 - 20.0).abs() < 8.0;
    let img = Image::new(w, h, |x, y| if inside(x, y) { 0.5 } else { 0.04 });
    let r = img.meter();
    let (m, ev) = reference_ev(&img, &IDENTITY);
    assert!((r.mean_luminance - m).abs() < 1e-9 * m);
    assert_eq!(r.ev_auto, ev);
    assert_eq!(r.ev_auto, CAMERA_GAIN_EV);
    let count = (0..w * h).filter(|i| inside(i % w, i / w)).count() as f64;
    let total = (w * h) as f64;
    let flat = ((count * 0.5_f64.ln() + (total - count) * 0.04_f64.ln()) / total).exp();
    assert!(
        r.mean_luminance > flat * 1.05,
        "{} vs unweighted {flat}",
        r.mean_luminance
    );
    // The weight itself: a single bright pixel at the centre against one in the corner.
    let mut centre = Image::flat(41, 41, 0.1);
    centre.rgb[3 * (20 * 41 + 20)..3 * (20 * 41 + 20) + 3].copy_from_slice(&[0.2; 3]);
    let mut corner = Image::flat(41, 41, 0.1);
    corner.rgb[0..3].copy_from_slice(&[0.2; 3]);
    // A lone outlier pixel is trimmed by the percentiles in both cases; use a block instead.
    for y in 18..23 {
        for x in 18..23 {
            centre.rgb[3 * (y * 41 + x)..3 * (y * 41 + x) + 3].copy_from_slice(&[0.2; 3]);
        }
    }
    for y in 0..5 {
        for x in 0..5 {
            corner.rgb[3 * (y * 41 + x)..3 * (y * 41 + x) + 3].copy_from_slice(&[0.2; 3]);
        }
    }
    assert!(centre.meter().mean_luminance > corner.meter().mean_luminance * 1.01);
}

#[test]
fn the_matrix_decides_the_luminance() {
    let img = Image::flat(30, 30, 0.09);
    // A gain of 2 in the chain doubles Y: the reported statistic follows, the exposure does not move.
    let double: Mat3 = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]];
    let a = auto_exposure(&img.preview(), &IDENTITY).unwrap();
    let b = auto_exposure(&img.preview(), &double).unwrap();
    assert!((b.mean_luminance / a.mean_luminance - 2.0).abs() < 1e-12);
    assert_eq!(a.ev_auto, CAMERA_GAIN_EV);
    assert_eq!(b.ev_auto, CAMERA_GAIN_EV);
    // Only the AWG4 Y row counts: a matrix that moves light between channels without changing Y of a
    // grey leaves the statistic alone.
    let swap: Mat3 = [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]];
    let c = auto_exposure(&img.preview(), &swap).unwrap();
    assert!((a.mean_luminance - c.mean_luminance).abs() < 1e-12);
    // The matrix does decide the highlight guard: the same picture through a bright matrix (Y = 8 / 2^1.11
    // and a bit over, on more than 0.5 % of the pixels) is lowered, through the plain one it is not.
    let bright = Image::flat(30, 30, 2.0);
    assert_eq!(bright.meter().guard_reduction_ev, 0.0);
    let r = auto_exposure(&bright.preview(), &double).unwrap();
    assert!(r.guard_reduction_ev > 0.0, "{r:?}");
}

#[test]
fn the_highlight_guard_follows_the_half_percent_rule() {
    // 10 000 pixels, a fraction of them at Y = 6 (not clipped, 13 after the gain), the rest middle grey.
    let (w, h) = (100, 100);
    let with_bright = |count: usize, level: f64| {
        Image::new(
            w,
            h,
            move |x, y| if y * w + x < count { level } else { 0.18 },
        )
    };
    // 0.5 % (50 pixels) is allowed: no change.
    let r = with_bright(50, 6.0).meter();
    assert_eq!(r.guard_reduction_ev, 0.0, "{r:?}");
    assert_eq!(r.ev_auto, CAMERA_GAIN_EV);
    // 51 pixels is more than 0.5 %: lowered until at most 50 are above 8, i.e. the 51st brightest
    // pixel (6) must land at 8: log2(8/6) = +0.415, which is 0.695 stops down from the gain.
    let r = with_bright(51, 6.0).meter();
    assert!(
        (r.ev_auto - (8.0_f64 / 6.0).log2()).abs() < 1e-5,
        "{}",
        r.ev_auto
    );
    assert!((r.guard_reduction_ev - (CAMERA_GAIN_EV - (8.0_f64 / 6.0).log2())).abs() < 1e-5);
    // A highlight that the gain brings to 8 or below does not count: 3.6 * 2^1.11 = 7.8.
    let r = with_bright(300, 3.6).meter();
    assert_eq!(r.guard_reduction_ev, 0.0, "{r:?}");
    // ... and one that only just exceeds it costs only the excess: 3.8 * 2^1.11 = 8.2.
    let r = with_bright(300, 3.8).meter();
    assert!(
        (r.ev_auto - (8.0_f64 / 3.8).log2()).abs() < 1e-5 && r.guard_reduction_ev < 0.05,
        "{r:?}"
    );
    // Very bright highlights: at most one stop.
    let r = with_bright(300, 100.0).meter();
    assert!((r.guard_reduction_ev - 1.0).abs() < 1e-9, "{r:?}");
    assert!((r.ev_auto - (r.ev_before_guard - 1.0)).abs() < 1e-9);
    // Masked (clipped) pixels are not counted: 300 blown-out pixels in the mask cause no guard.
    let mut masked = with_bright(300, 100.0);
    for i in 0..300 {
        masked.mask[i] = 1;
    }
    assert_eq!(masked.meter().guard_reduction_ev, 0.0);
    // The guard never raises the exposure.
    let r = Image::flat(20, 20, 0.001).meter();
    assert_eq!(r.guard_reduction_ev, 0.0);
    assert_eq!(r.ev_auto, r.ev_before_guard);
}

#[test]
fn the_buffers_are_checked() {
    let rgb = vec![0.1_f32; 3 * 10];
    let err = auto_exposure(
        &PreviewRgb {
            width: 4,
            height: 3,
            rgb: &rgb,
            clip_mask: None,
        },
        &IDENTITY,
    )
    .unwrap_err();
    assert_eq!(
        err,
        ExposureError::BufferSize {
            width: 4,
            height: 3,
            expected: 36,
            found: 30
        }
    );
    assert!(err.to_string().contains("36"));
    let rgb = vec![0.1_f32; 3 * 12];
    let mask = vec![0_u8; 11];
    let err = auto_exposure(
        &PreviewRgb {
            width: 4,
            height: 3,
            rgb: &rgb,
            clip_mask: Some(&mask),
        },
        &IDENTITY,
    )
    .unwrap_err();
    assert!(matches!(
        err,
        ExposureError::MaskSize {
            expected: 12,
            found: 11,
            ..
        }
    ));
    assert!(
        preview_median_luminance(
            &PreviewRgb {
                width: 5,
                height: 5,
                rgb: &rgb,
                clip_mask: None
            },
            &IDENTITY,
            0.0
        )
        .is_err()
    );
    assert_eq!(
        jpeg_median_luminance(&[1, 2, 3, 4]),
        Err(ExposureError::NotRgb(4))
    );
}

#[test]
fn the_exposure_setting_resolves_against_the_automatic_value() {
    assert_eq!(ExposureSetting::Auto.resolve(0.8), 0.8);
    assert_eq!(ExposureSetting::Manual { ev: -1.5 }.resolve(0.8), -1.5);
    assert_eq!(ExposureSetting::default(), ExposureSetting::Auto);
    assert_eq!(
        serde_json::to_string(&ExposureSetting::Auto).unwrap(),
        r#"{"kind":"auto"}"#
    );
    let m: ExposureSetting = serde_json::from_str(r#"{"kind":"manual","ev":0.25}"#).unwrap();
    assert_eq!(m, ExposureSetting::Manual { ev: 0.25 });
}

#[test]
fn a_manual_exposure_is_held_to_the_range_of_the_control() {
    // docs/01: Manual(ev) -4 to +4. (The white balance has the same kind of check.)
    assert_eq!(
        ExposureSetting::manual(-4.0),
        Ok(ExposureSetting::Manual { ev: -4.0 })
    );
    assert!(ExposureSetting::manual(4.0).is_ok());
    assert!(ExposureSetting::manual(0.01).is_ok());
    for ev in [
        4.01,
        -4.01,
        100.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let err = ExposureSetting::manual(ev).unwrap_err();
        assert!(matches!(err, ExposureError::EvOutOfRange(_)), "{ev}");
        assert!(err.to_string().contains("-4 to 4"));
        assert!(ExposureSetting::Manual { ev }.validate().is_err());
    }
    assert!(ExposureSetting::Auto.validate().is_ok());
    // The automatic value may be as low as -4 (the guard) and the range holds it.
    assert_eq!(exposure::EXPOSURE_MIN_EV, -4.0);
    assert_eq!(exposure::EXPOSURE_MAX_EV, 4.0);
}

// ---------------------------------------------------------------------------------------------
// Burst consistency
// ---------------------------------------------------------------------------------------------

fn shot(shutter_s: f64, iso: f64) -> ShotSettings {
    ShotSettings {
        shutter_s,
        f_number: 5.6,
        iso,
        exposure_compensation_ev: 0.0,
    }
}

fn frame(camera: &str, time_s: Option<f64>, settings: ShotSettings) -> BurstFrame {
    BurstFrame {
        camera: camera.to_string(),
        capture_time_s: time_s,
        settings,
    }
}

#[test]
fn a_burst_shares_the_median_ev() {
    let s = shot(1.0 / 250.0, 100.0);
    let frames: Vec<BurstFrame> = (0..5)
        .map(|i| frame("A1", Some(100.0 + 0.1 * f64::from(i)), s))
        .collect();
    let ev = [0.5, 0.9, 0.1, 0.4, 2.0];
    let shared = share_burst_ev(&frames, &ev);
    assert_eq!(shared, vec![0.5; 5]);
    // Even groups use the mean of the two middle values.
    let shared = share_burst_ev(&frames[..4], &ev[..4]);
    assert_eq!(shared, vec![0.45; 4]);
    // One frame is its own group.
    assert_eq!(share_burst_ev(&frames[..1], &ev[..1]), vec![0.5]);
    assert!(share_burst_ev(&[], &[]).is_empty());
}

#[test]
fn a_burst_keeps_the_constant_gain_and_a_guarded_frame_takes_the_median() {
    // Five frames of one burst whose key differs by many stops (the subject moves, the light changes): every
    // frame has the same exposure and the shared one is that constant, as before the median and after it.
    let s = shot(1.0 / 500.0, 100.0);
    let frames: Vec<BurstFrame> = (0..5)
        .map(|i| frame("A1", Some(100.0 + 0.1 * f64::from(i)), s))
        .collect();
    let keys = [0.004, 0.03, 0.18, 0.9, 1.4];
    let own: Vec<f64> = keys
        .iter()
        .map(|&v| Image::flat(40, 30, v).meter().ev_auto)
        .collect();
    assert_eq!(own, vec![CAMERA_GAIN_EV; 5]);
    assert_eq!(share_burst_ev(&frames, &own), vec![CAMERA_GAIN_EV; 5]);
    // One frame with a sky that clips (its guard acted) is the odd one out: the group median is still the
    // constant, so that frame takes the camera gain too (the spec orders the sharing after the guard).
    let mut own = own;
    let guarded = Image::new(100, 100, |x, y| if y * 100 + x < 300 { 6.0 } else { 0.18 }).meter();
    assert!(guarded.ev_auto < CAMERA_GAIN_EV);
    own[2] = guarded.ev_auto;
    assert_eq!(share_burst_ev(&frames, &own), vec![CAMERA_GAIN_EV; 5]);
    // Two guarded frames out of five still leave the median at the constant; three move it.
    own[3] = guarded.ev_auto;
    assert_eq!(share_burst_ev(&frames, &own), vec![CAMERA_GAIN_EV; 5]);
    own[4] = guarded.ev_auto;
    assert_eq!(share_burst_ev(&frames, &own), vec![guarded.ev_auto; 5]);
}

#[test]
fn burst_grouping_rules() {
    let s = shot(1.0 / 125.0, 400.0);
    let t0 = 1_700_000_000.0;
    let frames = vec![
        frame("A1", Some(t0), s),                               // 0: group A
        frame("A1", Some(t0 + 1.75), s),                        // 1: within 2 s of 0
        frame("A1", Some(t0 + 3.5), s), // 2: within 2 s of 1: still the burst
        frame("A1", Some(t0 + 3.5 + BURST_WINDOW_S), s), // 3: exactly 2 s after 2: still in
        frame("A1", Some(t0 + 3.5 + BURST_WINDOW_S + 2.25), s), // 4: more than 2 s later: new group
        frame("A7C", Some(t0 + 0.5), s), // 5: other camera
        frame("A1", Some(t0 + 0.5), shot(1.0 / 125.0, 800.0)), // 6: other ISO
        frame(
            "A1",
            Some(t0 + 0.6),
            ShotSettings {
                exposure_compensation_ev: 0.7,
                ..s
            },
        ), // 7: other EC
        frame("A1", Some(t0 + 0.6), ShotSettings { f_number: 8.0, ..s }), // 8: other aperture
        frame("A1", None, s),           // 9: no time
        frame("A1", None, s),           // 10: no time
    ];
    let g = burst_groups(&frames);
    assert_eq!(g[0], g[1]);
    assert_eq!(g[1], g[2]);
    assert_eq!(g[2], g[3]);
    assert_ne!(g[3], g[4]);
    for other in [5, 6, 7, 8] {
        assert_ne!(g[other], g[0], "frame {other}");
    }
    assert_ne!(g[9], g[10], "frames without a time are alone");
    assert_ne!(g[9], g[0]);
    // Numbered by first appearance in the input.
    assert_eq!(g[0], 0);
    let mut seen = Vec::new();
    for &x in &g {
        if !seen.contains(&x) {
            assert_eq!(x, seen.len());
            seen.push(x);
        }
    }
}

#[test]
fn bursts_do_not_depend_on_the_input_order_and_interleave_with_other_cameras() {
    let s = shot(1.0 / 500.0, 200.0);
    let frames = vec![
        frame("B", Some(10.3), s),
        frame("A", Some(10.2), s),
        frame("B", Some(10.1), s),
        frame("A", Some(10.0), s),
        frame("B", Some(30.0), s),
    ];
    let ev = [3.0, 1.0, 2.0, 0.0, 5.0];
    let shared = share_burst_ev(&frames, &ev);
    // B: 10.1 and 10.3 share the mean of 3 and 2; the one at 30 s is alone. A: mean of 1 and 0.
    assert_eq!(shared, vec![2.5, 0.5, 2.5, 0.5, 5.0]);
    // Reversed input gives the same sharing.
    let mut f2 = frames.clone();
    f2.reverse();
    let mut e2 = ev;
    e2.reverse();
    let mut s2 = share_burst_ev(&f2, &e2);
    s2.reverse();
    assert_eq!(s2, shared);
    // A different frame of the other camera in between does not break a burst.
    let s_other = shot(1.0 / 60.0, 100.0);
    let mixed = vec![
        frame("A", Some(0.0), s),
        frame("A", Some(0.5), s_other),
        frame("A", Some(1.0), s),
    ];
    let g = burst_groups(&mixed);
    assert_eq!(g[0], g[2]);
    assert_ne!(g[0], g[1]);
}

#[test]
fn burst_grouping_matches_a_quadratic_reference_on_random_shoots() {
    // Reference: sort by time; a frame joins the group of the latest earlier frame with the same camera and
    // identical settings when that frame is at most BURST_WINDOW_S before it, otherwise it starts a group.
    // Times are multiples of 0.25 s so the 2 s boundary is exact.
    let mut rng = Rng(0xB0_57);
    for round in 0..200 {
        let count = 1 + (rng.next_f64() * 24.0) as usize;
        let mut frames = Vec::new();
        let mut clock = 1000.0;
        for _ in 0..count {
            clock += 0.25 * (rng.next_f64() * 14.0).floor();
            let camera = ["A", "B"][(rng.next_f64() * 2.0) as usize];
            let settings = ShotSettings {
                shutter_s: [1.0 / 250.0, 1.0 / 125.0][(rng.next_f64() * 2.0) as usize],
                f_number: 5.6,
                iso: [100.0, 200.0][(rng.next_f64() * 2.0) as usize],
                exposure_compensation_ev: 0.0,
            };
            let time = if rng.next_f64() < 0.1 {
                None
            } else {
                Some(clock)
            };
            frames.push(frame(camera, time, settings));
        }
        // Shuffle the input order (Fisher-Yates) so the order of the list is not the order in time.
        for i in (1..frames.len()).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            frames.swap(i, j);
        }
        let got = burst_groups(&frames);

        let mut order: Vec<usize> = (0..frames.len())
            .filter(|&i| frames[i].capture_time_s.is_some())
            .collect();
        order.sort_by(|&a, &b| {
            let (ta, tb) = (
                frames[a].capture_time_s.unwrap(),
                frames[b].capture_time_s.unwrap(),
            );
            ta.total_cmp(&tb).then(a.cmp(&b))
        });
        let mut want: Vec<usize> = (0..frames.len()).collect(); // own group by default
        for (pos, &i) in order.iter().enumerate() {
            let previous = order[..pos].iter().rev().find(|&&j| {
                frames[j].camera == frames[i].camera
                    && frames[j].settings.same_as(&frames[i].settings)
            });
            if let Some(&j) = previous
                && frames[i].capture_time_s.unwrap() - frames[j].capture_time_s.unwrap()
                    <= BURST_WINDOW_S
            {
                want[i] = want[j];
            }
        }
        // The same partition, whatever the labels.
        for a in 0..frames.len() {
            for b in 0..frames.len() {
                assert_eq!(
                    got[a] == got[b],
                    want[a] == want[b],
                    "round {round}: frames {a} and {b}\n{frames:?}\n{got:?}"
                );
            }
        }
        // Labels count from 0 in order of first appearance.
        let mut seen = 0;
        for &g in &got {
            assert!(g <= seen, "round {round}: {got:?}");
            if g == seen {
                seen += 1;
            }
        }
    }
}

#[test]
fn identical_settings_means_the_same_numbers() {
    let a = shot(1.0 / 3.0, 100.0);
    let b = shot(0.333_333_333_333_333_3, 100.0);
    assert!(a.same_as(&b));
    assert!(!a.same_as(&shot(1.0 / 3.1, 100.0)));
    assert!(!a.same_as(&shot(1.0 / 3.0, 125.0)));
    // NaN times never group.
    let frames = vec![frame("A", Some(f64::NAN), a), frame("A", Some(f64::NAN), a)];
    let g = burst_groups(&frames);
    assert_ne!(g[0], g[1]);
    assert_eq!(share_burst_ev(&frames, &[1.0, 2.0]), vec![1.0, 2.0]);
}

// ---------------------------------------------------------------------------------------------
// The embedded-JPEG comparison
// ---------------------------------------------------------------------------------------------

#[test]
fn srgb_decoding_matches_the_standard() {
    assert_eq!(srgb_to_linear(0.0), 0.0);
    assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-15);
    assert!((srgb_to_linear(0.5) - 0.214_041_14).abs() < 1e-8);
    assert!((srgb_to_linear(0.04045) - 0.04045 / 12.92).abs() < 1e-15);
    assert!((srgb_to_linear(118.0 / 255.0) - 0.181_164).abs() < 1e-5);
}

#[test]
fn the_median_luminance_of_a_jpeg_is_in_linear_light() {
    // Four equal blocks: the median is the mean of the two middle ones, in linear light, not in codes.
    let (w, h, pixels) = decode_jpeg(&from_hex(GREY_JPEG_HEX));
    assert_eq!((w, h), (16, 16));
    assert_eq!(pixels.len(), 16 * 16 * 3);
    let median = jpeg_median_luminance(&pixels).unwrap().unwrap();
    let expect = 0.5 * (srgb_to_linear(90.0 / 255.0) + srgb_to_linear(150.0 / 255.0));
    assert!((median / expect - 1.0).abs() < 0.02, "{median} vs {expect}");
    // Working on the code values instead (the mean code 120 decoded) would be 8 % off.
    assert!((srgb_to_linear(120.0 / 255.0) / expect - 1.0).abs() > 0.05);

    // Colours: Rec.709 luminance of the decoded channels.
    let (_, _, pixels) = decode_jpeg(&from_hex(COLOR_JPEG_HEX));
    let lum = |c: [f64; 3]| {
        0.2126 * srgb_to_linear(c[0] / 255.0)
            + 0.7152 * srgb_to_linear(c[1] / 255.0)
            + 0.0722 * srgb_to_linear(c[2] / 255.0)
    };
    let mut ys = [
        lum([200.0, 40.0, 40.0]),
        lum([40.0, 200.0, 40.0]),
        lum([40.0, 40.0, 200.0]),
        lum([128.0, 128.0, 128.0]),
    ];
    ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let expect = 0.5 * (ys[1] + ys[2]);
    let median = jpeg_median_luminance(&pixels).unwrap().unwrap();
    assert!((median / expect - 1.0).abs() < 0.05, "{median} vs {expect}");
}

#[test]
fn a_preview_median_is_taken_over_every_pixel_at_the_exposure() {
    let img = Image::new(20, 20, |x, _| if x < 10 { 0.1 } else { 0.4 });
    let mut masked = Image::new(20, 20, |x, _| if x < 10 { 0.1 } else { 0.4 });
    for y in 0..20 {
        masked.mask[y * 20 + 15] = 1;
    }
    for ev in [-1.0, 0.0, 1.5] {
        let m = preview_median_luminance(&img.preview(), &IDENTITY, ev)
            .unwrap()
            .unwrap();
        // Half the pixels at 0.1 and half at 0.4: the median is the mean of the two.
        let expect = 0.25 * ev.exp2();
        assert!((m / expect - 1.0).abs() < 1e-6, "{ev}: {m}");
        // The mask is ignored: the JPEG's median includes every pixel.
        let masked_m = preview_median_luminance(&masked.preview(), &IDENTITY, ev)
            .unwrap()
            .unwrap();
        assert_eq!(m, masked_m);
    }
    let empty = PreviewRgb {
        width: 0,
        height: 0,
        rgb: &[],
        clip_mask: None,
    };
    assert_eq!(
        preview_median_luminance(&empty, &IDENTITY, 0.0).unwrap(),
        None
    );
    assert_eq!(jpeg_median_luminance(&[]).unwrap(), None);
}

#[test]
fn the_error_is_in_stops_and_signed() {
    assert_eq!(exposure_error_ev(0.36, 0.18), 1.0);
    assert_eq!(exposure_error_ev(0.09, 0.18), -1.0);
    assert_eq!(exposure_error_ev(0.18, 0.18), 0.0);
}

#[test]
fn the_gate_counts_the_fraction_within_the_tolerance_and_lists_the_outliers() {
    let entry = |name: &str, err_ev: f64| GateEntry {
        name: name.to_string(),
        ours_median_y: 0.18 * err_ev.exp2(),
        reference_median_y: 0.18,
    };
    // Eight of ten within half a stop (one exactly at the limit), two outside.
    let mut entries: Vec<GateEntry> = (0..7)
        .map(|i| entry(&format!("ok{i}"), -0.4 + 0.1 * f64::from(i)))
        .collect();
    entries.push(entry("limit", 0.5));
    entries.push(entry("bright", 0.9));
    entries.push(entry("dark", -1.6));
    let report = evaluate_gate(&entries, GATE_TOLERANCE_EV, GATE_REQUIRED_FRACTION);
    assert_eq!(report.evaluated, 10);
    assert_eq!(report.within, 8);
    assert!((report.fraction - 0.8).abs() < 1e-12);
    assert!(report.passes, "80 % is enough");
    let names: Vec<&str> = report.outliers.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(names, ["dark", "bright"], "worst first");
    assert!((report.outliers[0].error_ev + 1.6).abs() < 1e-12);

    // One more outlier and it fails.
    entries.push(entry("bright2", 0.6));
    let report = evaluate_gate(&entries, GATE_TOLERANCE_EV, GATE_REQUIRED_FRACTION);
    assert_eq!(report.within, 8);
    assert_eq!(report.evaluated, 11);
    assert!(!report.passes);

    // Entries without a usable median are skipped and named; an empty set never passes.
    let bad = vec![
        entry("fine", 0.0),
        GateEntry {
            name: "no jpeg".to_string(),
            ours_median_y: 0.2,
            reference_median_y: 0.0,
        },
        GateEntry {
            name: "nan".to_string(),
            ours_median_y: f64::NAN,
            reference_median_y: 0.2,
        },
    ];
    let report = evaluate_gate(&bad, GATE_TOLERANCE_EV, GATE_REQUIRED_FRACTION);
    assert_eq!((report.evaluated, report.within), (1, 1));
    assert_eq!(report.skipped, ["no jpeg", "nan"]);
    assert!(report.passes);
    let none = evaluate_gate(&[], GATE_TOLERANCE_EV, GATE_REQUIRED_FRACTION);
    assert!(!none.passes);
    assert_eq!(none.fraction, 0.0);
}

#[test]
fn the_whole_gate_pipeline_on_synthetic_scenes_with_planted_camera_offsets() {
    // 20 synthetic scenes (grey pixels, so Rec.709 and AWG4 luminance agree). The "camera JPEG" is the
    // scene exposed at EV_auto plus a planted offset, sRGB-encoded to 8 bits and measured the way a
    // decoded JPEG is. The gate must see the planted offsets (to the 8-bit quantisation) and classify
    // the scenes accordingly.
    let mut rng = Rng(0x00_6A7E);
    let mut entries = Vec::new();
    let mut planted = Vec::new();
    for scene in 0..20 {
        let (w, h) = (48, 32);
        let base = 0.01 + 0.12 * rng.next_f64();
        let spread = 1.0 + 3.0 * rng.next_f64();
        let img = Image::new(w, h, |x, y| {
            base * spread.powf(x as f64 / w as f64)
                * (0.6 + 0.8 * ((x * 7 + y * 13) % 11) as f64 / 11.0)
        });
        let r = img.meter();
        let ours = preview_median_luminance(&img.preview(), &IDENTITY, r.ev_auto)
            .unwrap()
            .unwrap();
        // 14 scenes agree with the camera to 0.3 EV, 3 are 0.7 to 1.2 EV off, 3 within 0.45 EV on the other side.
        let offset = match scene {
            0..=13 => (rng.next_f64() - 0.5) * 0.6,
            14..=16 => 0.7 + 0.5 * rng.next_f64(),
            _ => -(0.3 + 0.15 * rng.next_f64()),
        };
        planted.push(offset);
        let gain = (r.ev_auto - offset).exp2();
        let codes: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let lin = (f64::from(img.rgb[3 * i]) * gain).clamp(0.0, 1.0);
                let code = if lin <= 0.003_130_8 {
                    12.92 * lin
                } else {
                    1.055 * lin.powf(1.0 / 2.4) - 0.055
                };
                let c = (code * 255.0).round() as u8;
                [c, c, c]
            })
            .collect();
        let reference = jpeg_median_luminance(&codes).unwrap().unwrap();
        entries.push(GateEntry {
            name: format!("scene{scene:02}"),
            ours_median_y: ours,
            reference_median_y: reference,
        });
    }
    for (e, offset) in entries.iter().zip(&planted) {
        let measured = exposure_error_ev(e.ours_median_y, e.reference_median_y);
        assert!(
            (measured - offset).abs() < 0.03,
            "{}: planted {offset:.3}, measured {measured:.3}",
            e.name
        );
    }
    let report = evaluate_gate(&entries, GATE_TOLERANCE_EV, GATE_REQUIRED_FRACTION);
    let expected_outliers: Vec<usize> = planted
        .iter()
        .enumerate()
        .filter(|(_, o)| o.abs() > GATE_TOLERANCE_EV + 0.03)
        .map(|(i, _)| i)
        .collect();
    // Scenes 14 to 16 were planted 0.7 to 1.2 EV off; nothing else is outside half a stop.
    assert_eq!(expected_outliers, [14, 15, 16]);
    let mut found: Vec<&str> = report.outliers.iter().map(|o| o.name.as_str()).collect();
    found.sort_unstable();
    assert_eq!(found, ["scene14", "scene15", "scene16"]);
    assert_eq!(report.within, 17);
    assert!((report.fraction - 0.85).abs() < 1e-12);
    assert!(report.passes, "17 of 20 is above the 80 % bar");
}

#[test]
fn the_constants_are_the_ones_of_the_spec() {
    assert_eq!(TARGET_GREY, 0.18);
    assert_eq!(CENTRE_SIGMA, 0.35);
    assert_eq!(CAMERA_GAIN_EV, 1.11);
    assert_eq!(exposure::EV_LIMIT, 3.0);
    assert_eq!(exposure::PERCENTILE_LOW, 0.01);
    assert_eq!(exposure::PERCENTILE_HIGH, 0.99);
    assert_eq!(exposure::GUARD_LEVEL, 8.0);
    assert_eq!(exposure::GUARD_FRACTION, 0.005);
    assert_eq!(exposure::GUARD_MAX_REDUCTION_EV, 1.0);
    assert_eq!(BURST_WINDOW_S, 2.0);
    assert_eq!(GATE_TOLERANCE_EV, 0.5);
    assert_eq!(GATE_REQUIRED_FRACTION, 0.8);
}

#[test]
fn a_full_size_eighth_resolution_preview_meters_in_well_under_a_second() {
    // A 61 MP frame at 1/8 resolution is 1188 x 792 pixels. The bound is far above the real time (tens of
    // milliseconds); it only guards against a quadratic mistake.
    let (w, h) = (1188, 792);
    let img = Image::new(w, h, |x, y| {
        0.02 + 0.3 * ((x * 31 + y * 17) % 97) as f64 / 97.0
    });
    let start = std::time::Instant::now();
    let r = auto_exposure(&img.preview(), &IDENTITY).unwrap();
    let elapsed = start.elapsed();
    eprintln!("auto exposure on {w} x {h}: {elapsed:?}");
    assert!(r.metered && r.unmasked_pixels == w * h);
    assert!(elapsed.as_secs_f64() < 3.0, "{elapsed:?}");
}
