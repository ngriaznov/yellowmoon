//! M1-06 review: the estimator on hostile input. Whatever the preview and the metadata hold (NaN, infinities, negative and
//! enormous values, one-pixel and one-row images, absurd gains and exposure settings), `estimate` must neither panic nor
//! hand a non-finite number to the colour chain.

use std::path::PathBuf;

use ym_color::illum::{IlluminantEstimate, IlluminantModel, RawPreview, ShotInfo};

fn a7r4() -> IlluminantModel {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    IlluminantModel::from_path(root.join("data/models/illum/sony-ilce-7rm4.json")).unwrap()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Every number an estimate carries must be usable by `SceneLight` and the chain.
fn assert_sane(e: &IlluminantEstimate, what: &str) {
    assert!(
        e.w_scene.iter().all(|c| c.is_finite() && *c > 0.0),
        "{what}: w_scene {:?}",
        e.w_scene
    );
    assert!((e.w_scene[1] - 1.0).abs() < 1e-15, "{what}: green");
    assert!(
        e.chroma_uv.iter().all(|c| c.is_finite()),
        "{what}: chroma {:?}",
        e.chroma_uv
    );
    assert!(
        e.covariance_uv.iter().flatten().all(|c| c.is_finite()),
        "{what}: covariance {:?}",
        e.covariance_uv
    );
    let p = e.class_probabilities();
    assert!(
        p.iter().all(|c| c.is_finite() && (0.0..=1.0).contains(c)),
        "{what}: classes {p:?}"
    );
    assert!(
        (p.iter().sum::<f64>() - 1.0).abs() < 1e-9,
        "{what}: classes sum {p:?}"
    );
    assert!(
        e.confidence.is_finite() && (0.0..=1.0).contains(&e.confidence),
        "{what}: confidence {}",
        e.confidence
    );
}

#[test]
fn hostile_pixels_sizes_and_metadata_never_panic_or_give_non_finite_output() {
    let model = a7r4();
    let specials = [
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        -1.0,
        0.0,
        1e-30,
        1e-40,
        1.0,
        3.0e38,
        f32::MAX,
        f32::MIN_POSITIVE,
    ];
    let shots = [
        ShotInfo::default(),
        ShotInfo {
            as_shot_gains: Some([2.0, 1.0, 1.5]),
            f_number: Some(4.0),
            exposure_time_s: Some(0.01),
            iso: Some(400.0),
        },
        ShotInfo {
            as_shot_gains: Some([1e-300, 1.0, 1e300]),
            ..ShotInfo::default()
        },
        ShotInfo {
            as_shot_gains: Some([1e300, 1.0, 1e-300]),
            ..ShotInfo::default()
        },
        ShotInfo {
            as_shot_gains: Some([f64::MIN_POSITIVE, 1.0, f64::MAX]),
            ..ShotInfo::default()
        },
        ShotInfo {
            f_number: Some(1e200),
            exposure_time_s: Some(1e-300),
            iso: Some(1e300),
            ..ShotInfo::default()
        },
        ShotInfo {
            f_number: Some(1e200),
            exposure_time_s: Some(1.0),
            iso: Some(100.0),
            ..ShotInfo::default()
        },
        ShotInfo {
            f_number: Some(4.0),
            exposure_time_s: Some(1e-320),
            iso: Some(100.0),
            ..ShotInfo::default()
        },
        ShotInfo {
            f_number: Some(1e-200),
            exposure_time_s: Some(1e300),
            iso: Some(1e-300),
            ..ShotInfo::default()
        },
    ];
    let mut rng = Lcg(42);
    let sizes = [
        (0usize, 0usize),
        (1, 1),
        (1, 9),
        (9, 1),
        (2, 2),
        (3, 3),
        (17, 5),
        (64, 48),
        (300, 200),
    ];
    for (width, height) in sizes {
        for round in 0..6 {
            let n = width * height;
            let mut rgb = Vec::with_capacity(3 * n);
            for _ in 0..3 * n {
                let v = match round {
                    0 => (rng.next() * 0.9) as f32,
                    1 => specials[(rng.next() * specials.len() as f64) as usize % specials.len()],
                    2 => {
                        if rng.next() < 0.1 {
                            specials[(rng.next() * specials.len() as f64) as usize % specials.len()]
                        } else {
                            (rng.next() * 0.5) as f32
                        }
                    }
                    3 => 0.2, // flat grey: no edges at all
                    4 => (rng.next() * 1e-3) as f32,
                    _ => ((rng.next() - 0.3) * 4.0) as f32,
                };
                rgb.push(v);
            }
            let mask: Vec<u8> = (0..n)
                .map(|_| u8::from(round == 2 && rng.next() < 0.5))
                .collect();
            for shot in &shots {
                for with_mask in [false, true] {
                    let preview = RawPreview {
                        width,
                        height,
                        rgb: &rgb,
                        clip_mask: with_mask.then_some(mask.as_slice()),
                    };
                    let est = model.estimate(&preview, shot).unwrap();
                    assert_sane(
                        &est,
                        &format!("{width}x{height} round {round} mask {with_mask} shot {shot:?}"),
                    );
                }
            }
        }
    }
}

#[test]
fn histograms_of_the_wrong_shape_are_an_error_not_a_panic() {
    use ym_color::illum::{HistogramSpec, Histograms};
    let model = a7r4();
    let n = model.histogram_spec().n_bins;
    let channels = model.histogram_spec().channels.len();
    let good = || Histograms {
        data: vec![1.0 / (n * n) as f64; channels * n * n],
        valid_fraction: vec![1.0; channels],
        in_range: vec![1.0; channels],
        mass: vec![1.0; channels],
    };
    // The right shape works.
    let ok = model
        .estimate_from_histograms(good(), None, None)
        .expect("a histogram of the model's shape");
    assert_sane(&ok.estimate, "flat histogram");
    // Every wrong shape is refused.
    let mut short = good();
    short.data.truncate(n * n);
    let mut no_fraction = good();
    no_fraction.valid_fraction.clear();
    let mut no_mass = good();
    no_mass.mass.clear();
    let empty = Histograms {
        data: Vec::new(),
        valid_fraction: Vec::new(),
        in_range: Vec::new(),
        mass: Vec::new(),
    };
    for (what, h) in [
        ("short data", short),
        ("no valid fraction", no_fraction),
        ("no mass", no_mass),
        ("empty", empty),
    ] {
        assert!(
            model.estimate_from_histograms(h, None, None).is_err(),
            "{what}"
        );
    }
    // A hand-made specification with fewer than two bins cannot be histogrammed into.
    let mut spec: HistogramSpec = model.histogram_spec().clone();
    for bins in [0usize, 1] {
        spec.n_bins = bins;
        let rgb = vec![0.2f32; 3 * 4 * 4];
        let preview = RawPreview {
            width: 4,
            height: 4,
            rgb: &rgb,
            clip_mask: None,
        };
        assert!(
            ym_color::illum::histogram::compute(&spec, &preview).is_err(),
            "{bins} bins"
        );
    }
}

#[test]
fn exposure_settings_that_overflow_are_ignored_like_missing_ones() {
    let overflow = ShotInfo {
        f_number: Some(1e200),
        exposure_time_s: Some(1e-300),
        iso: Some(100.0),
        ..ShotInfo::default()
    };
    assert_eq!(overflow.bv(), None);
    let denormal = ShotInfo {
        f_number: Some(4.0),
        exposure_time_s: Some(1e-320),
        iso: Some(100.0),
        ..ShotInfo::default()
    };
    assert_eq!(denormal.bv(), None);
    let fine = ShotInfo {
        f_number: Some(4.0),
        exposure_time_s: Some(1.0 / 60.0),
        iso: Some(100.0),
        ..ShotInfo::default()
    };
    assert!(fine.bv().is_some_and(f64::is_finite));
}
