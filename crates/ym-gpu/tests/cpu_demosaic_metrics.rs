//! Unit tests of the demosaic gate's measurement code (`tests/common/demosaic_metrics.rs`): the
//! zipper and false-colour metrics, the sRGB decode, the noise-model scaling. The gate itself (the
//! tables of `docs/05` R6) is `cargo run -p ym-gpu --example demosaic_gate`.
//!
//! The module's own tests run here (it is included, not a dependency). On top of them this file
//! checks that the metrics rank the real methods the way the literature says: bilinear zippers
//! and shows false colour at saturated edges, the directional methods much less.

#[path = "common/demosaic_metrics.rs"]
mod demosaic_metrics;

use demosaic_metrics::{
    BORDER_PX, LabImage, PRIMARY, SENSITIVITY, artifacts, artifacts_row, demosaic_sets_root,
    linearize, load_set, mean_finite, psnr_row, synthetic_set,
};
use ym_gpu::cpu::RgbImage;
use ym_gpu::cpu::demosaic::Method;
use ym_gpu::cpu::synth;

fn column(label: &str) -> usize {
    PRIMARY
        .iter()
        .position(|v| v.label == label)
        .expect("a column of PRIMARY")
}

#[test]
fn the_columns_cover_the_methods_the_rule_compares() {
    let methods: Vec<Method> = PRIMARY.iter().map(|v| v.method).collect();
    for m in [
        Method::Bilinear,
        Method::Menon,
        Method::Lmmse,
        Method::AdaptiveRatio,
        Method::LogGuided,
    ] {
        assert!(methods.contains(&m), "{m:?} is missing from PRIMARY");
    }
    // The sensitivity columns vary the noise model of methods that read one.
    for v in SENSITIVITY {
        assert!(matches!(
            v.method,
            Method::AdaptiveRatio | Method::LogGuided
        ));
    }
}

#[test]
fn zipper_and_false_colour_rank_bilinear_below_the_directional_methods() {
    // Saturated colour regions with straight, slanted and curved edges: where bilinear zippers.
    let truth = synth::edges(192, 192);
    let rows = artifacts_row(&truth, &PRIMARY);
    let (bilinear, menon, lmmse, ar) = (
        rows[column("bilinear")],
        rows[column("menon")],
        rows[column("lmmse")],
        rows[column("adaptive_ratio")],
    );
    assert!(bilinear.edge_px > 100, "edge pixels: {}", bilinear.edge_px);
    assert_eq!(
        bilinear.edge_px, menon.edge_px,
        "the denominators come from the ground truth alone"
    );
    for (name, a) in [("menon", menon), ("lmmse", lmmse), ("adaptive_ratio", ar)] {
        assert!(
            a.zipper_pct < bilinear.zipper_pct,
            "{name}: zipper {:.2} % against bilinear {:.2} %",
            a.zipper_pct,
            bilinear.zipper_pct
        );
        assert!(
            a.chroma_de00_mean < bilinear.chroma_de00_mean,
            "{name}: chroma dE00 {:.3} against bilinear {:.3}",
            a.chroma_de00_mean,
            bilinear.chroma_de00_mean
        );
    }
}

#[test]
fn the_psnr_row_has_one_value_per_column_and_scale_does_not_break_it() {
    let (_, truth) = synthetic_set()
        .into_iter()
        .next()
        .expect("a synthetic image");
    for scale in demosaic_metrics::SCALES {
        let row = psnr_row(&truth, scale, &PRIMARY);
        assert_eq!(row.len(), PRIMARY.len());
        assert!(
            row.iter().all(|v| v.is_finite() && *v > 5.0),
            "scale {scale}: {row:?}"
        );
    }
    // The Lab conversion keeps the image size.
    let lab = LabImage::new(&truth);
    assert_eq!(lab.lab.len(), truth.width * truth.height);
    assert_eq!(BORDER_PX, 10);
}

/// Images at or below the border size have no interior: the rates are `NaN` (no pixel to count)
/// rather than a panic or a division by zero, and the mean helper skips them.
#[test]
fn images_without_an_interior_give_no_rate_and_no_panic() {
    for (w, h) in [
        (1, 1),
        (2, 2),
        (1, 40),
        (40, 1),
        (15, 15),
        (2 * BORDER_PX, 2 * BORDER_PX),
    ] {
        let mut img = RgbImage::new(w, h);
        img.data.fill(0.3);
        let lab = LabImage::new(&img);
        let a = artifacts(&lab, &lab, BORDER_PX);
        assert_eq!(a.edge_px, 0, "{w}x{h}");
        assert!(
            a.zipper_pct.is_nan(),
            "{w}x{h}: no edge pixels, no zipper rate"
        );
        assert!(a.false_colour_pct.is_nan(), "{w}x{h}: no interior, no rate");
        assert!(a.chroma_de00_mean.is_nan(), "{w}x{h}");
    }
    // One interior pixel is enough for a (zero) rate.
    let mut img = RgbImage::new(2 * BORDER_PX + 1, 2 * BORDER_PX + 1);
    img.data.fill(0.3);
    let lab = LabImage::new(&img);
    let a = artifacts(&lab, &lab, BORDER_PX);
    assert_eq!(a.false_colour_pct, 0.0);
    assert!(mean_finite([a.false_colour_pct, f64::NAN]).abs() < 1e-12);
}

/// The measurement protocol against the numbers in `docs/05` R6, on Kodak when it is there
/// (`tools/fetch/demosaic-sets.sh`): the gamma-encoded values must reproduce the M1-12 table
/// (bilinear 29.82, Menon 38.88, LMMSE 39.16) and the sRGB-decoded values the demosaic gate's
/// table at scale 1 (Menon 36.14, LMMSE 36.41, adaptive ratio 36.38, log-guided 32.53). A change of
/// the decode, the border, the clamp or the noise model of the gate shows up here. Run with
/// `cargo test -p ym-gpu --test cpu_demosaic_metrics --release -- --ignored --nocapture`.
#[test]
#[ignore = "needs testdata/demosaic/kodak (tools/fetch/demosaic-sets.sh)"]
fn the_protocol_reproduces_the_numbers_of_docs_05_on_kodak() {
    let images = load_set(&demosaic_sets_root().join("kodak"));
    if images.is_empty() {
        eprintln!(
            "skipped: no Kodak images in {}",
            demosaic_sets_root().display()
        );
        return;
    }
    assert_eq!(images.len(), 24, "the Kodak set has 24 images");
    let mean_over = |linear: bool| -> Vec<f64> {
        let mut mean = vec![0.0; PRIMARY.len()];
        for (_, img) in &images {
            let truth = if linear { linearize(img) } else { img.clone() };
            for (m, v) in mean.iter_mut().zip(psnr_row(&truth, 1.0, &PRIMARY)) {
                *m += v / images.len() as f64;
            }
        }
        mean
    };
    let column = |label: &str| {
        PRIMARY
            .iter()
            .position(|v| v.label == label)
            .expect("column")
    };
    let gamma = mean_over(false);
    for (label, want) in [("bilinear", 29.82), ("menon", 38.88), ("lmmse", 39.16)] {
        let got = gamma[column(label)];
        assert!(
            (got - want).abs() < 0.006,
            "gamma {label}: {got:.4} against {want}"
        );
    }
    let linear = mean_over(true);
    for (label, want) in [
        ("menon", 36.14),
        ("lmmse", 36.41),
        ("adaptive_ratio", 36.38),
        ("log_guided", 32.53),
    ] {
        let got = linear[column(label)];
        assert!(
            (got - want).abs() < 0.006,
            "linear {label}: {got:.4} against {want}"
        );
    }
}
