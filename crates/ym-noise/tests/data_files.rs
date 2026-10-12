//! The shipped noise data and the Python fixture, read through the Rust loader.

use std::path::Path;

use ym_noise::priors::{
    BPrior, BodyClass, CameraContext, ClassAssumptions, PriorSource, PriorStore,
};

fn data_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/cameras/noise")
}

/// `crates/ym-noise/tests/data/384-ilce-1.json` is a synthetic body file written by
/// `training/noise/test_public_priors.py`; copy it next to the shipped gain-regime table.
fn store_with_fixture() -> (tempfile::TempDir, PriorStore) {
    let dir = tempfile::tempdir().unwrap();
    std::fs::copy(
        data_dir().join("gain-regimes.json"),
        dir.path().join("gain-regimes.json"),
    )
    .unwrap();
    std::fs::copy(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/384-ilce-1.json"),
        dir.path().join("384-ilce-1.json"),
    )
    .unwrap();
    let store = PriorStore::load_dir(dir.path()).unwrap();
    (dir, store)
}

#[test]
fn the_shipped_directory_loads_and_every_body_gets_a_prior() {
    let store = PriorStore::load_dir(&data_dir()).unwrap();
    assert_eq!(store.gain_regimes().bodies.len(), 13);
    // No body file exists until the corpus has been processed (M2-01): the class prior applies.
    assert!(store.is_empty());
    let p = store.prior_for(&CameraContext {
        model_id: Some(384),
        model: Some("ILCE-1".into()),
        iso: Some(800),
        class: None,
    });
    assert_eq!(p.source, PriorSource::Class(BodyClass::FullFrame));
    let want = ClassAssumptions::a_at_iso100(BodyClass::FullFrame) * 8.0;
    assert!((p.a_center().unwrap() / want - 1.0).abs() < 1e-12);
}

#[test]
fn the_python_fixture_is_read_by_the_rust_loader() {
    let (_dir, store) = store_with_fixture();
    let file = store.body(384).expect("the fixture body file");
    assert_eq!(file.model, "ILCE-1");
    assert_eq!(file.regimes.len(), 2);
    assert_eq!(file.regimes[0].iso_min, 100);
    assert_eq!(file.regimes[0].iso_max, Some(499));
    assert_eq!(file.regimes[1].iso_max, None);

    // Low-gain regime, ISO 160: both fits present.
    let ctx = |iso| CameraContext {
        model_id: Some(384),
        model: Some("ILCE-1".into()),
        iso: Some(iso),
        class: None,
    };
    let p = store.prior_for(&ctx(160));
    assert_eq!(
        p.source,
        PriorSource::PublicData {
            model_id: 384,
            regime: 0
        }
    );
    // The synthetic truth is a = 3e-5 * ISO / 100 (see the Python test): the prior follows it.
    let a = p.a_center().unwrap();
    assert!((a / (3.0e-5 * 1.6) - 1.0).abs() < 0.10, "a prior {a}");
    assert!((p.log_a.unwrap().sigma - 0.25).abs() < 1e-12);
    match p.b {
        BPrior::Gaussian { mean, sigma } => {
            assert!(mean > 0.0);
            assert!((sigma / mean - 0.4).abs() < 1e-12);
        }
        other => panic!("expected a Gaussian b prior, got {other:?}"),
    }
    // Extended low ISO 50 uses the gain of ISO 100.
    let p50 = store.prior_for(&ctx(50));
    let p100 = store.prior_for(&ctx(100));
    assert_eq!(p50.a_center(), p100.a_center());
    // High-gain regime above the switch (500).
    let p = store.prior_for(&ctx(1600));
    assert_eq!(
        p.source,
        PriorSource::PublicData {
            model_id: 384,
            regime: 1
        }
    );
    let a = p.a_center().unwrap();
    assert!((a / (3.0e-5 * 16.0) - 1.0).abs() < 0.10, "a prior {a}");
    // ISO 25600 is above the A1's "processed" threshold (16000): widened.
    assert!(store.prior_for(&ctx(25_600)).widened);
    assert!(!store.prior_for(&ctx(1600)).widened);
    // A body without a file keeps the class prior.
    let other = store.prior_for(&CameraContext {
        model_id: Some(375),
        model: Some("ILCE-7RM4".into()),
        iso: Some(400),
        class: None,
    });
    assert_eq!(other.source, PriorSource::Class(BodyClass::FullFrame));
}
