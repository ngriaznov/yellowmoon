//! A missing or unusable ONNX Runtime library is an error, never a panic (feature `ort`).
//!
//! This is a separate test binary on purpose: ONNX Runtime is loaded once per process and the first attempt
//! decides (`ort_backend::LIBRARY`), so the failing attempt must not share a process with the parity test.
#![cfg(feature = "ort")]

mod common;

use ym_ml::backends::ort_backend::OrtJdd;
use ym_ml::model::{MlError, TileShape};

#[test]
fn a_missing_library_is_an_error_not_a_panic_and_stays_one() {
    let onnx = common::data_dir().join("golden_small.onnx");
    let shape = TileShape::square(32);
    // 1. A library path that does not exist.
    match OrtJdd::with_library("/nonexistent/libonnxruntime.so", &onnx, shape, 32) {
        Err(MlError::Backend {
            backend: "ort",
            detail,
        }) => {
            assert!(
                detail.contains("libonnxruntime"),
                "unhelpful message: {detail}"
            );
        }
        Err(other) => panic!("wrong error: {other}"),
        Ok(_) => panic!("loading a missing library succeeded"),
    }
    // 2. The failure is remembered. A second attempt, with the default library lookup or with any other
    //    path, reports it again; it used to run into ort's half-initialised library state and panic.
    assert!(OrtJdd::new(&onnx, shape, 32).is_err());
    assert!(OrtJdd::with_library("/also/missing/libonnxruntime.so", &onnx, shape, 32).is_err());
}
