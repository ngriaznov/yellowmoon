//! The GPU backends behind `TileModel` (burn and ort); the M0 benchmark decides which one ships
//! (`docs/07-ml-models.md` section Runtime).
//!
//! Both are optional cargo features, **off by default**, so the default workspace build compiles neither:
//!
//! * feature `burn`: `burn_backend::BurnJdd`, the network written with burn 0.22 tensors on the wgpu
//!   backend, weights from the same safetensors file as the CPU runtime;
//! * feature `ort`: `ort_backend::OrtJdd`, ONNX Runtime 2.x loaded dynamically (`ORT_DYLIB_PATH`) running
//!   the static-shape ONNX file exported by `training/jdd/export.py`.
//!
//! The pure-Rust CPU runtime ([`crate::cpu_runtime`]) is always available.

#[cfg(feature = "burn")]
pub mod burn_backend;
#[cfg(feature = "ort")]
pub mod ort_backend;

/// Names of the backends compiled into this build, for logs and the benchmark.
pub fn available() -> Vec<&'static str> {
    #[allow(unused_mut)]
    let mut v = vec!["cpu"];
    #[cfg(feature = "burn")]
    v.push("burn");
    #[cfg(feature = "ort")]
    v.push("ort");
    v
}

#[cfg(test)]
mod tests {
    #[test]
    fn cpu_is_always_available() {
        assert!(super::available().contains(&"cpu"));
    }
}
