//! Shared helpers of the integration tests: fixture paths, a tiny deterministic RNG and a frame generator.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use safetensors::SafeTensors;
use ym_ml::model::{TileInputs, TileShape};
use ym_ml::tiles::Frame;

pub fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
}

pub fn read(name: &str) -> Vec<u8> {
    std::fs::read(data_dir().join(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

/// Float tensors of a safetensors file by name, with their shapes.
pub fn load_tensors(bytes: &[u8]) -> HashMap<String, (Vec<usize>, Vec<f32>)> {
    let st = SafeTensors::deserialize(bytes).expect("valid safetensors");
    st.tensors()
        .into_iter()
        .map(|(name, t)| {
            let v = t
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            (name, (t.shape().to_vec(), v))
        })
        .collect()
}

pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / (1u64 << 24) as f32
    }
}

/// A plausible random frame: balanced planes in [0, 1.5], small sigma and qstep, sparse clip mask.
pub fn random_frame(width: usize, height: usize, seed: u64) -> Frame {
    let mut rng = Lcg(seed);
    let mut f = Frame::zeros(width, height);
    for c in 0..4 {
        for v in f.plane_mut(c) {
            *v = rng.next_f32() * 1.5;
        }
    }
    for v in f.plane_mut(4) {
        *v = rng.next_f32() * 0.05;
    }
    for v in f.plane_mut(5) {
        *v = rng.next_f32() * 0.01;
    }
    for v in f.plane_mut(6) {
        *v = if rng.next_f32() > 0.95 { 1.0 } else { 0.0 };
    }
    f
}

pub fn tile_from(shape: TileShape, data: &[f32]) -> TileInputs {
    TileInputs::new(shape, data.to_vec()).expect("tile length")
}
