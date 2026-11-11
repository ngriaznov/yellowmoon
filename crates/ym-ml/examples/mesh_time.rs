//! Times Face Mesh V2 and Selfie Multiclass (`docs/27` S-09, S-10): one network pass each, on 4 threads and on 1.
//!
//! ```text
//! cargo run --release -p ym-ml --example mesh_time -- <weights dir>
//! ```

use std::time::Instant;

use ym_ml::mediapipe::facemesh::FaceMeshNet;
use ym_ml::mediapipe::selfie::SelfieNet;

fn time(label: &str, f: impl Fn()) {
    f();
    let n = 10;
    let t = Instant::now();
    for _ in 0..n {
        f();
    }
    println!(
        "{label}: {:.1} ms",
        t.elapsed().as_secs_f64() * 1e3 / f64::from(n)
    );
}

fn main() {
    let dir = std::path::PathBuf::from(std::env::args().nth(1).expect("weights dir"));
    let mesh = FaceMeshNet::load(&dir).expect("mesh");
    let selfie = SelfieNet::load(&dir).expect("selfie");
    let input: Vec<f32> = (0..256 * 256 * 3)
        .map(|i| ((i * 7919) % 255) as f32 / 255.0)
        .collect();
    let one = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("pool");
    time("Face Mesh V2, one 256 x 256 pass, 4 threads", || {
        mesh.forward(&input).expect("forward");
    });
    one.install(|| {
        time("Face Mesh V2, 1 thread", || {
            mesh.forward(&input).expect("forward");
        })
    });
    time("Selfie Multiclass, one 256 x 256 pass, 4 threads", || {
        selfie.forward(&input).expect("forward");
    });
    one.install(|| {
        time("Selfie Multiclass, 1 thread", || {
            selfie.forward(&input).expect("forward");
        })
    });
}
