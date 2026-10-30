//! The GPU colour kernel against the committed OpenColorIO vectors (`crates/ym-color/tests/data`).
//!
//! * M1-10, LUT gate: tetrahedral interpolation of our own synthetic cubes against OpenColorIO's
//!   tetrahedral result, in the kernel's f32 arithmetic (and with it the kernel's CDL-free LUT path,
//!   shapers and domains).
//!
//! Inputs that must survive f16 storage are not forced through it: the f32 device runs the exact
//! vectors; f16 devices run them too and report their (slightly larger) difference.

mod common;

use std::path::PathBuf;

use common::chains::identity_tone;
use common::gpus;
use serde_json::Value;
use std::sync::Arc;
use ym_color::consts::linalg::IDENTITY;
use ym_color::cube::parse_lut;
use ym_color::display::{ChainParams, ColorChain, LookSetting};
use ym_color::logc::logc4_decode_rgb;
use ym_color::look::{LookInput, LookTransform};
use ym_gpu::cpu::RgbImage;
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};

const CODE10: f64 = 1.0 / 1023.0;

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../ym-color/tests/data")
}

fn triplets(v: &Value) -> Vec<[f64; 3]> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|row| {
            let r = row.as_array().expect("triplet");
            let f = |i: usize| r[i].as_f64().expect("number") as f32 as f64;
            [f(0), f(1), f(2)]
        })
        .collect()
}

fn image_of(px: &[[f64; 3]]) -> RgbImage {
    let inter: Vec<f32> = px.iter().flatten().map(|&v| v as f32).collect();
    RgbImage::from_interleaved(px.len(), 1, &inter).expect("image")
}

fn render_stage(
    dev: &common::TestGpu,
    params: &ChainParams,
    img: &RgbImage,
    stop: Stop,
) -> Vec<f32> {
    let chain = ColorChain::new(params).expect("chain");
    let img = img.clone();
    let out = dev
        .gpu
        .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
            let p = PreparedChain::new(ctx, &chain)?;
            p.render(ctx, &img, (0, 0), ColorOutput::Stage(stop))
        })
        .expect("render");
    match out.data {
        PixelData::F32Planar(v) => v,
        other => panic!("{other:?}"),
    }
}

#[test]
fn lut_gate_against_opencolorio_tetrahedral_vectors() {
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(data_dir().join("lut/lut_vectors.json")).expect("vectors"),
    )
    .expect("json");
    for dev in gpus() {
        // LogC4 codes above ~1.3 decode beyond f16's range: the f32 device takes every vector, the
        // f16 devices are covered by the parity tests in gpu_color.rs.
        if dev.mode != StorageMode::F32 {
            continue;
        }
        let mut worst_overall = 0.0f64;
        for case in doc["cases"].as_array().expect("cases") {
            let file = case["file"].as_str().expect("file");
            let text = std::fs::read_to_string(data_dir().join("lut").join(file)).expect("cube");
            let lut = parse_lut(&text).expect("parse");
            let look = LookTransform::new(LookInput::LogC4, None, Some(lut)).expect("look");
            let inputs = triplets(&case["inputs"]);
            let expected = triplets(&case["outputs"]);
            // Camera values that make C8's output equal the LUT input: decode the LogC4 codes.
            let cams: Vec<[f64; 3]> = inputs.iter().map(|&c| logc4_decode_rgb(c)).collect();
            let mut params = ChainParams::new(IDENTITY, identity_tone());
            params.look = Some(LookSetting::new(Arc::new(look), 100.0));
            let img = image_of(&cams);
            let g = render_stage(dev, &params, &img, Stop::C10);
            let n = inputs.len();
            let mut worst = 0.0f64;
            for i in 0..n {
                for ch in 0..3 {
                    worst = worst.max((f64::from(g[ch * n + i]) - expected[i][ch]).abs());
                }
            }
            eprintln!(
                "{file}: {n} vectors, worst |GPU - OCIO| {:.2e} ({:.4} codes at 10 bits)",
                worst,
                worst * 1023.0
            );
            worst_overall = worst_overall.max(worst);
            // The vector inputs reach 1.5 in places; going through LogC4 decode and encode in f32
            // costs a few 1e-7, the LUT arithmetic about the same.
            assert!(worst <= 1.0e-4, "{file}: worst difference {worst}");
        }
        eprintln!(
            "LUT gate: worst over all cubes {worst_overall:.2e} ({:.4} codes at 10 bits; gate 1 code = {CODE10:.1e})",
            worst_overall * 1023.0
        );
    }
}

#[test]
fn hardware_style_trilinear_interpolation_would_fail_the_lut_gate() {
    // A negative control: on the warp cube the vectors separate tetrahedral from trilinear by
    // far more than the gate, so the 4e-7 agreement above can only come from tetrahedral maths.
    let doc: Value = serde_json::from_str(
        &std::fs::read_to_string(data_dir().join("lut/lut_vectors.json")).expect("vectors"),
    )
    .expect("json");
    let case = doc["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .find(|c| c["file"] == "warp_17.cube")
        .expect("warp cube");
    let text = std::fs::read_to_string(data_dir().join("lut/warp_17.cube")).expect("cube");
    let lut = parse_lut(&text).expect("parse");
    let (inputs, expected) = (triplets(&case["inputs"]), triplets(&case["outputs"]));
    let mut worst = 0.0f64;
    for (i, e) in inputs.iter().zip(&expected) {
        let t = lut.cube.sample_trilinear(*i);
        for ch in 0..3 {
            worst = worst.max((t[ch] - e[ch]).abs());
        }
    }
    eprintln!(
        "trilinear on the warp cube: {:.2e} from OpenColorIO's tetrahedral ({:.2} codes at 10 bits)",
        worst,
        worst * 1023.0
    );
    assert!(
        worst > 1.5 * CODE10,
        "the vectors no longer separate the two methods: {worst}"
    );
}

#[test]
fn f16_rounded_tables_would_exceed_the_parity_target() {
    // The decision of kernels::color: cubes, shapers and curve tables are f32 storage because an
    // f16 table rounds values in [0.5, 1) by up to 2^-12 = 2.4e-4, more than the 2e-4 gate.
    let curve = ym_color::curve::Curve::new(&[(0.0, 0.0), (0.5, 0.55), (1.0, 1.0)]).expect("curve");
    let table = curve.bake();
    let worst = table
        .values()
        .iter()
        .map(|&v| (f64::from(common::f16_round(v as f32)) - v).abs())
        .fold(0.0, f64::max);
    eprintln!("largest f16 rounding error of a baked curve table: {worst:.2e}");
    assert!(worst > 2.0e-4, "{worst}");
    assert!(worst <= 2.5e-4);
}
