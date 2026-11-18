//! The two classes of vectors of the raw algorithms (`docs/29-raw-constants.md` F-06 and RC-04) and the
//! generator of the second one.
//!
//! * **Oracle** (`tests/cpu_data/<algorithm>/`): made by the development oracle, never edited. It runs
//!   under `Tuning::ORACLE` (the cubics the vectors were made with, for the log-guided demosaic) and
//!   is the proof of the *structure*: borders, phases, ties and decision logic.
//! * **Self** (`tests/cpu_data/<algorithm>_self/`): made by [`generate`] from our own Rust under
//!   `Tuning::DEFAULT` (the shipped cubics), with the inputs, parameters, field names and tolerances
//!   of the oracle set. It is the regression of the shipped constants and is never called an
//!   oracle: when a fit changes a constant, the set is regenerated (`cargo run -p ym-gpu --example
//!   gen_self_vectors`) and the diff shows what moved.
//!
//! The generator reads each oracle file, runs our code on its `input` and `params` and writes a file
//! of the same fields in which only the results (`expected`, `expected_float`, the `derived` block)
//! come from us. This module is included by the tests (`mod self_gen;`) and by the example
//! (`#[path]`), so the tests regenerate with the very code that wrote the committed files.

#![allow(dead_code)]

use std::path::{Path, PathBuf};

use serde_json::{Map, Number, Value};
use ym_gpu::cpu::demosaic::adaptive_ratio::{self, Derived, Params, ShadowMatch};
use ym_gpu::cpu::demosaic::log_guided;
use ym_gpu::cpu::demosaic::{CaCurves, DemosaicInputs, RadialCurve};
use ym_gpu::cpu::output_sharpen::{ExtraStage, Method, Mode, OutputSharpen};
use ym_gpu::cpu::sharpen::diffraction::{DiffractionParams, DiffractionPlan, Tuning as DiffTuning};
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn, Placement, RgbImage};
use ym_noise::model::NoiseModel;

/// Which of the two sets of a package a test reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    /// The unedited vectors of the development oracle, run under `Tuning::ORACLE`.
    Oracle,
    /// The vectors our own code writes, run under `Tuning::DEFAULT`.
    SelfSet,
}

/// Both classes, in the order the tests run them.
pub const CLASSES: [Class; 2] = [Class::Oracle, Class::SelfSet];

impl Class {
    /// The folder of `package` for this class, under `tests/cpu_data/`.
    pub fn folder(self, package: Package) -> &'static str {
        match self {
            Class::Oracle => package.oracle_folder(),
            Class::SelfSet => package.self_folder(),
        }
    }

    /// The directory of `package` for this class.
    pub fn dir(self, package: Package) -> PathBuf {
        data_root().join(self.folder(package))
    }

    /// The constants this class runs under: `oracle` for the oracle set, `shipped` for the self set.
    pub fn pick<T>(self, oracle: T, shipped: T) -> T {
        match self {
            Class::Oracle => oracle,
            Class::SelfSet => shipped,
        }
    }

    /// A word for messages.
    pub fn label(self) -> &'static str {
        match self {
            Class::Oracle => "oracle",
            Class::SelfSet => "self",
        }
    }
}

/// The four packages that have vectors.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Package {
    /// Spec 01, the adaptive ratio demosaic.
    AdaptiveRatio,
    /// Spec 02, the log-guided demosaic.
    LogGuided,
    /// Spec 03, the diffraction deconvolution.
    Diffraction,
    /// Spec 04, the output sharpening.
    OutputSharpen,
}

/// Every package, in spec order.
pub const PACKAGES: [Package; 4] = [
    Package::AdaptiveRatio,
    Package::LogGuided,
    Package::Diffraction,
    Package::OutputSharpen,
];

impl Package {
    /// The folder of the oracle set.
    pub const fn oracle_folder(self) -> &'static str {
        match self {
            Package::AdaptiveRatio => "adaptive_ratio_demosaic",
            Package::LogGuided => "log_guided_demosaic",
            Package::Diffraction => "diffraction_deconvolution",
            Package::OutputSharpen => "output_sharpening",
        }
    }

    /// The folder of the self set.
    pub const fn self_folder(self) -> &'static str {
        match self {
            Package::AdaptiveRatio => "adaptive_ratio_demosaic_self",
            Package::LogGuided => "log_guided_demosaic_self",
            Package::Diffraction => "diffraction_deconvolution_self",
            Package::OutputSharpen => "output_sharpening_self",
        }
    }

    /// Whether the constants the shipped build runs differ from the ones the oracle vectors were made
    /// with. For the log-guided demosaic that is the pair of cubics (`FixedCubics::SHIPPED` against
    /// `ORACLE_CUBICS`, `docs/29` RC-01); the other packages compare their `Tuning`.
    pub fn default_differs_from_oracle(self) -> bool {
        use ym_gpu::cpu::{demosaic, output_sharpen, sharpen};
        match self {
            Package::AdaptiveRatio => {
                demosaic::adaptive_ratio::Tuning::DEFAULT
                    != demosaic::adaptive_ratio::Tuning::ORACLE
            }
            Package::LogGuided => log_guided::FixedCubics::SHIPPED != log_guided::ORACLE_CUBICS,
            Package::Diffraction => {
                sharpen::diffraction::Tuning::DEFAULT != sharpen::diffraction::Tuning::ORACLE
            }
            Package::OutputSharpen => {
                output_sharpen::Tuning::DEFAULT != output_sharpen::Tuning::ORACLE
            }
        }
    }
}

/// `crates/ym-gpu/tests/cpu_data`.
pub fn data_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/cpu_data")
}

/// The `.json` files of a vector folder, sorted by name.
pub fn json_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("vector directory {}: {e}", dir.display()))
        .map(|e| e.expect("directory entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    files
}

/// Reads a vector file.
pub fn read_vector(path: &Path) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("vector file {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("vector json {}: {e}", path.display()))
}

// ---------------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------------

/// The text of a vector file: the top-level fields one per line, each value compact. One format, so
/// that the committed bytes and a regeneration can be compared.
pub fn render(v: &Value) -> String {
    let Value::Object(map) = v else {
        panic!("a vector is an object");
    };
    let mut out = String::from("{\n");
    let n = map.len();
    for (i, (key, value)) in map.iter().enumerate() {
        out.push_str("  ");
        out.push_str(&serde_json::to_string(key).expect("key"));
        out.push_str(": ");
        out.push_str(&serde_json::to_string(value).expect("value"));
        out.push_str(if i + 1 < n { ",\n" } else { "\n" });
    }
    out.push_str("}\n");
    out
}

/// An `f32` as a JSON number with nine significant digits, as the oracle's files hold their `f32`
/// values (`%.9g`): enough for the value to read back as the same `f32`, and the same digits as the
/// oracle's text, so that an unchanged value is an unchanged line.
fn f32v(x: f32) -> Value {
    assert!(x.is_finite(), "a vector holds finite numbers, got {x}");
    let text = format!("{x:.8e}");
    let parsed: f64 = text.parse().expect("an f32 prints as a number");
    Value::Number(Number::from_f64(parsed).expect("finite"))
}

fn f64v(x: f64) -> Value {
    assert!(x.is_finite(), "a vector holds finite numbers, got {x}");
    Value::Number(Number::from_f64(x).expect("finite"))
}

fn uint(x: usize) -> Value {
    Value::Number(Number::from(x as u64))
}

fn f32s(xs: &[f32]) -> Value {
    Value::Array(xs.iter().map(|&x| f32v(x)).collect())
}

fn f64s(xs: &[f64]) -> Value {
    Value::Array(xs.iter().map(|&x| f64v(x)).collect())
}

// ---------------------------------------------------------------------------------------------
// Reading what the oracle files hold
// ---------------------------------------------------------------------------------------------

fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("number {key} in {v}"))
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

fn doubles(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

fn object_mut(v: &mut Value) -> &mut Map<String, Value> {
    v.as_object_mut().expect("an object")
}

/// Sets `key`, which the oracle's file has too: the self set keeps the oracle's fields.
fn set(map: &mut Map<String, Value>, key: &str, value: Value) {
    let slot = map
        .get_mut(key)
        .unwrap_or_else(|| panic!("the oracle file has no field {key}"));
    *slot = value;
}

/// The keys of an object, for the check that no field was added or left stale.
fn keys(v: &Value) -> Vec<String> {
    v.as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}

/// Fails when `block` has a field that is neither in `refreshed` (which the generator recomputes) nor
/// in `structure` (which it keeps as the oracle states it): a new field of the oracle's files must not
/// be copied silently into the self set.
fn assert_fields(what: &str, block: &Map<String, Value>, refreshed: &[&str], structure: &[&str]) {
    let mut have: Vec<&str> = block.keys().map(String::as_str).collect();
    let mut want: Vec<&str> = refreshed.iter().chain(structure).copied().collect();
    have.sort_unstable();
    want.sort_unstable();
    assert_eq!(have, want, "{what}: the generator knows these fields");
}

/// Marks a file as ours, without touching its fields.
fn mark_self(doc: &mut Value) {
    if let Some(Value::String(s)) = object_mut(doc).get_mut("description") {
        s.push_str(" [self set: made by gen_self_vectors from our own Rust under the shipped constants; not an oracle]");
    }
}

// ---------------------------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------------------------

/// Which constants the generator runs our code under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Regime {
    /// What ships: `Tuning::DEFAULT` and the shipped cubics. The self set is written under it.
    Shipped,
    /// What the oracle vectors were made with: `Tuning::ORACLE` and `ORACLE_CUBICS`. Only the tests
    /// use it, to prove the generator right: its output must agree with the oracle files.
    Oracle,
}

/// The self vector of one oracle vector: the same document with the results made by our code under
/// `regime`.
pub fn vector(package: Package, oracle: &Value, regime: Regime) -> Value {
    match package {
        Package::AdaptiveRatio => adaptive_ratio_vector(oracle, regime),
        Package::LogGuided => log_guided_vector(oracle, regime),
        Package::Diffraction => diffraction_vector(oracle, regime),
        Package::OutputSharpen => output_sharpening_vector(oracle, regime),
    }
}

/// Writes the self set of `package` into `out` (created if missing) from the oracle files and our
/// code under the shipped constants, and returns the names of the files written.
pub fn generate(package: Package, out: &Path) -> Vec<String> {
    std::fs::create_dir_all(out).unwrap_or_else(|e| panic!("create {}: {e}", out.display()));
    let mut written = Vec::new();
    for path in json_files(&Class::Oracle.dir(package)) {
        let oracle = read_vector(&path);
        let doc = vector(package, &oracle, Regime::Shipped);
        assert_eq!(
            keys(&doc),
            keys(&oracle),
            "{}: the self file has the oracle's fields",
            path.display()
        );
        let name = path.file_name().expect("file name").to_owned();
        std::fs::write(out.join(&name), render(&doc))
            .unwrap_or_else(|e| panic!("write {}: {e}", name.to_string_lossy()));
        written.push(name.to_string_lossy().into_owned());
    }
    written
}

/// Writes the self sets of all four packages under `root` (`<root>/<package>_self/`).
pub fn generate_all(root: &Path) -> Vec<(Package, Vec<String>)> {
    PACKAGES
        .iter()
        .map(|&p| (p, generate(p, &root.join(p.self_folder()))))
        .collect()
}

// --- spec 01 -----------------------------------------------------------------------------------

fn adaptive_ratio_vector(oracle: &Value, regime: Regime) -> Value {
    let params = &oracle["params"];
    let input = &oracle["input"];
    let (w, h) = (
        input["width"].as_u64().expect("width") as usize,
        input["height"].as_u64().expect("height") as usize,
    );
    let cfa = Cfa::from_name(input["cfa"].as_str().expect("cfa")).expect("known pattern");
    let mut mosaic = Mosaic::from_data(w, h, cfa, floats(&input["data"])).expect("mosaic");
    let g = doubles(&params["gains"]);
    let n = &params["noise"];
    let noise = NoiseDn::from_model(&NoiseModel {
        a: num(n, "a"),
        b: num(n, "b"),
        delta: [0.0; 4],
        sigma_row: num(n, "sigma_row"),
        q: num(n, "q"),
    });
    let shadow_match = match params["shadow_match"].as_str().expect("shadow_match") {
        "auto" => ShadowMatch::Auto,
        "on" => ShadowMatch::On,
        "off" => ShadowMatch::Off,
        other => panic!("unknown shadow_match {other}"),
    };
    let run_params = Params {
        detail: num(params, "detail") as f32,
        strength: num(params, "strength") as f32,
        shadow_match,
        shadow_sharpening: params["shadow_sharpening"].as_bool().expect("bool"),
        tuning: match regime {
            Regime::Shipped => adaptive_ratio::Tuning::DEFAULT,
            Regime::Oracle => adaptive_ratio::Tuning::ORACLE,
        },
    };
    let curves = (!params["ca"].is_null()).then(|| {
        let ca = &params["ca"];
        let knots = &ca["lens_knots"];
        mosaic.place = Placement {
            x0: ca["origin_x"].as_u64().expect("origin_x") as usize,
            y0: ca["origin_y"].as_u64().expect("origin_y") as usize,
            frame_width: knots["frame_width"].as_u64().expect("frame width") as usize,
            frame_height: knots["frame_height"].as_u64().expect("frame height") as usize,
        };
        let curve = |c: &Value| {
            RadialCurve::new(doubles(&c["radius_px"]), doubles(&c["shift_px"])).expect("curve")
        };
        CaCurves::new(
            num(ca, "centre_x"),
            num(ca, "centre_y"),
            curve(&ca["red"]),
            curve(&ca["blue"]),
        )
        .expect("curves")
    });
    let gains = [g[0] as f32, g[1] as f32, g[2] as f32];
    let inputs = DemosaicInputs {
        gains,
        noise,
        ca: curves.as_ref(),
    };
    let out = adaptive_ratio::demosaic_whole(&mosaic, &inputs, &run_params).expect("demosaic");

    // The f64 constructor reproduces the oracle's derivation; the f32 one is what the demosaic runs.
    let d = Derived::with_f64_gains([g[0], g[1], g[2]], noise, &run_params).expect("derived");
    let mut doc = oracle.clone();
    let derived = object_mut(
        object_mut(&mut doc["params"])
            .get_mut("derived")
            .expect("derived block"),
    );
    assert_fields(
        "adaptive ratio derived",
        derived,
        &[
            "rho_r",
            "rho_b",
            "g_dn",
            "r_dn",
            "K",
            "s",
            "T",
            "k",
            "snr18_db",
            "shadow_match",
            "halo",
        ],
        &[],
    );
    set(derived, "rho_r", f64v(d.rho_r));
    set(derived, "rho_b", f64v(d.rho_b));
    set(derived, "g_dn", f64v(d.g_dn));
    set(derived, "r_dn", f64v(d.r_dn));
    set(derived, "K", f64v(d.offset1_dn));
    set(derived, "s", f64v(d.snr_target));
    set(derived, "T", f64v(d.gate_dn));
    set(derived, "k", f64v(d.offset3_dn));
    set(derived, "snr18_db", f64v(d.snr18_db));
    set(derived, "shadow_match", Value::Bool(d.shadow_match));
    set(derived, "halo", uint(d.halo_px));
    set_planar(&mut doc["expected"], &out);
    mark_self(&mut doc);
    doc
}

/// Replaces the `data` of an `expected` block with `img`'s planes, after checking its shape.
fn set_planar(expected: &mut Value, img: &RgbImage) {
    assert_eq!(
        (
            num(expected, "width") as usize,
            num(expected, "height") as usize
        ),
        (img.width, img.height),
        "the output has the size the oracle's file states"
    );
    set(object_mut(expected), "data", f32s(&img.data));
}

// --- spec 02 -----------------------------------------------------------------------------------

fn log_guided_vector(oracle: &Value, regime: Regime) -> Value {
    let params = &oracle["params"];
    let n = &params["noise"];
    let mut model = NoiseModel::new(num(n, "a"), num(n, "b"));
    model.sigma_row = num(n, "sigma_row");
    model.q = num(n, "q");
    let noise = NoiseDn::from_model(&model);
    let input = &oracle["input"];
    let (w, h) = (
        input["width"].as_u64().expect("width") as usize,
        input["height"].as_u64().expect("height") as usize,
    );
    let cfa = Cfa::from_name(input["cfa"].as_str().expect("cfa")).expect("cfa name");
    let mosaic = Mosaic::from_data(w, h, cfa, floats(&input["data"])).expect("mosaic");
    let detail = params["guide_detail"].as_i64().expect("guide_detail") as i32;
    let headroom = params["headroom_log2"].as_u64().expect("headroom_log2") as u8;

    // The fixed form under the shipped cubics; the float form has none.
    let cubics = match regime {
        Regime::Shipped => log_guided::FixedCubics::SHIPPED,
        Regime::Oracle => log_guided::ORACLE_CUBICS,
    };
    let fixed = log_guided::demosaic_fixed_with(&mosaic, noise, detail, headroom, cubics);
    let float = log_guided::demosaic_float(&mosaic, noise, detail);
    let d = log_guided::derive(noise, detail, headroom);
    let df = log_guided::derive(noise, detail, 0);

    let mut doc = oracle.clone();
    let derived = object_mut(
        object_mut(&mut doc["params"])
            .get_mut("derived")
            .expect("derived block"),
    );
    assert_fields(
        "log-guided derived",
        derived,
        &[
            "headroom",
            "read_noise_dn",
            "p",
            "p_float_form",
            "k",
            "kernel",
        ],
        &["kernel_divisor"],
    );
    set(derived, "headroom", f64v(d.headroom));
    set(derived, "read_noise_dn", f64v(d.read_noise_dn));
    set(derived, "p", uint(d.p as usize));
    set(derived, "p_float_form", uint(df.p as usize));
    set(derived, "k", Value::Number(Number::from(i64::from(d.k))));
    set(
        derived,
        "kernel",
        Value::Array(
            d.kernel
                .iter()
                .map(|&k| Value::Number(Number::from(i64::from(k))))
                .collect(),
        ),
    );
    // `kernel_divisor` is structure (1024), not a tuned value: kept as the oracle states it.
    set_planar(&mut doc["expected"], &fixed);
    set_planar(&mut doc["expected_float"], &float);
    mark_self(&mut doc);
    doc
}

// --- spec 03 -----------------------------------------------------------------------------------

fn diffraction_plan(params: &Value, regime: Regime) -> Option<DiffractionPlan> {
    let g0 = floats(&params["g0"]);
    let luma = floats(&params["luma"]);
    let n = &params["noise"];
    let model = NoiseModel {
        a: num(n, "a"),
        b: num(n, "b"),
        delta: [0.0; 4],
        sigma_row: num(n, "sigma_row"),
        q: num(n, "q"),
    };
    assert_eq!(num(params, "working_scale"), 65535.0);
    let p = DiffractionParams {
        pixel_pitch_um: [
            num(params, "pixel_pitch_x_um"),
            num(params, "pixel_pitch_y_um"),
        ],
        f_number: num(params, "f_number"),
        scale: num(params, "scale"),
        strength: num(params, "strength") as f32,
        tuning: match regime {
            Regime::Shipped => &DiffTuning::DEFAULT,
            Regime::Oracle => &DiffTuning::ORACLE,
        },
    };
    DiffractionPlan::new(
        &p,
        [g0[0], g0[1], g0[2]],
        &NoiseDn::from_model(&model),
        [luma[0], luma[1], luma[2]],
    )
}

fn diffraction_vector(oracle: &Value, regime: Regime) -> Value {
    let plan = diffraction_plan(&oracle["params"], regime);
    let image = (!oracle["input"].is_null()).then(|| {
        let input = &oracle["input"];
        let (w, h) = (num(input, "width") as usize, num(input, "height") as usize);
        let img = RgbImage::from_planar(w, h, floats(&input["data"])).expect("input image");
        plan.as_ref()
            .map_or_else(|| img.clone(), |p| p.apply_whole(&img))
    });
    let mut doc = oracle.clone();
    let expected = object_mut(&mut doc["expected"]);
    let on = plan.as_ref();
    let known: [(&str, Option<Value>); 11] = [
        ("active", Some(uint(usize::from(on.is_some())))),
        ("radius", Some(uint(on.map_or(0, DiffractionPlan::radius)))),
        (
            "gauss_sigma",
            Some(f32v(on.map_or(0.0, DiffractionPlan::gauss_sigma))),
        ),
        (
            "gauss_radius",
            Some(uint(on.map_or(0, DiffractionPlan::gauss_radius))),
        ),
        (
            "halo_px",
            Some(uint(on.map_or(0, DiffractionPlan::halo_px))),
        ),
        (
            "sigma0_dn",
            Some(f64v(on.map_or(0.0, DiffractionPlan::sigma0_dn))),
        ),
        (
            "sigma1_dn",
            Some(f64v(on.map_or(0.0, DiffractionPlan::sigma1_dn))),
        ),
        (
            "r80_um",
            Some(f64v(on.map_or(0.0, DiffractionPlan::r80_um))),
        ),
        (
            "support_um",
            Some(f64v(on.map_or(0.0, DiffractionPlan::support_um))),
        ),
        (
            "kernel",
            Some(f32s(on.map_or(&[][..], DiffractionPlan::kernel))),
        ),
        ("data", image.as_ref().map(|img| f32s(&img.data))),
    ];
    for (key, value) in known {
        if !expected.contains_key(key) {
            continue;
        }
        let value = value.unwrap_or_else(|| panic!("{key} needs an image"));
        set(expected, key, value);
    }
    // `width`, `height` and `layout` are the input's, unchanged; any other field is unknown here.
    for key in expected.keys() {
        assert!(
            [
                "active",
                "radius",
                "gauss_sigma",
                "gauss_radius",
                "halo_px",
                "sigma0_dn",
                "sigma1_dn",
                "r80_um",
                "support_um",
                "kernel",
                "data",
                "width",
                "height",
                "layout",
            ]
            .contains(&key.as_str()),
            "the generator does not know the field {key}"
        );
    }
    mark_self(&mut doc);
    doc
}

// --- spec 04 -----------------------------------------------------------------------------------

fn f32_of(v: &Value, key: &str) -> f32 {
    num(v, key) as f32
}

fn sharpen_method(v: &Value) -> Method {
    Method::from_id(v["method_id"].as_u64().expect("method_id") as u32).expect("method")
}

fn method_name(m: Method) -> &'static str {
    match m {
        Method::HardThreshold => "hard_threshold",
        Method::SmoothGain => "smooth_gain",
        Method::GradientRamp => "gradient_ramp",
    }
}

fn output_sharpen_of(params: &Value, regime: Regime) -> (OutputSharpen, f32) {
    let mode = match params["mode"].as_str().expect("mode") {
        "screen" => Mode::Screen,
        "print" => Mode::Print,
        other => panic!("mode {other}"),
    };
    let lw = params["luma_weights"].as_array().expect("luma weights");
    let extra_stages = params["extra_stages"]
        .as_array()
        .expect("extra stages")
        .iter()
        .map(|e| ExtraStage {
            method: sharpen_method(e),
            auto_method: e["auto_method"].as_bool().expect("auto_method"),
            amount_pct: f32_of(e, "amount_pct"),
            radius_px: f32_of(e, "radius_px"),
            threshold_levels: f32_of(e, "threshold_levels"),
            halo_pct: f32_of(e, "halo_pct"),
        })
        .collect();
    (
        OutputSharpen {
            mode,
            method: sharpen_method(params),
            amount_pct: f32_of(params, "amount_pct"),
            radius_px: f32_of(params, "radius_px"),
            threshold_levels: f32_of(params, "threshold_levels"),
            halo_pct: f32_of(params, "halo_pct"),
            print_distance: f32_of(params, "print_distance"),
            luma_weights: [0, 1, 2].map(|i| lw[i].as_f64().expect("weight") as f32),
            extra_stages,
            // The reference compares exactly; the tie tolerance is ours (spec 7.1 item 5).
            halo_tie_tolerance: 0.0,
            tuning: match regime {
                Regime::Shipped => ym_gpu::cpu::output_sharpen::Tuning::DEFAULT,
                Regime::Oracle => ym_gpu::cpu::output_sharpen::Tuning::ORACLE,
            },
        },
        f32_of(params, "scale"),
    )
}

fn output_sharpening_vector(oracle: &Value, regime: Regime) -> Value {
    let (os, scale) = output_sharpen_of(&oracle["params"], regime);
    let plan = os.plan(scale).expect("plan");
    let input = &oracle["input"];
    let img = RgbImage::from_planar(
        num(input, "width") as usize,
        num(input, "height") as usize,
        floats(&input["data"]),
    )
    .expect("input image");
    let out = os.apply(&img, scale).expect("apply");

    let stages: Vec<Value> = plan
        .stages()
        .iter()
        .enumerate()
        .map(|(i, st)| {
            let mut m = Map::new();
            m.insert("output".into(), Value::Bool(st.output));
            m.insert(
                "method".into(),
                Value::String(method_name(st.method).into()),
            );
            m.insert("method_id".into(), uint(usize::from(st.method.id())));
            m.insert("amount".into(), f32v(st.amount));
            m.insert("sigma".into(), f32v(st.sigma));
            m.insert("threshold".into(), f32v(st.threshold));
            m.insert("halo".into(), f32v(st.halo));
            m.insert("kernel".into(), f64s(&plan.kernel(i)));
            m.insert("gain_kernel".into(), f64s(&plan.gain_kernel(i)));
            Value::Object(m)
        })
        .collect();
    let mut doc = oracle.clone();
    let derived = object_mut(
        object_mut(&mut doc["params"])
            .get_mut("derived")
            .expect("derived block"),
    );
    assert_fields(
        "output sharpening derived",
        derived,
        &[
            "max_taps",
            "halo_enabled",
            "halo_px",
            "halo_kernel",
            "stages",
        ],
        &[],
    );
    set(derived, "max_taps", uint(plan.max_taps()));
    set(derived, "halo_enabled", Value::Bool(plan.halo_enabled()));
    set(derived, "halo_px", uint(plan.halo_px()));
    set(derived, "halo_kernel", f64s(&plan.halo_kernel()));
    set(derived, "stages", Value::Array(stages));
    set_planar(&mut doc["expected"], &out);
    mark_self(&mut doc);
    doc
}
