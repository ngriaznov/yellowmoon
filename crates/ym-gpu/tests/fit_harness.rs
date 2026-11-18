//! The fit harness of `docs/29-raw-constants.md` (task RC-05): the overrides by field name, the data note,
//! the CA scenes as a library, and the three examples `fit_demosaic`, `fit_diffraction` and `fit_sharpen`
//! on tiny scenes. The examples are compiled into this test as modules, so what runs here is the code of
//! the examples, not a copy.

// The examples are compiled into this test as modules, and each loads the shared files of tests/common
// itself, so the same file is a module of this crate several times.
#![allow(clippy::float_cmp, clippy::duplicate_mod)]

use ym_gpu::cpu::demosaic::adaptive_ratio;
use ym_gpu::cpu::output_sharpen::{self, OutputSharpen};
use ym_gpu::cpu::sharpen::diffraction;
use ym_gpu::cpu::{RgbImage, synth};

#[path = "common/ca_scenes.rs"]
mod ca_scenes;
#[path = "common/demosaic_metrics.rs"]
mod demosaic_metrics;
#[path = "common/fit.rs"]
mod fit;
#[path = "../examples/fit_demosaic.rs"]
#[allow(dead_code, unused_imports)]
mod fit_demosaic_ex;
#[path = "../examples/fit_diffraction.rs"]
#[allow(dead_code, unused_imports)]
mod fit_diffraction_ex;
#[path = "../examples/fit_sharpen.rs"]
#[allow(dead_code, unused_imports)]
mod fit_sharpen_ex;

use fit::{Common, Evaluation, Fields};

fn args(list: &[&str]) -> Vec<String> {
    list.iter().map(|s| (*s).to_owned()).collect()
}

/// The names of the fields in the `Debug` text of a struct (`Tuning { a: 1.0, b: 2 }`).
fn debug_field_names<T: std::fmt::Debug>(t: &T) -> Vec<String> {
    let text = format!("{t:?}");
    let inner = &text[text.find('{').expect("a struct") + 1..text.rfind('}').expect("a struct")];
    inner
        .split(", ")
        .map(|f| f.split(':').next().expect("name").trim().to_owned())
        .collect()
}

fn check_fields<T: Fields + std::fmt::Debug + PartialEq>(base: &T) {
    // Every field of the struct is named, and nothing else is.
    let mut named: Vec<&str> = T::NAMES.to_vec();
    let mut real = debug_field_names(base);
    named.sort_unstable();
    real.sort();
    assert_eq!(
        named, real,
        "Fields::NAMES must list every field of the struct"
    );
    // Reading a field and writing it back changes nothing.
    for name in T::NAMES {
        let mut t = *base;
        let v = t.get(name).expect("a field reads");
        t.set(name, v).expect("a field takes its own value");
        assert_eq!(&t, base, "{name} does not round-trip");
    }
}

#[test]
fn every_field_of_every_tuning_is_named_and_round_trips() {
    check_fields(&adaptive_ratio::Tuning::ORACLE);
    check_fields(&diffraction::Tuning::ORACLE);
    check_fields(&output_sharpen::Tuning::ORACLE);
}

fn refuses<T: Fields + std::fmt::Debug>(base: &T, name: &str, v: f64) -> String {
    Evaluation {
        label: "x".to_owned(),
        sets: vec![(name.to_owned(), v)],
    }
    .apply(base)
    .expect_err("must be refused")
}

#[test]
fn an_unknown_field_name_is_refused_and_the_message_lists_the_fields() {
    let e = refuses(&adaptive_ratio::Tuning::ORACLE, "no_such_field", 1.0);
    assert!(e.contains("unknown Tuning field `no_such_field`"), "{e}");
    assert!(
        e.contains("sat_dominance") && e.contains("second_diff_weight"),
        "{e}"
    );
    let e = refuses(&diffraction::Tuning::ORACLE, "sat_dominance", 1.0);
    assert!(
        e.contains("unknown Tuning field") && e.contains("sigma_factor"),
        "{e}"
    );
    let e = refuses(&output_sharpen::Tuning::ORACLE, "sigma_factor", 1.0);
    assert!(
        e.contains("unknown Tuning field") && e.contains("sigma_slope"),
        "{e}"
    );
}

#[test]
fn a_value_a_field_cannot_hold_is_refused() {
    let d = diffraction::Tuning::ORACLE;
    assert!(refuses(&d, "line_angles", 2.5).contains("whole number"));
    assert!(refuses(&d, "line_angles", -1.0).contains("whole number"));
    assert!(refuses(&d, "lambda_min_nm", 1e12).contains("whole number"));
    assert!(refuses(&output_sharpen::Tuning::ORACLE, "taps_min", 3.5).contains("whole number"));
    let a = adaptive_ratio::Tuning::ORACLE;
    assert!(refuses(&a, "sat_dominance", f64::NAN).contains("not a finite number"));
    assert!(refuses(&a, "sat_dominance", f64::INFINITY).contains("not a finite number"));
    assert!(refuses(&output_sharpen::Tuning::ORACLE, "sigma_slope", 1e300).contains("f32"));
    // A good value goes through, an integer field takes a whole number.
    let mut d = d;
    d.set("line_angles", 12.0).expect("whole number");
    assert_eq!(d.line_angles, 12);
}

#[test]
fn the_command_line_reads_set_rel_and_eval() {
    let a = args(&[
        "--from",
        "default",
        "--set",
        "sat_dominance=0.8",
        "--set",
        "chroma_floor=0.05",
        "--rel",
        "second_diff_weight=0.9,1.1",
        "--eval",
        "mine@shadow_amount=0.5,shadow_laplacian_scale=4",
        "--held",
        "--photo",
        "x.ARW",
    ]);
    let mut c = Common::default();
    let mut left = Vec::new();
    let mut i = 0;
    while i < a.len() {
        if !c.take(&a, &mut i).expect("well formed") {
            left.push(a[i].clone());
        }
        i += 1;
    }
    assert_eq!(left, ["--photo", "x.ARW"], "what the example reads itself");
    assert!(c.held && !c.list);
    let base = adaptive_ratio::Tuning::ORACLE;
    let evals = c.evaluations(&base).expect("known fields");
    let labels: Vec<&str> = evals.iter().map(|(e, _)| e.label.as_str()).collect();
    assert_eq!(
        labels,
        [
            "second_diff_weight x0.9",
            "second_diff_weight x1.1",
            "mine",
            "set"
        ]
    );
    // 1.6 x 0.9 is the 1.44 a person types, not 1.4400000000000002.
    assert_eq!(evals[0].1.second_diff_weight, 1.44);
    assert_eq!(evals[1].1.second_diff_weight, 1.76);
    assert_eq!(evals[2].1.shadow_amount, 0.5);
    assert_eq!(evals[2].1.shadow_laplacian_scale, 4.0);
    assert_eq!(
        (evals[3].1.sat_dominance, evals[3].1.chroma_floor),
        (0.8, 0.05)
    );
    // Nothing else moved.
    assert_eq!(evals[3].1.second_diff_weight, base.second_diff_weight);
}

#[test]
fn a_relative_value_of_an_f32_field_is_the_decimal_a_person_types() {
    // 0.7 is an f32; 0.7 x 0.9 must give the f32 of 0.63, which is what the audit's tables used.
    let mut c = Common::default();
    let a = args(&["--rel", "sigma_slope=0.9"]);
    let mut i = 0;
    assert!(c.take(&a, &mut i).expect("well formed"));
    let t = c
        .evaluations(&output_sharpen::Tuning::ORACLE)
        .expect("known")[0]
        .1;
    assert_eq!(t.sigma_slope, 0.63f32);
}

#[test]
fn bad_arguments_are_refused_before_anything_runs() {
    let mut c = Common::default();
    for bad in [
        &["--set", "no_equals_sign"][..],
        &["--set", "sat_dominance=abc"],
        &["--rel", "sat_dominance"],
        &["--rel", "sat_dominance=x"],
        &["--eval", "no-at-sign"],
        &["--from", "elsewhere"],
        &["--set"],
    ] {
        let a = args(bad);
        let mut i = 0;
        assert!(c.take(&a, &mut i).is_err(), "{bad:?}");
    }
    // An unknown field is found when the evaluations are built, not after the scenes have loaded.
    let a = args(&["--set", "no_such_field=1"]);
    let mut i = 0;
    assert!(c.take(&a, &mut i).expect("well formed"));
    let e = c
        .evaluations(&adaptive_ratio::Tuning::ORACLE)
        .expect_err("unknown field");
    assert!(e.contains("unknown Tuning field"), "{e}");
}

#[test]
fn the_report_has_a_data_note() {
    // docs/29 F-07: every report says which data it used, which it skipped, that nothing is committed
    // and that the noise models are stand-ins.
    let note = fit::data_note(
        "fit_demosaic",
        &[("kodak".to_owned(), 24)],
        &[("pixelshift200".to_owned(), "not present".to_owned())],
    );
    assert_eq!(note["kind"], "data_note");
    let text = note["note"].as_str().expect("a note");
    assert!(text.contains("never committed"));
    assert!(text.contains("stand-ins") && text.contains("no real-sensor ground truth"));
    assert_eq!(note["used"][0]["set"], "kodak");
    assert_eq!(note["skipped"][0]["set"], "pixelshift200");
}

// ---------------------------------------------------------------------------------------------
// fit_demosaic
// ---------------------------------------------------------------------------------------------

#[test]
fn fit_demosaic_runs_on_a_tiny_scene_and_a_moved_constant_moves_it() {
    use fit_demosaic_ex::{Group, Source, clean_scene, dark_noisy_scene, psnr_means, run_scene};
    let truth = synth::natural(48, 48, 1);
    let scenes = vec![
        clean_scene(Group::A, Source::Synthetic, &truth),
        clean_scene(Group::A, Source::McMaster, &truth),
        dark_noisy_scene(Group::B, Source::Synthetic, &truth, 77),
        dark_noisy_scene(Group::B2, Source::Synthetic, &truth, 1077),
    ];
    let base = adaptive_ratio::Tuning::ORACLE;
    let run = |t: &adaptive_ratio::Tuning| -> Vec<RgbImage> {
        scenes.iter().map(|s| run_scene(s, t)).collect()
    };
    let (a, b) = (run(&base), run(&base));
    assert_eq!(a, b, "deterministic");
    let mut moved = base;
    moved.set("sat_dominance", 1.2).expect("known field");
    moved.set("shadow_amount", 1.5).expect("known field");
    let c = run(&moved);
    assert_ne!(a[0], c[0], "the clean scene follows the dominance");
    assert_ne!(a[2], c[2], "the dark scene follows the shadow amount");
    let means = psnr_means(&scenes, &a);
    let get = |k: &str| means.iter().find(|(n, _)| *n == k).expect("key").1;
    assert!(get("A").is_finite() && get("A_fit").is_finite() && get("A_held").is_finite());
    assert!(get("B").is_finite() && get("B2").is_finite());
    assert_eq!(
        get("A_fit"),
        get("A_held"),
        "the same picture in both splits"
    );
    assert!(get("P").is_nan(), "no PixelShift200 scene in this run");
}

#[test]
fn fit_demosaic_skips_a_missing_set_and_reads_pixelshift200_when_it_is_there() {
    use fit_demosaic_ex::{Group, load_scenes};
    let root = std::env::temp_dir().join(format!("ym-fit-harness-{}", std::process::id()));
    std::fs::create_dir_all(&root).expect("temp folder");
    let none = load_scenes(&root, false, &[], false);
    let skipped: Vec<&str> = none.skipped.iter().map(|(n, _)| n.as_str()).collect();
    assert!(skipped.contains(&"pixelshift200"), "{skipped:?}");
    assert!(skipped.contains(&"kodak") && skipped.contains(&"mcmaster"));
    assert!(none.used.iter().all(|(n, _)| !n.starts_with("pixelshift")));
    // The procedural set is always there (7 images), and every third one is also a B scene.
    assert_eq!(none.scenes.len(), 7 + 3);
    // `--cells` adds the A images at the two other signal scales of the gate's nine cells (RC-07).
    let cells = load_scenes(&root, false, &[], true);
    assert_eq!(cells.scenes.len(), 7 + 7 * 2 + 3);
    let scales: Vec<f64> = cells
        .scenes
        .iter()
        .filter(|s| matches!(s.group, Group::Cell01 | Group::Cell001))
        .map(|s| s.scale)
        .collect();
    assert_eq!(scales.len(), 14);
    assert!(
        scales
            .iter()
            .all(|&v| (v - 0.1).abs() < 1e-6 || (v - 0.01).abs() < 1e-6)
    );
    // A folder with one image.
    let ps = root.join("pixelshift200");
    std::fs::create_dir_all(&ps).expect("folder");
    let file = std::fs::File::create(ps.join("one.png")).expect("png");
    let mut enc = png::Encoder::new(std::io::BufWriter::new(file), 16, 16);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let pixels: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 251) as u8).collect();
    enc.write_header()
        .expect("header")
        .write_image_data(&pixels)
        .expect("data");
    let some = load_scenes(&root, true, &[], false);
    assert!(
        some.used
            .iter()
            .any(|(n, k)| n == "pixelshift200" && *k == 1)
    );
    assert!(some.skipped.iter().all(|(n, _)| n != "pixelshift200"));
    assert_eq!(some.scenes.len(), 7 + 1 + 3 * 2, "A, P, B and B2");
    // A missing photograph is reported, not fatal.
    let bad = load_scenes(&root, false, &[root.join("missing.ARW")], false);
    assert!(bad.skipped.iter().any(|(n, _)| n.starts_with("photo ")));
    std::fs::remove_dir_all(&root).expect("clean up the temp folder");
}

#[test]
fn the_ca_scenes_are_a_library_and_take_the_tuning() {
    use ca_scenes::{ca_noise_model, ca_path_result, ca_scene_set};
    let set = ca_scene_set((240, 160), 0.0, 2024);
    assert_eq!(set.len(), 5);
    let noise = ym_gpu::cpu::NoiseDn::from_model(&ca_noise_model(0.0));
    let base = adaptive_ratio::Tuning::ORACLE;
    let (r1, _) = ca_path_result(&set[1], noise, &base);
    let (r2, _) = ca_path_result(&set[1], noise, &base);
    assert_eq!(r1.psnr_rgb_db, r2.psnr_rgb_db, "deterministic");
    assert!(r1.psnr_rgb_db.is_finite() && r1.red_rms_px.is_finite());
    let mut moved = base;
    moved.set("second_diff_weight", 10.0).expect("known field");
    let (r3, _) = ca_path_result(&set[1], noise, &moved);
    assert_ne!(
        r1.psnr_rgb_db, r3.psnr_rgb_db,
        "the wiring check of docs/29: A1 reaches the CA path"
    );
}

// ---------------------------------------------------------------------------------------------
// fit_diffraction
// ---------------------------------------------------------------------------------------------

#[test]
fn fit_diffraction_evaluates_four_cases_and_a_moved_constant_changes_them() {
    use fit_diffraction_ex::{build_scenes, deltas, run};
    let sc = build_scenes(false);
    assert_eq!(sc.cases.len(), 4);
    let base_t = diffraction::Tuning::ORACLE;
    let base = run(&sc, &base_t);
    let again = run(&sc, &base_t);
    assert_eq!(base.len(), again.len());
    for ((r, o), (r2, o2)) in base.iter().zip(&again) {
        assert_eq!((r.text, r.mtf, r.noise), (r2.text, r2.mtf, r2.noise));
        assert_eq!(o, o2);
    }
    // The base against itself is exactly zero.
    let zero = deltas(&sc, &base, &base, false).expect("fit cases");
    assert_eq!(zero["text_psnr_db"], 0.0);
    assert_eq!(zero["max_dout"], 0.0);
    // sigma_factor 0.75 -> 0.78 (the exact factor for a Gaussian) moves the output.
    let mut moved = base_t;
    moved.set("sigma_factor", 0.78).expect("known field");
    let res = run(&sc, &moved);
    let d = deltas(&sc, &res, &base, false).expect("fit cases");
    assert!(d["max_dout"].as_f64().expect("number") > 0.0);
    assert_ne!(d["mtf50_cpp"], 0.0);
    assert!(
        deltas(&sc, &res, &base, true).is_none(),
        "no held-out case without --held"
    );
    // --held adds the same four with a second noise seed.
    let held = build_scenes(true);
    assert_eq!(held.cases.len(), 8);
    assert!(held.cases[4..].iter().all(|c| c.held));
}

// ---------------------------------------------------------------------------------------------
// fit_sharpen
// ---------------------------------------------------------------------------------------------

#[test]
fn fit_sharpen_screen_preset_of_a_tuning_is_the_screen_preset() {
    use fit_sharpen_ex::screen_with;
    assert_eq!(
        screen_with(&output_sharpen::Tuning::DEFAULT),
        OutputSharpen::screen()
    );
    let mut t = output_sharpen::Tuning::ORACLE;
    t.set("screen_radius_px", 1.5).expect("known field");
    t.set("screen_amount_pct", 80.0).expect("known field");
    let o = screen_with(&t);
    assert_eq!((o.radius_px, o.amount_pct), (1.5, 80.0));
    assert_eq!(o.tuning, t);
}

#[test]
fn fit_sharpen_runs_on_tiny_images_and_the_edge_profile_follows_the_sigma_map() {
    use fit_sharpen_ex::{edge_image, edge_numbers, run};
    let images = vec![synth::natural(64, 64, 1), synth::edges(64, 64)];
    let base_t = output_sharpen::Tuning::ORACLE;
    let a = run(&images, &base_t);
    assert_eq!(a.len(), 4, "two images at two scales");
    assert_eq!(a, run(&images, &base_t), "deterministic");
    let mut moved = base_t;
    moved.set("sigma_slope", 0.9).expect("known field");
    assert_ne!(a, run(&images, &moved));
    let edge = edge_image();
    let e0 = edge_numbers(&edge, &base_t);
    let e1 = edge_numbers(&edge, &moved);
    for scale in ["1", "0.4"] {
        for key in [
            "rise_10_90_px",
            "overshoot_pct",
            "undershoot_pct",
            "noise_gain",
        ] {
            assert!(e0[scale][key].is_number(), "{scale} {key}");
        }
    }
    assert_ne!(e0["1"]["rise_10_90_px"], e1["1"]["rise_10_90_px"]);
    assert!(
        e0["1"]["noise_gain"].as_f64().expect("number") > 1.0,
        "sharpening adds noise"
    );
}

// ---------------------------------------------------------------------------------------------
// The driver
// ---------------------------------------------------------------------------------------------

#[test]
fn the_search_driver_passes_its_own_tests() {
    let script =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tools/fit/test_search.py");
    let Ok(python) = std::process::Command::new("python3")
        .arg("--version")
        .output()
    else {
        eprintln!("python3 is not installed: skipping the driver's tests");
        return;
    };
    assert!(python.status.success());
    let out = std::process::Command::new("python3")
        .arg("-I")
        .arg(&script)
        .output()
        .expect("run the driver's tests");
    assert!(
        out.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
