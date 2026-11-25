//! The EoMT small runtime against the original model (`docs/34` sections 4b.1 and 10, RG-03).
//!
//! The fixture is `tests/data/eomt/` (committed, 260 KB): three small pictures made procedurally (a landscape, a street with a
//! person, a shore; 3:2, 2:3 and square) and what Hugging Face's `EomtForUniversalSegmentation` and `EomtImageProcessor` give for
//! them on the CPU in f32 with the f16 weights widened (`tools/eomt/parity_fixture.py`, test tooling: nothing of it runs in the app
//! or in `ym`). The weights are `data/models/eomt/` (`tools/fetch/eomt.sh`). Without them the gates that need the network print a plain
//! "SKIPPED" line and pass; `YM_REQUIRE_EOMT=1` makes a missing file a failure (CI sets it once the release exists). The class
//! table's test needs only the fixture and always runs.
//!
//! `YM_EOMT_EXTRA_FIXTURE=/path` adds the fixture folder a `--photos` run of the script wrote (real photographs; for a local check,
//! nothing of it is committed).
//!
//! The comparison is whole-path: our resize and padding, our network, our post-processing, against the original's. What differs is
//! (a) the 8-bit resize, equal to the original's on all but about 0.1 % of the samples, which are one level off, and (b) float
//! rounding in 12 blocks of attention. Every gate says which quantity it bounds and the worst value measured when the bound was set.
//!
//! The network is large: run these with `cargo test -p ym-ml --release --test eomt` (a dev build is optimised too, but slower).

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use serde_json::Value;
use ym_ml::eomt::{
    CLASS_NAMES, CLASSES, Eomt, Raw, UNLABELLED, WEIGHTS_FILE, element_of, input_tensor,
    kept_queries, synthetic_weights,
};

// -------------------------------------------------------------------------------------------------
// The gates
// -------------------------------------------------------------------------------------------------
//
// Bounds set from the first runs: the worst value over the three committed pictures and the eleven real photographs and film stills of
// the local check (`YM_EOMT_EXTRA_FIXTURE`), on the Linux machine, 2026-11-25, and the bound is 2 to 4 times it. The proposal made before
// measuring was "the element masks equal on at least 99 % of the pixels" (`docs/34` section 10); what was measured is far better, so
// the bound is 99.9 %. A gate that fails is reported, not loosened.

/// Samples of the model input that differ from the original's by more than one level of 8 bits: none. (Worst measured: 1 level.)
const INPUT_MAX_LEVELS: i64 = 1;
/// The share of the input samples that differ at all (by one level). Worst measured: 0.35 % (a film still), 0 % on the committed set.
const INPUT_MAX_DIFFERING: f64 = 0.01;
/// The class logits of the sampled queries: the largest absolute difference. Worst measured: 6.95e-2 (a film still); 6.1e-3 on the
/// committed set.
const CLASS_LOGIT_MAX_ABS: f64 = 0.15;
/// The class logits: RMS of the difference over the RMS of the original's. Worst measured: 1.8e-3.
const CLASS_LOGIT_REL_RMS: f64 = 4e-3;
/// The mask logits at the sampled points: the largest absolute difference. Worst measured: 0.56 (a film still, logits of order 10);
/// 0.018 on the committed set.
const MASK_LOGIT_MAX_ABS: f64 = 1.2;
/// The mask logits: RMS of the difference over the RMS of the original's. Worst measured: 1.7e-3.
const MASK_LOGIT_REL_RMS: f64 = 4e-3;
/// The probability of each kept query: the largest absolute difference. Worst measured: 3.2e-3.
const SCORE_MAX_ABS: f64 = 8e-3;
/// The label map: the share of the pixels with the same class as the original's post-processing. Worst measured: 99.974 %.
const CLASS_AGREEMENT_MIN: f64 = 0.999;
/// The element map (the class table of `docs/34` section 4b.2 applied to both): the share of the pixels with the same element. Worst
/// measured: 99.974 % (the classes that merge into one element can only raise it).
const ELEMENT_AGREEMENT_MIN: f64 = 0.999;
/// The picture whose model input is the original's to the bit (a 192 x 192 square scaled up: the 8-bit resize is exact there) gives
/// the **network's own parity**: class logits within this (measured 2.5e-5), mask logits within the next (measured 1.0e-4), and the
/// label map equal on every pixel.
const EXACT_INPUT_CLASS_MAX_ABS: f64 = 5e-4;
const EXACT_INPUT_MASK_MAX_ABS: f64 = 2e-3;

// -------------------------------------------------------------------------------------------------
// The fixture
// -------------------------------------------------------------------------------------------------

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn weights_dir() -> PathBuf {
    repo().join("data/models/eomt")
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/eomt")
}

fn require() -> bool {
    std::env::var_os("YM_REQUIRE_EOMT").is_some_and(|v| v != "0" && !v.is_empty())
}

struct Picture {
    name: String,
    width: usize,
    height: usize,
    rgb: Vec<u8>,
    fit: (usize, usize),
    reference_classes: Vec<u8>,
    input_samples: Vec<i64>,
    queries: Vec<usize>,
    class_logits: Vec<Vec<f64>>,
    mask_samples: Vec<Vec<f64>>,
    kept: Vec<(usize, u8, f64)>,
}

struct Fixture {
    id2label: Vec<String>,
    strides: (usize, usize),
    mask: (usize, usize, usize),
    pictures: Vec<Picture>,
}

/// An 8-bit PNG: (width, height, channels, bytes).
fn read_png(path: &Path) -> (usize, usize, usize, Vec<u8>) {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .expect("png info");
    let mut buf = vec![0; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.bit_depth, png::BitDepth::Eight, "{}", path.display());
    let channels = info.color_type.samples();
    buf.truncate(info.width as usize * info.height as usize * channels);
    (info.width as usize, info.height as usize, channels, buf)
}

fn numbers(v: &Value) -> Vec<f64> {
    v.as_array()
        .expect("an array")
        .iter()
        .map(|x| x.as_f64().expect("a number"))
        .collect()
}

fn load_fixture(dir: &Path) -> Fixture {
    let text = std::fs::read_to_string(dir.join("parity.json"))
        .unwrap_or_else(|e| panic!("{}: {e}", dir.join("parity.json").display()));
    let doc: Value = serde_json::from_str(&text).expect("parity.json is JSON");
    let at = |key: &str| doc[key].as_u64().expect(key) as usize;
    let pictures = doc["images"]
        .as_array()
        .expect("images")
        .iter()
        .map(|i| {
            let (w, h, channels, bytes) = read_png(&dir.join(i["png"].as_str().expect("png")));
            assert_eq!(
                (w, h),
                (
                    i["width"].as_u64().unwrap() as usize,
                    i["height"].as_u64().unwrap() as usize
                )
            );
            let rgb = if channels == 3 {
                bytes
            } else {
                bytes
                    .chunks(channels)
                    .flat_map(|p| [p[0], p[0], p[0]])
                    .collect()
            };
            let (cw, ch, cc, classes) =
                read_png(&dir.join(i["classes_png"].as_str().expect("classes")));
            let fit = numbers(&i["fit"]);
            assert_eq!((ch, cw, cc), (fit[0] as usize, fit[1] as usize, 1));
            Picture {
                name: i["name"].as_str().expect("name").to_owned(),
                width: w,
                height: h,
                rgb,
                fit: (fit[0] as usize, fit[1] as usize),
                reference_classes: classes,
                input_samples: numbers(&i["input_samples"])
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
                queries: numbers(&i["queries"]).iter().map(|&v| v as usize).collect(),
                class_logits: i["class_logits"]
                    .as_array()
                    .expect("class logits")
                    .iter()
                    .map(numbers)
                    .collect(),
                mask_samples: i["mask_samples"]
                    .as_array()
                    .expect("mask samples")
                    .iter()
                    .map(numbers)
                    .collect(),
                kept: i["kept"]
                    .as_array()
                    .expect("kept")
                    .iter()
                    .map(|k| {
                        (
                            k["query"].as_u64().unwrap() as usize,
                            k["label"].as_u64().unwrap() as u8,
                            k["score"].as_f64().unwrap(),
                        )
                    })
                    .collect(),
            }
        })
        .collect();
    Fixture {
        id2label: doc["id2label"]
            .as_array()
            .expect("id2label")
            .iter()
            .map(|v| v.as_str().expect("a name").to_owned())
            .collect(),
        strides: (at("row_stride"), at("col_stride")),
        mask: (at("mask_offset"), at("mask_stride"), at("mask_count")),
        pictures,
    }
}

fn fixtures() -> &'static Vec<(String, Fixture)> {
    static FIXTURES: OnceLock<Vec<(String, Fixture)>> = OnceLock::new();
    FIXTURES.get_or_init(|| {
        let mut all = vec![("committed".to_owned(), load_fixture(&fixture_dir()))];
        if let Some(extra) = std::env::var_os("YM_EOMT_EXTRA_FIXTURE") {
            all.push(("extra".to_owned(), load_fixture(Path::new(&extra))));
        }
        all
    })
}

/// The model, or `None` after printing why not (or a panic when `YM_REQUIRE_EOMT` is set).
fn model() -> Option<&'static Eomt> {
    static MODEL: OnceLock<Option<Eomt>> = OnceLock::new();
    MODEL
        .get_or_init(|| {
            if !weights_dir().join(WEIGHTS_FILE).exists() {
                let msg = format!(
                    "SKIPPED: no EoMT weights in {} (run tools/fetch/eomt.sh)",
                    weights_dir().display()
                );
                assert!(!require(), "{msg}");
                println!("{msg}");
                return None;
            }
            Some(Eomt::load(&weights_dir()).expect("the weights load and match SHA256SUMS"))
        })
        .as_ref()
}

// -------------------------------------------------------------------------------------------------
// The measurements
// -------------------------------------------------------------------------------------------------

#[derive(Debug)]
struct Measured {
    set: String,
    name: String,
    input_max_levels: i64,
    input_differing: f64,
    class_max_abs: f64,
    class_rel_rms: f64,
    mask_max_abs: f64,
    mask_rel_rms: f64,
    score_max_abs: f64,
    same_kept: bool,
    class_agreement: f64,
    element_agreement: f64,
    kept: usize,
}

fn rel_rms(ours: &[f64], theirs: &[f64]) -> (f64, f64) {
    let mut sq = 0.0;
    let mut sq_ref = 0.0;
    let mut worst = 0.0f64;
    for (&a, &b) in ours.iter().zip(theirs) {
        sq += (a - b) * (a - b);
        sq_ref += b * b;
        worst = worst.max((a - b).abs());
    }
    ((sq / sq_ref.max(1e-300)).sqrt(), worst)
}

fn measure(model: &Eomt, set: &str, fx: &Fixture, p: &Picture) -> Measured {
    // The input.
    let (input, fit) = input_tensor(&p.rgb, p.width, p.height);
    assert_eq!(fit, p.fit, "{}: the size the picture is fitted to", p.name);
    const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
    const STD: [f32; 3] = [0.229, 0.224, 0.225];
    let (mut max_levels, mut differing, mut n) = (0i64, 0usize, 0usize);
    let mut sample = p.input_samples.iter();
    for y in (0..640).step_by(fx.strides.0) {
        for x in (0..640).step_by(fx.strides.1) {
            for c in 0..3 {
                let v = input[(y * 640 + x) * 3 + c];
                let level = ((v * STD[c] + MEAN[c]) * 255.0).round() as i64;
                let want = *sample.next().expect("a sample");
                max_levels = max_levels.max((level - want).abs());
                differing += usize::from(level != want);
                n += 1;
            }
        }
    }
    assert!(
        sample.next().is_none(),
        "{}: all the samples are used",
        p.name
    );
    // The network and the post-processing.
    let raw: Raw = model.infer(&input).expect("runs");
    let labels = model.label_map(&raw, fit.0, fit.1);
    // Class logits of the sampled queries.
    let (mut ours, mut theirs) = (vec![], vec![]);
    for (row, &q) in p.queries.iter().enumerate() {
        for k in 0..134 {
            ours.push(f64::from(raw.class_logits[q * 134 + k]));
            theirs.push(p.class_logits[row][k]);
        }
    }
    let (class_rel, class_max) = rel_rms(&ours, &theirs);
    // Mask logits at the sampled points.
    let (offset, stride, count) = fx.mask;
    let (mut ours, mut theirs) = (vec![], vec![]);
    for (row, &q) in p.queries.iter().enumerate() {
        let mut at = 0;
        for y in 0..count {
            for x in 0..count {
                let (gy, gx) = (offset + stride * y, offset + stride * x);
                ours.push(f64::from(raw.mask_logits[q * 160 * 160 + gy * 160 + gx]));
                theirs.push(p.mask_samples[row][at]);
                at += 1;
            }
        }
    }
    let (mask_rel, mask_max) = rel_rms(&ours, &theirs);
    // The kept queries.
    let kept = kept_queries(&raw.class_logits);
    let same_kept = kept.len() == p.kept.len()
        && kept
            .iter()
            .zip(&p.kept)
            .all(|(a, b)| a.query == b.0 && a.label == b.1);
    let score_max = kept
        .iter()
        .zip(&p.kept)
        .map(|(a, b)| (f64::from(a.score) - b.2).abs())
        .fold(0.0, f64::max);
    // The label map and the element map.
    assert_eq!(labels.classes.len(), p.reference_classes.len());
    let total = labels.classes.len() as f64;
    let same = labels
        .classes
        .iter()
        .zip(&p.reference_classes)
        .filter(|(a, b)| a == b)
        .count() as f64;
    let same_element = labels
        .classes
        .iter()
        .zip(&p.reference_classes)
        .filter(|&(&a, &b)| element_of(a) == element_of(b))
        .count() as f64;
    Measured {
        set: set.to_owned(),
        name: p.name.clone(),
        input_max_levels: max_levels,
        input_differing: differing as f64 / n as f64,
        class_max_abs: class_max,
        class_rel_rms: class_rel,
        mask_max_abs: mask_max,
        mask_rel_rms: mask_rel,
        score_max_abs: score_max,
        same_kept,
        class_agreement: same / total,
        element_agreement: same_element / total,
        kept: p.kept.len(),
    }
}

fn results() -> Option<&'static Vec<Measured>> {
    static RESULTS: OnceLock<Option<Vec<Measured>>> = OnceLock::new();
    RESULTS
        .get_or_init(|| {
            let model = model()?;
            let mut out = vec![];
            for (set, fx) in fixtures() {
                for p in &fx.pictures {
                    let m = measure(model, set, fx, p);
                    println!(
                        "{}/{}: input max {} level(s), {:.3} % of the samples differ; class logits max |d| {:.2e}, rel RMS {:.2e}; \
                         mask logits max |d| {:.2e}, rel RMS {:.2e}; kept {} (same {}), score max |d| {:.2e}; \
                         class map agrees on {:.4} %, element map on {:.4} %",
                        m.set,
                        m.name,
                        m.input_max_levels,
                        m.input_differing * 100.0,
                        m.class_max_abs,
                        m.class_rel_rms,
                        m.mask_max_abs,
                        m.mask_rel_rms,
                        m.kept,
                        m.same_kept,
                        m.score_max_abs,
                        m.class_agreement * 100.0,
                        m.element_agreement * 100.0
                    );
                    out.push(m);
                }
            }
            Some(out)
        })
        .as_ref()
}

// -------------------------------------------------------------------------------------------------
// The tests
// -------------------------------------------------------------------------------------------------

/// The class table of `docs/34` section 4b.2 against the checkpoint's `id2label`, as the fixture script read it from the checkpoint's
/// `config.json`. Needs no weights.
#[test]
fn the_class_table_is_the_checkpoints_id2label() {
    for (set, fx) in fixtures() {
        assert_eq!(fx.id2label.len(), CLASSES, "{set}");
        for (id, name) in fx.id2label.iter().enumerate() {
            assert_eq!(CLASS_NAMES[id], name, "{set}: class {id}");
        }
    }
}

#[test]
fn the_fixture_is_what_the_documentation_says() {
    let fx = &fixtures()[0].1;
    assert_eq!(fx.pictures.len(), 3);
    for p in &fx.pictures {
        assert_eq!(p.queries.len(), 16, "{}", p.name);
        assert!(!p.kept.is_empty(), "{} has kept queries", p.name);
        assert!(p.reference_classes.iter().any(|&c| c != UNLABELLED));
        // Every kept query is among the sampled ones.
        for (q, _, _) in &p.kept {
            assert!(p.queries.contains(q), "{}: query {q}", p.name);
        }
    }
    // The sizes exercise the fit: landscape, portrait and square.
    let fits: Vec<(usize, usize)> = fx.pictures.iter().map(|p| p.fit).collect();
    assert_eq!(fits, vec![(427, 640), (640, 427), (640, 640)]);
    assert!(CLASS_NAMES.len() == 133 && UNLABELLED == 133);
}

#[test]
fn the_input_is_the_originals_to_within_one_level() {
    let Some(results) = results() else { return };
    for m in results {
        assert!(
            m.input_max_levels <= INPUT_MAX_LEVELS,
            "{}/{}: input differs by {} levels",
            m.set,
            m.name,
            m.input_max_levels
        );
        assert!(
            m.input_differing <= INPUT_MAX_DIFFERING,
            "{}/{}: {:.3} % of the input samples differ",
            m.set,
            m.name,
            m.input_differing * 100.0
        );
    }
}

/// The network alone: on the picture whose input equals the original's (`shore`), the logits agree to a few 1e-5 and the label map to
/// the last pixel. What the other pictures add is the resize's one-level differences and float rounding through 12 blocks.
#[test]
fn the_network_alone_matches_the_original_where_the_input_is_the_same() {
    let Some(results) = results() else { return };
    let m = results
        .iter()
        .find(|m| m.set == "committed" && m.name == "shore")
        .expect("the shore picture");
    assert_eq!(m.input_max_levels, 0, "the shore's input is the original's");
    assert!(
        m.class_max_abs <= EXACT_INPUT_CLASS_MAX_ABS,
        "class logits max |d| {:.3e}",
        m.class_max_abs
    );
    assert!(
        m.mask_max_abs <= EXACT_INPUT_MASK_MAX_ABS,
        "mask logits max |d| {:.3e}",
        m.mask_max_abs
    );
    assert_eq!(m.class_agreement, 1.0);
}

#[test]
fn the_logits_match_the_original() {
    let Some(results) = results() else { return };
    for m in results {
        let who = format!("{}/{}", m.set, m.name);
        assert!(
            m.class_max_abs <= CLASS_LOGIT_MAX_ABS,
            "{who}: class logits max |d| {:.3e}",
            m.class_max_abs
        );
        assert!(
            m.class_rel_rms <= CLASS_LOGIT_REL_RMS,
            "{who}: class logits rel RMS {:.3e}",
            m.class_rel_rms
        );
        assert!(
            m.mask_max_abs <= MASK_LOGIT_MAX_ABS,
            "{who}: mask logits max |d| {:.3e}",
            m.mask_max_abs
        );
        assert!(
            m.mask_rel_rms <= MASK_LOGIT_REL_RMS,
            "{who}: mask logits rel RMS {:.3e}",
            m.mask_rel_rms
        );
    }
}

#[test]
fn the_same_queries_are_kept_with_the_same_classes() {
    let Some(results) = results() else { return };
    for m in results {
        let who = format!("{}/{}", m.set, m.name);
        assert!(
            m.same_kept,
            "{who}: the kept queries differ from the original's"
        );
        assert!(
            m.score_max_abs <= SCORE_MAX_ABS,
            "{who}: score max |d| {:.3e}",
            m.score_max_abs
        );
    }
}

#[test]
fn the_label_map_and_the_element_map_agree_with_the_original() {
    let Some(results) = results() else { return };
    for m in results {
        let who = format!("{}/{}", m.set, m.name);
        assert!(
            m.class_agreement >= CLASS_AGREEMENT_MIN,
            "{who}: the class map agrees on {:.4} % only",
            m.class_agreement * 100.0
        );
        assert!(
            m.element_agreement >= ELEMENT_AGREEMENT_MIN,
            "{who}: the element map agrees on {:.4} % only",
            m.element_agreement * 100.0
        );
    }
}

/// The real weights give the same bits on 1, 2 and 8 threads and on a second run (the synthetic-weights twin is in
/// `src/eomt/tests.rs`): the model input, the logits and the label map.
#[test]
fn the_real_weights_give_the_same_bits_on_1_2_and_8_threads_and_on_a_second_run() {
    let Some(model) = model() else { return };
    let p = &fixtures()[0].1.pictures[0];
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| {
                let (input, (th, tw)) = input_tensor(&p.rgb, p.width, p.height);
                let raw = model.infer(&input).expect("runs");
                let labels = model.label_map(&raw, th, tw);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
                (
                    bits(&input),
                    bits(&raw.class_logits),
                    bits(&raw.mask_logits),
                    labels,
                )
            })
    };
    let reference = run(1);
    for (label, threads) in [
        ("2 threads", 2),
        ("8 threads", 8),
        ("a second run on 1 thread", 1),
    ] {
        let again = run(threads);
        assert!(
            reference.0 == again.0,
            "{}: the input differs on {label}",
            p.name
        );
        assert!(
            reference.1 == again.1,
            "{}: the class logits differ on {label}",
            p.name
        );
        assert!(
            reference.2 == again.2,
            "{}: the mask logits differ on {label}",
            p.name
        );
        assert!(
            reference.3 == again.3,
            "{}: the label map differs on {label}",
            p.name
        );
    }
    println!(
        "{}: input, logits and label map bit-identical on 1, 2 and 8 threads and on a second run ({} mask values)",
        p.name,
        reference.2.len()
    );
}

#[test]
fn the_public_api_runs_on_synthetic_weights() {
    let model = Eomt::from_bytes(&synthetic_weights(5), None).expect("loads");
    let rgb = vec![120u8; 33 * 21 * 3];
    let labels = model.segment(&rgb, 33, 21).expect("runs");
    assert_eq!((labels.width, labels.height), (640, 407));
    assert_eq!(labels.classes.len(), 640 * 407);
    assert!(labels.classes.iter().all(|&c| usize::from(c) <= CLASSES));
}
