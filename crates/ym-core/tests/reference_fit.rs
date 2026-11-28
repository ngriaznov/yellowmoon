//! The Reference Grade fit through the core (`docs/34-reference-grade.md` sections 5 and 10, RG-06): a reference chosen and held per
//! node for the session, the fit through the photo's real chain, its result written as one patch, and the properties of the "fit:
//! properties" gate: the identity photo gives the identity, the fit is no worse than the neutral search, every number stays in
//! range, the knots stay monotone, **one thread and many give the same bits and so do two runs**, and Cancel changes nothing.
//!
//! The pictures are synthetic (the test's own raw scenes) and the scene is a **fake scene model** (`SceneModel::fake_default`: sky
//! above, vegetation and built surfaces in the middle, ground below, a subject disc with skin), so no weights are needed. The pure
//! fit is tested in `ym_color::refgrade`; the real networks in `crates/ym-ml/tests`.
mod common;

use std::path::PathBuf;
use std::sync::Arc;

use common::*;
use serde_json::json;
use ym_core::app::{Core, CoreOptions};
use ym_core::pipeline::thumbs::encode_png_rgb8;
use ym_core::pipeline::{PhotoSource, SceneModel};
use ym_core::reference_fit::{FitEvent, FitFailure, FitOutcome, FitStatus, apply_numbers};
use ym_core::session::{OpenOptions, PhotoId};
use ym_core::settings::{DevelopSettings, ReferenceGradeNode, SettingsPatch};
use ym_gpu::queue::CancelToken;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";
const WIDTH: u32 = 640;
const HEIGHT: u32 = 420;

/// A fake scene with no skin: sky above, vegetation and built surfaces in the middle, ground below with a patch of water, and a
/// subject disc in the centre. (A synthetic patchwork has no skin tones, which the skin walls would rightly object to.)
fn scene_without_skin() -> Arc<SceneModel> {
    counting_scene_without_skin(Arc::new(std::sync::atomic::AtomicUsize::new(0)))
}

/// [`scene_without_skin`], counting how many pictures it was asked to segment.
fn counting_scene_without_skin(calls: Arc<std::sync::atomic::AtomicUsize>) -> Arc<SceneModel> {
    use ym_ml::scene::{BUILT, GROUND, SKY, SUBJECT, VEGETATION, WATER};
    SceneModel::fake(
        "no skin",
        false,
        Arc::new(move |w, h, _rgb, _| {
            calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut bits = vec![0u8; w * h];
            for y in 0..h {
                for x in 0..w {
                    let (u, v) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
                    let mut b = if v < 0.33 {
                        SKY
                    } else if v < 0.66 {
                        if u < 0.5 { VEGETATION } else { BUILT }
                    } else if u > 0.7 && v > 0.8 {
                        WATER
                    } else {
                        GROUND
                    };
                    if (u - 0.5).hypot(v - 0.5) < 0.18 {
                        b |= SUBJECT;
                    }
                    bits[y * w + x] = b;
                }
            }
            Ok(bits)
        }),
    )
}

struct Rig {
    core: Core,
    _app: tempfile::TempDir,
    folder: tempfile::TempDir,
    ids: Vec<PhotoId>,
}

/// A core with `n` synthetic photos open and the fake scene model.
fn rig(n: usize) -> Rig {
    rig_with(n, scene_without_skin())
}

/// [`rig`] with the scene model given.
fn rig_with(n: usize, model: Arc<SceneModel>) -> Rig {
    let app = tempfile::tempdir().expect("app root");
    let folder = tempfile::tempdir().expect("photos");
    for k in 0..n {
        let scene = Scene::new(
            WIDTH,
            HEIGHT,
            light(PROFILE, 4800.0 + 700.0 * k as f64),
            Layout::Patchwork,
            0.5,
        )
        .noise(2e-4, 2e-6)
        .seed(21 + k as u64);
        scene.write(folder.path(), &format!("DSC{:05}.ARW", k + 1), MODEL);
    }
    let mut options = CoreOptions::new(data_root(), app.path().to_path_buf());
    options.render.scene_model = Some(model);
    options.render.data_root = Some(data_root());
    options.decode_threads = 2;
    options.thumbnails = false;
    let core = Core::open(options).expect("a core");
    let out = core
        .session
        .open(&[folder.path().to_path_buf()], OpenOptions::default());
    assert_eq!(out.added.len(), n);
    settle(&core.session);
    Rig {
        core,
        _app: app,
        folder,
        ids: out.added,
    }
}

/// The photo's settings with a Reference Grade node (id 2) after Main, reading Main and read by the Output, with `body`'s fields.
fn with_node(base: &DevelopSettings, body: serde_json::Value) -> DevelopSettings {
    let mut n = json!({ "id": 2, "kind": "reference_grade", "pos": [2, 0.0] });
    n.as_object_mut()
        .expect("an object")
        .extend(body.as_object().expect("an object").clone());
    let mut doc = serde_json::to_value(base).expect("settings");
    doc["nodes"] = json!({ "items": [ { "id": 1 }, n ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
        { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] });
    serde_json::from_value::<DevelopSettings>(doc)
        .expect("a document")
        .validated()
        .expect("a valid graph")
        .0
}

impl Rig {
    fn settings(&self, k: usize) -> DevelopSettings {
        self.core.session.view(self.ids[k]).expect("view").settings
    }

    /// Adds the node to photo `k` (through a patch, as the app does).
    fn add_node(&self, k: usize, body: serde_json::Value) {
        let now = self.settings(k);
        let next = with_node(&now, body);
        self.core
            .update_settings(self.ids[k], &SettingsPatch::between(&now, &next))
            .expect("the node is added");
    }

    fn source(&self, k: usize) -> PhotoSource {
        PhotoSource::from_session(&self.core.session, self.ids[k]).expect("a source")
    }

    fn node(&self, k: usize) -> ReferenceGradeNode {
        self.settings(k)
            .nodes
            .reference_grade(2)
            .expect("the node")
            .clone()
    }

    /// The photo as it is shown (sRGB 8-bit, whole), through its own settings.
    fn picture(&self, k: usize) -> ym_core::apple_auto::AdvisorPicture {
        self.core
            .renderer
            .advisor_picture(&self.source(k), &CancelToken::new())
            .expect("a picture")
    }

    /// A file holding a picture, as a reference.
    fn file_of(&self, name: &str, p: &ym_core::apple_auto::AdvisorPicture) -> PathBuf {
        let path = self.folder.path().join(name);
        std::fs::write(
            &path,
            encode_png_rgb8(p.width, p.height, &p.rgb).expect("a png"),
        )
        .expect("write");
        path
    }
}

fn pool(threads: usize) -> rayon::ThreadPool {
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("a pool")
}

/// The mean CIELAB distance between two 8-bit pictures of one size.
fn mean_delta_e(a: &[u8], b: &[u8]) -> f64 {
    use ym_color::refgrade::measure::LabConv;
    let conv = LabConv::new();
    let lab = |p: &[u8]| {
        conv.px_of_codes([
            f64::from(p[0]) / 255.0,
            f64::from(p[1]) / 255.0,
            f64::from(p[2]) / 255.0,
        ])
        .lab
    };
    let n = a.len() / 3;
    (0..n)
        .map(|i| {
            let (x, y) = (lab(&a[3 * i..3 * i + 3]), lab(&b[3 * i..3 * i + 3]));
            ((x[0] - y[0]).powi(2) + (x[1] - y[1]).powi(2) + (x[2] - y[2]).powi(2)).sqrt()
        })
        .sum::<f64>()
        / n as f64
}

fn fit_with(rig: &Rig, k: usize, threads: usize) -> Result<FitOutcome, FitFailure> {
    pool(threads).install(|| {
        rig.core
            .reference_fit(rig.ids[k], 2, &CancelToken::new(), &mut |_| {})
    })
}

// ---------------------------------------------------------------------------------------------
// The reference, held per node
// ---------------------------------------------------------------------------------------------

#[test]
fn a_reference_is_held_per_node_for_the_session_and_dropped_with_the_node() {
    let r = rig(2);
    r.add_node(0, json!({}));
    let other = r.picture(1);
    let path = r.file_of("ref.png", &other);
    // Nothing chosen: the fit says so and writes nothing.
    let before = r.settings(0);
    assert_eq!(
        r.core
            .reference_fit(r.ids[0], 2, &CancelToken::new(), &mut |_| {})
            .unwrap_err(),
        FitFailure::NoReference
    );
    assert_eq!(r.settings(0), before);
    // A file: the label is its name; the notes are the reference's.
    let info = r
        .core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .expect("a reference");
    assert_eq!(info.label, "ref.png");
    assert!(info.notes.is_empty() && !info.grey);
    assert_eq!(r.core.reference_info(r.ids[0], 2), Some(info));
    // A node that is not a Reference Grade node cannot have one.
    assert_eq!(
        r.core
            .reference_choose_file(r.ids[0], 1, &path, None)
            .unwrap_err(),
        FitFailure::NotReferenceGrade(1)
    );
    // A photo of the strip: not the photo being graded; its label is the caller's.
    let same = r
        .core
        .reference_choose_photo(r.ids[0], 2, r.ids[0], "Photo 1 of 2", &CancelToken::new())
        .unwrap_err();
    assert_eq!(
        same.to_string(),
        "The reference can't be used: That's the photo you're grading. Choose another."
    );
    assert_eq!(
        r.core.reference_info(r.ids[0], 2).map(|i| i.label),
        Some("ref.png".into())
    );
    let snap = r
        .core
        .reference_choose_photo(r.ids[0], 2, r.ids[1], "Photo 2 of 2", &CancelToken::new())
        .expect("a snapshot");
    assert_eq!(snap.label, "Photo 2 of 2");
    // An unreadable file keeps the earlier reference.
    let junk = r.folder.path().join("junk.png");
    std::fs::write(&junk, b"not a picture").unwrap();
    let err = r
        .core
        .reference_choose_file(r.ids[0], 2, &junk, None)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "The reference can't be used: Yellowmoon reads JPEG, PNG and TIFF pictures. Save this one as one of those, then choose it again."
    );
    assert_eq!(
        r.core.reference_info(r.ids[0], 2).map(|i| i.label),
        Some("Photo 2 of 2".into())
    );
    // The reference is in no setting of the photo: the settings document has no word of it.
    let doc = serde_json::to_string(&r.settings(0)).unwrap();
    assert!(
        !doc.contains("ref.png") && !doc.contains("Photo 2"),
        "{doc}"
    );
    // Removing the node drops the reference; so does clear.
    let now = r.settings(0);
    let mut gone = now.clone();
    gone.nodes = DevelopSettings::default().nodes;
    r.core
        .update_settings(r.ids[0], &SettingsPatch::between(&now, &gone))
        .expect("the node is removed");
    assert!(r.core.reference_info(r.ids[0], 2).is_none());
    r.add_node(0, json!({}));
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    assert!(r.core.reference_clear(r.ids[0], 2) && !r.core.reference_clear(r.ids[0], 2));
}

// ---------------------------------------------------------------------------------------------
// The fit
// ---------------------------------------------------------------------------------------------

#[test]
fn the_photos_own_picture_as_the_reference_gives_the_identity_and_already_close() {
    let r = rig(1);
    // The person's own fields are in the chain the fit matches through, and the fit leaves them alone.
    r.add_node(
        0,
        json!({ "name": "Mine", "trim": { "exposure_ev": 0.1 }, "grade": { "contrast": 8 }, "strength": 70 }),
    );
    let own = r.picture(0);
    let path = r.file_of("own.png", &own);
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    let before = r.node(0);
    let out = fit_with(&r, 0, 2).expect("a fit");
    assert_eq!(out.status, FitStatus::AlreadyClose);
    assert_eq!(
        out.status.text(),
        "The photo is already close to the reference. Nothing was changed."
    );
    assert!(out.result.identity, "{:?}", out.result.x);
    // The identity is written: the fitted fields are the node's defaults, and nothing else moved.
    let after = r.node(0);
    let mut identity = before.clone();
    apply_numbers(
        &mut identity,
        &ym_color::refgrade::params::to_numbers(&[0.0; 49]),
    );
    assert_eq!(after, identity);
    assert_eq!(after.grade.curve, ReferenceGradeNode::default().grade.curve);
    assert!(after.hue_curves.is_default() && after.elements.is_default());
    assert_eq!(out.result.total, out.result.identity_total);
}

/// The photo graded by a known grade of the node's own family, through the real chain, as a reference file for the ungraded photo.
fn known_grade() -> serde_json::Value {
    json!({
        "grade": {
            "saturation": 30,
            "curve": { "luma": [[0, 0], [0.2136, 0.1836], [0.2784, 0.2784], [0.4080, 0.4380], [1, 1]],
                       "red": [[0, 0], [0.2784, 0.2984], [1, 1]],
                       "green": [[0, 0], [1, 1]],
                       "blue": [[0, 0], [0.2784, 0.2584], [1, 1]] } },
        "hue_curves": { "hue_vs_sat": [ { "hue_deg": 0, "value": 1.15 }, { "hue_deg": 120, "value": 0.9 }, { "hue_deg": 240, "value": 1.1 } ] },
        "elements": { "sky": { "exposure_ev": -0.3, "hue_deg": 10, "chroma": 1.2 } }
    })
}

#[test]
fn a_known_grade_is_recovered_closer_than_the_identity() {
    let r = rig(2);
    // Photo 0 is the photo; photo 1 is the same scene graded (a second copy of the same raw, with the known grade), and its
    // picture is the reference.
    r.add_node(0, json!({}));
    let ungraded = r.picture(0);
    // The graded picture of photo 0's own raw: grade the same photo, take its picture, then set the grade back to the identity.
    let identity_body = serde_json::to_value(r.node(0)).unwrap();
    r.add_node(0, known_grade());
    let graded = r.picture(0);
    r.add_node(0, identity_body);
    assert_eq!(
        r.picture(0).rgb,
        ungraded.rgb,
        "the node at its identity is the ungraded photo"
    );
    let path = r.file_of("graded.png", &graded);
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();

    let identity_distance = mean_delta_e(&ungraded.rgb, &graded.rgb);
    let out = fit_with(&r, 0, 4).expect("a fit");
    assert_eq!(out.status, FitStatus::Fitted, "{:?}", out.result.x);
    let fitted = r.picture(0);
    let fitted_distance = mean_delta_e(&fitted.rgb, &graded.rgb);
    println!(
        "recovery: mean dE76 to the graded picture, identity {identity_distance:.3}, fitted {fitted_distance:.3}; objective {:.3} -> {:.3}; start {:?}; {} candidates",
        out.result.identity_total, out.result.total, out.result.start, out.result.evals
    );
    // The bound: the fitted photo is at most 92 % as far from the graded picture as the ungraded one is (measured 88 % on this
    // scene: the fit copies the look's statistics, never the picture's brightness or its pixels, `docs/34` section 16.8), and the
    // objective falls to at most a third of the identity's (measured 21 %).
    assert!(
        fitted_distance < 0.92 * identity_distance,
        "fitted {fitted_distance:.3} against the identity's {identity_distance:.3}"
    );
    assert!(
        out.result.total < 0.33 * out.result.identity_total,
        "{} against {}",
        out.result.total,
        out.result.identity_total
    );
    assert!(out.result.total <= out.result.neutral_total);
    // The skin walls had no skin to look at and said so.
    assert!(
        out.notes
            .iter()
            .any(|n| n == "No skin was found, so Skin protection did nothing."),
        "{:?}",
        out.notes
    );
}

/// The properties of a fit's result (`docs/34` section 10, "fit: properties"), on the recovery fit.
#[test]
fn a_fit_is_no_worse_than_the_neutral_search_keeps_its_numbers_in_range_and_its_knots_monotone() {
    let r = rig(2);
    r.add_node(0, json!({}));
    let path = r.file_of("other.png", &r.picture(1));
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    let before = r.settings(0);
    let mut events: Vec<FitEvent> = Vec::new();
    let out = pool(4)
        .install(|| {
            r.core
                .reference_fit(r.ids[0], 2, &CancelToken::new(), &mut |e| events.push(e))
        })
        .expect("a fit");
    let res = &out.result;
    // No worse than the neutral search, no worse than the identity; the lower of the two starts, a tie to neutral.
    assert!(res.total <= res.neutral_total && res.total <= res.identity_total);
    assert_eq!(res.total, res.neutral_total.min(res.look_total));
    if res.look_total >= res.neutral_total {
        assert_eq!(res.start, ym_color::refgrade::search::Start::Neutral);
    }
    // Every number in range; an element that failed the need test is exactly 0 in the neutral search.
    assert!(res.x.iter().all(|v| (-1.0..=1.0).contains(v)));
    let neutral = &res.searches[0];
    for k in 0..6 {
        if !neutral.need.local[k] {
            assert_eq!(neutral.x[30 + 3 * k..33 + 3 * k], [0.0; 3], "element {k}");
        }
    }
    if !neutral.need.local[6] {
        assert_eq!(neutral.x[48], 0.0);
    }
    // The node as written: its curves hold end points, strictly increasing inputs and non-decreasing outputs; the hue curves are
    // six knots at 0, 60, ... 300 or empty; the settings accepted every value without clamping (the patch went through validation).
    let node = r.node(0);
    for c in [
        &node.grade.curve.luma,
        &node.grade.curve.red,
        &node.grade.curve.green,
        &node.grade.curve.blue,
    ] {
        assert_eq!(
            (c.first(), c.last()),
            (Some(&[0.0, 0.0]), Some(&[1.0, 1.0]))
        );
        assert!(
            c.windows(2)
                .all(|w| w[1][0] > w[0][0] && w[1][1] >= w[0][1]),
            "{c:?}"
        );
    }
    for h in [&node.hue_curves.hue_vs_hue, &node.hue_curves.hue_vs_sat] {
        assert!(
            h.is_empty()
                || h.iter().map(|p| p.hue_deg).collect::<Vec<_>>()
                    == [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
        );
    }
    let numbers = ym_color::refgrade::params::to_numbers(&res.x);
    let mut want = before.nodes.reference_grade(2).unwrap().clone();
    apply_numbers(&mut want, &numbers);
    assert_eq!(node, want, "the node holds exactly the fit's numbers");
    // One patch, one undo step, and the person's other fields stayed.
    assert_eq!(SettingsPatch::between(&before, &r.settings(0)), out.applied);
    r.core.update_settings(r.ids[0], &out.undo).expect("undo");
    assert_eq!(r.settings(0), before);
    // The events: the scene, then the three searches in order, never going back.
    assert_eq!(events.first(), Some(&FitEvent::Scene));
    assert_eq!(
        events.first().map(FitEvent::text).as_deref(),
        Some("Finding what is in the pictures…")
    );
    let searches: Vec<u8> = events
        .iter()
        .filter_map(|e| match e {
            FitEvent::Search { search, .. } => Some(*search),
            FitEvent::Scene => None,
        })
        .collect();
    assert!(searches.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(searches.first(), Some(&1));
    assert_eq!(searches.last(), Some(&3));
    assert_eq!(
        events.iter().rev().find_map(|e| match e {
            FitEvent::Search { search, of, .. } => Some(format!("Fitting… {search} of {of}")),
            FitEvent::Scene => None,
        }),
        Some("Fitting… 3 of 3".to_owned())
    );
}

/// One thread and many give the same bits, and so do two runs (`AGENTS.md` rule 8, `docs/34` section 5.7).
#[test]
fn one_thread_and_many_and_two_runs_give_the_same_bits() {
    let r = rig(2);
    r.add_node(0, json!({}));
    let path = r.file_of("other.png", &r.picture(1));
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    let before = r.settings(0);
    let run = |threads: usize| {
        let out = fit_with(&r, 0, threads).expect("a fit");
        let node = r.node(0);
        r.core.update_settings(r.ids[0], &out.undo).expect("undo");
        assert_eq!(r.settings(0), before);
        (
            out.result.x.map(f64::to_bits),
            out.result.total.to_bits(),
            out.result.evals,
            node,
        )
    };
    let one = run(1);
    let many = run(4);
    let again = run(3);
    assert_eq!(one, many, "one thread and four");
    assert_eq!(one, again, "one thread and three");
    assert_eq!(run(1), one, "a second run on one thread");
}

/// Cancel changes nothing: the settings are those of before, the reference is still held, and the next fit runs.
#[test]
fn cancel_changes_nothing() {
    let r = rig(2);
    r.add_node(0, json!({}));
    let path = r.file_of("other.png", &r.picture(1));
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    let before = r.settings(0);
    // Cancelled while the scene is made, and in the middle of the second search.
    for at in [0usize, 1, 2] {
        let cancel = CancelToken::new();
        let mut seen = 0usize;
        let err = r
            .core
            .reference_fit(r.ids[0], 2, &cancel, &mut |e| {
                let fire = match (at, e) {
                    (0, FitEvent::Scene) => true,
                    (1, FitEvent::Search { search: 1, .. }) => seen > 2,
                    (2, FitEvent::Search { search: 2, .. }) => true,
                    _ => false,
                };
                seen += 1;
                if fire {
                    cancel.cancel();
                }
            })
            .unwrap_err();
        assert_eq!(err, FitFailure::Cancelled, "cancelled at {at}");
        assert_eq!(
            err.to_string(),
            "The fit was cancelled. Nothing was changed."
        );
        assert_eq!(r.settings(0), before, "cancelled at {at}");
        assert!(r.core.reference_info(r.ids[0], 2).is_some());
    }
}

/// Failures write nothing.
#[test]
fn a_node_that_is_not_in_the_picture_writes_nothing() {
    let r = rig(2);
    // A node nothing reads: not connected to the Output.
    let now = r.settings(0);
    let mut doc = serde_json::to_value(&now).unwrap();
    doc["nodes"] = json!({ "items": [ { "id": 1 }, { "id": 2, "kind": "reference_grade", "pos": [2, 0.0] } ], "links": [
        { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
        { "from": { "node": 1, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] });
    let next = serde_json::from_value::<DevelopSettings>(doc)
        .unwrap()
        .validated()
        .unwrap()
        .0;
    r.core
        .update_settings(r.ids[0], &SettingsPatch::between(&now, &next))
        .expect("a graph");
    let path = r.file_of("other.png", &r.picture(1));
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    let before = r.settings(0);
    let err = fit_with(&r, 0, 2).unwrap_err();
    assert_eq!(err, FitFailure::NotConnected, "{err:?}");
    assert_eq!(r.settings(0), before);
}

/// The evaluation tool's inputs (`docs/34` section 10.1, RG-10): the photo with **every pixel** of the analysis picture, and the
/// held reference as the fit measures it. The pixels of the whole picture are the fit's own: a candidate rendered through both gives
/// the same pixel at every place the fit's sample has.
#[test]
fn the_evaluation_photo_has_every_pixel_and_agrees_with_the_fits_sample() {
    use ym_color::refgrade::measure::Px;
    use ym_color::refgrade::search::CandidateRenderer;
    let r = rig(2);
    r.add_node(0, json!({}));
    let cancel = CancelToken::new();
    let sampled = r
        .core
        .renderer
        .reference_fit_photo(&r.source(0), 2, &cancel)
        .expect("the fit's photo");
    let full = r
        .core
        .reference_eval_photo(r.ids[0], 2, &cancel)
        .expect("the evaluation photo");
    assert_eq!((full.width, full.height), (sampled.width, sampled.height));
    assert_eq!(full.sample.len(), full.width * full.height);
    assert_eq!(full.states.len(), full.sample.len());
    assert_eq!(full.scene.len(), full.sample.len());
    assert_eq!(full.sample[0], (0, 0));
    assert_eq!(full.sample[full.width + 1], (1, 1));
    assert_eq!(full.mask_ok, sampled.mask_ok);
    // A candidate with a global grade and a sky grade, rendered through both.
    let node = r.node(0);
    let mut x = [0.0; 49];
    x[3] = 0.4;
    x[10] = -0.3;
    x[17] = 0.3;
    x[24] = 0.2;
    x[31] = 0.2;
    let mut a = ym_core::reference_fit::ChainCandidates::new(&sampled, &node);
    let mut b = ym_core::reference_fit::ChainCandidates::new(&full, &node);
    let (mut pa, mut pb) = (
        vec![Px::default(); a.sample_len()],
        vec![Px::default(); b.sample_len()],
    );
    a.render(&x, &mut pa).expect("a render");
    b.render(&x, &mut pb).expect("a render");
    for (i, &(px, py)) in sampled.sample.iter().enumerate() {
        assert_eq!(pa[i], pb[py * full.width + px], "pixel ({px}, {py})");
        assert_eq!(
            sampled.scene.class[i],
            full.scene.class[py * full.width + px]
        );
        assert_eq!(sampled.scene.skin[i], full.scene.skin[py * full.width + px]);
    }
    // The grade is not the identity somewhere (the test sees something).
    let mut ident = vec![Px::default(); b.sample_len()];
    b.render(&[0.0; 49], &mut ident).expect("a render");
    assert!(pb.iter().zip(&ident).any(|(p, q)| p != q));
    // The coverage the skin and whites walls are gated by (RG-13) is counted on every pixel of the analysis picture, with the node at
    // its identity: the same shares the whole-picture render gives, from either photo.
    let n = ident.len() as f64;
    let whites = ident
        .iter()
        .filter(|q| ym_color::refgrade::objective::is_white(q))
        .count() as f64
        / n;
    let skin = full.scene.skin.iter().filter(|&&s| s).count() as f64 / n;
    assert_eq!(full.coverage, sampled.coverage);
    assert!(
        (sampled.coverage.skin - skin).abs() < 1e-12,
        "{:?}",
        sampled.coverage
    );
    assert!(
        (sampled.coverage.whites - whites).abs() < 2e-3,
        "{:?} against {whites}",
        sampled.coverage
    );
    // The reference side: nothing before a reference is chosen, then every pixel of its analysis picture.
    assert_eq!(
        r.core.reference_measured(r.ids[0], 2, &cancel).unwrap_err(),
        FitFailure::NoReference
    );
    let path = r.file_of("graded.png", &r.picture(1));
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .expect("a reference");
    let m = r
        .core
        .reference_measured(r.ids[0], 2, &cancel)
        .expect("the measured reference");
    assert_eq!(m.px.len(), m.width * m.height);
    assert_eq!(m.scene.len(), m.px.len());
    assert!(m.width.max(m.height) <= 512);
    // Asking again gives the same bits (the scene is made once and kept).
    let again = r.core.reference_measured(r.ids[0], 2, &cancel).unwrap();
    assert_eq!(m.px, again.px);
    assert_eq!(m.scene, again.scene);
}

/// Where the time of a fit goes, on this machine (a Linux figure; run with `--ignored --nocapture`): one candidate's render and its
/// measures, and a whole fit at 1 and 4 threads.
#[test]
#[ignore = "timing: run by hand"]
fn the_time_of_a_fit_and_of_its_parts() {
    use std::time::Instant;
    use ym_color::refgrade::measure::Px;
    use ym_color::refgrade::search::CandidateRenderer;
    let r = rig(2);
    r.add_node(0, json!({}));
    let path = r.file_of("graded.png", &r.picture(1));
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    let src = r.source(0);
    let t = Instant::now();
    let photo = r
        .core
        .renderer
        .reference_fit_photo(&src, 2, &CancelToken::new())
        .expect("a photo");
    println!(
        "the photo's side (the analysis picture, the scene, the soft masks, the states): {:.2} s",
        t.elapsed().as_secs_f64()
    );
    let node = r.node(0);
    for threads in [1usize, 4] {
        let mut cand = ym_core::reference_fit::ChainCandidates::new(&photo, &node);
        let mut out = vec![Px::default(); cand.sample_len()];
        let mut x = [0.0; 49];
        x[17] = 0.3;
        x[24] = 0.2;
        x[31] = 0.2;
        let n = 100;
        let t = Instant::now();
        pool(threads).install(|| {
            for _ in 0..n {
                cand.render(&x, &mut out).unwrap();
            }
        });
        println!(
            "{threads} threads: one candidate's render of 8192 pixels {:.2} ms",
            t.elapsed().as_secs_f64() * 1000.0 / n as f64
        );
    }
    for threads in [1usize, 4] {
        let t = Instant::now();
        let out = fit_with(&r, 0, threads).expect("a fit");
        println!(
            "{threads} threads: a whole fit {:.1} s, {} candidates ({:.2} ms each); start {:?}",
            t.elapsed().as_secs_f64(),
            out.result.evals,
            t.elapsed().as_secs_f64() * 1000.0 / out.result.evals as f64,
            out.result.start
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Pinned fits (RG-11: faster, and not one bit different)
// ---------------------------------------------------------------------------------------------

/// FNV-1a, 64 bit: a fixed, dependency-free hash for the pins (std's hashers promise nothing across releases).
fn fnv1a(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

/// A hash of everything a fit decides: its 49 numbers, its objective and the three searches' ends (as bits), the candidates it
/// evaluated, which start won, and the node's fitted fields as the settings document writes them.
fn pin_of(out: &FitOutcome, node: &ReferenceGradeNode) -> u64 {
    let r = &out.result;
    let mut bytes: Vec<u8> = Vec::new();
    for v in r.x {
        bytes.extend(v.to_bits().to_le_bytes());
    }
    for v in [r.total, r.identity_total, r.neutral_total, r.look_total] {
        bytes.extend(v.to_bits().to_le_bytes());
    }
    for s in &r.searches {
        for v in s.x {
            bytes.extend(v.to_bits().to_le_bytes());
        }
        bytes.extend(s.after.total().to_bits().to_le_bytes());
    }
    bytes.extend((r.evals as u64).to_le_bytes());
    bytes.push(u8::from(r.start == ym_color::refgrade::search::Start::Look));
    let fitted =
        json!({ "grade": node.grade, "hue_curves": node.hue_curves, "elements": node.elements });
    bytes.extend(serde_json::to_vec(&fitted).expect("json"));
    fnv1a(&bytes)
}

/// Fits photo 0 on 1 and on 4 threads, checks the two give one pin, and returns it.
fn pin_on_threads(r: &Rig) -> u64 {
    let before = r.settings(0);
    let mut pins = Vec::new();
    for threads in [1usize, 4] {
        let out = fit_with(r, 0, threads).expect("a fit");
        pins.push(pin_of(&out, &r.node(0)));
        r.core.update_settings(r.ids[0], &out.undo).expect("undo");
        assert_eq!(r.settings(0), before);
    }
    assert_eq!(pins[0], pins[1], "one thread and four");
    pins[0]
}

/// The fitted settings of four fixture fits, as the fit made them before RG-11 (the speed task): the same bits now, on 1 thread and
/// on 4. A change to the fit's results changes these, and has to be a decision of its own (`AGENTS.md` rule 8, `docs/34` section 5.7).
#[test]
#[cfg_attr(
    not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64")
    )),
    ignore = "the pins are per platform (libm); none recorded for this target"
)]
fn four_fixture_fits_give_the_bits_they_gave_before_the_speed_task() {
    // The known grade, recovered.
    let r = rig(2);
    r.add_node(0, json!({}));
    let identity_body = serde_json::to_value(r.node(0)).unwrap();
    r.add_node(0, known_grade());
    let graded = r.picture(0);
    r.add_node(0, identity_body);
    let path = r.file_of("graded.png", &graded);
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    let recovery = pin_on_threads(&r);
    // Another photo's picture as a file, and as a photo of the strip.
    let path = r.file_of("other.png", &r.picture(1));
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    let other_file = pin_on_threads(&r);
    r.core
        .reference_choose_photo(r.ids[0], 2, r.ids[1], "Photo 2 of 2", &CancelToken::new())
        .unwrap();
    let other_photo = pin_on_threads(&r);
    // The person's own fields in the chain, strength 70 and no skin protection.
    r.add_node(
        0,
        json!({ "trim": { "exposure_ev": 0.1 }, "grade": { "contrast": 8 }, "strength": 70, "skin_protection": false }),
    );
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .unwrap();
    let own_fields = pin_on_threads(&r);
    let got = [recovery, other_file, other_photo, own_fields];
    println!("PINS {got:016x?}");
    assert_eq!(got, PINNED_FITS, "{got:016x?}");
}

// The pins are per platform: the fit calls `ln`, `exp`, `cbrt`, `atan2`, `hypot` and `powf`, and Apple's libm and glibc can differ
// in the last bit of each. On one machine the fit is deterministic (`one_thread_and_many_and_two_runs_give_the_same_bits`). The pins of a platform guard that
// platform against regressions.

/// The pins of [`four_fixture_fits_give_the_bits_they_gave_before_the_speed_task`]: recovery, a file, a photo of the strip, own fields.
/// Linux x86_64: recorded at the base commit 6e8f9d2f, before RG-11, so they prove the speed work changed no bit.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const PINNED_FITS: [u64; 4] = [
    0xefe3_4191_2c9d_d6b7,
    0x30c7_6e01_a349_b55a,
    0x30c7_6e01_a349_b55a,
    0x9563_285a_8f7c_360f,
];

/// The same pins on macOS aarch64: recorded by CI run 45888389844 at e1ed3a69, after RG-11. They still stand for "before": RG-11
/// reorders no floating-point operation and moved each value's expression without changing it, so the same functions on the same
/// platform give the same bits, as the Linux pins show. From now on they guard against regressions.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
const PINNED_FITS: [u64; 4] = [
    0xdc0b_9a47_d8f3_f21d,
    0x08d3_0138_5763_dc11,
    0x08d3_0138_5763_dc11,
    0x2d11_551f_21a1_31e9,
];

/// No pins are recorded for other targets; the test is ignored there.
#[cfg(not(any(
    all(target_os = "linux", target_arch = "x86_64"),
    all(target_os = "macos", target_arch = "aarch64")
)))]
const PINNED_FITS: [u64; 4] = [0; 4];

/// The scenes are made once per photo and once per reference, and kept for the next fit (`docs/34` sections 4b.1 and 5.7, RG-11): a
/// second fit of the same photo to the same reference asks the segmenter for nothing; a new reference is segmented once, and the
/// photo is not segmented again.
#[test]
fn the_scene_is_made_once_per_photo_and_once_per_reference() {
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    let calls = Arc::new(AtomicUsize::new(0));
    let r = rig_with(2, counting_scene_without_skin(Arc::clone(&calls)));
    r.add_node(0, json!({}));
    let first = r.file_of("first.png", &r.picture(1));
    let second = r.file_of("second.png", &r.picture(0));
    r.core
        .reference_choose_file(r.ids[0], 2, &first, None)
        .unwrap();
    let start = calls.load(SeqCst);
    let fit = |label: &str| {
        let out = fit_with(&r, 0, 2).expect(label);
        r.core.update_settings(r.ids[0], &out.undo).expect("undo");
        calls.load(SeqCst)
    };
    // The first fit makes the photo's scene (unless the background has) and the reference's.
    let after_first = fit("a first fit");
    assert!(after_first > start, "the first fit segments the reference");
    // The second fit to the same held reference makes neither.
    assert_eq!(
        fit("a second fit"),
        after_first,
        "a second fit to the same reference"
    );
    // Another reference: its scene is made, once; the photo's is kept.
    r.core
        .reference_choose_file(r.ids[0], 2, &second, None)
        .unwrap();
    let after_new = fit("a fit to a new reference");
    assert_eq!(after_new, after_first + 1, "one new picture to segment");
    assert_eq!(fit("and again"), after_new);
}

// ---------------------------------------------------------------------------------------------
// Match brightness (docs/34 section 5.8, RG-16)
// ---------------------------------------------------------------------------------------------

/// Sets the Input's exposure of photo `k` to `ev` stops, through a patch.
fn set_exposure(r: &Rig, k: usize, ev: f64) {
    let now = r.settings(k);
    let mut next = now.clone();
    next.input_mut().exposure = ym_core::settings::ExposureSetting::manual_ev(ev);
    r.core
        .update_settings(r.ids[k], &SettingsPatch::between(&now, &next))
        .expect("the exposure is set");
}

/// The relative luminance of a picture's median L*, as the fit reads it.
fn median_y(p: &ym_core::apple_auto::AdvisorPicture) -> f64 {
    use ym_color::refgrade::brightness::luminance_of_lstar;
    use ym_color::refgrade::measure::{LabConv, median_lightness};
    let conv = LabConv::new();
    let px: Vec<_> = p
        .rgb
        .chunks_exact(3)
        .map(|c| {
            conv.px_of_codes([
                f64::from(c[0]) / 255.0,
                f64::from(c[1]) / 255.0,
                f64::from(c[2]) / 255.0,
            ])
        })
        .collect();
    luminance_of_lstar(median_lightness(&px))
}

/// A rig whose photo 0 is `under_ev` stops under the picture it is to match, with that picture chosen as the node's reference
/// (the same scene at `reference_ev` stops of exposure). Returns the rig and the reference's median luminance.
fn dark_photo_rig(under_ev: f64, reference_ev: f64) -> (Rig, f64) {
    let r = rig(1);
    r.add_node(0, json!({}));
    set_exposure(&r, 0, reference_ev);
    let reference = r.picture(0);
    let path = r.file_of("bright.png", &reference);
    set_exposure(&r, 0, reference_ev - under_ev);
    r.core
        .reference_choose_file(r.ids[0], 2, &path, None)
        .expect("a reference");
    (r, median_y(&reference))
}

/// Fits photo 0 at `percent` Match brightness, from the node's defaults, and returns the outcome and the photo as it is shown.
fn fit_at(r: &Rig, percent: f64) -> (FitOutcome, ym_core::apple_auto::AdvisorPicture) {
    let now = r.settings(0);
    let mut next = now.clone();
    if let Some(n) = next.nodes.reference_grade_mut(2) {
        n.match_brightness = percent;
    }
    r.core
        .update_settings(r.ids[0], &SettingsPatch::between(&now, &next))
        .expect("the field is set");
    let out = fit_with(r, 0, 4).expect("a fit");
    let shown = r.picture(0);
    r.core.update_settings(r.ids[0], &out.undo).expect("undo");
    (out, shown)
}

/// Fits a photo `under_ev` stops under its reference at 0, 25, 50, 75 and 100 %, checks the move of each (Δ rises with the slider
/// toward the reference, the bisection is within 0.01 stop of its target) and returns the held gap and the shown median move of each
/// fit in stops: `(gap, [(percent, move)])`.
fn slider_run(under_ev: f64) -> (f64, Vec<(f64, f64)>) {
    let (r, y_r) = dark_photo_rig(under_ev, 0.0);
    let y_p = median_y(&r.picture(0));
    let gap = (y_r / y_p).log2();
    println!("under {under_ev}: gap between the photo and the reference {gap:.3} stops");
    assert!(
        gap.abs() > 0.2 && gap.abs() < 1.5,
        "the scene's gap is under the hold: {gap}"
    );
    let sign = gap.signum();
    let mut last_asked = f64::NEG_INFINITY;
    let mut shown_moves: Vec<(f64, f64)> = Vec::new();
    for (percent, share) in [
        (0.0, 0.0),
        (25.0, 0.25),
        (50.0, 0.5),
        (75.0, 0.75),
        (100.0, 1.0),
    ] {
        let (out, shown) = fit_at(&r, percent);
        let b = out.brightness;
        let got = (median_y(&shown) / y_p).log2();
        println!(
            "{percent:>5} %: asked {:.3}, displayed move of the sample {:.3}, shift {:.3}, fitted picture moved {got:.3}, {:?}",
            b.asked_ev, b.moved_ev, b.shift_ev, out.status
        );
        shown_moves.push((percent, got));
        assert_eq!(b.percent, percent);
        // The asked move grows toward the reference: it rises for a brighter reference and falls for a darker one.
        assert!(
            sign * b.asked_ev >= last_asked,
            "the move goes toward the reference as the slider rises"
        );
        last_asked = sign * b.asked_ev;
        assert!(!b.limited);
        if percent == 0.0 {
            // Nothing new ran: today's fit, and nothing moved.
            assert_eq!(b, ym_color::refgrade::brightness::BrightnessMove::none());
            continue;
        }
        // The bisection's result is within 0.01 stop of its target.
        assert!((b.moved_ev - b.asked_ev).abs() < 0.01, "{b:?}");
        assert!((b.asked_ev - share * b.gap_ev).abs() < 1e-12, "{b:?}");
    }
    (gap, shown_moves)
}

/// Row (3): the shown median move at `b` is within 0.1 stop of `m0 + b (g - m0)`, `m0` being the 0 % fit's own shown move; it is
/// monotone in `b` (toward the reference); 100 % is within 0.1 stop of the reference's median.
fn check_even_slider(gap: f64, moves: &[(f64, f64)]) {
    let m0 = moves[0].1;
    for &(percent, got) in &moves[1..] {
        let b = percent / 100.0;
        let want = m0 + b * (gap - m0);
        assert!(
            (got - want).abs() < 0.1,
            "{percent} %: moved {got:.3}, half-way rule wants {want:.3} (m0 {m0:.3}, gap {gap:.3})"
        );
    }
    let sign = gap.signum();
    assert!(
        moves.windows(2).all(|w| sign * w[1].1 >= sign * w[0].1),
        "not monotone: {moves:?}"
    );
    let at_100 = moves.last().expect("measured").1;
    assert!(
        (at_100 - gap).abs() < 0.1,
        "100 %: moved {at_100:.3}, the reference is {gap:.3} away"
    );
}

/// Rows (2) and (3): a photo a stop under its reference.
#[test]
fn a_photo_a_stop_under_moves_evenly_from_the_zero_percent_fit_to_the_reference() {
    let (gap, moves) = slider_run(1.0);
    println!("shown moves by percent: {moves:?}, gap {gap:.3}");
    check_even_slider(gap, &moves);
}

/// Row (3) on a photo half a stop under its reference.
#[test]
fn a_photo_half_a_stop_under_goes_monotonically_to_the_reference() {
    let (gap, moves) = slider_run(0.5);
    println!("shown moves by percent: {moves:?}, gap {gap:.3}");
    check_even_slider(gap, &moves);
}

/// Row (3) on a photo a stop over its reference: the gap is negative and the move falls as the slider rises.
#[test]
fn a_photo_a_stop_over_goes_monotonically_down_to_the_reference() {
    let (gap, moves) = slider_run(-1.0);
    println!("shown moves by percent: {moves:?}, gap {gap:.3}");
    assert!(gap < 0.0);
    check_even_slider(gap, &moves);
}

/// Row (2): the gap is held at 1.5 stops, and the report says so.
#[test]
fn a_gap_over_one_and_a_half_stops_is_held_and_reported() {
    let (r, _) = dark_photo_rig(3.5, 1.0);
    let (out, _) = fit_at(&r, 100.0);
    let b = out.brightness;
    println!("{b:?}");
    assert!(b.limited, "{b:?}");
    assert!((b.gap_ev - 1.5).abs() < 1e-12 && (b.asked_ev - 1.5).abs() < 1e-12);
    assert!((b.moved_ev - 1.5).abs() < 0.01, "{b:?}");
    // The other way: a bright photo and a dark reference.
    let (r, _) = dark_photo_rig(-3.5, 0.0);
    let (out, _) = fit_at(&r, 50.0);
    let b = out.brightness;
    assert!(b.limited && (b.gap_ev + 1.5).abs() < 1e-12, "{b:?}");
    assert!(
        (b.asked_ev + 0.75).abs() < 1e-12 && (b.moved_ev + 0.75).abs() < 0.01,
        "{b:?}"
    );
}

/// Row (2): `P'` is the photo rendered with the exposure shift, at the node's identity: a render of the photo with that Exposure
/// in the node's own trim, bit for bit.
#[test]
fn the_moved_photo_is_a_render_with_that_exposure() {
    use ym_color::refgrade::measure::Px;
    use ym_color::refgrade::search::CandidateRenderer;
    let r = rig(2);
    r.add_node(0, json!({}));
    let cancel = CancelToken::new();
    let photo = r
        .core
        .renderer
        .reference_fit_photo(&r.source(0), 2, &cancel)
        .expect("the photo");
    let node = r.node(0);
    for d in [-1.25, 0.4, 1.0] {
        let mut shifted = ym_core::reference_fit::ChainCandidates::new(&photo, &node);
        let mut got = vec![Px::default(); shifted.sample_len()];
        shifted
            .render_shifted(&[0.0; 49], d, &mut got)
            .expect("a render");
        // The same photo with the node's own Exposure at `d`, fitted through the settings as the person would set it.
        r.add_node(0, json!({ "trim": { "exposure_ev": d } }));
        let with_trim = r
            .core
            .renderer
            .reference_fit_photo(&r.source(0), 2, &cancel)
            .expect("the photo");
        let node_d = r.node(0);
        let mut plain = ym_core::reference_fit::ChainCandidates::new(&with_trim, &node_d);
        let mut want = vec![Px::default(); plain.sample_len()];
        plain.render(&[0.0; 49], &mut want).expect("a render");
        assert_eq!(got, want, "d = {d}");
        r.add_node(0, json!({}));
    }
    // A shift of 0 is the plain render, bit for bit.
    let mut a = ym_core::reference_fit::ChainCandidates::new(&photo, &node);
    let mut b = ym_core::reference_fit::ChainCandidates::new(&photo, &node);
    let (mut pa, mut pb) = (
        vec![Px::default(); a.sample_len()],
        vec![Px::default(); b.sample_len()],
    );
    a.render(&[0.0; 49], &mut pa).unwrap();
    b.render_shifted(&[0.0; 49], 0.0, &mut pb).unwrap();
    assert_eq!(pa, pb);
}

/// Determinism (`AGENTS.md` rule 8): one thread and four, and two runs, give the same bits at 50 %, the move included.
#[test]
fn a_fit_with_match_brightness_gives_the_same_bits_on_one_thread_and_many_and_twice() {
    let (r, _) = dark_photo_rig(1.0, 0.0);
    let now = r.settings(0);
    let mut next = now.clone();
    next.nodes
        .reference_grade_mut(2)
        .expect("the node")
        .match_brightness = 50.0;
    r.core
        .update_settings(r.ids[0], &SettingsPatch::between(&now, &next))
        .unwrap();
    let mut pins = Vec::new();
    for threads in [1usize, 4, 4] {
        let out = fit_with(&r, 0, threads).expect("a fit");
        let b = out.brightness;
        pins.push((
            pin_of(&out, &r.node(0)),
            b.shift_ev.to_bits(),
            b.moved_ev.to_bits(),
            b.asked_ev.to_bits(),
        ));
        r.core.update_settings(r.ids[0], &out.undo).expect("undo");
    }
    assert_eq!(pins[0], pins[1], "one thread and four");
    assert_eq!(pins[1], pins[2], "two runs");
}
