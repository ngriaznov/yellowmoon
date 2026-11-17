//! Albums in the core (`docs/31`): saving and opening a session, the settings as stored (Auto stays Auto), the provider's
//! answers, finding the photos, a kept failed entry, fingerprints, refusals, the edited state, the file and its schema.

#![allow(clippy::field_reassign_with_default)]

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::rich::{every_mask, rich};
use common::{Layout, Scene, light};
use serde_json::{Value, json};
use ym_core::album::{
    Album, AlbumError, AlbumPhotoState, AlbumState, CHANGED_SENTENCE, FINGERPRINT_BYTES,
    Fingerprint, MAX_ALBUM_BYTES, TOO_LARGE_SENTENCE, TOO_NEW_SENTENCE, write_album,
};
use ym_core::app::{Core, CoreError, CoreOptions};
use ym_core::apple_auto::{AppleAuto, FakeAdvisor};
use ym_core::auto_provider::{AutoProvider, ProviderAdvisors};
use ym_core::export_queue::{ExportEvent, ExportJob, ExportSink};
use ym_core::session::{ApplyScope, OpenOptions, PhotoId, PhotoStatus};
use ym_core::settings::{
    CorrectorNode, DevelopSettings, FilmNode, GradePatch, NodeId, NodeLink, NodePatch,
    PercentSetting, SettingsPatch,
};
use ym_export::ExportOptions;
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

struct Rig {
    core: Core,
    app: tempfile::TempDir,
    root: tempfile::TempDir,
    events: std::sync::mpsc::Receiver<ExportEvent>,
}

fn rig_with(advisors: ProviderAdvisors) -> Rig {
    let app = tempfile::tempdir().expect("app root");
    let root = tempfile::tempdir().expect("root");
    let (sink, events) = ExportSink::channel();
    let mut options = CoreOptions::new(common::data_root(), app.path().to_path_buf());
    options.decode_threads = 2;
    options.thumbnails = false;
    options.export_events = sink;
    options.home = Some(root.path().to_path_buf());
    options.advisors = advisors;
    Rig {
        core: Core::open(options).expect("core"),
        app,
        root,
        events,
    }
}

fn rig() -> Rig {
    rig_with(ProviderAdvisors::default())
}

fn fake_advisors() -> ProviderAdvisors {
    ProviderAdvisors::default().with(AutoProvider::Apple, Arc::new(FakeAdvisor))
}

/// Writes the named photos into `dir`, each a scene of its own.
fn write_photos(dir: &Path, names: &[&str]) -> Vec<PathBuf> {
    std::fs::create_dir_all(dir).expect("folder");
    names
        .iter()
        .enumerate()
        .map(|(k, name)| {
            let scene = Scene::new(
                300,
                200,
                light(PROFILE, 4600.0 + 300.0 * k as f64),
                Layout::Patchwork,
                0.25 + 0.03 * k as f64,
            )
            .noise(2e-4, 1e-5)
            .seed(k as u64 + 7);
            let path = dir.join(name);
            scene
                .builder(MODEL)
                .capture_time(&format!("2026:10:04 12:{:02}:00", k % 60), "07", None)
                .write_to(&path, &scene.mosaic())
                .expect("write");
            path
        })
        .collect()
}

impl Rig {
    fn open(&self, dir: &Path) -> Vec<PhotoId> {
        let out = self
            .core
            .session
            .open(&[dir.to_path_buf()], OpenOptions::default());
        assert!(
            self.core.session.wait_idle(Duration::from_secs(120)),
            "developed"
        );
        out.added
    }

    fn settle(&self) {
        assert!(
            self.core.session.wait_idle(Duration::from_secs(120)),
            "developed"
        );
    }

    fn settings(&self, id: PhotoId) -> DevelopSettings {
        self.core.session.settings(id).expect("settings").settings
    }

    fn set(&self, id: PhotoId, s: DevelopSettings) {
        self.core.session.replace_settings(id, s).expect("replaced");
    }

    fn patch(&self, id: PhotoId, p: &SettingsPatch) {
        self.core.session.update_settings(id, p).expect("patched");
    }

    fn edited(&self) -> bool {
        self.core.album_state().expect("an album").edited
    }

    fn names(&self) -> Vec<String> {
        self.core
            .session
            .state()
            .photos
            .into_iter()
            .map(|p| p.file_name)
            .collect()
    }

    /// The photos as an album would save them: path, settings, picks.
    fn stored(&self) -> Vec<(PathBuf, DevelopSettings, bool, bool)> {
        self.core
            .session
            .capture_photos()
            .into_iter()
            .map(|c| (c.path, c.settings, c.picked, c.current))
            .collect()
    }
}

fn read_json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("read")).expect("json")
}

/// Every file under `root`, relative, with its size.
fn tree(root: &Path) -> Vec<(String, u64)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, u64)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else {
                let rel = p
                    .strip_prefix(root)
                    .expect("under root")
                    .to_string_lossy()
                    .into_owned();
                out.push((rel, e.metadata().map(|m| m.len()).unwrap_or(0)));
            }
        }
    }
    let mut v = Vec::new();
    walk(root, root, &mut v);
    v.sort();
    v
}

fn grade(g: GradePatch) -> SettingsPatch {
    SettingsPatch::main(NodePatch {
        grade: Some(g),
        ..NodePatch::default()
    })
}

/// A session of ten photos in two folders (the strip is not in path order: the second folder was added later), with a
/// different kind of setting on each, some picks and the current photo.
fn ten_photos() -> (Rig, PathBuf) {
    let r = rig();
    let b = r.root.path().join("Shoot B");
    let a = r.root.path().join("Shoot A");
    write_photos(
        &b,
        &[
            "B0006.ARW",
            "B0007.ARW",
            "B0008.ARW",
            "B0009.ARW",
            "B0010.ARW",
        ],
    );
    write_photos(
        &a,
        &[
            "A0001.ARW",
            "A0002.ARW",
            "A0003.ARW",
            "A0004.ARW",
            "A0005.ARW",
        ],
    );
    r.open(&b);
    r.open(&a);
    let ids = r.core.session.photo_ids();
    assert_eq!(ids.len(), 10);
    // 0: every field, with a look this library does not have, every kind of mask and the white pick, the crop and the turn.
    r.set(ids[0], rich());
    // 1: a manual white balance and a crop.
    r.patch(ids[1], &SettingsPatch::white_balance(4350.0, 6.0));
    r.patch(
        ids[1],
        &SettingsPatch::input(NodePatch {
            crop: Some(Some(ym_core::settings::Crop {
                rect: ym_core::settings::NormRect {
                    x: 0.1,
                    y: 0.15,
                    width: 0.6,
                    height: 0.7,
                },
                angle_deg: 1.5,
                aspect: ym_core::settings::CropAspect::Free,
                auto: false,
            })),
            orientation: Some(ym_core::settings::Orientation::Deg90),
            ..NodePatch::default()
        }),
    );
    // 2: a graph with a Film node and a Layer.
    let mut s = DevelopSettings::with_look(r.core.looks.effective_default());
    s.nodes.films.insert(4, FilmNode::default());
    s.nodes.mixers.insert(
        5,
        ym_core::settings::MixerNode::new_of(ym_core::settings::NodeKind::Layer).expect("layer"),
    );
    let mut sky = CorrectorNode::default();
    sky.name = "Sky".to_owned();
    s.nodes.extra.insert(2, sky);
    s.main_mut().masks = every_mask();
    r.set(ids[2], s);
    // 3: nothing touched: Auto everywhere.
    // 4: a look at half strength.
    r.patch(ids[4], &SettingsPatch::look_at(LookId::neutral(), 50.0));
    // 5 to 9: assorted manual values.
    r.patch(ids[5], &SettingsPatch::exposure_ev(0.8));
    r.patch(ids[6], &SettingsPatch::sharpen(120.0, 1.1));
    r.patch(ids[7], &SettingsPatch::noise_reduction(10.0, 20.0, 30.0));
    r.patch(
        ids[8],
        &grade(GradePatch {
            contrast: Some(PercentSetting::manual_value(-20.0)),
            saturation: Some(PercentSetting::manual_value(33.0)),
            ..GradePatch::default()
        }),
    );
    r.patch(ids[9], &SettingsPatch::contrast(15.0));
    for i in [0, 3, 4, 9] {
        r.core.session.set_picked(ids[i], true).expect("pick");
    }
    r.core.session.set_current(ids[6]).expect("current");
    let albums = r.root.path().join("Albums");
    std::fs::create_dir_all(&albums).expect("folder");
    (r, albums.join("Lisbon.ymalbum"))
}

// ---------------------------------------------------------------------------------------------
// Save and open
// ---------------------------------------------------------------------------------------------

#[test]
fn a_session_of_ten_photos_with_every_kind_of_setting_comes_back_as_it_was() {
    let (a, path) = ten_photos();
    let before = a.stored();
    let names = a.names();
    assert_eq!(names[0], "B0006.ARW", "the strip is not in path order");
    assert!(
        a.core.album_state().is_none(),
        "no album until the first save"
    );
    let saved = a.core.album_save(&path).expect("saved");
    assert_eq!(
        saved,
        AlbumState {
            path: path.clone(),
            name: "Lisbon".into(),
            edited: false
        }
    );
    assert!(!a.edited() && a.core.album_state() == Some(saved));

    // Another computer: no looks, no presets, nothing.
    let b = rig();
    let outcome = b
        .core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    assert_eq!(b.names(), names, "the strip's order");
    assert_eq!(outcome.photos.len(), 10);
    assert!(
        outcome
            .photos
            .iter()
            .all(|(_, s)| *s == AlbumPhotoState::Found)
    );
    assert_eq!(
        outcome.album,
        AlbumState {
            path: path.clone(),
            name: "Lisbon".into(),
            edited: false
        }
    );
    let after = b.stored();
    assert_eq!(after.len(), 10);
    for (i, (x, y)) in before.iter().zip(&after).enumerate() {
        assert_eq!(x.0, y.0, "photo {i}: the same file");
        assert_eq!(x.1, y.1, "photo {i}: every setting");
        assert_eq!(x.2, y.2, "photo {i}: picked");
        assert_eq!(x.3, y.3, "photo {i}: current");
    }
    assert!(before[6].3 && after[6].3, "the current photo");
    // The settings in the session are what the file said, with the white pick's sample taken once developed.
    let ids = b.core.session.photo_ids();
    let r0 = b.core.session.settings(ids[0]).expect("developed");
    assert_eq!(r0.settings, before[0].1);
    assert!(r0.settings.input().white_pick.is_some());
    assert!(!b.edited(), "developing the photos is not an edit");
    // A look this library does not have stays as stored and is not replaced silently.
    assert_eq!(
        b.core
            .session
            .settings(ids[0])
            .unwrap()
            .settings
            .output()
            .look
            .as_str(),
        "user:film-warm"
    );
    assert!(
        b.core
            .looks
            .check_usable(&LookId::parse("user:film-warm").unwrap())
            .is_err()
    );
    // The photos came back developed, and a thing of the photo's own is there: the Object clicks and the Brush strokes.
    let masks = &after[0].1.main().masks;
    assert!(masks.items.iter().any(
        |r| matches!(&r.shape, ym_core::settings::MaskShape::Object(o) if !o.points.is_empty())
    ));
    assert!(masks.items.iter().any(
        |r| matches!(&r.shape, ym_core::settings::MaskShape::Brush(br) if !br.strokes.is_empty())
    ));
    // Nothing of it is in the application folder.
    assert!(tree(a.app.path()).is_empty() && tree(b.app.path()).is_empty());
}

#[test]
fn settings_are_stored_unpinned_and_auto_stays_auto() {
    let (a, path) = ten_photos();
    a.core.album_save(&path).expect("saved");
    let doc = read_json(&path);
    let photos = doc["photos"].as_array().expect("photos");
    // Photo 3 was never touched: every Auto of the Input and Main is Auto in the file.
    let items = photos[3]["settings"]["nodes"]["items"]
        .as_array()
        .expect("items");
    let input = &items[0];
    for key in ["exposure", "white_balance", "sharpen", "noise_reduction"] {
        assert_eq!(input[key], json!({ "kind": "auto" }), "input.{key}");
    }
    assert_eq!(input["light_source"], "auto");
    let g = &items[1]["grade"];
    for key in ["contrast", "highlights", "shadows", "black", "saturation"] {
        assert_eq!(g[key], json!({ "kind": "auto" }), "main.{key}");
    }
    assert_eq!(g["color_balance"]["shadows"], json!({ "kind": "auto" }));
    // A manual value is as it was set.
    assert_eq!(
        photos[5]["settings"]["nodes"]["items"][0]["exposure"],
        json!({ "kind": "manual", "ev": 0.8 })
    );
    // The crop, the turn and the white pick are the photo's own and are kept.
    let first = &photos[0]["settings"]["nodes"]["items"][0];
    assert!(
        first["crop"].is_object() && first["orientation"] == 90 && first["white_pick"].is_array()
    );
    // Picks, the current photo and the versions.
    let picked: Vec<bool> = photos
        .iter()
        .map(|p| p["picked"].as_bool().unwrap())
        .collect();
    assert_eq!(
        picked,
        [
            true, false, false, true, true, false, false, false, false, true
        ]
    );
    assert_eq!(doc["current"], 6);
    assert_eq!(doc["yellowmoon_album"], 1);
    assert_eq!(
        doc["written_by"]["pipeline"],
        ym_core::version::PIPELINE_VERSION
    );
    assert_eq!(
        doc["written_by"]["version"],
        ym_core::version::app_version()
    );
    // Absolute and relative paths, and the fingerprint of each file.
    let p0 = &photos[0];
    assert!(
        p0["absolute"]
            .as_str()
            .unwrap()
            .ends_with("Shoot B/B0006.ARW")
    );
    assert_eq!(p0["path"], "../Shoot B/B0006.ARW");
    let size = std::fs::metadata(p0["absolute"].as_str().unwrap())
        .unwrap()
        .len();
    assert_eq!(p0["fingerprint"]["size"], size);
    assert_eq!(p0["fingerprint"]["head_sha256"].as_str().unwrap().len(), 64);
    assert!(
        p0.get("auto").is_none(),
        "Default Auto's rule has no answer to keep"
    );
}

#[test]
fn the_same_session_is_the_same_bytes_and_a_saved_album_opened_exports_the_same_pictures() {
    let a = rig();
    let dir = a.root.path().join("Trip");
    write_photos(&dir, &["T1.ARW", "T2.ARW", "T3.ARW"]);
    let ids = a.open(&dir);
    a.patch(ids[0], &SettingsPatch::exposure_ev(0.5));
    a.patch(ids[1], &SettingsPatch::look_at(LookId::neutral(), 60.0));
    let mut s = DevelopSettings::with_look(a.core.looks.effective_default());
    let mut node = CorrectorNode::default();
    node.name = "Warm".to_owned();
    node.trim.temperature = 15.0;
    s.nodes.extra.insert(2, node);
    s.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(2)),
        NodeLink::picture(NodeId::Corrector(2), NodeId::Output),
    ];
    a.set(ids[2], s);
    let path = a.root.path().join("Trip.ymalbum");
    a.core.album_save(&path).expect("saved");
    let first = std::fs::read(&path).unwrap();
    a.core.album_save(&path).expect("saved again");
    assert_eq!(
        std::fs::read(&path).unwrap(),
        first,
        "the same session is the same bytes"
    );
    assert!(first.ends_with(b"}\n"));
    assert_eq!(
        first.iter().filter(|b| **b == b'\n').count(),
        1,
        "compact: one line"
    );

    let export = |r: &Rig, to: &Path| {
        let job = ExportJob {
            scope: ApplyScope::All,
            options: ExportOptions {
                destination: Some(to.to_path_buf()),
                ..ExportOptions::default()
            },
        };
        r.core.export_start(&job).expect("queued");
        let deadline = std::time::Instant::now() + Duration::from_secs(180);
        loop {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if r.events.recv_timeout(left).expect("finished") == ExportEvent::Busy(false) {
                break;
            }
        }
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(to)
            .unwrap()
            .flatten()
            .map(|e| {
                (
                    e.file_name().to_string_lossy().into_owned(),
                    std::fs::read(e.path()).unwrap(),
                )
            })
            .collect();
        files.sort();
        files
    };
    let out_a = a.root.path().join("out a");
    let from_a = export(&a, &out_a);
    let b = rig();
    b.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    let from_b = export(&b, &b.root.path().join("out b"));
    assert_eq!(from_a.len(), 3);
    assert_eq!(
        from_a, from_b,
        "an album opened from a saved session renders the same bytes"
    );
}

// ---------------------------------------------------------------------------------------------
// The provider's answers
// ---------------------------------------------------------------------------------------------

#[test]
fn the_providers_answer_comes_back_and_nothing_is_asked_on_opening_but_rerun_asks() {
    let a = rig_with(fake_advisors());
    let dir = a.root.path().join("Ask");
    write_photos(&dir, &["Q1.ARW", "Q2.ARW", "Q3.ARW"]);
    let ids = a.open(&dir);
    assert_eq!(
        a.core.session.apple_auto_asks(),
        3,
        "the provider answered every photo"
    );
    let resolved_a: Vec<_> = ids
        .iter()
        .map(|id| a.core.session.settings(*id).unwrap())
        .collect();
    assert!(ids.iter().all(|id| matches!(
        a.core.session.apple_result_of(*id),
        Some(AppleAuto::Answered { .. })
    )));
    let path = a.root.path().join("Ask.ymalbum");
    a.core.album_save(&path).expect("saved");
    // The settings are stored as Auto, not as Apple Auto's values, and the answer is stored beside them.
    let doc = read_json(&path);
    for photo in doc["photos"].as_array().unwrap() {
        let auto = &photo["auto"];
        assert_eq!(auto["provider"], "apple");
        assert_eq!(auto["model"], FakeAdvisor::MODEL);
        assert_eq!(auto["result"], "answered");
        assert!(auto["steps"].is_object() && auto["scene"].is_string() && auto["plan"].is_string());
        assert!(auto["plan"].as_str().unwrap().starts_with("Test plan"));
        let items = photo["settings"]["nodes"]["items"].as_array().unwrap();
        assert_eq!(items[1]["grade"]["contrast"], json!({ "kind": "auto" }));
    }

    // Opened elsewhere with the same provider: nothing is asked, and the photos have the answers (the same estimates).
    let b = rig_with(fake_advisors());
    b.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    assert_eq!(
        b.core.session.apple_auto_asks(),
        0,
        "opening an album asks no provider"
    );
    let ids_b = b.core.session.photo_ids();
    for (i, id) in ids_b.iter().enumerate() {
        let theirs = b.core.session.apple_result_of(*id).expect("an answer");
        let ours = a.core.session.apple_result_of(ids[i]).expect("an answer");
        assert_eq!(
            theirs.advisor_json(),
            ours.advisor_json(),
            "photo {i}'s answer"
        );
        let (x, y) = (&resolved_a[i], b.core.session.settings(*id).unwrap());
        assert_eq!(x.values, y.values, "photo {i}: the same estimates");
        assert!(matches!(theirs, AppleAuto::Answered { .. }));
    }
    assert!(!b.edited());
    // Saving it again keeps the answers (only the relative paths differ: the album is in another folder).
    let path2 = b.root.path().join("Again.ymalbum");
    b.core.album_save(&path2).expect("saved");
    let photos_of = |doc: &Value| {
        let mut photos = doc["photos"].clone();
        for photo in photos.as_array_mut().unwrap() {
            photo["path"] = Value::Null;
        }
        photos
    };
    assert_eq!(photos_of(&read_json(&path2)), photos_of(&doc));
    // Re-run asks the chosen provider again, for that photo.
    b.core.session.rerun_auto(ids_b[1]).expect("rerun");
    b.settle();
    assert_eq!(b.core.session.apple_auto_asks(), 1);
    assert!(matches!(
        b.core.session.apple_result_of(ids_b[1]),
        Some(AppleAuto::Answered { .. })
    ));
    assert_eq!(b.core.session.apple_auto_asks(), 1, "only that photo");
}

#[test]
fn a_photo_without_an_answer_is_graded_by_default_autos_rule_and_never_asked() {
    // Saved by a session without a provider: no answer in the file.
    let a = rig();
    let dir = a.root.path().join("Plain");
    write_photos(&dir, &["P1.ARW", "P2.ARW"]);
    a.open(&dir);
    let path = a.root.path().join("Plain.ymalbum");
    a.core.album_save(&path).expect("saved");
    assert!(
        read_json(&path)["photos"]
            .as_array()
            .unwrap()
            .iter()
            .all(|p| p.get("auto").is_none())
    );
    // Opened where a provider would act: it is not asked, whatever the preferences choose.
    let b = rig_with(fake_advisors());
    b.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    assert_eq!(b.core.session.apple_auto_asks(), 0);
    let ids = b.core.session.photo_ids();
    for id in &ids {
        assert!(b.core.session.apple_result_of(*id).is_none());
        assert!(
            !b.core.session.summary(*id).unwrap().apple_pending,
            "nothing is waiting for an answer"
        );
        assert_eq!(
            b.core
                .session
                .settings(*id)
                .unwrap()
                .values
                .main
                .grade
                .contrast,
            0.0
        );
    }
    // Photos added to the album later are graded as any new photo is.
    let more = b.root.path().join("More");
    write_photos(&more, &["M1.ARW"]);
    b.open(&more);
    assert_eq!(
        b.core.session.apple_auto_asks(),
        1,
        "only the new photo is asked"
    );
    // New instructions for the model ask every photo again, an album's too.
    b.core
        .session
        .set_apple_auto_instructions("Be kind to the highlights.");
    b.settle();
    assert_eq!(b.core.session.apple_auto_asks(), 1 + 3);
}

// ---------------------------------------------------------------------------------------------
// Exporting an opened album (docs/31 "As built": an export never waits for an answer nobody will ask for)
// ---------------------------------------------------------------------------------------------

/// Starts an export and waits for the queue to go idle, at most 90 s: a hang fails the test instead of the suite.
/// Returns the files written, by name.
fn export_within(r: &Rig, scope: ApplyScope, to: &Path) -> Vec<(String, Vec<u8>)> {
    let job = ExportJob {
        scope,
        options: ExportOptions {
            destination: Some(to.to_path_buf()),
            ..ExportOptions::default()
        },
    };
    let id = r.core.export_start(&job).expect("queued");
    let deadline = std::time::Instant::now() + Duration::from_secs(90);
    loop {
        let left = deadline.saturating_duration_since(std::time::Instant::now());
        match r.events.recv_timeout(left) {
            Ok(ExportEvent::Busy(false)) => break,
            Ok(_) => {}
            Err(_) => {
                r.core.export_cancel(id);
                panic!(
                    "the export did not finish in 90 s: it is waiting for an answer nobody asks for"
                );
            }
        }
    }
    if let Some(sum) = r.core.export_summary(id)
        && !sum.errors.is_empty()
    {
        eprintln!("export errors: {:?}", sum.errors);
    }
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(to)
        .map(|d| {
            d.flatten()
                .map(|e| {
                    (
                        e.file_name().to_string_lossy().into_owned(),
                        std::fs::read(e.path()).unwrap(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files
}

#[test]
fn an_opened_album_exports_when_the_provider_is_active_and_a_photo_has_no_stored_answer() {
    // Saved by a session without a provider: no answer in the file. Opened where a provider acts.
    let a = rig();
    let dir = a.root.path().join("Plain");
    write_photos(&dir, &["P1.ARW", "P2.ARW", "P3.ARW"]);
    a.open(&dir);
    let path = a.root.path().join("Plain.ymalbum");
    a.core.album_save(&path).expect("saved");
    let b = rig_with(fake_advisors());
    b.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    let files = export_within(&b, ApplyScope::All, &b.root.path().join("out"));
    assert_eq!(files.len(), 3, "every photo is exported");
    assert_eq!(b.core.session.apple_auto_asks(), 0, "nothing was asked");
    // The current photo alone, too.
    let one = export_within(&b, ApplyScope::This, &b.root.path().join("one"));
    assert_eq!(one.len(), 1);
    // The file is the picture the viewport shows: the same bytes as the session without a provider.
    let from_a = export_within(&a, ApplyScope::All, &a.root.path().join("out"));
    assert_eq!(files, from_a, "Default Auto's grade, as on screen");
}

#[test]
fn an_album_photo_with_no_answer_is_not_pending_and_only_rerun_makes_it_so() {
    let a = rig();
    let dir = a.root.path().join("NoAnswer");
    write_photos(&dir, &["N1.ARW", "N2.ARW"]);
    a.open(&dir);
    let path = a.root.path().join("NoAnswer.ymalbum");
    a.core.album_save(&path).expect("saved");
    let b = rig_with(fake_advisors());
    b.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    let ids = b.core.session.photo_ids();
    for id in &ids {
        let served = b.core.session.estimates(*id).expect("estimates");
        assert_eq!(served.apple, AppleAuto::Off, "graded by the built-in rule");
        assert_eq!(served.provider, AutoProvider::Yellowmoon);
        assert!(!b.core.session.summary(*id).unwrap().apple_pending);
    }
    // The pixels are the built-in grade the photo already showed (the session without a provider).
    let files = export_within(&b, ApplyScope::All, &b.root.path().join("out"));
    let from_a = export_within(&a, ApplyScope::All, &a.root.path().join("out"));
    assert_eq!(files.len(), 2);
    assert_eq!(files, from_a, "the export bytes are unchanged");
    assert_eq!(b.core.session.apple_auto_asks(), 0, "nothing is asked");
    // Re-run unfreezes it and asks: pending while asked, then answered.
    b.core.session.rerun_auto(ids[0]).expect("rerun");
    b.settle();
    assert_eq!(b.core.session.apple_auto_asks(), 1, "asked once");
    assert!(matches!(
        b.core.session.apple_result_of(ids[0]),
        Some(AppleAuto::Answered { .. })
    ));
    assert_eq!(
        b.core.session.estimates(ids[1]).unwrap().apple,
        AppleAuto::Off,
        "the other photo is still not asked"
    );
}

#[test]
fn an_opened_album_exports_with_its_stored_answers_and_rerun_still_asks() {
    let a = rig_with(fake_advisors());
    let dir = a.root.path().join("Ask");
    write_photos(&dir, &["Q1.ARW", "Q2.ARW", "Q3.ARW"]);
    a.open(&dir);
    let path = a.root.path().join("Ask.ymalbum");
    a.core.album_save(&path).expect("saved");
    let from_a = export_within(&a, ApplyScope::All, &a.root.path().join("out"));
    let b = rig_with(fake_advisors());
    b.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    let from_b = export_within(&b, ApplyScope::All, &b.root.path().join("out"));
    assert_eq!(from_a.len(), 3);
    assert_eq!(from_a, from_b, "the stored answers are used");
    assert_eq!(b.core.session.apple_auto_asks(), 0);
    // Re-run asks the provider again and the export then uses that answer.
    let ids = b.core.session.photo_ids();
    b.core.session.rerun_auto(ids[0]).expect("rerun");
    let again = export_within(&b, ApplyScope::This, &b.root.path().join("again"));
    assert_eq!(again.len(), 1);
    assert_eq!(
        b.core.session.apple_auto_asks(),
        1,
        "Re-run asked the provider"
    );
    assert!(matches!(
        b.core.session.apple_result_of(ids[0]),
        Some(AppleAuto::Answered { .. })
    ));
}

#[test]
fn an_opened_album_with_a_missing_photo_exports_the_others_with_a_provider() {
    let (_r, path, shoot) = trip();
    std::fs::remove_file(shoot.join("D2.ARW")).expect("remove the file");
    let b = rig_with(fake_advisors());
    b.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    let files = export_within(&b, ApplyScope::All, &b.root.path().join("out"));
    assert_eq!(
        files.len(),
        2,
        "the missing photo is skipped, the others exported"
    );
}

/// Photos shot in bursts (frames a few tenths of a second apart share an exposure), each frame a scene of its own.
fn write_burst_photos(dir: &Path, names: &[&str], burst_len: usize) -> Vec<PathBuf> {
    std::fs::create_dir_all(dir).expect("folder");
    names
        .iter()
        .enumerate()
        .map(|(k, name)| {
            let scene = Scene::new(
                300,
                200,
                light(PROFILE, 4600.0 + 300.0 * k as f64),
                Layout::Patchwork,
                0.25 + 0.03 * k as f64,
            )
            .noise(2e-4, 1e-5)
            .seed(k as u64 + 7);
            let path = dir.join(name);
            let (group, frame) = (k / burst_len, k % burst_len);
            scene
                .builder(MODEL)
                .capture_time(
                    &format!("2026:10:04 12:{:02}:00", group % 60),
                    &format!("{}", 10 + 10 * frame),
                    None,
                )
                .write_to(&path, &scene.mosaic())
                .expect("write");
            path
        })
        .collect()
}

#[test]
fn an_opened_album_of_bursts_with_edits_nodes_and_unanswered_photos_exports_every_photo() {
    // Saved by a session where a provider acted: every photo has its answer. Two of them lose it in the file, as a failed answer is
    // not kept, then the album is opened where the provider acts.
    let a = rig_with(fake_advisors());
    let dir = a.root.path().join("Bursts");
    let names: Vec<String> = (0..12).map(|k| format!("B{k:02}.ARW")).collect();
    let names: Vec<&str> = names.iter().map(String::as_str).collect();
    write_burst_photos(&dir, &names, 4);
    let ids = a.open(&dir);
    a.patch(ids[1], &SettingsPatch::exposure_ev(0.89));
    let mut s = a.settings(ids[2]);
    let mut sky = CorrectorNode::default();
    sky.name = "Sky".to_owned();
    s.nodes.extra.insert(2, sky);
    a.set(ids[2], s);
    a.patch(ids[5], &SettingsPatch::look_at(LookId::neutral(), 86.0));
    let path = a.root.path().join("Bursts.ymalbum");
    a.core.album_save(&path).expect("saved");
    let mut doc = read_json(&path);
    for k in [1, 5, 9] {
        doc["photos"][k]
            .as_object_mut()
            .expect("an entry")
            .remove("auto");
    }
    std::fs::write(&path, serde_json::to_vec(&doc).unwrap()).expect("rewritten");
    let b = rig_with(fake_advisors());
    b.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    let files = export_within(&b, ApplyScope::All, &b.root.path().join("out"));
    assert_eq!(
        files.iter().map(|f| f.0.clone()).collect::<Vec<_>>().len(),
        12,
        "every photo is exported: {:?}",
        files.iter().map(|f| f.0.clone()).collect::<Vec<_>>()
    );
    assert_eq!(b.core.session.apple_auto_asks(), 0, "nothing was asked");
    for (n, id) in b.core.session.photo_ids().into_iter().enumerate() {
        b.core.session.set_current(id).expect("current");
        let one = export_within(&b, ApplyScope::This, &b.root.path().join(format!("one{n}")));
        assert_eq!(one.len(), 1, "photo {n} exports alone");
    }
}

// ---------------------------------------------------------------------------------------------
// Finding the photos
// ---------------------------------------------------------------------------------------------

/// An album of three photos saved in `Albums/` beside `Shoot/`.
fn trip() -> (Rig, PathBuf, PathBuf) {
    let r = rig();
    let shoot = r.root.path().join("Shoot");
    write_photos(&shoot, &["D1.ARW", "D2.ARW", "D3.ARW"]);
    let ids = r.open(&shoot);
    r.patch(ids[0], &SettingsPatch::exposure_ev(0.1));
    r.patch(ids[1], &SettingsPatch::exposure_ev(0.2));
    r.patch(ids[2], &SettingsPatch::exposure_ev(0.3));
    r.core.session.set_picked(ids[1], true).unwrap();
    let albums = r.root.path().join("Albums");
    std::fs::create_dir_all(&albums).unwrap();
    let path = albums.join("Trip.ymalbum");
    r.core.album_save(&path).expect("saved");
    (r, path, shoot)
}

fn edit_album(path: &Path, f: impl FnOnce(&mut Value)) {
    let mut doc = read_json(path);
    f(&mut doc);
    std::fs::write(path, doc.to_string()).unwrap();
}

#[test]
fn the_relative_path_wins_and_the_absolute_path_is_the_fallback() {
    let (r, path, _shoot) = trip();
    let first = r.core.session.capture_photos()[0].path.clone();
    // A decoy copy of the first photo elsewhere, which the absolute path names.
    let decoy_dir = r.root.path().join("Decoys");
    std::fs::create_dir_all(&decoy_dir).unwrap();
    let decoy = decoy_dir.join("D1.ARW");
    let other = Scene::new(300, 200, light(PROFILE, 6100.0), Layout::Patchwork, 0.5)
        .noise(2e-4, 1e-5)
        .seed(99);
    other
        .builder(MODEL)
        .write_to(&decoy, &other.mosaic())
        .expect("write");
    edit_album(&path, |d| {
        d["photos"][0]["absolute"] = json!(decoy.to_str().unwrap())
    });
    let b = rig();
    b.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    let found = b.core.session.capture_photos();
    // The path as it would be had the photo been opened by hand: the name the person gave, symbolic
    // links not resolved (`Session::open` keeps `absolute(path)`; the set of open files is keyed by
    // the canonical form). On macOS the temporary folder is `/var/...`, a link to `/private/var/...`,
    // so comparing with a canonicalised path would be wrong.
    assert_eq!(found[0].path, first, "the relative path wins");
    assert_eq!(
        found[0].path.canonicalize().ok(),
        first.canonicalize().ok(),
        "and it is the same file as the one opened by hand"
    );
    assert!(
        found[0].path.starts_with(r.root.path().join("Shoot"))
            || found[0].path.ends_with("Shoot/D1.ARW")
    );
    // The relative path is wrong: the absolute path is used.
    edit_album(&path, |d| {
        d["photos"][0]["path"] = json!("../Nowhere/D1.ARW");
        d["photos"][0]["absolute"] = json!(decoy.to_str().unwrap());
    });
    let c = rig();
    let out = c
        .core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    assert_eq!(
        out.photos[0].1,
        AlbumPhotoState::Changed,
        "the decoy is another file"
    );
    assert!(
        c.core.session.capture_photos()[0]
            .path
            .ends_with("Decoys/D1.ARW")
    );
    // No relative path at all (another volume): the absolute path.
    edit_album(&path, |d| {
        d["photos"][0]["path"] = Value::Null;
        d["photos"][0]["absolute"] = json!(first.to_str().unwrap());
    });
    let e = rig();
    let out = e
        .core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    assert_eq!(out.photos[0].1, AlbumPhotoState::Found);
    assert_eq!(e.core.session.capture_photos()[0].path, first);
}

#[test]
fn a_missing_photo_is_a_kept_failed_entry_that_saving_again_writes_back() {
    let (r, path, shoot) = trip();
    let original = read_json(&path);
    let gone = shoot.join("D2.ARW");
    std::fs::remove_file(&gone).expect("remove the file");
    let b = rig_with(fake_advisors());
    let out = b
        .core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    b.settle();
    assert_eq!(
        b.names(),
        ["D1.ARW", "D2.ARW", "D3.ARW"],
        "its place is kept"
    );
    let absolute = original["photos"][1]["absolute"]
        .as_str()
        .unwrap()
        .to_owned();
    assert_eq!(
        out.photos[1].1,
        AlbumPhotoState::Missing {
            absolute: absolute.clone()
        }
    );
    let ids = b.core.session.photo_ids();
    // The tile says CAN'T OPEN and the cell names the file the album had.
    match b.core.session.status(ids[1]).unwrap() {
        PhotoStatus::Skipped { reason } => assert_eq!(reason, format!("Not found: {absolute}")),
        other => panic!("{other:?}"),
    }
    assert!(
        b.core.session.settings(ids[1]).is_err(),
        "it cannot be developed"
    );
    // Its settings, pick and place are kept.
    let kept = b.core.session.capture_photos();
    assert_eq!(kept[1].settings, r.settings(r.core.session.photo_ids()[1]));
    assert!(kept[1].picked);
    // Saving again does not drop it: the file says what it said.
    let again = b.root.path().join("Again.ymalbum");
    b.core.album_save(&again).expect("saved");
    let after = read_json(&again);
    let mut want = original["photos"].clone();
    let mut got = after["photos"].clone();
    for list in [&mut want, &mut got] {
        for photo in list.as_array_mut().unwrap() {
            photo["path"] = Value::Null; // another folder
        }
    }
    assert_eq!(
        got, want,
        "the entry, its fingerprint and every other photo are as they were"
    );
    // Taking it out of the strip is a change to the album.
    assert!(!b.edited());
    b.core.remove_photos(&[ids[1]]);
    assert!(b.edited());
    assert_eq!(b.names(), ["D1.ARW", "D3.ARW"]);
    b.core.album_save(&again).expect("saved");
    assert_eq!(read_json(&again)["photos"].as_array().unwrap().len(), 2);
    assert!(!b.edited());
    // A strip of nothing but missing photos is still an album.
    for f in ["D1.ARW", "D3.ARW"] {
        std::fs::remove_file(shoot.join(f)).unwrap();
    }
    let c = rig();
    c.core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    assert_eq!(c.core.session.photo_ids().len(), 3);
    assert!(c.core.session.current().is_some());
}

#[test]
fn a_file_that_changed_gets_a_notice_and_its_settings_still_apply() {
    let r = rig();
    let shoot = r.root.path().join("Big");
    std::fs::create_dir_all(&shoot).unwrap();
    for (k, name) in ["G1.ARW", "G2.ARW"].iter().enumerate() {
        // About 2 MB each: larger than the two 64 KiB ends.
        let scene = Scene::new(
            1000,
            700,
            light(PROFILE, 5000.0 + 200.0 * k as f64),
            Layout::Patchwork,
            0.3,
        )
        .noise(2e-4, 1e-5);
        scene
            .builder(MODEL)
            .write_to(&shoot.join(name), &scene.mosaic())
            .expect("write");
    }
    let ids = r.open(&shoot);
    r.patch(ids[0], &SettingsPatch::exposure_ev(0.4));
    r.patch(ids[1], &SettingsPatch::exposure_ev(-0.4));
    let path = r.root.path().join("Big.ymalbum");
    r.core.album_save(&path).expect("saved");
    let expected = r.stored();
    // Change the end of a file: the tail fingerprint differs.
    let victim = shoot.join("G2.ARW");
    let mut bytes = std::fs::read(&victim).unwrap();
    let n = bytes.len();
    assert!(
        n as u64 > 4 * FINGERPRINT_BYTES,
        "the file is larger than its two ends: {n}"
    );
    bytes[n - 5] ^= 0xFF;
    std::fs::write(&victim, &bytes).unwrap();
    let b = rig();
    let out = b
        .core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    assert_eq!(
        out.photos
            .iter()
            .map(|(_, s)| s.clone())
            .collect::<Vec<_>>(),
        [AlbumPhotoState::Found, AlbumPhotoState::Changed]
    );
    assert_eq!(
        CHANGED_SENTENCE,
        "This file has changed since the album was saved."
    );
    assert_eq!(b.stored()[1].1, expected[1].1, "its settings apply");
    // The fingerprint says whether it is this file, not whether it is byte for byte the same: a change in the middle of a
    // large file is not seen.
    let first = shoot.join("G1.ARW");
    let f0 = Fingerprint::of(&first).unwrap();
    let mut bytes = std::fs::read(&first).unwrap();
    let mid = bytes.len() / 2;
    assert!(mid as u64 > FINGERPRINT_BYTES && (bytes.len() - mid) as u64 > FINGERPRINT_BYTES);
    bytes[mid] ^= 0xFF;
    std::fs::write(&first, &bytes).unwrap();
    assert_eq!(Fingerprint::of(&first).unwrap(), f0);
}

#[test]
fn a_fingerprint_is_the_size_and_the_hashes_of_the_first_and_last_64_kib() {
    let dir = tempfile::tempdir().unwrap();
    let small = dir.path().join("small.bin");
    std::fs::write(&small, b"abc").unwrap();
    let f = Fingerprint::of(&small).unwrap();
    // SHA-256("abc"), the whole file for both ends when it is smaller than 64 KiB.
    let abc = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    assert_eq!(
        (f.size, f.head_sha256.as_str(), f.tail_sha256.as_str()),
        (3, abc, abc)
    );
    let big = dir.path().join("big.bin");
    let data: Vec<u8> = (0..200_000u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(&big, &data).unwrap();
    let f = Fingerprint::of(&big).unwrap();
    assert_eq!(f.size, 200_000);
    let sha = |b: &[u8]| {
        use sha2::{Digest, Sha256};
        Sha256::digest(b)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect::<String>()
    };
    let n = FINGERPRINT_BYTES as usize;
    assert_eq!(f.head_sha256, sha(&data[..n]));
    assert_eq!(f.tail_sha256, sha(&data[data.len() - n..]));
    // A file that is not there.
    assert!(Fingerprint::of(&dir.path().join("none")).is_err());
    // An empty file.
    let empty = dir.path().join("empty.bin");
    std::fs::write(&empty, b"").unwrap();
    let e = Fingerprint::of(&empty).unwrap();
    assert_eq!(e.size, 0);
}

#[test]
fn a_photo_listed_twice_is_one_photo() {
    let (_r, path, _shoot) = trip();
    edit_album(&path, |d| {
        let first = d["photos"][0].clone();
        d["photos"].as_array_mut().unwrap().push(first);
        d["current"] = json!(3);
    });
    let b = rig();
    let out = b
        .core
        .album_open(&path, OpenOptions::default())
        .expect("opened");
    assert_eq!(out.photos.len(), 3);
    assert_eq!(b.names(), ["D1.ARW", "D2.ARW", "D3.ARW"]);
    // The current photo of the second listing is the photo's one place.
    assert_eq!(
        b.core.session.current(),
        Some(b.core.session.photo_ids()[0])
    );
}

// ---------------------------------------------------------------------------------------------
// What is refused
// ---------------------------------------------------------------------------------------------

#[test]
fn a_refused_album_leaves_the_session_as_it_was() {
    let (a, path) = ten_photos();
    a.core.album_save(&path).expect("saved");
    let ids = a.core.session.photo_ids();
    let before = (
        a.names(),
        a.stored(),
        a.core.album_state(),
        a.core.session.picked(),
    );
    let dir = tempfile::tempdir().unwrap();
    let write = |name: &str, text: &str| {
        let p = dir.path().join(name);
        std::fs::write(&p, text).unwrap();
        p
    };
    let good = read_json(&path);
    let with = |f: &dyn Fn(&mut Value)| {
        let mut d = good.clone();
        f(&mut d);
        d.to_string()
    };
    let big = dir.path().join("big.ymalbum");
    std::fs::File::create(&big)
        .unwrap()
        .set_len(MAX_ALBUM_BYTES + 1)
        .unwrap();
    for (file, sentence) in [
        (
            write("v2.ymalbum", &with(&|d| d["yellowmoon_album"] = json!(2))),
            TOO_NEW_SENTENCE,
        ),
        (
            write("extra.ymalbum", &with(&|d| d["colour"] = json!(1))),
            TOO_NEW_SENTENCE,
        ),
        (
            write(
                "field.ymalbum",
                &with(&|d| d["photos"][0]["settings"]["nodes"]["items"][0]["teleport"] = json!(1)),
            ),
            TOO_NEW_SENTENCE,
        ),
        (
            write("junk.ymalbum", "not json"),
            "This file isn't a Yellowmoon album.",
        ),
        (
            write("empty.ymalbum", ""),
            "This file isn't a Yellowmoon album.",
        ),
        (
            write("noversion.ymalbum", "{\"photos\":[]}"),
            "This file isn't a Yellowmoon album.",
        ),
        (
            write("nophotos.ymalbum", "{\"yellowmoon_album\":1}"),
            "This file isn't a Yellowmoon album.",
        ),
        (
            write(
                "flat.ymalbum",
                &with(&|d| d["photos"][0]["settings"] = json!({ "exposure": { "kind": "auto" } })),
            ),
            "This file isn't a Yellowmoon album.",
        ),
        (big, TOO_LARGE_SENTENCE),
        (
            dir.path().join("gone.ymalbum"),
            "Yellowmoon couldn't read this file. Check that it is still there, then try again.",
        ),
    ] {
        let e = a
            .core
            .album_open(&file, OpenOptions::default())
            .expect_err("refused");
        assert_eq!(e.to_string(), sentence, "{}", file.display());
        assert_eq!(e.scope(), "album");
        assert!(matches!(e, CoreError::Album(_)));
    }
    assert_eq!(
        a.core.session.photo_ids(),
        ids,
        "the session was not replaced"
    );
    assert_eq!(
        (
            a.names(),
            a.stored(),
            a.core.album_state(),
            a.core.session.picked()
        ),
        before
    );
    // More than 100,000 photos is too large too (the check is on the list, before anything is read).
    let many = format!(
        "{{\"yellowmoon_album\":1,\"photos\":[{}]}}",
        vec!["{}"; 100_001].join(",")
    );
    assert!(matches!(Album::from_json(&many), Err(AlbumError::TooLarge)));
    // An older album reads: what is missing takes its default.
    let old = r#"{"yellowmoon_album":1,"photos":[{"absolute":"/nowhere/A.ARW","settings":{}}]}"#;
    let album = Album::from_json(old).expect("reads");
    assert_eq!(album.photos[0].settings, DevelopSettings::default());
    assert!(!album.photos[0].picked && album.photos[0].path.is_none() && album.current.is_none());
    assert_eq!(album.photos[0].fingerprint, Fingerprint::unknown());
    // A number out of range is clamped like any settings document.
    let loud = r#"{"yellowmoon_album":1,"photos":[{"absolute":"/nowhere/A.ARW","settings":{"nodes":{"items":[{"id":"input","exposure":{"kind":"manual","ev":40}}]}}}]}"#;
    assert_eq!(
        Album::from_json(loud).unwrap().photos[0]
            .settings
            .input()
            .exposure,
        ym_core::settings::ExposureSetting::manual_ev(4.0)
    );
}

#[cfg(unix)]
#[test]
fn a_path_that_is_not_utf8_refuses_the_save_and_writes_nothing() {
    use std::os::unix::ffi::OsStrExt;
    let r = rig();
    let dir = r.root.path().join("Odd");
    std::fs::create_dir(&dir).unwrap();
    let scene =
        Scene::new(300, 200, light(PROFILE, 5200.0), Layout::Patchwork, 0.3).noise(2e-4, 1e-5);
    let odd = dir.join(std::ffi::OsStr::from_bytes(b"DSC\xff01.ARW"));
    if let Err(e) = scene.builder(MODEL).write_to(&odd, &scene.mosaic()) {
        // APFS (macOS) refuses a file name that is not valid UTF-8 ("Illegal byte sequence"):
        // there is nothing to test on such a file system.
        eprintln!("skipped: this file system refuses a non-UTF-8 file name ({e})");
        return;
    }
    let out = r
        .core
        .session
        .open(std::slice::from_ref(&dir), OpenOptions::default());
    assert_eq!(out.added.len(), 1);
    let path = r.root.path().join("Odd.ymalbum");
    let e = r.core.album_save(&path).expect_err("refused");
    assert!(
        matches!(&e, CoreError::Album(AlbumError::PathNotUtf8 { .. })),
        "{e:?}"
    );
    assert!(
        e.to_string()
            .starts_with("Yellowmoon can't save the path of DSC")
    );
    assert!(
        e.to_string()
            .ends_with(" in an album. Rename the file or its folder, then try again.")
    );
    assert!(!path.exists());
    assert!(
        r.core.album_state().is_none(),
        "the session is not an album"
    );
}

#[test]
fn a_write_that_fails_changes_nothing_and_says_so() {
    let (r, path, _shoot) = trip();
    let before = std::fs::read(&path).unwrap();
    r.patch(
        r.core.session.photo_ids()[0],
        &SettingsPatch::exposure_ev(1.0),
    );
    assert!(r.edited());
    // The album's folder is gone.
    let missing = r.root.path().join("Gone/Trip.ymalbum");
    let e = r.core.album_save(&missing).expect_err("fails");
    let s = e.to_string();
    assert!(
        s.starts_with("Yellowmoon couldn't save the album to "),
        "{s}"
    );
    assert!(s.ends_with(". Check that you can save there and that the disk has free space, then try again. Nothing was changed."), "{s}");
    assert!(!missing.exists());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(r.edited(), "still edited, and still the same album");
    assert_eq!(r.core.album_state().unwrap().path, path);
    // The folder holds no temporary file.
    assert_eq!(tree(path.parent().unwrap()).len(), 1);
}

// ---------------------------------------------------------------------------------------------
// The edited state
// ---------------------------------------------------------------------------------------------

#[test]
fn every_change_of_the_album_bumps_the_edited_state_a_save_clears_it_and_undo_does_not() {
    let (r, path, _shoot) = trip();
    let ids = r.core.session.photo_ids();
    assert!(!r.edited(), "a saved album is not edited");
    let save = || {
        r.core.album_save(&path).expect("saved");
        assert!(!r.edited());
    };
    // What is not a change: moving to another photo, reading, listing, asking for a preset's rings.
    r.core.session.set_current(ids[2]).unwrap();
    let _ = r.core.session.settings(ids[0]);
    let _ = r.core.presets_list(Some(ids[0]));
    let _ = r.core.session.state();
    assert!(!r.edited());

    let more = r.root.path().join("More");
    write_photos(&more, &["Z1.ARW"]);
    type Step<'a> = (&'a str, Box<dyn Fn() + 'a>);
    let mut steps: Vec<Step<'_>> = Vec::new();
    let c = &r.core;
    steps.push((
        "a slider",
        Box::new(|| {
            c.update_settings(ids[0], &SettingsPatch::exposure_ev(-0.7))
                .unwrap();
        }),
    ));
    steps.push((
        "a look",
        Box::new(|| {
            c.apply_look(ApplyScope::All, &LookId::neutral(), 75.0)
                .unwrap();
        }),
    ));
    steps.push((
        "a node patch",
        Box::new(|| {
            c.update_settings(
                ids[1],
                &SettingsPatch::default().and(
                    NodeId::Corrector(2),
                    NodePatch {
                        kind: Some(ym_core::settings::NodeKind::Corrector),
                        ..NodePatch::default()
                    },
                ),
            )
            .unwrap();
        }),
    ));
    steps.push((
        "a pick",
        Box::new(|| c.session.set_picked(ids[0], true).unwrap()),
    ));
    steps.push((
        "an unpick",
        Box::new(|| c.session.set_picked(ids[0], false).unwrap()),
    ));
    steps.push(("pick all", Box::new(|| c.session.pick_all())));
    steps.push(("clear picks", Box::new(|| c.session.clear_picks())));
    steps.push((
        "a paste",
        Box::new(|| {
            let patch = c.copy_settings(ids[0]).unwrap();
            c.paste_settings(&[ids[2]], &patch).unwrap();
        }),
    ));
    steps.push((
        "a preset applied",
        Box::new(|| {
            c.preset_save(
                "Recipe",
                &[ym_core::presets::PresetGroup::Light],
                &ym_core::presets::PresetSource::Photo { photo_id: ids[0] },
                true,
            )
            .unwrap();
            c.preset_apply("Recipe", &[ids[1]]).unwrap();
        }),
    ));
    steps.push((
        "Re-run",
        Box::new(|| {
            c.session.rerun_auto(ids[1]).unwrap();
        }),
    ));
    steps.push((
        "Auto crop's patch",
        Box::new(|| {
            c.update_settings(
                ids[2],
                &SettingsPatch::input(NodePatch {
                    crop: Some(Some(ym_core::settings::Crop {
                        rect: ym_core::settings::NormRect {
                            x: 0.0,
                            y: 0.0,
                            width: 0.5,
                            height: 0.5,
                        },
                        angle_deg: 0.0,
                        aspect: ym_core::settings::CropAspect::Free,
                        auto: true,
                    })),
                    ..NodePatch::default()
                }),
            )
            .unwrap();
        }),
    ));
    steps.push((
        "a photo removed",
        Box::new(|| {
            c.remove_photos(&[ids[2]]);
        }),
    ));
    steps.push((
        "a photo put back",
        Box::new(|| {
            c.restore_photos();
        }),
    ));
    steps.push((
        "a photo added",
        Box::new(|| {
            c.session
                .open(std::slice::from_ref(&more), OpenOptions::default());
        }),
    ));
    // The preset's recipe was written in the temporary application folder; the album is what is under test.
    for (what, step) in &steps {
        step();
        assert!(r.edited(), "{what} is a change to the album");
        save();
    }
    // Undo does not clear the state: it is one more change.
    let id = r.core.session.photo_ids()[0];
    let report = r
        .core
        .paste_settings(&[id], &SettingsPatch::exposure_ev(1.9))
        .unwrap();
    assert!(r.edited());
    save();
    r.core
        .update_settings(id, &report.pasted[0].before)
        .unwrap();
    assert!(r.edited(), "undo is a change too");
    save();
    // Bumps are counted.
    let count = r.core.session.edit_count();
    r.core.session.set_picked(id, true).unwrap();
    r.core.session.set_picked(id, true).unwrap();
    assert_eq!(
        r.core.session.edit_count(),
        count + 1,
        "picking a picked photo changes nothing"
    );
}

#[test]
fn the_albums_listeners_hear_when_it_becomes_edited_is_saved_opened_and_closed() {
    let (r, path, _shoot) = trip();
    let heard = Arc::new(AtomicUsize::new(0));
    let h = Arc::clone(&heard);
    r.core.on_album_changed(move || {
        h.fetch_add(1, Ordering::SeqCst);
    });
    let id = r.core.session.photo_ids()[0];
    r.patch(id, &SettingsPatch::exposure_ev(0.9));
    assert_eq!(heard.load(Ordering::SeqCst), 1, "it became edited");
    r.patch(id, &SettingsPatch::exposure_ev(0.8));
    r.core.session.set_picked(id, true).unwrap();
    assert_eq!(
        heard.load(Ordering::SeqCst),
        1,
        "already edited: nothing new to say"
    );
    r.core.album_save(&path).unwrap();
    assert_eq!(heard.load(Ordering::SeqCst), 2, "saved");
    r.core.album_open(&path, OpenOptions::default()).unwrap();
    assert_eq!(
        heard.load(Ordering::SeqCst),
        4,
        "the old album closed and the new one opened"
    );
    r.core.close_session();
    assert_eq!(heard.load(Ordering::SeqCst), 5, "closed");
    assert!(r.core.album_state().is_none());
}

#[test]
fn a_session_opened_from_photos_is_no_album_until_it_is_saved_and_closing_it_is_as_before() {
    let r = rig();
    let dir = r.root.path().join("Plain");
    write_photos(&dir, &["N1.ARW"]);
    let ids = r.open(&dir);
    assert!(r.core.album_state().is_none());
    r.patch(ids[0], &SettingsPatch::exposure_ev(0.3));
    assert!(r.core.album_state().is_none(), "editing is no album");
    let path = r.root.path().join("Plain.ymalbum");
    r.core.album_save(&path).unwrap();
    assert_eq!(r.core.album_state(), Some(AlbumState::new(&path, false)));
    assert_eq!(r.core.album_state().unwrap().name, "Plain");
    r.core.close_session();
    assert!(r.core.album_state().is_none());
}

// ---------------------------------------------------------------------------------------------
// Nothing else is written
// ---------------------------------------------------------------------------------------------

#[test]
fn a_whole_cycle_writes_only_the_album_and_nothing_in_the_application_folder() {
    let (a, path) = ten_photos();
    a.core.album_save(&path).unwrap();
    let b = rig();
    b.core.album_open(&path, OpenOptions::default()).unwrap();
    b.settle();
    let id = b.core.session.photo_ids()[0];
    b.patch(id, &SettingsPatch::exposure_ev(0.4));
    let again = b.root.path().join("Again.ymalbum");
    b.core.album_save(&again).unwrap();
    b.core.close_session();
    // The application folders hold no album, no list of recent albums and no autosave: nothing at all.
    assert!(tree(a.app.path()).is_empty(), "{:?}", tree(a.app.path()));
    assert!(tree(b.app.path()).is_empty(), "{:?}", tree(b.app.path()));
    // The albums are where the person put them; the photos are untouched.
    let names: BTreeSet<String> = tree(b.root.path())
        .into_iter()
        .map(|(n, _)| n)
        .filter(|n| n.ends_with(".ymalbum"))
        .collect();
    assert_eq!(names, BTreeSet::from(["Again.ymalbum".to_owned()]));
    assert!(
        tree(a.root.path())
            .iter()
            .any(|(n, _)| n == "Albums/Lisbon.ymalbum")
    );
    // Saving writes no temporary file beside the album.
    assert_eq!(tree(&a.root.path().join("Albums")).len(), 1);
}

#[test]
fn saving_never_touches_an_original() {
    let (r, path, shoot) = trip();
    let files: Vec<_> = ["D1.ARW", "D2.ARW", "D3.ARW"]
        .iter()
        .map(|n| {
            (
                shoot.join(n),
                std::fs::read(shoot.join(n)).unwrap(),
                std::fs::metadata(shoot.join(n))
                    .unwrap()
                    .modified()
                    .unwrap(),
            )
        })
        .collect();
    r.patch(
        r.core.session.photo_ids()[0],
        &SettingsPatch::exposure_ev(0.6),
    );
    r.core.album_save(&path).unwrap();
    let b = rig();
    b.core.album_open(&path, OpenOptions::default()).unwrap();
    b.settle();
    for (p, bytes, modified) in files {
        assert_eq!(std::fs::read(&p).unwrap(), bytes);
        assert_eq!(std::fs::metadata(&p).unwrap().modified().unwrap(), modified);
    }
    assert_eq!(
        tree(&shoot).len(),
        3,
        "nothing was written next to the photos"
    );
}

// ---------------------------------------------------------------------------------------------
// The schema and the wire types
// ---------------------------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .unwrap()
        .to_path_buf()
}

fn validator() -> jsonschema::Validator {
    let read = |name: &str| -> Value {
        serde_json::from_str(
            &std::fs::read_to_string(repo_root().join("crates/ym-core/schema").join(name)).unwrap(),
        )
        .unwrap()
    };
    let registry = jsonschema::Registry::new()
        .add(
            "https://yellowmoon.app/schema/settings.schema.json",
            read("settings.schema.json"),
        )
        .expect("a uri")
        .prepare()
        .expect("a registry");
    jsonschema::options()
        .with_registry(&registry)
        .build(&read("album.schema.json"))
        .expect("the schema is a valid schema")
}

fn problems(v: &jsonschema::Validator, doc: &Value) -> Vec<String> {
    v.iter_errors(doc)
        .take(5)
        .map(|e| format!("{}: {e}", e.instance_path()))
        .collect()
}

#[test]
fn what_the_core_writes_validates_against_the_schema_and_what_it_refuses_does_not() {
    let v = validator();
    // With answers, and with every kind of setting.
    let a = rig_with(fake_advisors());
    let dir = a.root.path().join("Schema");
    write_photos(&dir, &["S1.ARW", "S2.ARW"]);
    let ids = a.open(&dir);
    a.set(ids[0], rich());
    let path = a.root.path().join("S.ymalbum");
    a.core.album_save(&path).unwrap();
    let doc = read_json(&path);
    assert!(problems(&v, &doc).is_empty(), "{:?}", problems(&v, &doc));
    assert!(doc["photos"][0].get("auto").is_some());
    let (b, path2) = ten_photos();
    b.core.album_save(&path2).unwrap();
    let doc2 = read_json(&path2);
    assert!(problems(&v, &doc2).is_empty(), "{:?}", problems(&v, &doc2));
    let with = |f: &dyn Fn(&mut Value)| {
        let mut d = doc.clone();
        f(&mut d);
        d
    };
    for (i, bad) in [
        with(&|d| d["yellowmoon_album"] = json!(2)),
        with(&|d| d["colour"] = json!(1)),
        with(&|d| {
            d.as_object_mut().unwrap().remove("photos");
        }),
        with(&|d| d["photos"][0]["picked"] = json!("yes")),
        with(&|d| d["photos"][0]["teleport"] = json!(1)),
        with(&|d| {
            d["photos"][0]
                .as_object_mut()
                .unwrap()
                .remove("fingerprint");
        }),
        with(&|d| d["photos"][0]["fingerprint"]["head_sha256"] = json!("XYZ")),
        with(&|d| d["photos"][0]["settings"]["exposure"] = json!(1)),
        with(&|d| d["photos"][0]["auto"]["provider"] = json!("yellowmoon")),
        with(&|d| d["photos"][0]["auto"]["result"] = json!("failed")),
        with(&|d| d["current"] = json!(-1)),
        with(&|d| d["written_by"]["extra"] = json!(1)),
    ]
    .iter()
    .enumerate()
    {
        assert!(!problems(&v, bad).is_empty(), "document {i} validates");
    }
}

fn ts_keys(source: &str, name: &str) -> BTreeSet<String> {
    let start = source
        .find(&format!("export interface {name} {{"))
        .unwrap_or_else(|| panic!("interface {name} in types.ts"));
    let body = &source[start..];
    let end = body.find("\n}").expect("end of the interface");
    body[..end]
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.strip_prefix("  ")?;
            if line.starts_with(' ') || line.starts_with('/') || line.starts_with('*') {
                return None;
            }
            let (key, _) = line.split_once(':')?;
            Some(key.trim_end_matches('?').to_owned())
        })
        .collect()
}

#[test]
fn the_album_state_has_the_fields_of_the_typescript_interface() {
    let path = repo_root().join("src/api/types.ts");
    let Ok(source) = std::fs::read_to_string(&path) else {
        eprintln!("{} is not there; skipping", path.display());
        return;
    };
    let state =
        serde_json::to_value(AlbumState::new(Path::new("/tmp/Lisbon.ymalbum"), true)).unwrap();
    let keys: BTreeSet<String> = state.as_object().unwrap().keys().cloned().collect();
    assert_eq!(ts_keys(&source, "AlbumState"), keys);
    assert_eq!(state["name"], "Lisbon");
    assert!(source.contains("| \"album\""), "the album error scope");
}

#[test]
fn the_album_api_round_trips_without_a_session() {
    // `to_json` and `from_json` are inverses, and `write_album` is the file the core reads back.
    let (a, path) = ten_photos();
    let album = Album::capture(&a.core.session, &path).expect("captured");
    assert_eq!(album.photos.len(), 10);
    assert_eq!(album.current, Some(6));
    let text = album.to_json().unwrap();
    assert_eq!(Album::from_json(&text).unwrap(), album);
    assert_eq!(Album::from_json(&text).unwrap().to_json().unwrap(), text);
    let out = a.root.path().join("Direct.ymalbum");
    write_album(&out, &album).unwrap();
    assert_eq!(std::fs::read_to_string(&out).unwrap(), text);
    assert_eq!(Album::read_file(&out).unwrap(), album);
}
