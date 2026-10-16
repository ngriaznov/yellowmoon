//! The look library service: listing, importing, removing, the default look, missing looks, the
//! display rendering, `apply_look` scopes and the `looks_changed` notification.
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::*;
use ym_color::look::LookInput;
use ym_color::lut::Lut3d;
use ym_core::looks::{ImportRequest, LookService, LooksError};
use ym_core::pipeline::{DisplayChoice, LookProvider};
use ym_core::preferences::{DisplayRendering, PreferencesPatch, PreferencesStore};
use ym_core::session::{ApplyScope, OpenOptions, Session};
use ym_looks::library::LookId;
use ym_looks::lock::LibraryLock;

struct Fixture {
    root: tempfile::TempDir,
    service: LookService,
    changes: Arc<AtomicUsize>,
}

fn fixture() -> Fixture {
    let root = tempfile::tempdir().expect("app root");
    let service = service_in(root.path());
    let changes = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&changes);
    service.on_change(move || {
        c.fetch_add(1, Ordering::SeqCst);
    });
    Fixture {
        root,
        service,
        changes,
    }
}

fn service_in(root: &Path) -> LookService {
    LookService::open(root, &data_root(), Arc::new(PreferencesStore::open(root))).expect("service")
}

fn write_cube(dir: &Path, name: &str, size: usize) -> PathBuf {
    let path = dir.join(name);
    let text = Lut3d::identity(size).expect("cube").to_cube_string(name);
    std::fs::write(&path, text).expect("write cube");
    path
}

fn import_cube(f: &Fixture, path: &Path) -> Result<ym_core::looks::LooksImportResult, LooksError> {
    f.service.import(
        path,
        &ImportRequest {
            input: Some(LookInput::LogC4),
            cancel: None,
        },
        &mut |_| {},
    )
}

fn changes(f: &Fixture) -> usize {
    f.changes.load(Ordering::SeqCst)
}

#[test]
fn the_built_in_looks_are_listed_and_signature_is_the_default() {
    let f = fixture();
    let list = f.service.list();
    // Neutral first, then Signature; both are built in and usable.
    let ids: Vec<&str> = list.iter().map(|l| l.id.as_str()).collect();
    assert_eq!(&ids[..2], ["builtin:neutral", "builtin:signature"]);
    let neutral = &list[0];
    assert_eq!(neutral.name, "Neutral");
    assert!(!neutral.is_default && !neutral.missing);
    assert_eq!(neutral.input, LookInput::LogC4);
    let signature = &list[1];
    assert_eq!(signature.name, "Signature");
    assert!(signature.is_default && !signature.missing);
    assert_eq!(signature.input, LookInput::LogC4);
    assert_eq!(list.iter().filter(|l| l.is_default).count(), 1);
    assert!(f.service.display_luts().is_empty());
    assert_eq!(f.service.effective_default().as_str(), "builtin:signature");
    assert_eq!(f.service.display_choice(), DisplayChoice::Aces2);
    // Both built-in looks load: Neutral is the identity, Signature moves colour but not mid grey.
    let provider = f.service.provider();
    let neutral_look = provider.look(&LookId::neutral()).expect("Neutral loads");
    assert!(neutral_look.transform.is_identity());
    let signature_look = provider
        .look(&LookId::factory_default())
        .expect("Signature loads");
    assert!(!signature_look.transform.is_identity());
    let warm = ym_color::logc::LogC4([0.5, 0.4, 0.3]);
    assert_ne!(
        signature_look.transform.eval_logc4(warm).expect("log look"),
        warm
    );
    // Nothing was written by looking.
    assert!(
        std::fs::read_dir(f.root.path())
            .expect("dir")
            .next()
            .is_none()
    );
}

/// A data folder with no `signature.json` (a broken bundle, a test): the default look is Neutral,
/// which exists in code, and the list says so.
#[test]
fn without_the_signature_file_the_default_is_neutral() {
    let data = tempfile::tempdir().expect("data");
    let builtin = data.path().join("looks").join("builtin");
    std::fs::create_dir_all(&builtin).expect("dir");
    std::fs::copy(
        data_root()
            .join("looks")
            .join("builtin")
            .join("neutral.json"),
        builtin.join("neutral.json"),
    )
    .expect("neutral.json");
    let root = tempfile::tempdir().expect("app root");
    let service = LookService::open(
        root.path(),
        data.path(),
        Arc::new(PreferencesStore::open(root.path())),
    )
    .expect("service");
    let list = service.list();
    assert_eq!(
        list.iter().map(|l| l.id.as_str()).collect::<Vec<_>>(),
        ["builtin:neutral"]
    );
    assert!(list[0].is_default);
    assert_eq!(service.effective_default().as_str(), "builtin:neutral");
    // The preference itself still says Signature; the file coming back makes it the default again.
    assert_eq!(
        PreferencesStore::open(root.path()).get().default_look,
        LookId::factory_default()
    );
}

#[test]
fn importing_a_cube_adds_a_look_notifies_once_and_is_idempotent() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    let cube = write_cube(src.path(), "Warm Film.cube", 5);

    let result = import_cube(&f, &cube).expect("import");
    assert_eq!(result.added.len(), 1);
    assert_eq!((result.display_luts_added, result.unchanged), (0, 0));
    let look = &result.added[0];
    assert_eq!(look.id.as_str(), "user:warm-film");
    assert_eq!(look.name, "Warm Film");
    assert!(!look.missing && !look.is_default);
    assert_eq!(changes(&f), 1, "looks_changed fired once");
    assert!(f.service.list().iter().any(|l| l.id == look.id));
    // The file was copied into the library and the manifest records its hash.
    assert!(f.root.path().join("looks").join("manifest.json").is_file());
    let stored = f
        .root
        .path()
        .join("looks")
        .join("user")
        .join("Warm Film.cube");
    assert_eq!(
        std::fs::read(&stored).expect("copy"),
        std::fs::read(&cube).expect("src")
    );
    // The render graph can use it, and an export can record its hash.
    let provider = f.service.provider();
    assert!(provider.look(&look.id).is_ok());
    let sha = provider.look_sha256(&look.id).expect("sha");
    assert_eq!(sha.len(), 64);
    // Built-in Neutral and Signature have a hash too: the one of their description files.
    for built_in in [LookId::neutral(), LookId::factory_default()] {
        assert_eq!(provider.look_sha256(&built_in).map(|s| s.len()), Some(64));
    }

    // Again: nothing added, nothing notified.
    let again = import_cube(&f, &cube).expect("import again");
    assert!(again.added.is_empty());
    assert_eq!(again.unchanged, 1);
    assert_eq!(changes(&f), 1);
}

#[test]
fn a_cube_without_an_answer_asks_the_question() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    let cube = write_cube(src.path(), "Mystery.cube", 5);
    let err = f
        .service
        .import(&cube, &ImportRequest::default(), &mut |_| {})
        .expect_err("needs the input");
    assert!(
        err.to_string()
            .starts_with("Say what these LUTs expect as input"),
        "{err}"
    );
    assert_eq!(err.exit_code(), 2);
    assert_eq!(changes(&f), 0);
    assert!(!f.root.path().join("looks").exists() || f.service.list().len() == 1);
    // A path that is not there, and a file that is not a LUT.
    let missing = f.service.import(
        &src.path().join("nope.cube"),
        &ImportRequest::default(),
        &mut |_| {},
    );
    assert!(matches!(missing, Err(LooksError::Import(_))));
    let junk = src.path().join("junk.cube");
    std::fs::write(&junk, "not a lut").expect("junk");
    assert!(import_cube(&f, &junk).is_err());
    assert_eq!(changes(&f), 0);
}

#[test]
fn the_default_look_is_a_preference_and_a_missing_one_is_not_used() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    let id = import_cube(&f, &write_cube(src.path(), "Warm.cube", 5))
        .expect("import")
        .added[0]
        .id
        .clone();
    let before = changes(&f);
    f.service.set_default(&id).expect("default");
    assert_eq!(changes(&f), before + 1);
    assert_eq!(f.service.effective_default(), id);
    let list = f.service.list();
    assert!(list.iter().find(|l| l.id == id).expect("look").is_default);
    assert!(
        !list
            .iter()
            .find(|l| l.id.as_str() == "builtin:signature")
            .expect("s")
            .is_default
    );
    // Persisted: a new store over the same folder sees it.
    assert_eq!(PreferencesStore::open(f.root.path()).get().default_look, id);
    // Setting it again changes nothing.
    f.service.set_default(&id).expect("again");
    assert_eq!(changes(&f), before + 1);

    // The look's file goes away. After a verify it is flagged and the default falls back.
    std::fs::remove_file(f.root.path().join("looks").join("user").join("Warm.cube")).expect("rm");
    f.service.verify().expect("verify");
    let missing = f
        .service
        .list()
        .into_iter()
        .find(|l| l.id == id)
        .expect("listed");
    assert!(
        missing.missing,
        "a look without its file stays in the list, disabled"
    );
    assert_eq!(f.service.effective_default().as_str(), "builtin:signature");
    assert!(
        f.service
            .list()
            .iter()
            .find(|l| l.id.as_str() == "builtin:signature")
            .expect("s")
            .is_default
    );
    let err = f.service.set_default(&id).expect_err("refused");
    assert_eq!(err.to_string(), "File missing or changed. Import again.");
    assert_eq!(err.exit_code(), 3);
    let err = f
        .service
        .check_usable(&LookId::parse("user:nothing").expect("id"))
        .expect_err("unknown");
    assert!(matches!(err, LooksError::NotFound(_)));
    assert_eq!(
        err.to_string(),
        "That look isn't in the library. Pick another look."
    );
    // The render graph is told the same.
    let problem = f.service.provider().look(&id).expect_err("unusable");
    assert!(
        problem.0.contains("File missing or changed. Import again."),
        "{problem}"
    );
}

#[test]
fn removing_a_look_deletes_its_file_and_resets_what_pointed_at_it() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    let id = import_cube(&f, &write_cube(src.path(), "Warm.cube", 5))
        .expect("import")
        .added[0]
        .id
        .clone();
    f.service.set_default(&id).expect("default");
    let before = changes(&f);
    f.service.remove(&id).expect("removed");
    assert_eq!(changes(&f), before + 1);
    assert!(f.service.list().iter().all(|l| l.id != id));
    assert!(
        !f.root
            .path()
            .join("looks")
            .join("user")
            .join("Warm.cube")
            .exists()
    );
    assert_eq!(
        PreferencesStore::open(f.root.path())
            .get()
            .default_look
            .as_str(),
        "builtin:signature"
    );
    assert_eq!(f.service.effective_default().as_str(), "builtin:signature");

    let err = f
        .service
        .remove(&LookId::builtin("neutral"))
        .expect_err("built in");
    assert!(err.to_string().contains("built in"), "{err}");
    assert!(f.service.remove(&id).is_err(), "already gone");
}

#[test]
fn apply_look_goes_to_this_photo_the_picked_photos_or_all_and_refuses_a_missing_look() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    let warm = import_cube(&f, &write_cube(src.path(), "Warm.cube", 5))
        .expect("import")
        .added[0]
        .id
        .clone();
    let photos = tempfile::tempdir().expect("photos");
    for k in 1..=4 {
        Scene::new(
            120,
            80,
            light("sony-ilce-7rm4", 5500.0),
            Layout::Uniform,
            0.4,
        )
        .write(photos.path(), &format!("DSC{k:05}.ARW"), "ILCE-7RM4");
    }
    let session = Session::new(base_env());
    let ids = session
        .open(&[photos.path().to_path_buf()], OpenOptions::default())
        .added;
    settle(&session);
    let look_of = |i: usize| session.settings(ids[i]).expect("settings").settings.look;
    let neutral = LookId::builtin("neutral");

    // This photo (the first is current).
    let changed = f
        .service
        .apply_look(&session, ApplyScope::This, &warm, 70.0)
        .expect("this");
    assert_eq!(changed, vec![ids[0]]);
    assert_eq!(look_of(0), warm);
    assert_eq!(
        session.settings(ids[0]).expect("s").settings.look_intensity,
        70.0
    );
    assert_eq!(look_of(1), neutral);
    // Picked.
    session.set_picked(ids[1], true).expect("pick");
    session.set_picked(ids[3], true).expect("pick");
    let changed = f
        .service
        .apply_look(&session, ApplyScope::Picked, &warm, 100.0)
        .expect("picked");
    assert_eq!(changed, vec![ids[1], ids[3]]);
    assert_eq!(
        (look_of(1), look_of(2), look_of(3)),
        (warm.clone(), neutral.clone(), warm.clone())
    );
    // All.
    let changed = f
        .service
        .apply_look(&session, ApplyScope::All, &neutral, 100.0)
        .expect("all");
    // The third photo already had it: only the ones that changed are reported.
    assert_eq!(changed, vec![ids[0], ids[1], ids[3]]);
    assert!((0..4).all(|i| look_of(i) == neutral));

    // A look whose file went away is not applied to anything.
    std::fs::remove_file(f.root.path().join("looks").join("user").join("Warm.cube")).expect("rm");
    f.service.verify().expect("verify");
    let err = f
        .service
        .apply_look(&session, ApplyScope::All, &warm, 100.0)
        .expect_err("refused");
    assert_eq!(err.to_string(), "File missing or changed. Import again.");
    assert!((0..4).all(|i| look_of(i) == neutral));
}

#[test]
fn a_look_can_be_found_by_id_slug_or_name() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    let id = import_cube(&f, &write_cube(src.path(), "Bleach Bypass.cube", 5))
        .expect("import")
        .added[0]
        .id
        .clone();
    for q in [
        "user:bleach-bypass",
        "bleach-bypass",
        "Bleach Bypass",
        "  bleach   BYPASS ",
    ] {
        assert_eq!(f.service.resolve(q).expect(q), id, "{q}");
    }
    assert_eq!(
        f.service.resolve("neutral").expect("n").as_str(),
        "builtin:neutral"
    );
    let err = f.service.resolve("no such look").expect_err("none");
    assert_eq!(err.exit_code(), 3);
}

#[test]
fn another_process_changing_the_library_is_picked_up_by_a_reload() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    // The CLI: a second service over the same folder.
    let cli = service_in(f.root.path());
    cli.import(
        &write_cube(src.path(), "FromCli.cube", 5),
        &ImportRequest {
            input: Some(LookInput::LogC4),
            cancel: None,
        },
        &mut |_| {},
    )
    .expect("cli import");
    assert!(
        f.service
            .list()
            .iter()
            .all(|l| l.id.as_str() != "user:fromcli")
    );
    assert!(f.service.reload(), "something changed");
    assert_eq!(changes(&f), 1);
    assert!(
        f.service
            .list()
            .iter()
            .any(|l| l.id.as_str() == "user:fromcli")
    );
    assert!(!f.service.reload(), "nothing more");
    assert_eq!(changes(&f), 1);
}

#[test]
fn reading_looks_does_not_wait_for_an_import_that_waits_for_the_library_lock() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    let cube = write_cube(src.path(), "Slow.cube", 5);
    // Another process holds the library lock.
    let looks_dir = f.root.path().join("looks");
    std::fs::create_dir_all(&looks_dir).expect("looks dir");
    let guard = LibraryLock::acquire(&looks_dir, Duration::from_secs(5)).expect("lock");
    let service = Arc::new(service_in(f.root.path()));
    let importing = {
        let service = Arc::clone(&service);
        std::thread::spawn(move || {
            service.import(
                &cube,
                &ImportRequest {
                    input: Some(LookInput::LogC4),
                    cancel: None,
                },
                &mut |_| {},
            )
        })
    };
    std::thread::sleep(Duration::from_millis(300));
    assert!(
        !importing.is_finished(),
        "the import is waiting for the lock"
    );
    // Meanwhile the renderer's reads and the UI's list answer at once.
    let started = std::time::Instant::now();
    for _ in 0..50 {
        assert!(service.provider().look(&LookId::builtin("neutral")).is_ok());
        assert!(!service.list().is_empty());
    }
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "{:?}",
        started.elapsed()
    );
    drop(guard);
    let result = importing.join().expect("thread").expect("import");
    assert_eq!(result.added.len(), 1);
}

#[test]
fn the_display_rendering_is_a_preference_that_needs_an_imported_display_lut() {
    let f = fixture();
    let err = f
        .service
        .set_display_rendering(DisplayRendering::ArriDisplay {
            id: LookId::parse("arri-display:logc4:rec709-g24").expect("id"),
        })
        .expect_err("not imported");
    assert_eq!(
        err.to_string(),
        "Those display LUTs aren't imported. Import ARRI's display LUTs first."
    );
    assert_eq!(f.service.display_choice(), DisplayChoice::Aces2);
    // Through the preferences patch as well.
    let patch = PreferencesPatch {
        display_rendering: Some(DisplayRendering::ArriDisplay {
            id: LookId::parse("arri-display:logc4:rec709-g24").expect("id"),
        }),
        ..PreferencesPatch::default()
    };
    assert!(f.service.preferences_set(&patch).is_err());
    assert_eq!(
        PreferencesStore::open(f.root.path())
            .get()
            .display_rendering,
        DisplayRendering::Aces2
    );
    // And a default look that is not in the library.
    let patch = PreferencesPatch::default_look(LookId::parse("user:nope").expect("id"));
    let err = f.service.preferences_set(&patch).expect_err("unknown look");
    assert!(matches!(err, LooksError::NotFound(_)));
}

/// A folder with two display LUTs the way ARRI names them.
fn display_luts_folder(root: &Path) -> PathBuf {
    let folder = root.join("LogC4 Display LUTs");
    std::fs::create_dir_all(&folder).expect("folder");
    for n in ["LogC4_to_Gamma24_Rec709.cube", "LogC4_to_P3_D65.cube"] {
        write_cube(&folder, n, 5);
    }
    folder
}

#[test]
fn imported_display_luts_can_be_chosen_as_the_display_rendering() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    let folder = display_luts_folder(src.path());
    let result = f
        .service
        .import(&folder, &ImportRequest::default(), &mut |_| {})
        .expect("import");
    assert!(result.added.is_empty(), "display LUTs are not looks");
    assert_eq!((result.display_luts_added, result.unchanged), (2, 0));
    assert_eq!(changes(&f), 1);
    let luts = f.service.display_luts();
    let names: Vec<_> = luts.iter().map(|d| d.name.as_str()).collect();
    assert_eq!(names, vec!["P3 · D65", "Rec.709 · Gamma 2.4"]);
    assert!(luts.iter().all(|d| d.input == LookInput::LogC4));
    assert!(
        f.service
            .list()
            .iter()
            .all(|l| !l.id.as_str().starts_with("arri-display"))
    );

    let id = luts[1].id.clone();
    assert_eq!(id.as_str(), "arri-display:logc4:rec709-g24");
    f.service
        .set_display_rendering(DisplayRendering::ArriDisplay { id: id.clone() })
        .expect("chosen");
    assert_eq!(
        f.service.display_choice(),
        DisplayChoice::ArriLut(id.clone())
    );
    assert!(f.service.provider().display_transform(&id).is_ok());
    assert_eq!(
        PreferencesStore::open(f.root.path())
            .get()
            .display_rendering,
        DisplayRendering::ArriDisplay { id: id.clone() }
    );

    // If its file goes away the ACES 2 transform is used again (the preference is kept until the
    // LUT is removed or the user changes it).
    let stored = f.root.path().join("looks").join("arri-display");
    let file = std::fs::read_dir(&stored)
        .expect("display dir")
        .flatten()
        .map(|e| e.path())
        .find(|p| p.to_string_lossy().contains("Gamma24"))
        .expect("the stored LUT");
    std::fs::remove_file(file).expect("rm");
    f.service.verify().expect("verify");
    assert_eq!(f.service.display_choice(), DisplayChoice::Aces2);
    assert!(f.service.provider().display_transform(&id).is_err());

    // Removing the LUT resets the preference.
    f.service.remove(&id).expect("removed");
    assert_eq!(
        PreferencesStore::open(f.root.path())
            .get()
            .display_rendering,
        DisplayRendering::Aces2
    );
    assert_eq!(f.service.display_luts().len(), 1);
}

#[test]
fn a_damaged_looks_list_is_logged_without_the_path_of_the_persons_folder() {
    // The problem's own text names the looks list in the person's library, and the application's
    // log can end up in a bug report (`docs/01`): the warning says what is wrong, not where.
    captured_log(); // installs the capture before the work starts
    let root = tempfile::tempdir().expect("app root");
    let looks = root.path().join("looks");
    std::fs::create_dir_all(&looks).expect("looks folder");
    std::fs::write(looks.join("manifest.json"), b"{ not json").expect("damaged list");

    let service = service_in(root.path());
    assert!(
        service
            .list()
            .iter()
            .all(|l| l.id.as_str().starts_with("builtin:")),
        "the built-in looks are listed, no imported one"
    );

    let log = captured_log();
    let mine: Vec<&str> = log
        .lines()
        .filter(|l| l.contains("the look library has a problem"))
        .collect();
    assert!(
        mine.iter().any(|l| l.contains("manifest")),
        "the problem is logged in:\n{log}"
    );
    let folder = root.path().display().to_string();
    assert!(
        !log.contains(&folder),
        "the person's folder reached the log:\n{log}"
    );
    assert!(!log.contains("manifest.json"), "{log}");
}

/// A data folder like the real one, but with a made-up fingerprint for `bytes`: the LogC4 library's
/// look 1110 "Gray Test". (The real fingerprints are ARRI's hashes, which no synthetic file has.)
fn data_with_a_fingerprint_for(bytes: &[u8]) -> tempfile::TempDir {
    let real = data_root();
    let data = tempfile::tempdir().expect("data");
    let builtin = data.path().join("looks").join("builtin");
    std::fs::create_dir_all(&builtin).expect("dir");
    for name in ["neutral.json", "signature.json"] {
        std::fs::copy(
            real.join("looks").join("builtin").join(name),
            builtin.join(name),
        )
        .expect("copy");
    }
    let text = std::fs::read_to_string(real.join("looks").join("arri-packages.json"))
        .expect("package list");
    let mut list: serde_json::Value = serde_json::from_str(&text).expect("json");
    for package in list["packages"].as_array_mut().expect("packages") {
        package["fingerprints"] = if package["id"] == "look-library-logc4" {
            serde_json::json!([{
                "path": "ARRI Look Library for LogC4 (3D-LUTs)/1110 Gray Test.cube",
                "size": bytes.len(),
                "lut_size": 5,
                "sha256": ym_looks::manifest::sha256_hex(bytes),
            }])
        } else {
            serde_json::json!([])
        };
    }
    std::fs::write(
        data.path().join("looks").join("arri-packages.json"),
        serde_json::to_vec(&list).expect("json"),
    )
    .expect("write");
    data
}

#[test]
fn an_own_look_that_an_arri_package_claims_stays_the_default_look() {
    // 0.1.0 and 0.1.1 did not know ARRI's files, so people imported them as "Your LUTs" and made
    // one the default. Importing the real package now turns that entry into the ARRI look, with
    // its ARRI id: the default look follows it instead of quietly becoming Signature.
    let f = fixture();
    let root = f.root.path();
    let src = tempfile::tempdir().expect("src");
    let cube = write_cube(src.path(), "Warm.cube", 5);
    let own = import_cube(&f, &cube).expect("own").added[0].id.clone();
    assert_eq!(own.as_str(), "user:warm");
    f.service.set_default(&own).expect("default");

    // The same bytes are now the LogC4 library's look 1110.
    let data = data_with_a_fingerprint_for(&std::fs::read(&cube).expect("bytes"));
    let prefs = Arc::new(PreferencesStore::open(root));
    let service = LookService::open(root, data.path(), Arc::clone(&prefs)).expect("service");
    let changes = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&changes);
    service.on_change(move || {
        c.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(service.effective_default(), own);

    let result = service
        .import(
            &cube,
            &ImportRequest {
                input: None,
                cancel: None,
            },
            &mut |_| {},
        )
        .expect("import");
    assert_eq!(result.added.len(), 1);
    let arri = result.added[0].id.clone();
    assert_eq!(arri.as_str(), "arri:1000:gray-test");
    // The entry moved: no `user:warm` any more, and the default look is the ARRI one.
    assert!(service.list().iter().all(|l| l.id != own));
    assert_eq!(prefs.get().default_look, arri);
    assert_eq!(service.effective_default(), arri);
    assert!(service.list().iter().any(|l| l.id == arri && l.is_default));
    assert_eq!(
        PreferencesStore::open(root).get().default_look,
        arri,
        "kept on disk"
    );
    // And the person is told, with the ids a script would name.
    assert!(
        result
            .notes
            .iter()
            .any(|n| n.contains("user:warm is now arri:1000:gray-test.")),
        "{:?}",
        result.notes
    );
    assert!(
        result
            .notes
            .iter()
            .any(|n| n.contains("Your default look user:warm is now arri:1000:gray-test")),
        "{:?}",
        result.notes
    );
    assert!(
        !result
            .notes
            .iter()
            .any(|n| n.contains("replaced by the 3D-LUT version")),
        "{:?}",
        result.notes
    );
    assert_eq!(changes.load(Ordering::SeqCst), 1, "one looks_changed");
}

#[test]
fn a_default_look_that_was_not_claimed_is_left_alone() {
    let f = fixture();
    let root = f.root.path();
    let src = tempfile::tempdir().expect("src");
    let cube = write_cube(src.path(), "Warm.cube", 5);
    let other = write_cube(src.path(), "Other.cube", 3);
    let own = import_cube(&f, &other).expect("own").added[0].id.clone();
    f.service.set_default(&own).expect("default");

    let data = data_with_a_fingerprint_for(&std::fs::read(&cube).expect("bytes"));
    let prefs = Arc::new(PreferencesStore::open(root));
    let service = LookService::open(root, data.path(), Arc::clone(&prefs)).expect("service");
    // Warm is imported as the ARRI look; nothing was claimed, so the default stays where it was.
    let result = service
        .import(
            &cube,
            &ImportRequest {
                input: None,
                cancel: None,
            },
            &mut |_| {},
        )
        .expect("import");
    assert_eq!(result.added[0].id.as_str(), "arri:1000:gray-test");
    assert_eq!(prefs.get().default_look, own);
    assert!(result.notes.is_empty(), "{:?}", result.notes);
}

#[test]
fn files_that_were_refused_while_the_rest_came_in_are_said_with_their_reasons() {
    let f = fixture();
    let src = tempfile::tempdir().expect("src");
    write_cube(src.path(), "Good.cube", 5);
    std::fs::write(src.path().join("Torn.cube"), "LUT_3D_SIZE 5\n0 0 0\n").expect("torn");
    let result = import_cube(&f, src.path()).expect("the good one is imported");
    assert_eq!(result.added.len(), 1);
    assert_eq!(result.added[0].id.as_str(), "user:good");
    assert_eq!(result.refusals.len(), 1, "{:?}", result.refusals);
    assert!(
        result.refusals[0].starts_with("Torn.cube cannot be used. "),
        "{:?}",
        result.refusals
    );
    // Said once, among the refusals and not among the notes.
    assert!(
        result.notes.iter().all(|n| !n.contains("Torn.cube")),
        "{:?}",
        result.notes
    );
    // A clean import has none.
    let clean = tempfile::tempdir().expect("clean");
    write_cube(clean.path(), "Fine.cube", 5);
    assert!(
        import_cube(&f, clean.path())
            .expect("import")
            .refusals
            .is_empty()
    );
}
