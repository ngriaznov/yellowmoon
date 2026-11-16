//! The `presets/` folder (`docs/30` section 6): atomic writes under the lock, names and file names, the sorted list, files that
//! cannot be read, rename, replace, import, export, the Trash and the folder's place.

mod common;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::rich::rich;
use ym_core::presets::{
    ImportExists, Preset, PresetError, PresetGroup, PresetStore, TOO_NEW_SENTENCE,
};
use ym_looks::import::trash::TrashError;
use ym_looks::library::{Platform, root_from};
use ym_looks::lock::LibraryLock;

fn preset(name: &str) -> Preset {
    Preset::from_settings(
        name,
        &rich(),
        &[PresetGroup::Light, PresetGroup::Color],
        None,
    )
    .expect("a preset")
}

fn other(name: &str) -> Preset {
    let mut s = rich();
    s.main_grade_mut().brightness = 33.0;
    Preset::from_settings(name, &s, &[PresetGroup::Light], None).expect("a preset")
}

fn rig() -> (tempfile::TempDir, PresetStore) {
    let root = tempfile::tempdir().expect("root");
    let store = PresetStore::open(root.path()).with_lock_timeout(Duration::from_millis(300));
    (root, store)
}

fn files(folder: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(folder)
        .map(|rd| {
            rd.flatten()
                .map(|e| e.file_name().to_string_lossy().into_owned())
                .collect()
        })
        .unwrap_or_default();
    v.sort();
    v
}

type TrashLog = Arc<Mutex<Vec<PathBuf>>>;

/// A Trash that moves the file into `dir` and remembers what it was asked to take.
fn fake_trash(
    dir: PathBuf,
) -> (
    impl Fn(&Path) -> Result<(), TrashError> + Send + Sync + 'static,
    TrashLog,
) {
    let log = Arc::new(Mutex::new(Vec::new()));
    let seen = Arc::clone(&log);
    (
        move |path: &Path| {
            std::fs::create_dir_all(&dir).expect("trash");
            let dest = dir.join(path.file_name().expect("name"));
            std::fs::rename(path, dest).expect("move");
            seen.lock().expect("log").push(path.to_path_buf());
            Ok(())
        },
        log,
    )
}

// ---------------------------------------------------------------------------------------------
// The folder
// ---------------------------------------------------------------------------------------------

#[test]
fn the_folder_is_presets_in_the_app_folder_and_nothing_is_created_until_a_write() {
    let (root, store) = rig();
    assert_eq!(store.folder(), root.path().join("presets"));
    let listing = store.list();
    assert!(listing.presets.is_empty() && listing.damaged.is_empty());
    assert!(store.get("Anything").is_err());
    assert!(store.path_of("Anything").is_none());
    assert!(
        !root.path().join("presets").exists(),
        "a read creates nothing"
    );
    store.save(&preset("Warm evening"), false).expect("saved");
    assert_eq!(
        files(&root.path().join("presets")),
        [".lock", "Warm evening.ympreset"],
        "the preset and the lock, no index, no cache, no temporary file"
    );
}

#[test]
fn yellowmoon_home_replaces_the_app_folder() {
    let home = tempfile::tempdir().expect("home");
    let value = OsString::from(home.path());
    let root = root_from(
        |key| (key == "YELLOWMOON_HOME").then(|| value.clone()),
        Platform::Unix,
    )
    .expect("a root");
    assert_eq!(root, home.path());
    let store = PresetStore::open(&root).with_lock_timeout(Duration::from_millis(300));
    store.save(&preset("Here"), false).expect("saved");
    assert!(home.path().join("presets").join("Here.ympreset").is_file());
    // Beside `looks/`, in the same folder.
    assert_eq!(store.folder().parent(), Some(home.path()));
}

// ---------------------------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------------------------

#[test]
fn a_name_is_trimmed_at_most_64_characters_and_unique_ignoring_case() {
    let (_root, store) = rig();
    let saved = store
        .save(&preset("  Warm evening  "), false)
        .expect("saved");
    assert_eq!(saved.preset.name, "Warm evening");
    assert_eq!(saved.file, "Warm evening.ympreset");
    let e = store
        .save(&preset("warm EVENING"), false)
        .expect_err("exists");
    assert!(matches!(e, PresetError::Exists(_)));
    assert_eq!(
        e.to_string(),
        "A preset called warm EVENING already exists."
    );
    assert_eq!(e.exit_code(), 2);
    assert_eq!(store.list().presets.len(), 1);
    // The name in the file is the truth, and a lookup ignores case.
    assert_eq!(
        store.get("WARM evening").expect("found").name,
        "Warm evening"
    );
    // 64 characters are fine, in any script; 65 are not.
    let long = "é".repeat(64);
    store.save(&preset(&long), false).expect("64 characters");
    assert!(Preset::from_settings(&"é".repeat(65), &rich(), &[PresetGroup::Light], None).is_err());
    for name in ["Café ☕", "日本語のプリセット", "Ünïcödé"] {
        store.save(&preset(name), false).expect(name);
        assert_eq!(store.get(name).expect("found").name, name);
    }
    assert!(store.get("café ☕").is_ok(), "Unicode names fold case too");
    assert!(matches!(
        store.save(&preset("CAFÉ ☕"), false),
        Err(PresetError::Exists(_))
    ));
}

#[test]
fn the_file_name_is_the_name_made_safe_for_every_file_system() {
    let safe = PresetStore::safe_file_name;
    assert_eq!(safe("Warm evening"), "Warm evening");
    assert_eq!(safe("a/b"), "a-b");
    assert_eq!(safe(r"a\b"), "a-b");
    assert_eq!(safe(r#"a:b*c?d"e<f>g|h"#), "a-b-c-d-e-f-g-h");
    assert_eq!(safe("a\u{7}b\tc"), "a-b-c", "control characters");
    assert_eq!(safe("a//b"), "a-b", "runs of dashes");
    assert_eq!(safe("a    b"), "a b", "runs of spaces");
    assert_eq!(
        safe("a - b"),
        "a - b",
        "a space, a dash and a space are not runs"
    );
    assert_eq!(
        safe("  .Hidden. "),
        "Hidden",
        "leading and trailing spaces and dots"
    );
    assert_eq!(safe("."), "preset");
    assert_eq!(safe(".."), "preset");
    assert_eq!(safe("   "), "preset");
    assert_eq!(safe("..."), "preset");
    let long = safe(&"é".repeat(150));
    assert_eq!(long.len(), 200, "cut to 200 bytes at a character boundary");
    assert_eq!(long.chars().count(), 100);
    let odd = safe(&format!("{}x", "é".repeat(99)));
    assert!(odd.len() <= 200 && odd.is_char_boundary(odd.len()));
    assert!(safe(&"a".repeat(300)).len() == 200);
}

#[test]
fn two_names_that_make_one_file_name_get_a_number_on_the_newer() {
    let (root, store) = rig();
    let a = store.save(&preset("a/b"), false).unwrap();
    let b = store.save(&preset("a:b"), false).unwrap();
    let c = store.save(&preset("A?B"), false).unwrap();
    let d = store.save(&preset("a|b"), false).unwrap();
    assert_eq!(a.file, "a-b.ympreset");
    assert_eq!(b.file, "a-b 2.ympreset");
    assert_eq!(
        c.file, "A-B 3.ympreset",
        "a case-insensitive disk would have one file for a-b and A-B"
    );
    assert_eq!(d.file, "a-b 4.ympreset");
    let names: Vec<_> = store
        .list()
        .presets
        .iter()
        .map(|s| s.preset.name.clone())
        .collect();
    assert_eq!(names, ["a/b", "a:b", "A?B", "a|b"]);
    assert_eq!(files(&root.path().join("presets")).len(), 5);
}

// ---------------------------------------------------------------------------------------------
// The list
// ---------------------------------------------------------------------------------------------

#[test]
fn the_list_is_sorted_by_name_ignoring_case() {
    let (_root, store) = rig();
    for name in ["banana", "Cherry", "apple", "Banana split", "Apple pie"] {
        store.save(&preset(name), false).unwrap();
    }
    let names: Vec<_> = store
        .list()
        .presets
        .iter()
        .map(|s| s.preset.name.clone())
        .collect();
    assert_eq!(
        names,
        ["apple", "Apple pie", "banana", "Banana split", "Cherry"]
    );
    // The list reads the folder every time: a file put there by hand shows up.
    let text = preset("From ym").to_json().unwrap();
    std::fs::write(store.folder().join("From ym.ympreset"), text).unwrap();
    assert_eq!(store.list().presets.len(), 6);
    assert!(store.get("from YM").is_ok());
    // A revision is the first 12 hex digits of the file's SHA-256, and changes with the file.
    let a = store.get_stored("apple").unwrap();
    assert_eq!(a.revision.len(), 12);
    assert!(a.revision.bytes().all(|b| b.is_ascii_hexdigit()));
    let b = store.save(&other("apple"), true).unwrap();
    assert_ne!(
        a.revision, b.revision,
        "a preset that is saved again has a new revision"
    );
}

#[test]
fn a_file_that_cannot_be_read_is_listed_as_damaged_and_never_stops_the_others() {
    let (_root, store) = rig();
    store.save(&preset("Good one"), false).unwrap();
    let folder = store.folder().to_path_buf();
    std::fs::write(folder.join("junk.ympreset"), "not json").unwrap();
    std::fs::write(folder.join("binary.ympreset"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
    std::fs::write(
        folder.join("future.ympreset"),
        r#"{"yellowmoon_preset":2,"name":"Future"}"#,
    )
    .unwrap();
    std::fs::write(folder.join("big.ympreset"), vec![b' '; 4 * 1024 * 1024 + 1]).unwrap();
    // The same name copied in by hand under another file name.
    std::fs::copy(
        folder.join("Good one.ympreset"),
        folder.join("Zcopy.ympreset"),
    )
    .unwrap();
    // Not preset files at all: left alone and not listed.
    std::fs::write(folder.join("notes.txt"), "hello").unwrap();
    std::fs::create_dir(folder.join("a folder.ympreset")).unwrap();
    let before: Vec<_> = ["junk", "binary", "future", "big", "Zcopy"]
        .iter()
        .map(|n| std::fs::read(folder.join(format!("{n}.ympreset"))).unwrap())
        .collect();

    let listing = store.list();
    assert_eq!(
        listing
            .presets
            .iter()
            .map(|s| s.preset.name.as_str())
            .collect::<Vec<_>>(),
        ["Good one"]
    );
    let reasons: Vec<(String, String)> = listing
        .damaged
        .iter()
        .map(|d| (d.file.clone(), d.reason.clone()))
        .collect();
    let reason = |file: &str| {
        reasons
            .iter()
            .find(|(f, _)| f == file)
            .unwrap_or_else(|| panic!("{file} in {reasons:?}"))
            .1
            .clone()
    };
    assert_eq!(
        reason("junk.ympreset"),
        "This file isn't a Yellowmoon preset."
    );
    assert_eq!(
        reason("binary.ympreset"),
        "This file isn't a Yellowmoon preset."
    );
    assert_eq!(reason("future.ympreset"), TOO_NEW_SENTENCE);
    assert_eq!(
        reason("big.ympreset"),
        "This file is too large to be a preset."
    );
    assert_eq!(reason("Zcopy.ympreset"), "Another preset has this name.");
    assert_eq!(listing.damaged.len(), 5, "{reasons:?}");

    // Never rewritten: saving another preset leaves them as they were.
    store.save(&preset("Second"), false).unwrap();
    let after: Vec<_> = ["junk", "binary", "future", "big", "Zcopy"]
        .iter()
        .map(|n| std::fs::read(folder.join(format!("{n}.ympreset"))).unwrap())
        .collect();
    assert_eq!(before, after);
    // A new preset whose file name a damaged file holds takes the next number.
    let n = store.save(&preset("junk"), false).unwrap();
    assert_eq!(n.file, "junk 2.ympreset");
}

// ---------------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------------

#[test]
fn replace_writes_over_the_file_of_the_same_name_and_leaves_the_others_alone() {
    let (_root, store) = rig();
    let a = store.save(&preset("Alpha"), false).unwrap();
    store.save(&preset("Beta"), false).unwrap();
    let beta = std::fs::read(store.folder().join("Beta.ympreset")).unwrap();
    let again = store.save(&other("alpha"), true).unwrap();
    assert_eq!(
        again.file, a.file,
        "the file of the same name, whatever the case"
    );
    assert_eq!(again.preset.name, "alpha");
    assert_eq!(store.list().presets.len(), 2);
    assert_eq!(
        std::fs::read(store.folder().join("Beta.ympreset")).unwrap(),
        beta
    );
    assert_eq!(
        store.get("alpha").unwrap().settings.main_grade().brightness,
        33.0,
        "the new values"
    );
    assert_eq!(
        files(store.folder()),
        [".lock", "Alpha.ympreset", "Beta.ympreset"]
    );
}

#[test]
fn a_write_that_fails_leaves_the_old_file_whole_and_says_so() {
    let (_root, good) = rig();
    good.save(&preset("Keep"), false).unwrap();
    let path = good.folder().join("Keep.ympreset");
    let old = std::fs::read(&path).unwrap();

    // The disk fails at the rename: the staged file is made and the rename is refused (what `atomic_write` does when the rename
    // fails: the temporary file goes and the destination is the old file).
    let failing = PresetStore::open(good.folder().parent().unwrap())
        .with_lock_timeout(Duration::from_millis(300))
        .with_writer(|dest, bytes| {
            let temp = ym_looks::manifest::stage_write(dest, bytes)?;
            std::fs::remove_file(temp)?;
            Err(std::io::Error::other("the rename was refused"))
        });
    let e = failing.save(&other("Keep"), true).expect_err("fails");
    assert!(matches!(e, PresetError::Write { .. }));
    let sentence = e.to_string();
    assert!(
        sentence.starts_with("Yellowmoon couldn't save the preset to ")
            && sentence.ends_with(
                ". Check that you can save there and that the disk has free space, then try again. Nothing was changed."
            ),
        "{sentence}"
    );
    assert!(sentence.contains(&good.folder().display().to_string()));
    assert_eq!(std::fs::read(&path).unwrap(), old, "the old file is whole");
    assert_eq!(
        files(good.folder()),
        [".lock", "Keep.ympreset"],
        "no stray temporary file"
    );
    assert!(good.get("Keep").is_ok());

    // A new name that fails writes nothing at all; a rename that fails keeps the old file.
    assert!(failing.save(&preset("New"), false).is_err());
    assert!(failing.rename("Keep", "Renamed").is_err());
    assert_eq!(files(good.folder()), [".lock", "Keep.ympreset"]);
    assert_eq!(std::fs::read(&path).unwrap(), old);
}

#[test]
fn the_real_write_is_atomic_and_leaves_no_temporary_file() {
    let (_root, store) = rig();
    for i in 0..5 {
        let mut p = preset("Same");
        p.settings.main_grade_mut().brightness = f64::from(i);
        store.save(&p, i > 0).unwrap();
        assert_eq!(files(store.folder()), [".lock", "Same.ympreset"]);
    }
}

#[test]
fn the_lock_excludes_a_second_writer_and_a_read_never_takes_it() {
    let (_root, store) = rig();
    store.save(&preset("First"), false).unwrap();
    let held = LibraryLock::acquire(store.folder(), Duration::from_secs(5)).expect("the lock");
    // A writer waits for the lock and gives up with a sentence; a reader is not held up.
    let started = std::time::Instant::now();
    let e = store.save(&preset("Second"), false).expect_err("busy");
    assert!(matches!(e, PresetError::Busy), "{e}");
    assert!(started.elapsed() >= Duration::from_millis(250));
    assert_eq!(
        e.to_string(),
        "Another Yellowmoon is still changing the presets. Wait for it to finish, then try again."
    );
    assert!(store.rename("First", "Renamed").is_err());
    assert!(store.remove("First").is_err());
    assert_eq!(store.list().presets.len(), 1, "a read goes on");
    assert!(store.get("First").is_ok());
    drop(held);
    store
        .save(&preset("Second"), false)
        .expect("the lock is free again");
    // Two writers on two threads both get in, one after the other.
    let root = store.folder().parent().unwrap().to_path_buf();
    let threads: Vec<_> = (0..4)
        .map(|i| {
            let root = root.clone();
            std::thread::spawn(move || {
                let s = PresetStore::open(&root).with_lock_timeout(Duration::from_secs(20));
                s.save(&preset(&format!("Thread {i}")), false)
                    .expect("saved");
            })
        })
        .collect();
    for t in threads {
        t.join().expect("thread");
    }
    assert_eq!(store.list().presets.len(), 6);
}

#[test]
fn rename_writes_the_new_file_and_then_removes_the_old_one() {
    let (_root, store) = rig();
    store.save(&preset("Warm evening"), false).unwrap();
    store.save(&preset("Cool morning"), false).unwrap();
    let r = store.rename("warm EVENING", "Golden hour").unwrap();
    assert_eq!(r.preset.name, "Golden hour");
    assert_eq!(r.file, "Golden hour.ympreset");
    assert_eq!(
        files(store.folder()),
        [".lock", "Cool morning.ympreset", "Golden hour.ympreset"]
    );
    assert_eq!(store.get("Golden hour").unwrap().groups, preset("x").groups);
    // A name that exists is refused; its own name in another case is not.
    let e = store
        .rename("Golden hour", "cool MORNING")
        .expect_err("exists");
    assert_eq!(
        e.to_string(),
        "A preset called cool MORNING already exists."
    );
    let c = store.rename("Golden hour", "GOLDEN HOUR").unwrap();
    assert_eq!(
        (c.preset.name.as_str(), c.file.as_str()),
        ("GOLDEN HOUR", "Golden hour.ympreset")
    );
    assert_eq!(store.list().presets.len(), 2, "one file for the one preset");
    assert!(matches!(
        store.rename("Nobody", "Somebody"),
        Err(PresetError::NotFound(_))
    ));
    assert!(matches!(
        store.rename("Cool morning", "   "),
        Err(PresetError::InvalidName(_))
    ));
    assert!(store.get("Cool morning").is_ok());
}

// ---------------------------------------------------------------------------------------------
// Import and export
// ---------------------------------------------------------------------------------------------

#[test]
fn import_adds_and_never_replaces_behind_the_persons_back() {
    let (_root, store) = rig();
    let src = tempfile::tempdir().unwrap();
    let write = |name: &str, text: &str| {
        let p = src.path().join(name);
        std::fs::write(&p, text).unwrap();
        p
    };
    let warm = write("warm.ympreset", &preset("Warm evening").to_json().unwrap());
    let o = store.import(&warm, false);
    assert_eq!(o.added, ["Warm evening"]);
    assert!(o.replaced.is_empty() && o.exists.is_empty() && o.refused.is_empty());
    assert_eq!(store.get("warm evening").unwrap(), preset("Warm evening"));

    // The same name again: not replaced, and the page is told where the file is.
    let newer = write("warm2.ympreset", &other("warm EVENING").to_json().unwrap());
    let o = store.import(&newer, false);
    assert!(o.added.is_empty() && o.replaced.is_empty());
    assert_eq!(
        o.exists,
        [ImportExists {
            name: "warm EVENING".into(),
            path: newer.display().to_string()
        }]
    );
    assert_eq!(
        store.get("Warm evening").unwrap(),
        preset("Warm evening"),
        "unchanged"
    );
    // With `replace` it is written over.
    let o = store.import(&newer, true);
    assert_eq!(o.replaced, ["warm EVENING"]);
    assert_eq!(store.list().presets.len(), 1);
    assert_eq!(
        store
            .get("Warm evening")
            .unwrap()
            .settings
            .main_grade()
            .brightness,
        33.0
    );

    // Refusals carry their sentence and the file's name.
    let old = write("old.ympreset", r#"{"yellowmoon_preset":2}"#);
    let junk = write("junk.ympreset", "nope");
    let gone = src.path().join("gone.ympreset");
    let o = store.import(&old, false);
    assert_eq!(o.refused[0].file, "old.ympreset");
    assert_eq!(o.refused[0].reason, TOO_NEW_SENTENCE);
    assert_eq!(
        store.import(&junk, false).refused[0].reason,
        "This file isn't a Yellowmoon preset."
    );
    let g = store.import(&gone, false);
    assert_eq!(g.refused[0].reason, "Yellowmoon couldn't read this file.");
    assert_eq!(store.list().presets.len(), 1, "nothing else was added");
}

#[test]
fn export_writes_the_file_atomically_and_overwrites_the_destination() {
    let (_root, store) = rig();
    store.save(&preset("Warm evening"), false).unwrap();
    let out = tempfile::tempdir().unwrap();
    let dest = out.path().join("Warm evening.ympreset");
    std::fs::write(&dest, "something else").unwrap();
    store.export("warm evening", &dest).expect("exported");
    assert_eq!(
        std::fs::read_to_string(&dest).unwrap(),
        std::fs::read_to_string(store.folder().join("Warm evening.ympreset")).unwrap()
    );
    assert_eq!(
        files(out.path()),
        ["Warm evening.ympreset"],
        "no temporary file"
    );
    assert!(matches!(
        store.export("Nobody", &dest),
        Err(PresetError::NotFound(_))
    ));
    // The exported file imports into another folder as the same preset.
    let (_r2, other_store) = rig();
    assert_eq!(other_store.import(&dest, false).added, ["Warm evening"]);
    assert_eq!(
        other_store.get("Warm evening").unwrap(),
        store.get("Warm evening").unwrap()
    );
}

// ---------------------------------------------------------------------------------------------
// Delete
// ---------------------------------------------------------------------------------------------

#[test]
fn delete_moves_the_file_to_the_trash() {
    let (root, store) = rig();
    let (trash, log) = fake_trash(root.path().join("Trash"));
    let store = store.with_trash(trash);
    store.save(&preset("Warm evening"), false).unwrap();
    store.save(&preset("Cool morning"), false).unwrap();
    store.remove("WARM evening").expect("removed");
    assert_eq!(
        log.lock().unwrap().as_slice(),
        [store.folder().join("Warm evening.ympreset")]
    );
    assert_eq!(files(&root.path().join("Trash")), ["Warm evening.ympreset"]);
    assert_eq!(
        store
            .list()
            .presets
            .iter()
            .map(|s| s.preset.name.as_str())
            .collect::<Vec<_>>(),
        ["Cool morning"]
    );
    assert!(matches!(
        store.remove("Warm evening"),
        Err(PresetError::NotFound(_))
    ));
    // A damaged file is removed by its file name; nothing outside the folder can be.
    std::fs::write(store.folder().join("junk.ympreset"), "x").unwrap();
    assert_eq!(store.list().damaged.len(), 1);
    store.remove_file("junk.ympreset").expect("removed");
    assert!(store.list().damaged.is_empty());
    for bad in [
        "../Trash/Warm evening.ympreset",
        "notes.txt",
        ".lock",
        "missing.ympreset",
        "a/b.ympreset",
    ] {
        assert!(store.remove_file(bad).is_err(), "{bad}");
    }
    assert_eq!(files(&root.path().join("Trash")).len(), 2);
}

#[test]
fn a_machine_without_a_trash_keeps_the_file_and_says_why() {
    let (_root, store) = rig();
    let store = store.with_trash(|path| {
        Err(TrashError::Failed {
            path: path.to_path_buf(),
            reason: "there is no Trash here".into(),
        })
    });
    store.save(&preset("Warm evening"), false).unwrap();
    let e = store.remove("Warm evening").expect_err("no trash");
    assert!(matches!(e, PresetError::Trash(_)));
    assert!(e.to_string().contains("there is no Trash here"), "{e}");
    assert!(store.get("Warm evening").is_ok(), "the file stays");
}

// ---------------------------------------------------------------------------------------------
// Names resolve like `--look`
// ---------------------------------------------------------------------------------------------

#[test]
fn a_query_matches_the_name_exactly_then_as_a_prefix_or_is_a_path() {
    let (_root, store) = rig();
    for n in ["Warm evening", "Warm", "Cool morning", "Cool night"] {
        store.save(&preset(n), false).unwrap();
    }
    // Exactly first: "Warm" is a preset, and a prefix of another.
    assert_eq!(store.resolve("warm").unwrap().name, "Warm");
    assert_eq!(store.resolve("Warm e").unwrap().name, "Warm evening");
    assert_eq!(store.resolve("cool m").unwrap().name, "Cool morning");
    // No match exits 3; several matches exit 2 and list them.
    let none = store.resolve("Nothing").expect_err("none");
    assert_eq!(
        (none.exit_code(), none.to_string().as_str()),
        (3, "There is no preset called Nothing.")
    );
    let many = store.resolve("cool").expect_err("several");
    assert_eq!(many.exit_code(), 2);
    assert_eq!(
        many.to_string(),
        "cool matches more than one preset: Cool morning, Cool night. Use more of the name."
    );
    // A path to a `.ympreset` file works without importing it; a path that is not a file exits 3.
    let src = tempfile::tempdir().unwrap();
    let file = src.path().join("elsewhere.ympreset");
    std::fs::write(&file, preset("Elsewhere").to_json().unwrap()).unwrap();
    assert_eq!(
        store.resolve(file.to_str().unwrap()).unwrap().name,
        "Elsewhere"
    );
    assert_eq!(store.list().presets.len(), 4, "not imported");
    let gone = store
        .resolve(src.path().join("no.ympreset").to_str().unwrap())
        .expect_err("gone");
    assert_eq!(gone.exit_code(), 3);
    let dir = store
        .resolve(src.path().to_str().unwrap())
        .expect_err("a folder is not a file");
    assert_eq!(dir.exit_code(), 3);
    std::fs::write(src.path().join("bad.ympreset"), "nope").unwrap();
    let bad = store
        .resolve(src.path().join("bad.ympreset").to_str().unwrap())
        .expect_err("bad");
    assert_eq!(
        (bad.exit_code(), bad.to_string().as_str()),
        (2, "This file isn't a Yellowmoon preset.")
    );
}
