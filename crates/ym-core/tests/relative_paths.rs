//! Paths given relative to the working directory (`ym render .`, `ym render DSC0001.ARW`) behave
//! like absolute ones. This file holds a single test because it changes the working directory of
//! the process.

mod common;

use common::{Layout, Scene, light, session_with_events, settle};
use ym_core::session::OpenOptions;

#[test]
fn relative_paths_name_the_folder_and_the_default_export_folder() {
    let root = tempfile::tempdir().unwrap();
    let folder = root.path().join("Lisbon");
    std::fs::create_dir(&folder).unwrap();
    let scene = Scene::new(
        64,
        48,
        light("sony-ilce-7rm4", 5200.0),
        Layout::Uniform,
        0.3,
    );
    scene.write(&folder, "DSC0001.ARW", "ILCE-7RM4");
    scene.write(&folder, "DSC0002.ARW", "ILCE-7RM4");
    let root_canonical = std::fs::canonicalize(root.path()).unwrap();

    std::env::set_current_dir(&folder).unwrap();

    // `.` is the folder itself.
    let (session, _rx) = session_with_events();
    session.open(&[".".into()], OpenOptions::default());
    settle(&session);
    let state = session.state();
    assert_eq!(state.photos.len(), 2);
    assert_eq!(state.folder_name.as_deref(), Some("Lisbon"));
    assert_eq!(
        state
            .default_export_folder
            .as_deref()
            .map(std::path::Path::new),
        Some(root_canonical.join("Lisbon export").as_path())
    );
    // The photos know where they are even if the process moves.
    let view = session.view(session.photo_ids()[0]).unwrap();
    assert!(view.path.is_absolute(), "{:?}", view.path);

    // Files given by relative name come from one folder, which has a name.
    let (session, _rx) = session_with_events();
    session.open(
        &["DSC0001.ARW".into(), "DSC0002.ARW".into()],
        OpenOptions::default(),
    );
    settle(&session);
    let state = session.state();
    assert_eq!(state.folder_name.as_deref(), Some("Lisbon"));
    assert_eq!(
        state
            .default_export_folder
            .as_deref()
            .map(std::path::Path::new),
        Some(root_canonical.join("Lisbon export").as_path())
    );

    // `..` from inside a sub folder.
    let inner = folder.join("inner");
    std::fs::create_dir(&inner).unwrap();
    std::env::set_current_dir(&inner).unwrap();
    let (session, _rx) = session_with_events();
    session.open(&["..".into()], OpenOptions::default());
    settle(&session);
    assert_eq!(session.state().folder_name.as_deref(), Some("Lisbon"));
}
