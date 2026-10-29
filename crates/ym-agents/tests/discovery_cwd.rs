//! A relative entry of `PATH` never gives a relative program (`docs/22` section 4, step 2). The test
//! changes the process's working folder, so it is the only test of this file.

mod common;

use std::ffi::OsString;
use std::time::Duration;

use common::{install, locations_in};
use tempfile::TempDir;
use ym_agents::discovery::{Discovery, Locations, locate};
use ym_agents::tool::{Tool, Tuning};

fn tuning() -> Tuning {
    let mut t = Tuning::for_tool(Tool::Claude);
    t.probe.version = Duration::from_secs(10);
    t
}

fn with_path(home: &std::path::Path, entries: &[OsString]) -> Locations {
    Locations {
        path: Some(std::env::join_paths(entries).unwrap()),
        ..locations_in(home)
    }
}

#[test]
fn a_relative_path_entry_is_never_the_program() {
    let home = TempDir::new().unwrap();
    // `ym` was started in a folder that holds a program called `claude` (on Windows `claude.exe` there
    // would be found first by the system's own search).
    let here = TempDir::new().unwrap();
    install(Tool::Claude, here.path(), &Tool::Claude.file_name());
    std::env::set_current_dir(here.path()).unwrap();

    // Only relative entries: `.`, a folder below it, and (on Unix, where an empty entry means the
    // current folder) an empty one. Nothing is found, and nothing from the current folder is run.
    let record = TempDir::new().unwrap();
    let mut relative = vec![OsString::from("."), OsString::from("bin")];
    if cfg!(unix) {
        relative.push(OsString::new());
    }
    let mut loc = with_path(home.path(), &relative);
    loc.shell = None;
    let t = tuning().with_env("YM_FAKE_RECORD", &record.path().to_string_lossy());
    assert_eq!(locate(Tool::Claude, &loc, &t), Discovery::NotInstalled);
    assert!(
        common::read_runs(record.path()).is_empty(),
        "the program of the current folder was run"
    );

    // A relative entry first and a real folder after it: the match of the real folder is the one.
    let good = TempDir::new().unwrap();
    let program = install(Tool::Claude, good.path(), &Tool::Claude.file_name());
    let mut entries = relative.clone();
    entries.push(good.path().as_os_str().to_owned());
    let found = locate(Tool::Claude, &with_path(home.path(), &entries), &tuning());
    let Discovery::Found { path, .. } = found else {
        panic!("not found: {found:?}");
    };
    assert!(path.is_absolute(), "{}", path.display());
    assert_eq!(path, program);
}
