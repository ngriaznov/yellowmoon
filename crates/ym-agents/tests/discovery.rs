//! Finding a program (`docs/22` section 4): each step of the search, with test doubles in folders of the
//! test's own and no environment variable touched.

mod common;

use std::time::Duration;

use common::{double, install, known_path, locations_in, read_runs};
use semver::Version;
use tempfile::TempDir;
use ym_agents::discovery::{Discovery, Os, locate};
use ym_agents::tool::{Tool, Tuning};

fn tuning() -> Tuning {
    let mut t = Tuning::for_tool(Tool::Claude);
    t.probe.version = Duration::from_secs(10);
    t.probe.shell = Duration::from_millis(600);
    t
}

fn found_path(d: &Discovery) -> &std::path::Path {
    match d {
        Discovery::Found { path, .. } => path,
        other => panic!("expected Found, got {other:?}"),
    }
}

#[test]
fn the_known_folder_is_found() {
    let home = TempDir::new().unwrap();
    let path = known_path(Tool::Claude, home.path());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::copy(double(Tool::Claude), &path).unwrap();
    let found = locate(Tool::Claude, &locations_in(home.path()), &tuning());
    assert_eq!(
        found,
        Discovery::Found {
            path: path.clone(),
            version: Version::new(2, 1, 310)
        }
    );
    // agy is not there.
    assert_eq!(
        locate(Tool::Agy, &locations_in(home.path()), &tuning()),
        Discovery::NotInstalled
    );
}

#[test]
fn path_is_searched_after_the_known_folders() {
    let home = TempDir::new().unwrap();
    let bin = TempDir::new().unwrap();
    let program = install(Tool::Agy, bin.path(), &Tool::Agy.file_name());
    let mut loc = locations_in(home.path());
    loc.path = Some(std::env::join_paths([bin.path()]).unwrap());
    let found = locate(Tool::Agy, &loc, &tuning());
    assert_eq!(found_path(&found), program);
}

#[test]
fn a_too_old_copy_does_not_hide_a_newer_one() {
    let home = TempDir::new().unwrap();
    let bin = TempDir::new().unwrap();
    // The known folder holds a version that is too old; PATH holds a good one.
    let old = known_path(Tool::Claude, home.path());
    let dir = old.parent().unwrap();
    std::fs::create_dir_all(dir).unwrap();
    std::fs::copy(double(Tool::Claude), &old).unwrap();
    std::fs::write(dir.join("version.txt"), "2.1.100\n").unwrap();
    let good = install(Tool::Claude, bin.path(), &Tool::Claude.file_name());
    let mut loc = locations_in(home.path());
    loc.path = Some(std::env::join_paths([bin.path()]).unwrap());
    assert_eq!(found_path(&locate(Tool::Claude, &loc, &tuning())), good);

    // Alone, the old copy is `too old`, with its version.
    loc.path = None;
    assert_eq!(
        locate(Tool::Claude, &loc, &tuning()),
        Discovery::TooOld {
            path: old,
            version: Version::new(2, 1, 100)
        }
    );
}

#[test]
fn the_highest_of_several_old_copies_is_reported() {
    let home = TempDir::new().unwrap();
    let bin = TempDir::new().unwrap();
    let first = known_path(Tool::Claude, home.path());
    std::fs::create_dir_all(first.parent().unwrap()).unwrap();
    std::fs::copy(double(Tool::Claude), &first).unwrap();
    std::fs::write(first.with_file_name("version.txt"), "2.0.5\n").unwrap();
    let second = install(Tool::Claude, bin.path(), &Tool::Claude.file_name());
    std::fs::write(bin.path().join("version.txt"), "2.1.50\n").unwrap();
    let mut loc = locations_in(home.path());
    loc.path = Some(std::env::join_paths([bin.path()]).unwrap());
    assert_eq!(
        locate(Tool::Claude, &loc, &tuning()),
        Discovery::TooOld {
            path: second,
            version: Version::new(2, 1, 50)
        }
    );
}

#[test]
fn a_program_that_prints_no_version_is_probe_failed() {
    let home = TempDir::new().unwrap();
    let path = known_path(Tool::Claude, home.path());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    // Not a program at all.
    std::fs::write(&path, b"this is not an executable").unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let found = locate(Tool::Claude, &locations_in(home.path()), &tuning());
    assert_eq!(found, Discovery::ProbeFailed { path });
}

#[test]
fn a_program_that_hangs_on_version_is_probe_failed_at_its_limit() {
    let home = TempDir::new().unwrap();
    let path = known_path(Tool::Claude, home.path());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::copy(double(Tool::Claude), &path).unwrap();
    let mut t = tuning().with_env("YM_FAKE_SCENARIO", "probe_hang");
    t.probe.version = Duration::from_millis(400);
    let started = std::time::Instant::now();
    let found = locate(Tool::Claude, &locations_in(home.path()), &t);
    assert_eq!(found, Discovery::ProbeFailed { path });
    assert!(started.elapsed() < Duration::from_secs(10));
}

#[test]
fn the_same_program_reached_twice_runs_once() {
    // Known folder and PATH name the same file: it is asked its version once.
    let home = TempDir::new().unwrap();
    let record = TempDir::new().unwrap();
    let path = known_path(Tool::Claude, home.path());
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::copy(double(Tool::Claude), &path).unwrap();
    let mut loc = locations_in(home.path());
    loc.path = Some(std::env::join_paths([path.parent().unwrap()]).unwrap());
    // Make the first copy too old so that the search goes on to the second route.
    std::fs::write(path.with_file_name("version.txt"), "1.0.0\n").unwrap();
    let t = tuning().with_env("YM_FAKE_RECORD", &record.path().to_string_lossy());
    let found = locate(Tool::Claude, &loc, &t);
    assert!(matches!(found, Discovery::TooOld { .. }), "{found:?}");
    assert_eq!(
        read_runs(record.path()).len(),
        1,
        "one --version for one file"
    );
}

// ----- the login shell (macOS and Linux) -----

#[cfg(unix)]
mod shell {
    use super::*;
    use std::ffi::OsString;
    use std::os::unix::fs::PermissionsExt;
    use ym_agents::discovery::Locations;

    /// A script that stands in for `$SHELL`, named `name` (its file name decides whether the shell is
    /// trusted).
    fn script(dir: &std::path::Path, name: &str, body: &str) -> OsString {
        let path = dir.join(name);
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).unwrap();
        path.into_os_string()
    }

    fn locations(home: &std::path::Path, shell: OsString) -> Locations {
        Locations {
            shell: Some(shell),
            ..locations_in(home)
        }
    }

    #[test]
    fn the_login_shell_finds_a_program_on_a_path_the_app_does_not_have() {
        let home = TempDir::new().unwrap();
        let tools = TempDir::new().unwrap();
        let program = install(Tool::Claude, tools.path(), "claude");
        // It prints a banner first and answers on the last line; `$1 $2` are the flags.
        let shell = script(
            tools.path(),
            "zsh",
            &format!(
                "echo 'Welcome to a very chatty shell'\necho 'alias claude=x'\necho {}",
                program.display()
            ),
        );
        let found = locate(Tool::Claude, &locations(home.path(), shell), &tuning());
        assert_eq!(found_path(&found), program);
    }

    #[test]
    fn a_shell_that_hangs_is_given_up_at_its_limit() {
        let home = TempDir::new().unwrap();
        let tools = TempDir::new().unwrap();
        // `sh` is asked once (only zsh and bash get the second, interactive attempt).
        let shell = script(tools.path(), "sh", "sleep 30");
        let started = std::time::Instant::now();
        let found = locate(Tool::Claude, &locations(home.path(), shell), &tuning());
        assert_eq!(found, Discovery::NotInstalled);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "{:?}",
            started.elapsed()
        );
    }

    #[test]
    fn the_interactive_attempt_is_made_for_zsh_when_the_first_gives_nothing() {
        let home = TempDir::new().unwrap();
        let tools = TempDir::new().unwrap();
        let program = install(Tool::Claude, tools.path(), "claude");
        // Only the interactive start-up (`-i`) sees the program, like an nvm set up in .zshrc.
        let shell = script(
            tools.path(),
            "zsh",
            &format!(
                "case \"$1\" in -i) echo {} ;; *) echo nothing ;; esac",
                program.display()
            ),
        );
        let found = locate(Tool::Claude, &locations(home.path(), shell), &tuning());
        assert_eq!(found_path(&found), program);
    }

    /// The login shell is started in a session of its own: with no controlling terminal an interactive
    /// zsh or bash cannot find itself in a background group of one and stop itself (`SIGTTIN`), which
    /// made `ym doctor` wait out the limit of 3 s and then the 2 s of grace for each program.
    #[cfg(target_os = "linux")]
    #[test]
    fn the_login_shell_has_a_session_of_its_own() {
        let home = TempDir::new().unwrap();
        let tools = TempDir::new().unwrap();
        let program = install(Tool::Claude, tools.path(), "claude");
        let marker = tools.path().join("ids");
        // The script is the shell: it writes its pid, its group and its session, then answers.
        let shell = script(
            tools.path(),
            "zsh",
            &format!(
                "read pid comm state ppid pgrp sid rest < /proc/$$/stat\n\
                 echo \"$$ $pgrp $sid\" >> {}\necho {}",
                marker.display(),
                program.display()
            ),
        );
        let found = locate(Tool::Claude, &locations(home.path(), shell), &tuning());
        assert_eq!(found_path(&found), program);
        let text = std::fs::read_to_string(&marker).unwrap();
        let first: Vec<u32> = text
            .lines()
            .next()
            .unwrap()
            .split_whitespace()
            .map(|n| n.parse().unwrap())
            .collect();
        assert_eq!(
            (first[1], first[2]),
            (first[0], first[0]),
            "the shell leads its group and its session"
        );
    }

    #[test]
    fn a_cancel_ends_the_shell_lookup() {
        let home = TempDir::new().unwrap();
        let tools = TempDir::new().unwrap();
        let shell = script(tools.path(), "sh", "sleep 30");
        let source = ym_agents::process::CancelSource::new();
        let mut t = tuning();
        t.probe.shell = Duration::from_secs(20);
        t.cancel = source.token();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(500));
            source.cancel();
        });
        let started = std::time::Instant::now();
        let found = locate(Tool::Claude, &locations(home.path(), shell), &t);
        canceller.join().unwrap();
        assert_eq!(found, Discovery::NotInstalled);
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the lookup went on for {:?}",
            started.elapsed()
        );
    }

    #[test]
    fn an_unknown_shell_is_not_asked() {
        let home = TempDir::new().unwrap();
        let tools = TempDir::new().unwrap();
        let program = install(Tool::Claude, tools.path(), "claude");
        let shell = script(tools.path(), "nu", &format!("echo {}", program.display()));
        let found = locate(Tool::Claude, &locations(home.path(), shell), &tuning());
        assert_eq!(found, Discovery::NotInstalled);
    }

    #[test]
    fn the_shell_is_not_asked_when_the_known_folders_have_a_good_program() {
        let home = TempDir::new().unwrap();
        let tools = TempDir::new().unwrap();
        let known = known_path(Tool::Claude, home.path());
        std::fs::create_dir_all(known.parent().unwrap()).unwrap();
        std::fs::copy(double(Tool::Claude), &known).unwrap();
        let marker = tools.path().join("shell-was-run");
        let shell = script(tools.path(), "zsh", &format!("touch {}", marker.display()));
        let found = locate(Tool::Claude, &locations(home.path(), shell), &tuning());
        assert_eq!(found_path(&found), known);
        assert!(!marker.exists());
    }
}

// ----- Windows' rules, run on any system with the double standing in for the `.exe` files -----

#[test]
fn a_cmd_shim_alone_is_an_unsafe_launcher_and_the_npm_layout_is_resolved() {
    let home = TempDir::new().unwrap();
    let npm = home.path().join("AppData").join("Roaming").join("npm");
    std::fs::create_dir_all(&npm).unwrap();
    std::fs::write(npm.join("claude.cmd"), b"@echo off\r\n").unwrap();
    let mut loc = locations_in(home.path());
    loc.os = Os::Windows;
    assert_eq!(
        locate(Tool::Claude, &loc, &tuning()),
        Discovery::UnsafeLauncher
    );

    // With the program next to the shim, the shim is resolved to it, never started.
    let real_dir = npm
        .join("node_modules")
        .join("@anthropic-ai")
        .join("claude-code")
        .join("bin");
    let real = install(Tool::Claude, &real_dir, "claude.exe");
    assert_eq!(found_path(&locate(Tool::Claude, &loc, &tuning())), real);
}

#[test]
fn windows_known_folders_for_both_programs() {
    let home = TempDir::new().unwrap();
    let mut loc = locations_in(home.path());
    loc.os = Os::Windows;
    let claude = install(
        Tool::Claude,
        &home.path().join(".local").join("bin"),
        "claude.exe",
    );
    assert_eq!(found_path(&locate(Tool::Claude, &loc, &tuning())), claude);
    let agy = install(
        Tool::Agy,
        &home
            .path()
            .join("AppData")
            .join("Local")
            .join("agy")
            .join("bin"),
        "agy.exe",
    );
    let mut t = tuning();
    t.probe.version = Duration::from_secs(10);
    assert_eq!(found_path(&locate(Tool::Agy, &loc, &t)), agy);
}
