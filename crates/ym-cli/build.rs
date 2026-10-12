//! Embeds the short git commit hash as `YM_GIT_SHA` for `yellowmoon --version`.
//!
//! Falls back to `unknown` when git or the repository is not available (for example when building
//! from a source archive).

use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR"),
    );
    let sha = git(&manifest_dir, &["rev-parse", "--short", "HEAD"]);
    println!(
        "cargo:rustc-env=YM_GIT_SHA={}",
        sha.as_deref().unwrap_or("unknown")
    );

    if sha.is_some() {
        emit_rerun_hints(&manifest_dir);
    }
}

/// Re-run when the checked-out commit changes: `HEAD` itself, the branch ref it points at, and
/// `packed-refs` (a branch ref may live only there).
fn emit_rerun_hints(manifest_dir: &Path) {
    let git_dir = git_path(manifest_dir, "--git-dir");
    let common_dir = git_path(manifest_dir, "--git-common-dir");

    if let Some(git_dir) = &git_dir {
        let head = git_dir.join("HEAD");
        rerun_if_exists(&head);
        // `HEAD` contains `ref: refs/heads/<branch>` unless it is detached.
        if let Ok(contents) = std::fs::read_to_string(&head)
            && let Some(reference) = contents.strip_prefix("ref: ")
        {
            let reference = reference.trim();
            for dir in [Some(git_dir), common_dir.as_ref()].into_iter().flatten() {
                rerun_if_exists(&dir.join(reference));
            }
        }
    }
    if let Some(common_dir) = &common_dir {
        rerun_if_exists(&common_dir.join("packed-refs"));
    }
}

/// Resolves the path printed by `git rev-parse <flag>` to an absolute path.
fn git_path(manifest_dir: &Path, flag: &str) -> Option<PathBuf> {
    let printed = git(manifest_dir, &["rev-parse", flag])?;
    Some(manifest_dir.join(printed))
}

/// Cargo reruns a build script on every build when it watches a missing path, so only existing
/// paths are registered.
fn rerun_if_exists(path: &Path) {
    if path.exists() {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// Runs git in `dir` and returns trimmed, non-empty stdout, or `None` on any failure.
fn git(dir: &Path, args: &[&str]) -> Option<String> {
    let output = Command::new("git")
        .args(args)
        .current_dir(dir)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8(output.stdout).ok()?;
    let text = text.trim();
    (!text.is_empty()).then(|| text.to_owned())
}
