//! Running the real `ym` binary on one file (`ym render --json --settings <entry settings>`), the way a user would.
//!
//! Every file is rendered by its own process: the automatic exposure and the pooled noise model of a photo depend on
//! the other photos of the same run (burst grouping by capture time, `docs/01`), so the same file rendered in a
//! different company would give different pixels and the golden hashes would depend on `--filter`.
//!
//! The process runs with an empty `YELLOWMOON_HOME`, so the user's preferences (default look, display rendering,
//! export defaults) and imported looks cannot change a golden render.
//!
//! The golden hashes pin the CPU reference: the process gets `--cpu` and `YELLOWMOON_GPU=off`, so that no graphics
//! device is opened and a machine with a GPU cannot move a hash (`docs/gpu-plan.md` section 7). GPU renders are
//! compared with the stored CPU pictures within a tolerance, in a mode of their own.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// How long one `ym render` may take before it is stopped (a hung process would otherwise hold a CI job until its own
/// time-out). `$GOLDEN_RENDER_TIMEOUT_SECS` replaces it; the largest photo of the golden sets renders in well under a
/// minute in a release build.
pub const RENDER_TIMEOUT: Duration = Duration::from_secs(900);

/// The time-out in force: `$GOLDEN_RENDER_TIMEOUT_SECS` (whole seconds, at least 1), else [`RENDER_TIMEOUT`].
#[must_use]
pub fn render_timeout() -> Duration {
    std::env::var("GOLDEN_RENDER_TIMEOUT_SECS")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .filter(|&secs| secs >= 1)
        .map_or(RENDER_TIMEOUT, Duration::from_secs)
}

/// A pipe read on its own thread: what has arrived so far is always available, and the end is announced.
struct Drained {
    buffer: std::sync::Arc<std::sync::Mutex<Vec<u8>>>,
    done: std::sync::mpsc::Receiver<()>,
}

impl Drained {
    fn start(mut pipe: impl Read + Send + 'static) -> Self {
        let buffer = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let (tx, done) = std::sync::mpsc::channel();
        let shared = std::sync::Arc::clone(&buffer);
        std::thread::spawn(move || {
            let mut chunk = [0_u8; 8192];
            while let Ok(n) = pipe.read(&mut chunk) {
                if n == 0 {
                    break;
                }
                if let Ok(mut b) = shared.lock() {
                    b.extend_from_slice(&chunk[..n]);
                }
            }
            let _ = tx.send(());
        });
        Self { buffer, done }
    }

    /// Everything read so far, after giving the end of the stream `grace` to arrive. A process that left a child
    /// behind holding the pipe open would otherwise keep the caller waiting for as long as that child lives.
    fn collect(self, grace: Duration) -> Vec<u8> {
        let _ = self.done.recv_timeout(grace);
        self.buffer.lock().map(|b| b.clone()).unwrap_or_default()
    }
}

/// Runs `cmd` to the end and returns its status, stdout and stderr; stops it (and says so) after `timeout`.
fn run_with_timeout(
    cmd: &mut Command,
    timeout: Duration,
) -> Result<(std::process::ExitStatus, Vec<u8>, Vec<u8>)> {
    let mut child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("starting the process")?;
    // The pipes are drained while the process runs, so a chatty process cannot fill them and block.
    let out = child.stdout.take().map(Drained::start);
    let err = child.stderr.take().map(Drained::start);
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.try_wait().context("waiting for the process")? {
            break Some(status);
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            break None;
        }
        std::thread::sleep(Duration::from_millis(10));
    };
    let grace = Duration::from_secs(2);
    let stdout = out.map(|d| d.collect(grace)).unwrap_or_default();
    let stderr = err.map(|d| d.collect(grace)).unwrap_or_default();
    match status {
        Some(status) => Ok((status, stdout, stderr)),
        None => bail!(
            "did not finish in {} s and was stopped (GOLDEN_RENDER_TIMEOUT_SECS changes the limit): {}",
            timeout.as_secs(),
            first_lines(&String::from_utf8_lossy(&stderr), 3)
        ),
    }
}

/// What to ask `ym render` to write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Output {
    /// A 16-bit sRGB PNG: the golden pictures (no dither, no lossy coding).
    Png16,
    /// A DNG with the look baked in (the end-to-end colour check reads its linear AWG4 pixels).
    Dng,
    /// An 8-bit sRGB JPEG at quality 92 (the DNG gate compares with it).
    Jpeg,
}

impl Output {
    fn args(self) -> &'static [&'static str] {
        match self {
            Self::Png16 => &["--format", "png", "--bits", "16", "--colour-space", "srgb"],
            Self::Dng => &["--format", "dng"],
            Self::Jpeg => &[
                "--format",
                "jpeg",
                "--quality",
                "92",
                "--colour-space",
                "srgb",
            ],
        }
    }
}

/// Where the binary and its data are.
#[derive(Clone, Debug)]
pub struct YmBinary {
    /// The `ym` executable.
    pub path: PathBuf,
    /// `YELLOWMOON_DATA`, when the data folder is not the workspace's (release builds).
    pub data: Option<PathBuf>,
}

/// The result of one `ym render` call.
#[derive(Clone, Debug)]
pub struct Rendered {
    /// The file that was written.
    pub output: PathBuf,
    /// The JSON object `ym` printed for the file.
    pub json: Value,
}

impl Rendered {
    /// The pipeline version the binary reported.
    #[must_use]
    pub fn pipeline_version(&self) -> String {
        self.json
            .get("pipeline_version")
            .and_then(Value::as_str)
            .unwrap_or("unknown")
            .to_owned()
    }
}

/// The workspace root (this crate is `tools/golden`).
#[must_use]
pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

/// Finds the `ym` binary: `explicit`, else `$YM_BIN`, else the newer of `target/release/ym` and `target/debug/ym`
/// of the workspace.
pub fn find_ym(explicit: Option<&Path>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        if p.is_file() {
            return Ok(p.to_path_buf());
        }
        bail!("{} is not a file", p.display());
    }
    if let Some(p) = std::env::var_os("YM_BIN").map(PathBuf::from) {
        if p.is_file() {
            return Ok(p);
        }
        bail!("$YM_BIN names {}, which is not a file", p.display());
    }
    let target = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| workspace_root().join("target"));
    let newest = ["release", "debug"]
        .iter()
        .map(|profile| target.join(profile).join("ym"))
        .filter_map(|p| {
            let modified = std::fs::metadata(&p).and_then(|m| m.modified()).ok()?;
            Some((modified, p))
        })
        .max();
    match newest {
        Some((_, p)) => Ok(p),
        None => bail!(
            "no ym binary under {}: run `cargo build -p ym-cli` or give --ym",
            target.display()
        ),
    }
}

impl YmBinary {
    /// Renders `input` into `out_dir` (created if missing) with `settings`, a `ym:Settings` document, and returns the
    /// file and the JSON the binary printed. `work` is a scratch folder for the settings file and the empty home.
    ///
    /// # Errors
    ///
    /// When the process cannot start, exits with an error or reports `status` other than `ok`.
    pub fn render(
        &self,
        input: &Path,
        settings: &Value,
        output: Output,
        out_dir: &Path,
        work: &Path,
    ) -> Result<Rendered> {
        std::fs::create_dir_all(work).with_context(|| format!("creating {}", work.display()))?;
        std::fs::create_dir_all(out_dir)
            .with_context(|| format!("creating {}", out_dir.display()))?;
        let home = work.join("home");
        std::fs::create_dir_all(&home).with_context(|| format!("creating {}", home.display()))?;
        let settings_path = work.join("settings.json");
        std::fs::write(&settings_path, serde_json::to_vec(settings)?)
            .with_context(|| format!("writing {}", settings_path.display()))?;
        let mut cmd = Command::new(&self.path);
        cmd.arg("render")
            .arg("--cpu")
            .arg("--json")
            .arg("--quiet")
            .arg("--settings")
            .arg(&settings_path)
            .args(output.args())
            .args(["--name", "{name}", "--overwrite", "--out"])
            .arg(out_dir)
            .arg(input)
            .env("YELLOWMOON_HOME", &home)
            .env("YELLOWMOON_GPU", "off")
            .stdin(Stdio::null());
        if let Some(data) = &self.data {
            cmd.env("YELLOWMOON_DATA", data);
        }
        let (exit, stdout, stderr) = run_with_timeout(&mut cmd, render_timeout())
            .with_context(|| format!("running {}", self.path.display()))?;
        let stdout = String::from_utf8_lossy(&stdout);
        let stderr = String::from_utf8_lossy(&stderr);
        let json: Option<Value> = stdout
            .lines()
            .rev()
            .find(|l| !l.trim().is_empty())
            .and_then(|l| serde_json::from_str(l).ok());
        let Some(json) = json else {
            bail!(
                "ym printed no JSON (exit {}): {}",
                exit,
                first_lines(&stderr, 6)
            );
        };
        let status = json.get("status").and_then(Value::as_str).unwrap_or("?");
        if status != "ok" || !exit.success() {
            let message = json
                .pointer("/error/message")
                .and_then(Value::as_str)
                .map(str::to_owned)
                .unwrap_or_else(|| first_lines(&stderr, 6));
            bail!("ym: {status} (exit {exit}): {message}");
        }
        let output_path = json
            .get("output")
            .and_then(Value::as_str)
            .map(PathBuf::from)
            .context("ym's JSON has no `output`")?;
        Ok(Rendered {
            output: output_path,
            json,
        })
    }
}

fn first_lines(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().take(n).collect();
    if lines.is_empty() {
        "(no message on stderr)".to_owned()
    } else {
        lines.join(" | ")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_binary_is_named() {
        let e = find_ym(Some(Path::new("/nonexistent/ym"))).unwrap_err();
        assert!(e.to_string().contains("/nonexistent/ym"));
    }

    #[cfg(unix)]
    #[test]
    fn a_hung_process_is_stopped_and_a_chatty_one_is_not_blocked() {
        let started = Instant::now();
        let e = run_with_timeout(
            Command::new("sh").args(["-c", "echo hello >&2; sleep 30"]),
            Duration::from_millis(300),
        )
        .unwrap_err();
        let text = format!("{e:#}");
        assert!(
            text.contains("did not finish") && text.contains("hello"),
            "{text}"
        );
        assert!(started.elapsed() < Duration::from_secs(10));
        // 4 MB on each pipe is far more than a pipe holds: the readers keep it from blocking
        let (status, out, err) = run_with_timeout(
            Command::new("sh").args([
                "-c",
                "head -c 4000000 /dev/zero; head -c 4000000 /dev/zero >&2; exit 3",
            ]),
            Duration::from_secs(60),
        )
        .unwrap();
        assert_eq!(status.code(), Some(3));
        assert_eq!((out.len(), err.len()), (4_000_000, 4_000_000));
    }

    #[test]
    fn the_time_out_has_a_default_and_an_override() {
        // (the environment variable is not set by the test suite; its parsing is the only logic)
        if std::env::var_os("GOLDEN_RENDER_TIMEOUT_SECS").is_none() {
            assert_eq!(render_timeout(), RENDER_TIMEOUT);
        }
    }

    #[test]
    fn a_golden_render_asks_for_the_cpu_reference() {
        // The command is built in `render`; the process cannot be run here, so look at it through a
        // stand-in binary that prints what it was given.
        let dir = std::env::temp_dir().join(format!("golden-ymrun-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("dir");
        let script = dir.join("ym");
        std::fs::write(
            &script,
            "#!/bin/sh\nprintf '{\"status\":\"error\",\"error\":{\"message\":\"args: %s gpu: %s\"}}\\n' \"$*\" \"$YELLOWMOON_GPU\"\nexit 1\n",
        )
        .expect("script");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).expect("chmod");
        }
        let ym = YmBinary {
            path: script,
            data: None,
        };
        let e = ym
            .render(
                Path::new("a.ARW"),
                &serde_json::json!({}),
                Output::Png16,
                &dir.join("out"),
                &dir.join("work"),
            )
            .unwrap_err();
        let text = format!("{e:#}");
        assert!(text.contains("args: render --cpu --json"), "{text}");
        assert!(text.contains("gpu: off"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn output_kinds_ask_for_their_format() {
        assert!(Output::Png16.args().contains(&"png"));
        assert!(Output::Png16.args().contains(&"16"));
        assert!(Output::Dng.args().contains(&"dng"));
        assert!(Output::Jpeg.args().contains(&"jpeg"));
    }
}
