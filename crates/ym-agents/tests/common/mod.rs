//! What the integration tests share: the two test doubles, pictures, and a rig that gives an advisor a
//! double with a scenario, a record folder and a temporary folder of its own.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use tempfile::TempDir;
use ym_agents::advisor::Source;
use ym_agents::discovery::{Locations, Os};
use ym_agents::tool::{Tool, Tuning};
use ym_agents::{AgyAdvisor, ClaudeAdvisor};
use ym_core::apple_auto::{AdvisorPicture, AutoAdvisor};

/// The test double of Claude Code.
pub const FAKE_CLAUDE: &str = env!("CARGO_BIN_EXE_ym-fake-claude");
/// The test double of `agy`.
pub const FAKE_AGY: &str = env!("CARGO_BIN_EXE_ym-fake-agy");

/// The double of `tool`.
pub fn double(tool: Tool) -> PathBuf {
    PathBuf::from(match tool {
        Tool::Claude => FAKE_CLAUDE,
        Tool::Agy => FAKE_AGY,
    })
}

/// A dark, flat picture: `FakeAdvisor` brightens it, opens its shadows, adds contrast and a teal.
pub fn dark_flat() -> AdvisorPicture {
    AdvisorPicture {
        width: 96,
        height: 64,
        rgb: vec![40; 96 * 64 * 3],
    }
}

/// A picture of two flat halves, one dark and one bright: mean luma about 0.55, a wide spread.
pub fn halves() -> AdvisorPicture {
    let (width, height) = (96, 64);
    let mut rgb = Vec::with_capacity(width * height * 3);
    for _ in 0..height {
        for x in 0..width {
            let v = if x < width / 2 { 30 } else { 250 };
            rgb.extend_from_slice(&[v, v, v]);
        }
    }
    AdvisorPicture { width, height, rgb }
}

/// A smooth colour gradient.
pub fn gradient(width: usize, height: usize) -> AdvisorPicture {
    let mut rgb = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            rgb.push((x * 255 / (width - 1)) as u8);
            rgb.push((y * 255 / (height - 1)) as u8);
            rgb.push(((x + y) * 255 / (width + height - 2)) as u8);
        }
    }
    AdvisorPicture { width, height, rgb }
}

/// What the double saw in one run.
#[derive(Debug, Clone)]
pub struct Run {
    pub argv: Vec<String>,
    pub cwd: String,
    pub files: Vec<String>,
    pub env: HashMap<String, String>,
    pub stdin: Vec<u8>,
}

impl Run {
    /// Whether this was a request (only a request has `--safe-mode` or `--sandbox`).
    pub fn is_request(&self) -> bool {
        self.argv
            .iter()
            .any(|a| a == "--safe-mode" || a == "--sandbox")
    }

    /// Whether this was a sign-in check (`claude auth status`, `agy -p /usage`).
    pub fn is_sign_in_check(&self) -> bool {
        self.argv == ["auth", "status"]
            || self
                .argv
                .starts_with(&["-p".to_owned(), "/usage".to_owned()])
    }

    /// Whether this was a `--version`.
    pub fn is_version(&self) -> bool {
        self.argv == ["--version"]
    }
}

/// An advisor over a double, with a record and a temporary folder of its own.
pub struct Rig<A> {
    pub advisor: Arc<A>,
    pub record: TempDir,
    pub temp: TempDir,
}

/// The tuning for `tool` with the scenario, a record folder and a temporary folder, and probe limits
/// short enough for a test.
pub fn tuning(tool: Tool, scenario: &str, record: &Path, temp: &Path) -> Tuning {
    let mut tuning = Tuning::for_tool(tool)
        .with_env("YM_FAKE_SCENARIO", scenario)
        .with_env("YM_FAKE_RECORD", &record.to_string_lossy());
    tuning.temp_dir = Some(temp.to_path_buf());
    tuning.min_interval = Duration::ZERO;
    tuning.probe.version = Duration::from_secs(10);
    tuning.probe.claude_auth = Duration::from_secs(10);
    tuning.probe.agy_usage = Duration::from_secs(10);
    tuning
}

impl Rig<ClaudeAdvisor> {
    /// Claude Code's advisor over its double in `scenario`.
    pub fn claude(scenario: &str) -> Self {
        Self::claude_with(scenario, |_| {})
    }

    /// As [`Rig::claude`], after `adjust` has changed the tuning.
    pub fn claude_with(scenario: &str, adjust: impl FnOnce(&mut Tuning)) -> Self {
        let (record, temp) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let mut t = tuning(Tool::Claude, scenario, record.path(), temp.path());
        adjust(&mut t);
        let advisor = ClaudeAdvisor::custom(Source::Program(double(Tool::Claude)), t);
        Self {
            advisor,
            record,
            temp,
        }
    }
}

impl Rig<AgyAdvisor> {
    /// Antigravity's advisor over its double in `scenario`.
    pub fn agy(scenario: &str) -> Self {
        Self::agy_with(scenario, |_| {})
    }

    /// As [`Rig::agy`], after `adjust` has changed the tuning.
    pub fn agy_with(scenario: &str, adjust: impl FnOnce(&mut Tuning)) -> Self {
        let (record, temp) = (TempDir::new().unwrap(), TempDir::new().unwrap());
        let mut t = tuning(Tool::Agy, scenario, record.path(), temp.path());
        adjust(&mut t);
        let advisor = AgyAdvisor::custom(Source::Program(double(Tool::Agy)), t);
        Self {
            advisor,
            record,
            temp,
        }
    }
}

impl<A: AutoAdvisor> Rig<A> {
    /// Runs the full probe and waits for it.
    pub fn probe(&self) {
        self.advisor.refresh();
        self.advisor.settle(Duration::from_secs(30));
    }

    /// As [`Rig::probe`], for a check the person asked for (Preferences opening).
    pub fn probe_asked(&self) {
        self.advisor.refresh_asked();
        self.advisor.settle(Duration::from_secs(30));
    }

    /// Every run the double saw, in the order they started.
    pub fn runs(&self) -> Vec<Run> {
        read_runs(self.record.path())
    }

    /// The runs that were requests.
    pub fn requests(&self) -> Vec<Run> {
        self.runs().into_iter().filter(Run::is_request).collect()
    }

    /// How many `ym-auto-*` folders are left in this rig's temporary folder.
    pub fn leftovers(&self) -> usize {
        leftovers(self.temp.path())
    }
}

/// The `ym-auto-*` folders in `dir`.
pub fn leftovers(dir: &Path) -> usize {
    std::fs::read_dir(dir)
        .map(|d| {
            d.filter_map(Result::ok)
                .filter(|e| e.file_name().to_string_lossy().starts_with("ym-auto-"))
                .count()
        })
        .unwrap_or(0)
}

/// The runs recorded in `record`, oldest first.
pub fn read_runs(record: &Path) -> Vec<Run> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(record)
        .map(|d| {
            d.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| {
                    p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("run-"))
                })
                .collect()
        })
        .unwrap_or_default();
    dirs.sort();
    dirs.iter()
        .map(|dir| {
            let text = |name: &str| std::fs::read_to_string(dir.join(name)).unwrap_or_default();
            let argv: Vec<String> = serde_json::from_str(&text("argv.json")).unwrap_or_default();
            let env = text("env.txt")
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_owned(), v.to_owned()))
                .collect();
            Run {
                argv,
                cwd: text("cwd.txt"),
                files: text("files.txt").lines().map(str::to_owned).collect(),
                env,
                stdin: std::fs::read(dir.join("stdin.bin")).unwrap_or_default(),
            }
        })
        .collect()
}

/// Locations for a home folder: the native rules, no `PATH`, no shell.
pub fn locations_in(home: &Path) -> Locations {
    Locations {
        os: Os::current(),
        home: Some(home.to_path_buf()),
        local_app_data: Some(home.join("AppData").join("Local")),
        app_data: Some(home.join("AppData").join("Roaming")),
        path: None,
        shell: None,
    }
}

/// Copies `tool`'s double to `dir/<name>` (made if needed) and returns the path.
pub fn install(tool: Tool, dir: &Path, name: &str) -> PathBuf {
    std::fs::create_dir_all(dir).unwrap();
    let target = dir.join(name);
    std::fs::copy(double(tool), &target).unwrap();
    target
}

/// The path the native installer would use for `tool` under `home`.
pub fn known_path(tool: Tool, home: &Path) -> PathBuf {
    if cfg!(windows) {
        match tool {
            Tool::Claude => home.join(".local").join("bin").join("claude.exe"),
            Tool::Agy => home
                .join("AppData")
                .join("Local")
                .join("agy")
                .join("bin")
                .join("agy.exe"),
        }
    } else {
        home.join(".local").join("bin").join(tool.program())
    }
}

/// Whether the process tree a `slow` double started is gone: its pids (`pidfile`) are not running and the
/// grandchild's heartbeat has stopped.
pub fn assert_tree_dead(pidfile: &Path) {
    let beat = PathBuf::from(format!("{}.beat", pidfile.display()));
    let before = std::fs::read_to_string(&beat).unwrap_or_default();
    std::thread::sleep(Duration::from_millis(400));
    let after = std::fs::read_to_string(&beat).unwrap_or_default();
    assert_eq!(before, after, "the grandchild is still beating");
    #[cfg(unix)]
    for pid in std::fs::read_to_string(pidfile).unwrap_or_default().lines() {
        assert!(!pid_running(pid), "pid {pid} is still running");
    }
}

/// The pids a double wrote to `pidfile`: its own first, then its grandchild's when it has one.
pub fn pids_in(pidfile: &Path) -> Vec<String> {
    std::fs::read_to_string(pidfile)
        .unwrap_or_default()
        .lines()
        .map(str::to_owned)
        .collect()
}

/// Waits for the `slow` double to have written its pid file and its grandchild's first heartbeat.
pub fn wait_for_slow(pidfile: &Path) {
    let beat = PathBuf::from(format!("{}.beat", pidfile.display()));
    for _ in 0..200 {
        if pidfile.is_file() && beat.is_file() {
            return;
        }
        std::thread::sleep(Duration::from_millis(25));
    }
    panic!("the slow double did not start");
}

/// Whether `pid` is a running process (a zombie is not).
#[cfg(unix)]
pub fn pid_running(pid: &str) -> bool {
    if let Ok(stat) = std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        // "<pid> (<name>) <state> ..."; the name may hold spaces and parentheses.
        let state = stat
            .rsplit(')')
            .next()
            .and_then(|rest| rest.trim().chars().next());
        return state != Some('Z');
    }
    if Path::new("/proc/self").exists() {
        return false;
    }
    std::process::Command::new("ps")
        .args(["-o", "stat=", "-p", pid])
        .output()
        .map(|o| {
            let text = String::from_utf8_lossy(&o.stdout);
            let stat = text.trim();
            !stat.is_empty() && !stat.starts_with('Z')
        })
        .unwrap_or(false)
}
