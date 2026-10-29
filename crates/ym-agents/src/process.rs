//! Starting the person's programs (`docs/22` section 5): the one place a child process is made.
//!
//! [`run`] starts a program in its own process group (Unix) or job object (Windows), feeds it standard
//! input from its own thread, reads standard output and standard error from two more with caps, and
//! polls the child against a deadline and a [`CancelToken`]. On expiry or cancel it ends the whole tree
//! and always reaps it. A [`CancelSource`] also keeps the children that took its tokens, so that
//! [`CancelSource::cancel`] tells them to stop before it returns, whatever their own threads are doing.
//! Nothing here goes through a shell, and no `.cmd` file is ever started.

use std::ffi::{OsStr, OsString};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime};

use process_wrap::std::{ChildWrapper, CommandWrap};

/// The most a program may write to standard output before it is killed (`docs/22` section 5).
pub const STDOUT_CAP_BYTES: usize = 4 * 1024 * 1024;

/// The longest single line of standard output.
pub const LINE_CAP_BYTES: usize = 1024 * 1024;

/// How much of the end of standard error is kept.
pub const STDERR_TAIL_BYTES: usize = 4 * 1024;

/// The longest message that reaches a log or a person, in characters.
pub const MESSAGE_MAX_CHARS: usize = 300;

/// How often the main loop looks at the child, the deadline and the cancel token.
const POLL: Duration = Duration::from_millis(20);

/// How long a program that was asked to stop (`SIGTERM`) has before the group is killed.
#[cfg(unix)]
const TERM_GRACE: Duration = Duration::from_secs(2);

/// How long to wait for a pipe that a grandchild may still hold after the child is gone.
const PIPE_GRACE: Duration = Duration::from_secs(1);

/// The age after which a leftover `ym-auto-*` folder is swept (`docs/22` section 2.1).
pub const STALE_AFTER: Duration = Duration::from_secs(10 * 60);

/// The prefix of every temporary folder this crate makes.
pub const WORKDIR_PREFIX: &str = "ym-auto-";

/// `SIGTERM` on every Unix Yellowmoon runs on (Linux and macOS agree).
#[cfg(unix)]
const SIGTERM: i32 = 15;

/// `SIGCONT`: 18 on Linux, 19 on macOS and the BSDs. A stopped process group does not act on a pending
/// `SIGTERM` until it is continued, so every `SIGTERM` is followed by this one.
#[cfg(all(unix, any(target_os = "linux", target_os = "android")))]
const SIGCONT: i32 = 18;
#[cfg(all(unix, not(any(target_os = "linux", target_os = "android"))))]
const SIGCONT: i32 = 19;

// ---------------------------------------------------------------------------------------------
// Cancelling
// ---------------------------------------------------------------------------------------------

/// A running child, shared between the run that started it (which polls and reaps it) and the
/// [`CancelSource`] whose token it took (which may tell it to stop).
type SharedChild = Arc<Mutex<Box<dyn ChildWrapper>>>;

/// What a [`CancelSource`] knows: how often it cancelled, and the children that are running now.
#[derive(Debug, Default)]
struct SourceState {
    epoch: AtomicU64,
    next_id: AtomicU64,
    children: Mutex<Vec<(u64, SharedChild)>>,
}

/// Whoever may cancel the runs that took a token from it: one per advisor. Cancelling ends every run
/// that was started before it and leaves later runs alone.
///
/// The source also keeps the children of its runs, from their spawn to their reaping, so that
/// [`CancelSource::cancel`] does not depend on the thread of each run to get round to it.
#[derive(Debug, Clone, Default)]
pub struct CancelSource {
    state: Arc<SourceState>,
}

impl CancelSource {
    /// A new source that has cancelled nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// A token for a run that starts now.
    #[must_use]
    pub fn token(&self) -> CancelToken {
        CancelToken {
            source: Some(Arc::clone(&self.state)),
            started: self.state.epoch.load(Ordering::SeqCst),
        }
    }

    /// Cancels every run that took its token before this call, and tells their children to stop
    /// **before it returns**: `SIGTERM` and `SIGCONT` to each process group on Unix (the `SIGCONT` so
    /// that a stopped group acts on the `SIGTERM`), `TerminateJobObject` on Windows. A caller that exits
    /// right after, the app quitting or `ym`'s Ctrl-C handler, has thereby told every child to go.
    ///
    /// What follows is the runs' own: each notices the cancel within 20 ms, sends `SIGKILL` to a group
    /// that has not gone 2 s after the `SIGTERM`, reaps it and removes its temporary folder.
    /// [`AutoAdvisor::wait_idle`](ym_core::apple_auto::AutoAdvisor::wait_idle) waits for that.
    ///
    /// A run that has not started its child yet starts none: its token says it is cancelled.
    pub fn cancel(&self) {
        // The epoch moves first. A run takes the list's lock to check its token and spawn, so it is
        // either already in the list below, or sees the new epoch and spawns nothing.
        self.state.epoch.fetch_add(1, Ordering::SeqCst);
        let children = lock(&self.state.children);
        for (_, child) in children.iter() {
            signal_stop(lock(child).as_mut());
        }
    }
}

/// Tells a run that it was cancelled. Cheap to clone and to poll.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    source: Option<Arc<SourceState>>,
    started: u64,
}

impl CancelToken {
    /// A token that is never cancelled.
    #[must_use]
    pub fn never() -> Self {
        Self::default()
    }

    /// Whether the source cancelled since this token was made.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.source
            .as_ref()
            .is_some_and(|state| state.epoch.load(Ordering::SeqCst) != self.started)
    }
}

/// The entry of a running child in its source's list. Dropping it takes the child out: after that
/// [`CancelSource::cancel`] cannot reach it, which is what makes it safe to reap (a reaped group's
/// number may be reused).
struct Registration {
    state: Arc<SourceState>,
    id: u64,
}

impl Drop for Registration {
    fn drop(&mut self) {
        lock(&self.state.children).retain(|(id, _)| *id != self.id);
    }
}

/// Tells a child to stop: `SIGTERM` then `SIGCONT` to its group (Unix), the job's termination
/// (Windows). Errors (the group is gone already) are ignored.
fn signal_stop(child: &mut dyn ChildWrapper) {
    #[cfg(unix)]
    {
        let _ = child.signal(SIGTERM);
        let _ = child.signal(SIGCONT);
    }
    #[cfg(not(unix))]
    {
        let _ = child.start_kill();
    }
}

// ---------------------------------------------------------------------------------------------
// The request
// ---------------------------------------------------------------------------------------------

/// Which environment changes a child gets (`docs/22` section 5). The environment is inherited; these
/// are the changes on top of it, and nothing is read from it or printed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvProfile {
    /// No change: the person's shell lookup.
    Inherit,
    /// Claude Code: its telemetry and update switches off, `NO_COLOR`, and the variables that
    /// describe a parent Claude Code session removed.
    Claude,
    /// Antigravity: `NO_COLOR` only (no telemetry switch is documented for it).
    Agy,
}

/// What to start and how (`docs/22` section 5).
#[derive(Debug, Clone)]
pub struct RunSpec {
    /// The executable: an absolute path, never a script.
    pub program: PathBuf,
    /// The arguments, one by one; never through a shell.
    pub args: Vec<OsString>,
    /// The environment changes.
    pub env: EnvProfile,
    /// Further variables set last: the tests' scenario for a test double, nothing in production.
    pub extra_env: Vec<(OsString, OsString)>,
    /// The working folder.
    pub cwd: PathBuf,
    /// The bytes written to standard input, which is then closed.
    pub stdin: Vec<u8>,
    /// The time limit; the tree is killed at expiry.
    pub timeout: Duration,
    /// Ends the run early.
    pub cancel: CancelToken,
    /// The cap on standard output.
    pub stdout_cap: usize,
    /// The cap on one line of standard output.
    pub line_cap: usize,
    /// Unix: start the program in a new session (`setsid`), so that it has no controlling terminal.
    /// The person's shell, asked where a program is, needs it: an interactive zsh or bash that finds
    /// itself in a background process group of a terminal stops itself (`SIGTTIN`) and waits for a
    /// foreground that never comes. Ignored on Windows; the program is still in a group of its own.
    pub new_session: bool,
}

impl RunSpec {
    /// A run of `program` in `cwd` with no arguments, no input, a minute to finish and the usual caps.
    #[must_use]
    pub fn new(program: impl Into<PathBuf>, cwd: impl Into<PathBuf>, env: EnvProfile) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            env,
            extra_env: Vec::new(),
            cwd: cwd.into(),
            stdin: Vec::new(),
            timeout: Duration::from_secs(60),
            cancel: CancelToken::never(),
            stdout_cap: STDOUT_CAP_BYTES,
            line_cap: LINE_CAP_BYTES,
            new_session: false,
        }
    }

    /// Adds arguments.
    #[must_use]
    pub fn args<I, S>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<OsStr>,
    {
        self.args
            .extend(args.into_iter().map(|a| a.as_ref().to_os_string()));
        self
    }

    /// Sets the time limit.
    #[must_use]
    pub fn timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Sets the bytes written to standard input.
    #[must_use]
    pub fn stdin(mut self, bytes: Vec<u8>) -> Self {
        self.stdin = bytes;
        self
    }

    /// Sets the token that ends the run early.
    #[must_use]
    pub fn cancel(mut self, cancel: CancelToken) -> Self {
        self.cancel = cancel;
        self
    }

    /// Sets the variables added after every other environment change.
    #[must_use]
    pub fn extra_env(mut self, vars: Vec<(OsString, OsString)>) -> Self {
        self.extra_env = vars;
        self
    }
}

/// How a run ended (`docs/22` section 5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// The program ended by itself; its code is in [`RunOutput::exit_code`] (`None` when a signal
    /// ended it).
    Exited,
    /// The time limit ran out and the tree was killed.
    TimedOut,
    /// The cancel token fired and the tree was killed.
    Cancelled,
    /// Standard output grew past its cap (or a line past its cap) and the tree was killed.
    OutputTooLarge,
    /// The program could not be started; the system's message, redacted and cut.
    SpawnFailed(String),
}

/// What a run produced.
#[derive(Debug, Clone)]
pub struct RunOutput {
    /// How it ended.
    pub outcome: Outcome,
    /// The exit code, when the program ended by itself with one.
    pub exit_code: Option<i32>,
    /// Standard output, up to the cap.
    pub stdout: Vec<u8>,
    /// The end of standard error, at most [`STDERR_TAIL_BYTES`].
    pub stderr: Vec<u8>,
    /// How long the run took.
    pub elapsed: Duration,
}

impl RunOutput {
    /// Whether the program ended by itself with exit code 0.
    #[must_use]
    #[cfg_attr(not(any(test, feature = "test-support")), allow(dead_code))]
    pub fn success(&self) -> bool {
        self.outcome == Outcome::Exited && self.exit_code == Some(0)
    }

    /// Standard output as text (lossy).
    #[must_use]
    pub fn stdout_text(&self) -> String {
        String::from_utf8_lossy(&self.stdout).into_owned()
    }

    /// The end of standard error as text, redacted and cut to [`MESSAGE_MAX_CHARS`] characters from
    /// its end (the last words are the useful ones).
    #[must_use]
    pub fn stderr_message(&self) -> String {
        tail_message(&String::from_utf8_lossy(&self.stderr))
    }
}

// ---------------------------------------------------------------------------------------------
// Redaction
// ---------------------------------------------------------------------------------------------

/// `text` with anything that looks like a token replaced by `…` (`docs/22` section 5): `sk-…`,
/// `sk-ant-…`, `AIza…`, `ya29.…`, the word after `Bearer`, a JWT (`eyJ…` with two dots), and any run of
/// 40 or more base64 characters (letters, digits, `-`, `_`, `+`, `=` and `/`): a hex or base64 value, or
/// an AWS-style secret key. A run that holds a `/` must also have upper and lower case letters and a
/// digit, so that an ordinary path of 40 characters or more survives.
#[must_use]
pub fn redact(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut after_bearer = false;
    let mut word = String::new();
    let flush = |word: &mut String, out: &mut String, after_bearer: &mut bool| {
        if word.is_empty() {
            return;
        }
        let secret = *after_bearer || looks_like_token(word);
        *after_bearer = word.eq_ignore_ascii_case("bearer");
        if secret {
            out.push('…');
        } else {
            out.push_str(word);
        }
        word.clear();
    };
    for c in text.chars() {
        // A word is a run of what tokens are made of, a `/` and a `.` included (a secret key has
        // slashes, a JWT and `ya29.` have dots); a space, a quote or a comma ends it.
        if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '+' | '=' | '/') {
            word.push(c);
        } else {
            flush(&mut word, &mut out, &mut after_bearer);
            out.push(c);
        }
    }
    flush(&mut word, &mut out, &mut after_bearer);
    out
}

/// The shortest run of base64 characters that is taken for a credential.
const SECRET_RUN: usize = 40;

/// Whether one word looks like a credential.
fn looks_like_token(word: &str) -> bool {
    const PREFIXES: [&str; 3] = ["sk-", "AIza", "ya29."];
    // What follows an `=` is a value of its own (`API_KEY=sk-ant-…`).
    for part in word.split('=') {
        if PREFIXES.iter().any(|p| part.starts_with(p)) {
            return true;
        }
        // A JWT: three base64url parts joined by dots, the first of which is `{"` encoded.
        if part.starts_with("eyJ") && part.matches('.').count() >= 2 {
            return true;
        }
    }
    // A long run, between dots: hex or base64 (any letters), or, when it holds a slash, one that is no
    // path (it has both cases and a digit).
    word.split('.').any(|run| {
        run.len() >= SECRET_RUN && (!run.contains('/') || has_upper_lower_and_digit(run))
    })
}

/// Whether `run` has an upper case letter, a lower case letter and a digit.
fn has_upper_lower_and_digit(run: &str) -> bool {
    run.bytes().any(|b| b.is_ascii_uppercase())
        && run.bytes().any(|b| b.is_ascii_lowercase())
        && run.bytes().any(|b| b.is_ascii_digit())
}

/// `text` cut to `max` characters, with `…` at the cut.
#[must_use]
pub fn clip(text: &str, max: usize) -> String {
    let text = text.trim();
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let mut cut: String = text.chars().take(max.saturating_sub(1)).collect();
    cut.push('…');
    cut
}

/// A message for a log or a person: redacted, trimmed, at most [`MESSAGE_MAX_CHARS`] characters.
#[must_use]
pub fn message(text: &str) -> String {
    clip(&redact(text), MESSAGE_MAX_CHARS)
}

/// The last [`MESSAGE_MAX_CHARS`] characters of `text`, redacted.
#[must_use]
pub fn tail_message(text: &str) -> String {
    let redacted = redact(text);
    let trimmed = redacted.trim();
    let count = trimmed.chars().count();
    if count <= MESSAGE_MAX_CHARS {
        return trimmed.to_owned();
    }
    let tail: String = trimmed.chars().skip(count - (MESSAGE_MAX_CHARS - 1)).collect();
    format!("…{tail}")
}

/// The JSON object a program printed (`docs/22` section 8.3): the last line that starts an object at
/// its beginning and holds one, an object that is printed over several lines (from its `{` line on)
/// included. What follows the object is not read: a banner or a warning before it, and a notice or
/// "Done" after it, do not matter. A line that starts like an object but is none is passed over.
#[must_use]
pub fn last_json_object(output: &[u8]) -> Option<serde_json::Value> {
    let text = String::from_utf8_lossy(output);
    let text = text.trim();
    // Every line that could start the object: at its first column, as the programs print it (an
    // object nested in a pretty-printed one is indented).
    let mut starts: Vec<usize> = text
        .match_indices('\n')
        .map(|(at, _)| at + 1)
        .filter(|at| text[*at..].starts_with('{'))
        .collect();
    if text.starts_with('{') {
        starts.insert(0, 0);
    }
    for &start in starts.iter().rev() {
        // The first value of what follows; the lines after it are left unread.
        let first = serde_json::Deserializer::from_str(&text[start..])
            .into_iter::<serde_json::Value>()
            .next();
        if let Some(Ok(value)) = first
            && value.is_object()
        {
            return Some(value);
        }
    }
    None
}

// ---------------------------------------------------------------------------------------------
// The temporary folder
// ---------------------------------------------------------------------------------------------

/// The private folder of one request (`docs/22` section 5): `cwd/` is the working folder and `in/` holds
/// the files the program is told about. Deleted when dropped, on every path including a panic.
#[derive(Debug)]
pub struct Workdir {
    dir: tempfile::TempDir,
    cwd: PathBuf,
    input: PathBuf,
}

impl Workdir {
    /// Makes `ym-auto-*` in the operating system's temporary folder (mode 0700 on Unix, set here: the
    /// `tempfile` crate leaves it to the umask) with its two empty subfolders.
    pub fn new() -> std::io::Result<Self> {
        Self::new_in(&std::env::temp_dir())
    }

    /// As [`Workdir::new`], in another folder (the tests'sweep).
    pub fn new_in(parent: &Path) -> std::io::Result<Self> {
        let mut builder = tempfile::Builder::new();
        builder.prefix(WORKDIR_PREFIX);
        // `tempfile` makes the folder with the process's umask; the user's alone is the spec's.
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            builder.permissions(std::fs::Permissions::from_mode(0o700));
        }
        let dir = builder.tempdir_in(parent)?;
        let cwd = dir.path().join("cwd");
        let input = dir.path().join("in");
        make_private_dir(&cwd)?;
        make_private_dir(&input)?;
        Ok(Self { dir, cwd, input })
    }

    /// The folder itself.
    #[must_use]
    pub fn path(&self) -> &Path {
        self.dir.path()
    }

    /// The working folder: empty for Claude Code, only `photo.jpg` for `agy`.
    #[must_use]
    pub fn cwd(&self) -> &Path {
        &self.cwd
    }

    /// The folder for the files the program is told about (`instructions.txt`, `schema.json`).
    #[must_use]
    pub fn input(&self) -> &Path {
        &self.input
    }
}

/// Makes `path` readable by the user alone (mode 0700 on Unix; on Windows the folder inherits its
/// parent's access, which is the user's own temporary folder).
fn make_private_dir(path: &Path) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::DirBuilderExt;
        std::fs::DirBuilder::new().mode(0o700).create(path)
    }
    #[cfg(not(unix))]
    {
        std::fs::create_dir(path)
    }
}

/// Deletes the `ym-auto-*` folders of `parent` whose last change is older than `older_than`, which
/// only a crash can leave behind; a newer one and any other folder stay. Returns how many it removed.
/// Errors (a folder of another user, one that vanished) are ignored.
pub fn sweep_stale_in(parent: &Path, older_than: Duration) -> usize {
    let Ok(entries) = std::fs::read_dir(parent) else {
        return 0;
    };
    let now = SystemTime::now();
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name();
        if !name.to_string_lossy().starts_with(WORKDIR_PREFIX) {
            continue;
        }
        // A link is never followed: only a real folder is ours.
        let Ok(meta) = std::fs::symlink_metadata(entry.path()) else {
            continue;
        };
        if !meta.is_dir() {
            continue;
        }
        let Ok(modified) = meta.modified() else {
            continue;
        };
        let age = now.duration_since(modified).unwrap_or(Duration::ZERO);
        if age >= older_than && std::fs::remove_dir_all(entry.path()).is_ok() {
            removed += 1;
        }
    }
    removed
}

/// [`sweep_stale_in`] for the operating system's temporary folder and [`STALE_AFTER`].
pub fn sweep_stale() -> usize {
    sweep_stale_in(&std::env::temp_dir(), STALE_AFTER)
}

// ---------------------------------------------------------------------------------------------
// Running
// ---------------------------------------------------------------------------------------------

/// Owns the child so that dropping it, on any path including a panic, ends the whole tree and reaps
/// it (`docs/22` section 5, *Panics*). `process-wrap`'s standard-library API has no kill-on-drop of
/// its own (its `KillOnDrop` is for Tokio).
struct Running {
    child: SharedChild,
    /// The entry in the cancel source's list, until the child is about to be reaped.
    registration: Option<Registration>,
    /// The tree was killed and reaped: dropping has nothing left to do (and must not signal a group
    /// whose number may have been reused).
    finished: bool,
}

impl Running {
    /// Looks at the child without blocking: its status when it has exited. This asks the native child
    /// (the innermost wrapper): the group wrapper's own `try_wait` waits for the whole group.
    fn exited(&self) -> std::io::Result<Option<std::process::ExitStatus>> {
        lock(&self.child).inner_mut().try_wait()
    }

    /// Kills every process of the tree at once.
    fn kill_tree(&mut self) {
        let _ = lock(&self.child).start_kill();
    }

    /// Asks the tree to stop politely where the system has such a thing (`SIGTERM` to the group, then
    /// `SIGCONT` so that a stopped group acts on it).
    fn ask_to_stop(&mut self) {
        signal_stop(lock(&self.child).as_mut());
    }

    /// Reaps the child and the rest of its tree, after it was killed or has exited. The child leaves
    /// the cancel source's list first: from then on nobody may signal its group.
    fn reap(&mut self) -> Option<std::process::ExitStatus> {
        self.registration.take();
        let status = lock(&self.child).wait().ok();
        self.finished = true;
        status
    }
}

impl Drop for Running {
    fn drop(&mut self) {
        if !self.finished {
            self.registration.take();
            let mut child = lock(&self.child);
            let _ = child.start_kill();
            let _ = child.wait();
        }
    }
}

/// Why a child was not started.
enum StartError {
    /// The token was cancelled before the start: nothing was spawned.
    Cancelled,
    /// The system could not start the program.
    Failed(std::io::Error),
}

/// Shared state of one run's reader threads.
struct Pipes {
    stdout: Arc<Mutex<Vec<u8>>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    too_large: Arc<AtomicBool>,
    done: [Arc<AtomicBool>; 3],
}

/// Starts the program and waits for it (`docs/22` section 5). Never panics; every way a run can end is an
/// [`Outcome`].
#[must_use]
pub fn run(spec: RunSpec) -> RunOutput {
    run_with(spec, || {})
}

/// [`run`], calling `after_spawn` once the child is running and before the first poll. The tests make it
/// panic, to see that the guard ends the tree of a request whose caller panics.
#[must_use]
pub fn run_with(spec: RunSpec, after_spawn: impl FnOnce()) -> RunOutput {
    let started = Instant::now();
    let fail = |outcome| RunOutput {
        outcome,
        exit_code: None,
        stdout: Vec::new(),
        stderr: Vec::new(),
        elapsed: started.elapsed(),
    };

    let mut wrap = CommandWrap::with_new(&spec.program, |command| {
        command
            .args(&spec.args)
            .current_dir(&spec.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        apply_env(command, spec.env, &spec.extra_env);
    });
    wrap_for_platform(&mut wrap, spec.new_session);

    // Argument lists and standard error are logged at debug level only, with anything that looks like a
    // token cut out (`docs/22` section 2.1).
    tracing::debug!(
        program = %spec.program.display(),
        args = %message(&spec.args.iter().map(|a| a.to_string_lossy()).collect::<Vec<_>>().join(" ")),
        "starting a program"
    );
    let (child, registration) = match start_child(&mut wrap, &spec.cancel) {
        Ok(started) => started,
        Err(StartError::Cancelled) => return fail(Outcome::Cancelled),
        Err(StartError::Failed(error)) => {
            return fail(Outcome::SpawnFailed(message(&error.to_string())));
        }
    };
    let (stdin, stdout, stderr) = {
        let mut child = lock(&child);
        (
            child.stdin().take(),
            child.stdout().take(),
            child.stderr().take(),
        )
    };
    let mut running = Running {
        child,
        registration,
        finished: false,
    };

    after_spawn();

    let pipes = Pipes {
        stdout: Arc::new(Mutex::new(Vec::new())),
        stderr: Arc::new(Mutex::new(Vec::new())),
        too_large: Arc::new(AtomicBool::new(false)),
        done: [
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicBool::new(false)),
        ],
    };

    // Standard input from its own thread: a program that fails early never reads it, and a full pipe
    // must not stop us from reading the output (a broken pipe is not an error).
    {
        let input = spec.stdin.clone();
        let done = Arc::clone(&pipes.done[0]);
        std::thread::spawn(move || {
            if let Some(mut stdin) = stdin {
                let _ = stdin.write_all(&input);
                let _ = stdin.flush();
            }
            done.store(true, Ordering::SeqCst);
        });
    }
    {
        let buffer = Arc::clone(&pipes.stdout);
        let too_large = Arc::clone(&pipes.too_large);
        let done = Arc::clone(&pipes.done[1]);
        let (cap, line_cap) = (spec.stdout_cap, spec.line_cap);
        std::thread::spawn(move || {
            if let Some(stdout) = stdout {
                read_capped(stdout, &buffer, cap, line_cap, &too_large);
            }
            done.store(true, Ordering::SeqCst);
        });
    }
    {
        let buffer = Arc::clone(&pipes.stderr);
        let done = Arc::clone(&pipes.done[2]);
        std::thread::spawn(move || {
            if let Some(stderr) = stderr {
                read_tail(stderr, &buffer, STDERR_TAIL_BYTES);
            }
            done.store(true, Ordering::SeqCst);
        });
    }

    let deadline = started + spec.timeout;
    let mut status = None;
    let outcome = loop {
        // The cancel comes first: `CancelSource::cancel` has already told the child to stop, so a
        // child that is gone may have been ended by it, and that is a cancel, not an exit.
        if spec.cancel.is_cancelled() {
            break Outcome::Cancelled;
        }
        if pipes.too_large.load(Ordering::SeqCst) {
            break Outcome::OutputTooLarge;
        }
        match running.exited() {
            Ok(Some(exit)) => {
                status = Some(exit);
                // The epoch moves before the signal is sent, so a child that the cancel ended shows
                // the cancel here.
                break if spec.cancel.is_cancelled() {
                    Outcome::Cancelled
                } else {
                    Outcome::Exited
                };
            }
            Ok(None) => {}
            // The child cannot be looked at: treat it as gone and let the reaping say the rest.
            Err(_) => break Outcome::Exited,
        }
        if Instant::now() >= deadline {
            break Outcome::TimedOut;
        }
        std::thread::sleep(POLL);
    };

    match outcome {
        Outcome::Exited => {
            // Anything the program left behind in its group would hold our pipes open.
            running.kill_tree();
        }
        _ => stop_tree(&mut running),
    }
    let reaped = running.reap();
    let exit = status.or(reaped);
    drop(running);

    // Give the readers a moment to hit end-of-file; a pipe a grandchild escaped with is left alone.
    let wait_until = Instant::now() + PIPE_GRACE;
    while pipes.done.iter().any(|d| !d.load(Ordering::SeqCst)) && Instant::now() < wait_until {
        std::thread::sleep(Duration::from_millis(5));
    }

    let stdout = lock(&pipes.stdout).clone();
    let stderr = lock(&pipes.stderr).clone();
    let outcome = if pipes.too_large.load(Ordering::SeqCst) {
        Outcome::OutputTooLarge
    } else {
        outcome
    };
    let output = RunOutput {
        exit_code: if outcome == Outcome::Exited {
            exit.and_then(|s| s.code())
        } else {
            None
        },
        outcome,
        stdout,
        stderr,
        elapsed: started.elapsed(),
    };
    tracing::debug!(
        outcome = ?output.outcome,
        exit_code = ?output.exit_code,
        elapsed_ms = output.elapsed.as_millis() as u64,
        stderr = %output.stderr_message(),
        "a program ended"
    );
    output
}

/// Starts the child and, when `cancel` has a source, puts it in the source's list before anything
/// else can look at that list. The list's lock is held from the check of the token to the entry, so
/// [`CancelSource::cancel`] either finds the child there or has moved the epoch first, in which case
/// nothing is started.
///
/// A program whose file is still open for writing ("text file busy", `ETXTBSY`) is tried again for up
/// to half a second. It happens while a program is being updated by its own updater, and whenever
/// another thread of this process forks while the file was being written.
fn start_child(
    wrap: &mut CommandWrap,
    cancel: &CancelToken,
) -> Result<(SharedChild, Option<Registration>), StartError> {
    let mut attempts = 0;
    loop {
        let mut list = cancel.source.as_ref().map(|state| lock(&state.children));
        if cancel.is_cancelled() {
            return Err(StartError::Cancelled);
        }
        match wrap.spawn() {
            Ok(child) => {
                let child: SharedChild = Arc::new(Mutex::new(child));
                let registration = match (cancel.source.as_ref(), list.as_mut()) {
                    (Some(state), Some(list)) => {
                        let id = state.next_id.fetch_add(1, Ordering::SeqCst);
                        list.push((id, Arc::clone(&child)));
                        Some(Registration {
                            state: Arc::clone(state),
                            id,
                        })
                    }
                    _ => None,
                };
                return Ok((child, registration));
            }
            Err(error)
                if error.kind() == std::io::ErrorKind::ExecutableFileBusy && attempts < 10 =>
            {
                attempts += 1;
                drop(list);
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(error) => return Err(StartError::Failed(error)),
        }
    }
}

/// Ends the tree after a time-out, a cancel or an oversized output: `SIGTERM` (and `SIGCONT`, for a
/// stopped group) to the group, up to two seconds for it to leave, then `SIGKILL` to the group
/// (Windows: the job is terminated at once).
fn stop_tree(running: &mut Running) {
    running.ask_to_stop();
    #[cfg(unix)]
    {
        let until = Instant::now() + TERM_GRACE;
        while Instant::now() < until {
            if matches!(running.exited(), Ok(Some(_)) | Err(_)) {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    running.kill_tree();
}

#[cfg(unix)]
fn wrap_for_platform(wrap: &mut CommandWrap, new_session: bool) {
    if new_session {
        // A session of its own is a group of its own, with no controlling terminal.
        wrap.wrap(process_wrap::std::ProcessSession);
    } else {
        wrap.wrap(process_wrap::std::ProcessGroup::leader());
    }
}

#[cfg(windows)]
fn wrap_for_platform(wrap: &mut CommandWrap, _new_session: bool) {
    use windows::Win32::System::Threading::CREATE_NO_WINDOW;
    // The flags are registered with the wrapper, so the job object's own start flag does not
    // overwrite them (the crate merges them in whichever order they were added).
    wrap.wrap(process_wrap::std::CreationFlags(CREATE_NO_WINDOW));
    wrap.wrap(process_wrap::std::JobObject);
}

#[cfg(not(any(unix, windows)))]
fn wrap_for_platform(_wrap: &mut CommandWrap, _new_session: bool) {}

/// Applies the environment changes of `profile` and then `extra` to `command`.
fn apply_env(command: &mut std::process::Command, profile: EnvProfile, extra: &[(OsString, OsString)]) {
    match profile {
        EnvProfile::Inherit => {}
        EnvProfile::Claude => {
            for (name, value) in [
                ("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC", "1"),
                // The next three are for versions that predate the first.
                ("DISABLE_TELEMETRY", "1"),
                ("DISABLE_ERROR_REPORTING", "1"),
                ("DISABLE_AUTOUPDATER", "1"),
                ("NO_COLOR", "1"),
            ] {
                command.env(name, value);
            }
            // Variables that describe a parent Claude Code session, when `ym` runs inside one. Only the
            // names are looked at; no value is read.
            for name in PARENT_SESSION_VARS {
                command.env_remove(name);
            }
            for (name, _) in std::env::vars_os() {
                if name.to_string_lossy().starts_with("CLAUDE_CODE_REMOTE") {
                    command.env_remove(&name);
                }
            }
        }
        EnvProfile::Agy => {
            command.env("NO_COLOR", "1");
        }
    }
    for (name, value) in extra {
        command.env(name, value);
    }
}

/// The variables of a parent Claude Code session that a child must not inherit (`docs/22` section 5;
/// the list is unverified, section 19).
const PARENT_SESSION_VARS: [&str; 4] = [
    "CLAUDECODE",
    "CLAUDE_CODE_ENTRYPOINT",
    "CLAUDE_CODE_SESSION_ID",
    "CLAUDE_CODE_CHILD_SESSION",
];

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Reads `source` into `buffer` up to `cap` bytes and `line_cap` bytes per line; sets `too_large` and
/// stops when either is passed.
fn read_capped(
    mut source: impl Read,
    buffer: &Mutex<Vec<u8>>,
    cap: usize,
    line_cap: usize,
    too_large: &AtomicBool,
) {
    let mut chunk = [0_u8; 16 * 1024];
    let mut line = 0_usize;
    loop {
        let n = match source.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        let mut guard = lock(buffer);
        for &byte in &chunk[..n] {
            if byte == b'\n' {
                line = 0;
            } else {
                line += 1;
                if line > line_cap {
                    too_large.store(true, Ordering::SeqCst);
                    return;
                }
            }
        }
        let room = cap.saturating_sub(guard.len());
        if n > room {
            guard.extend_from_slice(&chunk[..room]);
            too_large.store(true, Ordering::SeqCst);
            return;
        }
        guard.extend_from_slice(&chunk[..n]);
    }
}

/// Reads `source` to its end keeping only the last `keep` bytes.
fn read_tail(mut source: impl Read, buffer: &Mutex<Vec<u8>>, keep: usize) {
    let mut chunk = [0_u8; 4 * 1024];
    loop {
        let n = match source.read(&mut chunk) {
            Ok(0) | Err(_) => return,
            Ok(n) => n,
        };
        let mut guard = lock(buffer);
        guard.extend_from_slice(&chunk[..n]);
        if guard.len() > keep {
            let excess = guard.len() - keep;
            guard.drain(..excess);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_are_redacted() {
        let text = "failed with sk-ant-api03-ABCDEFG and Bearer abc.def.ghi, key AIzaSyA1234 \
                    and ya29.a0AbC and 0123456789abcdef0123456789abcdef01234567 ok";
        let out = redact(text);
        assert!(!out.contains("ABCDEFG"), "{out}");
        assert!(!out.contains("abc.def.ghi"), "{out}");
        assert!(!out.contains("AIzaSyA1234"), "{out}");
        assert!(!out.contains("a0AbC"), "{out}");
        assert!(!out.contains("0123456789abcdef"), "{out}");
        assert!(out.contains("failed with"), "{out}");
        assert!(out.ends_with(" ok"), "{out}");
    }

    #[test]
    fn ordinary_words_and_paths_survive() {
        let text = "Claude Code could not be started: No such file or directory (os error 2) in /var/folders/ab/cd/T/ym-auto-x1";
        assert_eq!(redact(text), text);
    }

    #[test]
    fn messages_are_cut() {
        let long = "word ".repeat(200);
        assert!(message(&long).chars().count() <= MESSAGE_MAX_CHARS);
        let tail = tail_message(&format!("{}END", "x y ".repeat(300)));
        assert!(tail.chars().count() <= MESSAGE_MAX_CHARS);
        assert!(tail.ends_with("END"));
    }

    #[test]
    fn the_json_object_is_found_after_a_banner_or_over_several_lines() {
        let one = br#"{"a":1}"#;
        assert_eq!(last_json_object(one).unwrap()["a"], 1);
        assert_eq!(last_json_object(b"banner\n{\"a\":2}\n").unwrap()["a"], 2);
        assert_eq!(last_json_object(b"banner\n\n{\n  \"a\": 3,\n  \"b\": {\"c\": 4}\n}\n").unwrap()["a"], 3);
        assert_eq!(last_json_object(b"{\n \"a\": 5\n}\n").unwrap()["a"], 5);
        // Another object earlier in the output is not the answer; the last one is.
        assert_eq!(last_json_object(b"{\"a\":1}\n{\"a\":6}\n").unwrap()["a"], 6);
        assert!(last_json_object(b"").is_none());
        assert!(last_json_object(b"no json here\n[1,2]\n").is_none());
    }

    #[test]
    fn secret_shapes_that_are_not_prefixed_are_redacted() {
        // An AWS-style secret key: 40 base64 characters with slashes.
        let aws = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY";
        assert_eq!(aws.len(), 40);
        let out = redact(&format!("denied: secret {aws} for user"));
        assert_eq!(out, "denied: secret … for user");
        let out = redact(&format!("aws_secret_access_key={aws}\n"));
        assert!(!out.contains("EXAMPLEKEY") && !out.contains("wJalr"), "{out}");
        // A JWT, with its dots, alone and after a header name or an equals sign.
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.c2lnbmF0dXJl";
        assert_eq!(redact(&format!("Authorization: {jwt}")), "Authorization: …");
        assert_eq!(redact(&format!("token={jwt}!")), "…!");
        assert!(!redact(&format!("jwt {jwt}")).contains("c2lnbmF0"));
        // Two dots without the `eyJ` start are a version or a host name.
        assert_eq!(redact("com.yellowmoon.app 2.1.310"), "com.yellowmoon.app 2.1.310");
        // A key after an equals sign.
        assert_eq!(redact("ANTHROPIC_API_KEY=sk-ant-abc"), "…");
        // A long value without slashes (hex, base64url) goes as before, whatever its case.
        assert_eq!(redact(&"a".repeat(40)), "…");
        assert_eq!(redact(&"a".repeat(39)), "a".repeat(39));
    }

    #[test]
    fn ordinary_paths_survive_however_long() {
        for path in [
            "/usr/local/bin/claude",
            "/opt/homebrew/bin/claude",
            // Long, lower case, with digits: no upper case, no secret.
            "/home/runner/work/yellowmoon/yellowmoon/target/debug/build/ym-agents-1234567890abcdef",
            // Long, mixed case, no digit.
            "/Users/Nikita/Library/Application/Support/Yellowmoon/Caches/Previews",
        ] {
            assert_eq!(redact(path), path);
            assert_eq!(redact(&format!("spawn {path}: not found")), format!("spawn {path}: not found"));
        }
    }

    #[test]
    fn the_json_object_is_the_last_one_whatever_follows_it() {
        // A trailing line (a notice, "Done") after a one-line object.
        let envelope = br#"{"status":"SUCCESS","n":1}"#;
        let mut with_notice = b"banner\n".to_vec();
        with_notice.extend_from_slice(envelope);
        with_notice.extend_from_slice(b"\nDone\n");
        assert_eq!(last_json_object(&with_notice).unwrap()["n"], 1);
        with_notice.extend_from_slice(b"A new version is available: 1.5.0\n\n");
        assert_eq!(last_json_object(&with_notice).unwrap()["n"], 1);
        // `claude auth status`: its JSON, then a line.
        assert_eq!(
            last_json_object(b"{\"loggedIn\":true}\nrun claude update\n").unwrap()["loggedIn"],
            true
        );
        // The last object wins, a line that only looks like one is passed over.
        assert_eq!(last_json_object(b"{\"n\":1}\n{\"n\":2}\n{not json\nDone\n").unwrap()["n"], 2);
        // Printed over several lines, then a trailing line.
        let pretty = b"banner\n{\n  \"n\": 3,\n  \"list\": [\n    {\"x\": 1}\n  ]\n}\nDone\n";
        assert_eq!(last_json_object(pretty).unwrap()["n"], 3);
        // An earlier one-line object and a later pretty-printed one: the later one.
        assert_eq!(last_json_object(b"{\"n\":1}\n{\n \"n\": 4\n}\n").unwrap()["n"], 4);
        // Nothing that is an object.
        assert!(last_json_object(b"Done\n[1,2]\n\"text\"\n").is_none());
        assert!(last_json_object(b"{not json\nDone\n").is_none());
    }

    #[test]
    fn a_cancel_source_cancels_earlier_tokens_only() {
        let source = CancelSource::new();
        let before = source.token();
        assert!(!before.is_cancelled());
        source.cancel();
        assert!(before.is_cancelled());
        assert!(!source.token().is_cancelled());
        assert!(!CancelToken::never().is_cancelled());
    }

    #[test]
    fn sweeping_removes_only_old_ym_auto_folders() {
        let parent = tempfile::tempdir().unwrap();
        let old = Workdir::new_in(parent.path()).unwrap();
        let other = parent.path().join("keep-me");
        std::fs::create_dir(&other).unwrap();
        // Nothing is old enough yet.
        assert_eq!(sweep_stale_in(parent.path(), Duration::from_secs(600)), 0);
        assert!(old.path().exists());
        // With a zero age everything of ours is old; the other folder is left alone.
        let old_path = old.path().to_path_buf();
        std::mem::forget(old);
        assert_eq!(sweep_stale_in(parent.path(), Duration::ZERO), 1);
        assert!(!old_path.exists());
        assert!(other.exists());
    }

    #[cfg(unix)]
    #[test]
    fn the_workdir_is_private_to_the_user() {
        use std::os::unix::fs::PermissionsExt;
        let parent = tempfile::tempdir().unwrap();
        let dir = Workdir::new_in(parent.path()).unwrap();
        for folder in [dir.path(), dir.cwd(), dir.input()] {
            let mode = std::fs::metadata(folder).unwrap().permissions().mode() & 0o777;
            assert_eq!(mode & 0o077, 0, "{} is {mode:o}", folder.display());
        }
    }

    #[test]
    fn the_workdir_is_deleted_on_drop() {
        let parent = tempfile::tempdir().unwrap();
        let path = {
            let dir = Workdir::new_in(parent.path()).unwrap();
            assert!(dir.cwd().is_dir() && dir.input().is_dir());
            assert_eq!(std::fs::read_dir(dir.cwd()).unwrap().count(), 0);
            dir.path().to_path_buf()
        };
        assert!(!path.exists());
    }
}
