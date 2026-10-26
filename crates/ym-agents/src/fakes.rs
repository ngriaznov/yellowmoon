//! What the two test doubles share (`docs/22` section 15.2). Included by `src/bin/ym-fake-claude.rs` and
//! `src/bin/ym-fake-agy.rs` with `#[path]`; nothing of the library uses it.
//!
//! The doubles imitate the output of Claude Code 2.1.310 and `agy` 1.4.7 as `docs/22` sections 7 and 8
//! describe it, and check what they are given: a failed check prints the reason on standard error and
//! exits 99. They are controlled by environment variables:
//!
//! | Variable | Meaning |
//! |---|---|
//! | `YM_FAKE_SCENARIO` | the behaviour, default `signed_in` |
//! | `YM_FAKE_RECORD=<folder>` | what the double saw: `argv.txt`, `cwd.txt`, `stdin.bin`, `env.txt`, `files.txt` of the last run, and the same in `run-<n>/` for every run |
//! | `YM_FAKE_PIDFILE=<file>` | the `slow` scenarios write their own pid and a grandchild's; the grandchild beats in `<file>.beat` |
//! | `YM_FAKE_VERSION` | replaces the version it prints |
//! | `YM_FAKE_FAILS=<n>` | the first n requests fail (for the breaker) |
//! | `YM_FAKE_COUNTER=<folder>` | where requests are counted, for `YM_FAKE_FAILS` and `late_signout` (default: the record folder) |
#![allow(dead_code)]

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Map, Value, json};

/// The scenario's name.
pub fn scenario() -> String {
    std::env::var("YM_FAKE_SCENARIO").unwrap_or_else(|_| "signed_in".to_owned())
}

/// Prints `reason` on standard error and exits 99: the request is not one the real program takes.
pub fn reject(reason: &str) -> ! {
    eprintln!("ym-fake: {reason}");
    std::process::exit(99);
}

/// The version to print: `YM_FAKE_VERSION`, else `default`.
pub fn version(default: &str) -> String {
    std::env::var("YM_FAKE_VERSION").unwrap_or_else(|_| default.to_owned())
}

/// Blocks for ever (a probe that never ends).
pub fn hang() -> ! {
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}

/// Claims the next number in `dir` by creating `<prefix><n>` exclusively, so that two processes never
/// get the same one.
pub fn claim(dir: &Path, prefix: &str) -> usize {
    let _ = fs::create_dir_all(dir);
    for n in 0.. {
        let path = dir.join(format!("{prefix}{n:04}"));
        if fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .is_ok()
        {
            return n;
        }
    }
    unreachable!("an unbounded range")
}

/// The folder where requests are counted, if the scenario needs one.
fn counter_dir() -> Option<PathBuf> {
    std::env::var_os("YM_FAKE_COUNTER")
        .or_else(|| std::env::var_os("YM_FAKE_RECORD"))
        .map(|d| PathBuf::from(d).join("counter"))
}

/// How many requests started before this one (0 for the first). Call once per request.
pub fn request_index() -> usize {
    match counter_dir() {
        Some(dir) => claim(&dir, "request-"),
        None => reject("this scenario counts requests: set YM_FAKE_COUNTER or YM_FAKE_RECORD"),
    }
}

/// How many requests have started so far (for the probes of `late_signout`).
pub fn requests_so_far() -> usize {
    match counter_dir() {
        Some(dir) => fs::read_dir(dir)
            .map(|d| d.filter_map(Result::ok).count())
            .unwrap_or(0),
        None => 0,
    }
}

/// Whether this request is one of the first `YM_FAKE_FAILS` and must fail.
pub fn must_fail(index: usize) -> bool {
    std::env::var("YM_FAKE_FAILS")
        .ok()
        .and_then(|n| n.parse::<usize>().ok())
        .is_some_and(|n| index < n)
}

// ---------------------------------------------------------------------------------------------
// The record
// ---------------------------------------------------------------------------------------------

/// Writes what the double saw to `YM_FAKE_RECORD`, when it is set.
pub struct Recorder {
    dirs: Vec<PathBuf>,
}

impl Recorder {
    /// Records the arguments, the working folder, the files in it and the environment of this run.
    pub fn start() -> Self {
        let Some(top) = std::env::var_os("YM_FAKE_RECORD").map(PathBuf::from) else {
            return Self { dirs: Vec::new() };
        };
        let _ = fs::create_dir_all(&top);
        let n = claim(&top.join("claims"), "run-");
        let run = top.join(format!("run-{n:04}"));
        let _ = fs::create_dir_all(&run);
        let dirs = vec![top, run];
        let recorder = Self { dirs };

        let args: Vec<String> = std::env::args_os()
            .skip(1)
            .map(|a| a.to_string_lossy().into_owned())
            .collect();
        recorder.write("argv.txt", args.join("\n").as_bytes());
        let cwd = std::env::current_dir().unwrap_or_default();
        recorder.write("cwd.txt", cwd.to_string_lossy().as_bytes());
        let mut names: Vec<String> = fs::read_dir(&cwd)
            .map(|d| {
                d.filter_map(Result::ok)
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        names.sort();
        recorder.write("files.txt", names.join("\n").as_bytes());

        let var = |name: &str| std::env::var(name).unwrap_or_else(|_| "unset".to_owned());
        let mut env = String::new();
        for name in [
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC",
            "DISABLE_TELEMETRY",
            "DISABLE_ERROR_REPORTING",
            "DISABLE_AUTOUPDATER",
            "NO_COLOR",
            "CLAUDECODE",
            "CLAUDE_CODE_ENTRYPOINT",
            "CLAUDE_CODE_SESSION_ID",
            "CLAUDE_CODE_CHILD_SESSION",
        ] {
            env.push_str(&format!("{name}={}\n", var(name)));
        }
        let remote = std::env::vars_os()
            .any(|(k, _)| k.to_string_lossy().starts_with("CLAUDE_CODE_REMOTE"));
        env.push_str(&format!(
            "CLAUDE_CODE_REMOTE*={}\n",
            if remote { "set" } else { "unset" }
        ));
        #[cfg(windows)]
        {
            // The creation flags cannot be read back from inside a process. What shows is whose console
            // this is: with CREATE_NO_WINDOW the child has a console of its own (or none), so at most
            // this process is attached to it; without the flag it shares its parent's.
            env.push_str(&format!("CONSOLE_PROCESSES={}\n", console_processes()));
        }
        recorder.write("env.txt", env.as_bytes());
        recorder
    }

    /// Writes a file in the record folders.
    pub fn write(&self, name: &str, bytes: &[u8]) {
        for dir in &self.dirs {
            let _ = fs::write(dir.join(name), bytes);
        }
    }
}

/// How many processes are attached to this process's console (Windows); 0 when it has none.
#[cfg(windows)]
fn console_processes() -> u32 {
    unsafe extern "system" {
        fn GetConsoleProcessList(list: *mut u32, count: u32) -> u32;
    }
    let mut pids = [0_u32; 8];
    // SAFETY: the buffer is valid for `pids.len()` entries, which is the count passed.
    unsafe { GetConsoleProcessList(pids.as_mut_ptr(), pids.len() as u32) }
}

// ---------------------------------------------------------------------------------------------
// The slow scenario
// ---------------------------------------------------------------------------------------------

/// The argument that makes a double act as the grandchild of `slow`.
pub const SLEEPER_ARG: &str = "--ym-fake-sleeper";

/// Runs as the grandchild when started with [`SLEEPER_ARG`]: beats into a file until it is killed.
pub fn maybe_sleeper() {
    let mut args = std::env::args_os().skip(1);
    if args.next().is_some_and(|a| a == SLEEPER_ARG) {
        let beat = args.next().map(PathBuf::from).unwrap_or_default();
        let mut n = 0_u64;
        loop {
            n += 1;
            let _ = fs::write(&beat, n.to_string());
            std::thread::sleep(Duration::from_millis(50));
        }
    }
}

/// The `slow` scenarios: starts a grandchild, writes both pids to `YM_FAKE_PIDFILE` and sleeps for ten
/// minutes.
pub fn slow() -> ! {
    if let Some(pidfile) = std::env::var_os("YM_FAKE_PIDFILE").map(PathBuf::from) {
        let beat = PathBuf::from(format!("{}.beat", pidfile.display()));
        let me = std::env::current_exe().unwrap_or_default();
        let child = std::process::Command::new(me)
            .arg(SLEEPER_ARG)
            .arg(&beat)
            .spawn();
        let grandchild = child.as_ref().map(std::process::Child::id).unwrap_or(0);
        let _ = fs::write(
            &pidfile,
            format!("{}\n{grandchild}\n", std::process::id()),
        );
        // The grandchild is left running: it is the process group's to end.
        std::mem::forget(child);
    }
    std::thread::sleep(Duration::from_secs(600));
    std::process::exit(0)
}

// ---------------------------------------------------------------------------------------------
// The picture and the answer
// ---------------------------------------------------------------------------------------------

/// A decoded picture.
pub struct Picture {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Decodes `bytes` as a baseline JPEG with one scan, or says why not.
pub fn decode_jpeg(bytes: &[u8]) -> Result<Picture, String> {
    if bytes.len() < 4 || bytes[..2] != [0xFF, 0xD8] {
        return Err("the picture does not start with FFD8".to_owned());
    }
    let scans = count_scans(bytes);
    if scans != 1 {
        return Err(format!("the picture has {scans} scans, not one"));
    }
    let mut decoder = zune_jpeg::JpegDecoder::new(std::io::Cursor::new(bytes.to_vec()));
    let rgb = decoder
        .decode()
        .map_err(|e| format!("the picture does not decode: {e:?}"))?;
    let info = decoder
        .info()
        .ok_or_else(|| "the picture has no header".to_owned())?;
    let (width, height) = (usize::from(info.width), usize::from(info.height));
    if rgb.len() != width * height * 3 {
        return Err(format!(
            "the picture decoded to {} bytes for {width} x {height}",
            rgb.len()
        ));
    }
    Ok(Picture { width, height, rgb })
}

/// How many start-of-scan markers (`FFDA`) the JPEG's segments hold.
fn count_scans(bytes: &[u8]) -> usize {
    let mut at = 2;
    let mut scans = 0;
    while at + 4 <= bytes.len() {
        if bytes[at] != 0xFF {
            break;
        }
        let marker = bytes[at + 1];
        if marker == 0xD9 {
            break;
        }
        let len = usize::from(bytes[at + 2]) << 8 | usize::from(bytes[at + 3]);
        if marker == 0xDA {
            scans += 1;
            // The entropy-coded data follows; skip to the next marker that is not a stuffed byte or a restart.
            at += 2 + len;
            while at + 1 < bytes.len() {
                if bytes[at] == 0xFF && bytes[at + 1] != 0 && !(0xD0..=0xD7).contains(&bytes[at + 1]) {
                    break;
                }
                at += 1;
            }
            continue;
        }
        at += 2 + len;
    }
    scans
}

/// The answer the doubles give for a picture, as the object of `docs/22` section 6.2 with Apple's case
/// names; the steps are those of `FakeAdvisor` (`ym_core::apple_auto`): from the mean sRGB luma `L` and
/// its standard deviation `S`.
pub fn answer_for(picture: &Picture, first_word: &str) -> Map<String, Value> {
    let pixels = picture.width * picture.height;
    let lumas = picture.rgb.chunks_exact(3).map(|p| {
        (0.2126 * f64::from(p[0]) + 0.7152 * f64::from(p[1]) + 0.0722 * f64::from(p[2])) / 255.0
    });
    let n = pixels.max(1) as f64;
    let (sum, sum_sq) = lumas.fold((0.0, 0.0), |(s, q), l| (s + l, q + l * l));
    let mean = sum / n;
    let sd = (sum_sq / n - mean * mean).max(0.0).sqrt();
    let flat = sd < 0.15;
    let mut a = Map::new();
    a.insert("scene".into(), json!("Test picture"));
    a.insert("plan".into(), json!(format!("Test plan: {first_word}")));
    a.insert(
        "exposure".into(),
        json!(if mean < 0.30 {
            "slightlyBrighter"
        } else if mean > 0.65 {
            "slightlyDarker"
        } else {
            "keep"
        }),
    );
    if mean > 0.5 {
        a.insert("highlights".into(), json!("darker"));
    }
    a.insert(
        "shadows".into(),
        json!(if mean < 0.30 { "brighter" } else { "keep" }),
    );
    a.insert("contrast".into(), json!(if flat { "more" } else { "keep" }));
    a.insert("warmth".into(), json!("keep"));
    a.insert("saturation".into(), json!("keep"));
    a.insert(
        "shadowColour".into(),
        json!(if flat { "teal" } else { "none" }),
    );
    if flat {
        a.insert("shadowColourStrength".into(), json!("subtle"));
    }
    a
}

/// The answer after the scenario's twist: `malformed` has an unknown case name, `extra_keys` has nulls
/// and a key that is not in the schema.
pub fn twist(mut answer: Map<String, Value>, scenario: &str) -> Map<String, Value> {
    match scenario {
        "malformed" => {
            answer.insert("exposure".into(), json!("muchBrightest"));
        }
        "extra_keys" => {
            answer.insert("blacks".into(), Value::Null);
            answer.insert("tint".into(), Value::Null);
            answer.insert("note".into(), json!("an extra note"));
        }
        _ => {}
    }
    answer
}

/// The first word of `text`, or the empty string.
pub fn first_word(text: &str) -> String {
    text.split_whitespace().next().unwrap_or("").to_owned()
}

/// Reads standard input to its end.
pub fn read_stdin() -> Vec<u8> {
    let mut bytes = Vec::new();
    let _ = std::io::stdin().read_to_end(&mut bytes);
    bytes
}

/// Writes `bytes` to standard output and flushes.
pub fn out(bytes: &[u8]) {
    let mut stdout = std::io::stdout().lock();
    let _ = stdout.write_all(bytes);
    let _ = stdout.flush();
}

/// Writes `lines` of 1 KB each, `megabytes` MB in all, to standard output (the `huge` and `backpressure`
/// scenarios). A line stays under the line cap.
pub fn flood(megabytes: usize) {
    let line = format!("{}\n", "x".repeat(1023));
    let mut stdout = std::io::stdout().lock();
    for _ in 0..megabytes * 1024 {
        if stdout.write_all(line.as_bytes()).is_err() {
            return;
        }
    }
    let _ = stdout.flush();
}

/// The text after `flag` in `args`, when `flag` is there.
pub fn value_of<'a>(args: &'a [String], flag: &str) -> Option<&'a String> {
    args.iter().position(|a| a == flag).and_then(|i| args.get(i + 1))
}

/// Whether `flag` is in `args`.
pub fn has(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

/// Rejects the run when `flag` is missing.
pub fn require(args: &[String], flag: &str) {
    if !has(args, flag) {
        reject(&format!("{flag} is missing"));
    }
}

/// Rejects the run unless `schema_text` is a JSON Schema of an object that requires `scene` and `plan`.
pub fn check_schema(schema_text: &str) {
    let Ok(schema) = serde_json::from_str::<Value>(schema_text) else {
        reject("the schema is not JSON");
    };
    if schema.get("type") != Some(&json!("object")) {
        reject("the schema's root is not an object");
    }
    let required = schema.get("required").and_then(Value::as_array);
    let has_both = required.is_some_and(|r| r.contains(&json!("scene")) && r.contains(&json!("plan")));
    if !has_both {
        reject("the schema does not require scene and plan");
    }
}
