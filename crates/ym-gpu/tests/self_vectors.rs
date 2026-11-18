//! The two classes of vectors of the raw algorithms (`docs/29-raw-constants.md` F-06, RC-04).
//!
//! * The oracle vectors (`tests/cpu_data/<algorithm>/`) are never edited; they run under
//!   `Tuning::ORACLE` and prove the structure.
//! * The self vectors (`tests/cpu_data/<algorithm>_self/`) are written by
//!   `examples/gen_self_vectors.rs` from our own code under `Tuning::DEFAULT` and pin the shipped
//!   constants.
//!
//! What is held here:
//!
//! * `the_committed_self_set_is_what_the_generator_writes`: the generator is deterministic. It writes
//!   the whole set into a temporary folder and the bytes equal the committed files, file for file.
//! * `the_generator_under_the_oracle_constants_agrees_with_the_oracle_set`: the generator is right.
//!   Run under `Tuning::ORACLE` (and the cubics of the oracle) it reproduces every oracle vector
//!   within that vector's tolerance, in its results and in its derived values, from the same inputs.
//! * `the_committed_self_set_agrees_with_the_oracle_set_while_the_constants_have_not_moved`: at
//!   `DEFAULT = ORACLE` the committed self set is the oracle set within its tolerance.
//! * `a_default_that_moved_needs_a_current_self_set` (F-06): when `Tuning::DEFAULT` differs from
//!   `ORACLE`, a missing or stale self set fails.

use std::path::{Path, PathBuf};

use serde_json::Value;

mod self_gen;
use self_gen::{Class, PACKAGES, Package, Regime};

/// A fresh folder under the system's temporary directory.
fn temp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ym-self-vectors-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temporary folder");
    dir
}

fn names(dir: &Path) -> Vec<String> {
    self_gen::json_files(dir)
        .iter()
        .map(|p| p.file_name().expect("name").to_string_lossy().into_owned())
        .collect()
}

/// What differs between a self set on disk and what the generator writes now: files that are
/// missing, files that should not be there and files whose bytes differ.
fn differences(committed: &Path, regenerated: &Path) -> Vec<String> {
    let mut out = Vec::new();
    if !committed.is_dir() {
        out.push(format!("the folder {} is missing", committed.display()));
        return out;
    }
    let (have, want) = (names(committed), names(regenerated));
    for name in &want {
        if !have.contains(name) {
            out.push(format!("{name}: missing"));
        }
    }
    for name in &have {
        if !want.contains(name) {
            out.push(format!("{name}: not written by the generator"));
        }
    }
    for name in want.iter().filter(|n| have.contains(n)) {
        let a = std::fs::read(committed.join(name)).expect("committed file");
        let b = std::fs::read(regenerated.join(name)).expect("regenerated file");
        if a != b {
            out.push(format!(
                "{name}: stale (the bytes differ from a regeneration)"
            ));
        }
    }
    out
}

const REGENERATE: &str = "run `cargo run -p ym-gpu --example gen_self_vectors` and commit the diff";

// ---------------------------------------------------------------------------------------------
// Deterministic generation
// ---------------------------------------------------------------------------------------------

#[test]
fn the_committed_self_set_is_what_the_generator_writes() {
    let root = temp_dir("regenerate");
    let mut total = 0;
    for (package, written) in self_gen::generate_all(&root) {
        total += written.len();
        let problems = differences(
            &Class::SelfSet.dir(package),
            &root.join(package.self_folder()),
        );
        assert!(
            problems.is_empty(),
            "{}: {problems:#?}; {REGENERATE}",
            package.self_folder()
        );
        // The same files as the oracle set, by name.
        assert_eq!(
            names(&Class::SelfSet.dir(package)),
            names(&Class::Oracle.dir(package)),
            "{}: the self set has the oracle set's files",
            package.self_folder()
        );
    }
    assert_eq!(total, 73, "18 + 16 + 23 + 16 vectors");
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn generating_twice_gives_the_same_bytes() {
    let (a, b) = (temp_dir("twice-a"), temp_dir("twice-b"));
    // The smallest package is enough: the writer and the numbers are the same code for all four.
    let package = Package::LogGuided;
    self_gen::generate(package, &a);
    self_gen::generate(package, &b);
    assert_eq!(differences(&a, &b), Vec::<String>::new());
    let _ = std::fs::remove_dir_all(&a);
    let _ = std::fs::remove_dir_all(&b);
}

#[test]
fn the_self_files_say_what_they_are() {
    for package in PACKAGES {
        for path in self_gen::json_files(&Class::SelfSet.dir(package)) {
            let v = self_gen::read_vector(&path);
            let d = v["description"].as_str().unwrap_or_default();
            assert!(
                d.contains("[self set:") && d.contains("not an oracle"),
                "{}: the description does not mark the file as a self vector",
                path.display()
            );
        }
        for path in self_gen::json_files(&Class::Oracle.dir(package)) {
            let v = self_gen::read_vector(&path);
            assert!(
                !v["description"]
                    .as_str()
                    .unwrap_or_default()
                    .contains("[self set:"),
                "{}: an oracle file is marked as a self vector",
                path.display()
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The generator agrees with the oracle
// ---------------------------------------------------------------------------------------------

/// The largest absolute difference of the fixed-point form of the log-guided demosaic between the
/// oracle's cubics and the shipped ones (`docs/29` RC-01: the old and the new triple differ by up to
/// 5 codes of log2). Measured 1.10e-3 (`detail_minus50`); the bound is the measurement rounded up, so a
/// bigger move shows. The float form has no cubics and agrees within its own tolerance.
const LOG_GUIDED_CUBIC_SHIFT: f64 = 1.2e-3;

/// Compares the results of `ours` with the oracle vector `oracle`: the inputs, the parameters, the
/// field names and the tolerances are the oracle's; the results agree within the oracle's tolerance.
/// `fixed_bound` replaces the tolerance of the log-guided fixed form, whose cubics differ from the
/// oracle's. Returns the problems found.
fn disagreements(
    name: &str,
    package: Package,
    oracle: &Value,
    ours: &Value,
    fixed_bound: Option<f64>,
) -> Vec<String> {
    let mut out = Vec::new();
    let keys =
        |v: &Value| -> Vec<String> { v.as_object().expect("object").keys().cloned().collect() };
    let (mut ka, mut kb) = (keys(oracle), keys(ours));
    ka.sort();
    kb.sort();
    if ka != kb {
        out.push(format!("{name}: fields {kb:?} against the oracle's {ka:?}"));
        return out;
    }
    for key in ["name", "input", "tolerance", "tolerance_float"] {
        if oracle.get(key) != ours.get(key) {
            out.push(format!("{name}: {key} is not the oracle's"));
        }
    }
    // The parameters are the oracle's; only the derived block is ours.
    let strip = |v: &Value| {
        let mut p = v["params"].clone();
        p.as_object_mut().map(|m| m.remove("derived"));
        p
    };
    if strip(oracle) != strip(ours) {
        out.push(format!("{name}: params are not the oracle's"));
    }
    // The oracle printed the derived values of the adaptive ratio demosaic, the diffraction planner
    // and the output sharpening in full (the CPU tests hold them to 1e-9 and 1e-15), and those of the
    // log-guided demosaic to 7 digits (they hold it to 1e-7).
    let derived_rel = if package == Package::LogGuided {
        1e-7
    } else {
        1e-9
    };
    approx(
        &format!("{name}: params.derived"),
        &oracle["params"]["derived"],
        &ours["params"]["derived"],
        &|a, b| (a - b).abs() <= derived_rel * b.abs() + 1e-15,
        &mut out,
    );
    let tol = oracle["tolerance"].as_f64().unwrap_or(0.0);
    let tol_float = oracle["tolerance_float"].as_f64().unwrap_or(tol);
    for (field, tolerance) in [
        ("expected", fixed_bound.unwrap_or(tol)),
        ("expected_float", tol_float),
    ] {
        if oracle.get(field).is_some() {
            approx(
                &format!("{name}: {field}"),
                &oracle[field],
                &ours[field],
                &|a, b| {
                    if tolerance == 0.0 {
                        // Exact vectors: the same number (both files print an `f32` with nine
                        // significant digits).
                        a == b
                    } else {
                        (a - b).abs() <= tolerance
                    }
                },
                &mut out,
            );
        }
    }
    out
}

/// Walks two JSON values of the same shape and reports each number pair `close` rejects (once per
/// array, with the worst pair) and every other difference.
fn approx(
    what: &str,
    a: &Value,
    b: &Value,
    close: &dyn Fn(f64, f64) -> bool,
    out: &mut Vec<String>,
) {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().expect("number"), y.as_f64().expect("number"));
            if !close(x, y) {
                out.push(format!("{what}: {x} against {y}"));
            }
        }
        (Value::Array(xs), Value::Array(ys)) => {
            if xs.len() != ys.len() {
                out.push(format!("{what}: {} entries against {}", xs.len(), ys.len()));
                return;
            }
            if xs.iter().all(Value::is_number) {
                let worst = xs
                    .iter()
                    .zip(ys)
                    .map(|(x, y)| (x.as_f64().expect("n"), y.as_f64().expect("n")))
                    .filter(|&(x, y)| !close(x, y))
                    .max_by(|p, q| (p.0 - p.1).abs().total_cmp(&(q.0 - q.1).abs()));
                if let Some((x, y)) = worst {
                    out.push(format!("{what}: {x} against {y} (and maybe more)"));
                }
            } else {
                for (i, (x, y)) in xs.iter().zip(ys).enumerate() {
                    approx(&format!("{what}[{i}]"), x, y, close, out);
                }
            }
        }
        (Value::Object(xs), Value::Object(ys)) => {
            for (k, x) in xs {
                match ys.get(k) {
                    Some(y) => approx(&format!("{what}.{k}"), x, y, close, out),
                    None => out.push(format!("{what}: no field {k}")),
                }
            }
            for k in ys.keys().filter(|k| !xs.contains_key(*k)) {
                out.push(format!("{what}: extra field {k}"));
            }
        }
        _ => {
            if a != b {
                out.push(format!("{what}: {a} against {b}"));
            }
        }
    }
}

#[test]
fn the_generator_under_the_oracle_constants_agrees_with_the_oracle_set() {
    // The proof that the generator is right: with the constants the oracle vectors were made with
    // it writes the oracle's numbers, within each vector's own tolerance.
    let mut checked = 0;
    for package in PACKAGES {
        for path in self_gen::json_files(&Class::Oracle.dir(package)) {
            let oracle = self_gen::read_vector(&path);
            let ours = self_gen::vector(package, &oracle, Regime::Oracle);
            let name = format!(
                "{}/{}",
                package.oracle_folder(),
                path.file_name().unwrap().to_string_lossy()
            );
            let problems = disagreements(&name, package, &oracle, &ours, None);
            assert!(problems.is_empty(), "{problems:#?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 73);
}

#[test]
fn the_committed_self_set_agrees_with_the_oracle_set_while_the_constants_have_not_moved() {
    let mut checked = 0;
    for package in PACKAGES {
        let moved = package.default_differs_from_oracle();
        // The log-guided demosaic's cubics have moved (RC-01): its fixed form is held to the size of
        // that move, and everything else (float form, derived values) to the oracle's tolerance.
        let fixed_bound = (package == Package::LogGuided).then_some(LOG_GUIDED_CUBIC_SHIFT);
        if moved && fixed_bound.is_none() {
            continue; // a fit moved a constant: the self set is meant to differ, see the F-06 test
        }
        for path in self_gen::json_files(&Class::Oracle.dir(package)) {
            let file = path.file_name().expect("name");
            let oracle = self_gen::read_vector(&path);
            let ours = self_gen::read_vector(&Class::SelfSet.dir(package).join(file));
            let name = format!("{}/{}", package.self_folder(), file.to_string_lossy());
            let problems = disagreements(&name, package, &oracle, &ours, fixed_bound);
            assert!(problems.is_empty(), "{problems:#?}");
            checked += 1;
        }
    }
    assert!(checked >= 16, "at least the log-guided set was compared");
}

// ---------------------------------------------------------------------------------------------
// F-06: a moved default needs a current self set
// ---------------------------------------------------------------------------------------------

/// The F-06 rule: when the shipped constants differ from the oracle's, the self set in `dir` must be
/// present and equal what the generator writes now. When they have not moved, the oracle set already
/// pins them and nothing is required.
fn self_set_is_current_if_the_default_moved(
    package: Package,
    default_differs: bool,
    dir: &Path,
) -> Result<(), String> {
    if !default_differs {
        return Ok(());
    }
    let fresh = temp_dir(&format!("f06-{}", package.self_folder()));
    self_gen::generate(package, &fresh);
    let problems = differences(dir, &fresh);
    let _ = std::fs::remove_dir_all(&fresh);
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "Tuning::DEFAULT differs from ORACLE for {} and its self set is missing or stale: {problems:?}; {REGENERATE}",
            package.oracle_folder()
        ))
    }
}

#[test]
fn a_default_that_moved_needs_a_current_self_set() {
    for package in PACKAGES {
        if let Err(e) = self_set_is_current_if_the_default_moved(
            package,
            package.default_differs_from_oracle(),
            &Class::SelfSet.dir(package),
        ) {
            panic!("{e}");
        }
    }
}

#[test]
fn the_f06_rule_rejects_a_missing_or_stale_set_once_the_default_has_moved() {
    let package = Package::AdaptiveRatio;
    let dir = temp_dir("f06-set");
    self_gen::generate(package, &dir);

    // A current set passes, whether or not the default moved.
    assert_eq!(
        self_set_is_current_if_the_default_moved(package, true, &dir),
        Ok(())
    );
    assert_eq!(
        self_set_is_current_if_the_default_moved(package, false, &dir),
        Ok(())
    );

    // A file that no longer matches what the generator writes (a constant moved, the set was not
    // regenerated): stale.
    let victim = dir.join("edge_knight.json");
    let text = std::fs::read_to_string(&victim).expect("file");
    std::fs::write(&victim, text.replacen("0.1", "0.2", 1)).expect("write");
    let err = self_set_is_current_if_the_default_moved(package, true, &dir).unwrap_err();
    assert!(err.contains("edge_knight.json: stale"), "{err}");
    assert_eq!(
        self_set_is_current_if_the_default_moved(package, false, &dir),
        Ok(())
    );

    // A missing file, and a missing folder.
    std::fs::write(&victim, text).expect("restore");
    std::fs::rename(&victim, dir.join("edge_knight.json.moved")).expect("move away");
    let err = self_set_is_current_if_the_default_moved(package, true, &dir).unwrap_err();
    assert!(err.contains("edge_knight.json: missing"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
    let err = self_set_is_current_if_the_default_moved(package, true, &dir).unwrap_err();
    assert!(err.contains("is missing"), "{err}");
    assert_eq!(
        self_set_is_current_if_the_default_moved(package, false, &dir),
        Ok(())
    );
}
