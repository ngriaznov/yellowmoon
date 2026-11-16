//! The registries of the raw-algorithm constants (`docs/29-raw-constants.md`, task RC-03).
//!
//! Every constant of the inventory (`docs/29` §2.1) has one home: a field of a `Tuning` struct
//! (`adaptive_ratio::Tuning`, `diffraction::Tuning`, `output_sharpen::Tuning`) or, for the two
//! cubics of the log-guided demosaic, `FixedCubics`. The CPU reads the fields; the kernel wrappers
//! put a WGSL `const` block generated from the same `Tuning` in front of the shader. This file holds
//! that arrangement to account:
//!
//! * the registries and the inventory table of `docs/29` list the same IDs, and the kind of every ID
//!   is the status the spec's derivation table gives it (F-08);
//! * the generated block of each package is the registry's device constants, and every literal in it
//!   is the same `f32` as the literal the shader had before the block existed (bit for bit, read the
//!   way the shader compiler reads it);
//! * no value that the registry marks as specific appears as a literal anywhere else in a package's
//!   CPU file, kernel wrapper or shader (F-03), except at the places the short allow-list names, each
//!   with its reason.
//!
//! The scan is by value, so it cannot see a copy of a generic number (`0.5`, `3.0`, `5.0`): those
//! registry entries are not `distinct`, and the check for them is that the code reads the field.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

use ym_gpu::cpu::demosaic::{adaptive_ratio, log_guided};
use ym_gpu::cpu::output_sharpen;
use ym_gpu::cpu::sharpen::diffraction;
use ym_gpu::cpu::tuning::{Constant, Kind};
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::{KernelId, shader_source};

/// One package of the inventory: its registry, its generated block and the files that hold its
/// constants.
struct Package {
    name: &'static str,
    registry: &'static [Constant],
    /// The generated WGSL block of `Tuning::DEFAULT` and of `Tuning::ORACLE`, `None` when the package
    /// has none.
    blocks: Option<(String, String)>,
    /// The spec whose derivation table gives the status of each ID.
    spec: &'static str,
    /// The kernel ids whose source must carry the block.
    kernel_prefix: &'static str,
    /// Files searched for stray copies (under `crates/ym-gpu/`).
    sources: &'static [&'static str],
}

fn packages() -> Vec<Package> {
    vec![
        Package {
            name: "adaptive_ratio",
            registry: adaptive_ratio::REGISTRY,
            blocks: Some((
                adaptive_ratio::Tuning::DEFAULT.wgsl_block(),
                adaptive_ratio::Tuning::ORACLE.wgsl_block(),
            )),
            spec: "docs/algorithms/01-adaptive-ratio-demosaic.md",
            kernel_prefix: "AdaptiveRatio",
            sources: &[
                "src/cpu/demosaic/adaptive_ratio.rs",
                "src/kernels/adaptive_ratio.rs",
                "src/shaders/adaptive_ratio.wgsl",
            ],
        },
        Package {
            name: "log_guided",
            registry: log_guided::REGISTRY,
            blocks: None,
            spec: "docs/algorithms/02-log-guided-demosaic.md",
            kernel_prefix: "",
            sources: &[],
        },
        Package {
            name: "diffraction",
            registry: diffraction::REGISTRY,
            blocks: Some((
                diffraction::Tuning::DEFAULT.wgsl_block(),
                diffraction::Tuning::ORACLE.wgsl_block(),
            )),
            spec: "docs/algorithms/03-diffraction-deconvolution.md",
            kernel_prefix: "Diffraction",
            sources: &[
                "src/cpu/sharpen/diffraction.rs",
                "src/kernels/diffraction.rs",
                "src/shaders/diffraction.wgsl",
            ],
        },
        Package {
            name: "output_sharpen",
            registry: output_sharpen::REGISTRY,
            blocks: Some((
                output_sharpen::Tuning::DEFAULT.wgsl_block(),
                output_sharpen::Tuning::ORACLE.wgsl_block(),
            )),
            spec: "docs/algorithms/04-output-sharpening.md",
            kernel_prefix: "OutputSharpen",
            sources: &[
                "src/cpu/output_sharpen.rs",
                "src/kernels/output_sharpen.rs",
                "src/shaders/output_sharpen.wgsl",
            ],
        },
    ]
}

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn repo_file(relative: &str) -> String {
    let path = crate_dir().join("../..").join(relative);
    std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

// ---------------------------------------------------------------------------------------------
// The inventory of docs/29 section 2.1
// ---------------------------------------------------------------------------------------------

/// The IDs of the rows of the inventory table of `docs/29` §2.1 (the text between its heading and
/// the heading of §2.2).
fn inventory_ids(doc: &str) -> BTreeSet<String> {
    let start = doc
        .find("### 2.1 The inventory")
        .expect("the inventory heading");
    let end = doc
        .find("### 2.2 ")
        .expect("the heading after the inventory");
    table_rows(&doc[start..end])
        .into_iter()
        .map(|(id, _)| id)
        .collect()
}

/// `(first cell, all cells)` of every table row whose first cell is an inventory ID (`A1`, `L3`,
/// `D14`, `O12`).
fn table_rows(text: &str) -> Vec<(String, Vec<String>)> {
    text.lines()
        .filter(|l| l.starts_with("| "))
        .filter_map(|l| {
            // An escaped bar is text in a cell, not a cell border.
            let line = l.replace("\\|", "\u{a6}");
            let cells: Vec<String> = line
                .trim()
                .trim_matches('|')
                .split('|')
                .map(|c| c.trim().to_string())
                .collect();
            let id = cells.first()?.clone();
            let mut chars = id.chars();
            let letter = chars.next()?;
            let digits: String = chars.collect();
            let is_id = "ALDO".contains(letter)
                && !digits.is_empty()
                && digits.chars().all(|c| c.is_ascii_digit());
            is_id.then_some((id, cells))
        })
        .collect()
}

#[test]
fn the_inventory_has_44_rows() {
    // The count of docs/29 §2.1: 15 + 3 + 14 + 12.
    let ids = inventory_ids(&repo_file("docs/29-raw-constants.md"));
    assert_eq!(ids.len(), 44, "{ids:?}");
}

#[test]
fn the_registries_and_the_inventory_list_the_same_rows() {
    let inventory = inventory_ids(&repo_file("docs/29-raw-constants.md"));
    let registered: BTreeSet<String> = packages()
        .iter()
        .flat_map(|p| p.registry.iter().map(|c| c.id.to_string()))
        .collect();
    let missing_from_registry: Vec<_> = inventory.difference(&registered).collect();
    let missing_from_inventory: Vec<_> = registered.difference(&inventory).collect();
    assert!(
        missing_from_registry.is_empty(),
        "rows of docs/29 section 2.1 with no registry entry: {missing_from_registry:?}"
    );
    assert!(
        missing_from_inventory.is_empty(),
        "registry entries with no row in docs/29 section 2.1: {missing_from_inventory:?}"
    );
}

#[test]
fn each_registry_holds_the_rows_of_its_spec() {
    // The ID letter says the package: A spec 01, L spec 02, D spec 03, O spec 04.
    for (p, letter) in packages().iter().zip(['A', 'L', 'D', 'O']) {
        for c in p.registry {
            assert!(
                c.id.starts_with(letter),
                "{}: {} ({}) is in the wrong registry",
                p.name,
                c.id,
                c.name
            );
        }
    }
}

#[test]
fn every_kind_is_the_status_of_the_specs_derivation_table() {
    // The derivation table of each spec (docs/29 RC-02) gives every ID one status. The registry's
    // kind is that status.
    let want = |s: &str| match s {
        "Maths" => Kind::Maths,
        "Fit" => Kind::Fit,
        "Design" => Kind::Design,
        "Guard" => Kind::Guard,
        other => panic!("unknown status {other}"),
    };
    let mut checked = BTreeSet::new();
    for p in packages() {
        let spec = repo_file(p.spec);
        let statuses: BTreeMap<String, Kind> = table_rows(&spec)
            .iter()
            .filter(|(_, cells)| cells.len() >= 5)
            .filter(|(_, cells)| ["Maths", "Fit", "Design", "Guard"].contains(&cells[2].as_str()))
            .map(|(id, cells)| (id.clone(), want(&cells[2])))
            .collect();
        assert!(
            !statuses.is_empty(),
            "{}: no derivation table in {}",
            p.name,
            p.spec
        );
        for c in p.registry {
            let kind = statuses
                .get(c.id)
                .unwrap_or_else(|| panic!("{}: {} has no row in {}", p.name, c.id, p.spec));
            assert_eq!(
                c.kind, *kind,
                "{}: {} ({}) is {:?} in the registry and {:?} in {}",
                p.name, c.id, c.name, c.kind, kind, p.spec
            );
            assert!(
                c.status.fits(c.kind),
                "{}: {} ({}) has the status {:?} for the kind {:?}",
                p.name,
                c.id,
                c.name,
                c.status,
                c.kind
            );
            checked.insert(c.id);
        }
    }
    assert_eq!(checked.len(), 44);
}

#[test]
fn names_are_unique_and_values_are_finite() {
    for p in packages() {
        let mut names = BTreeSet::new();
        for c in p.registry {
            assert!(names.insert(c.name), "{}: {} twice", p.name, c.name);
            assert!(c.value.is_finite(), "{}: {}", p.name, c.name);
        }
    }
}

#[test]
fn the_centre_weight_expression_gives_the_121_and_123_of_the_spec() {
    // A5: 121 and 123 are not constants. They are `1 + A + B` and `1 + A + B + 2` at t = 1.
    let value = |name: &str| {
        adaptive_ratio::REGISTRY
            .iter()
            .find(|c| c.name == name)
            .unwrap_or_else(|| panic!("{name}"))
            .value
    };
    assert_eq!(value("centre_weight_at_zero_misregistration"), 121.0);
    assert_eq!(value("centre_divisor_at_zero_misregistration"), 123.0);
    let t = adaptive_ratio::Tuning::DEFAULT;
    assert_eq!(
        t.centre_weight_at_zero_misregistration(),
        1.0 + t.centre_weight_linear + t.centre_weight_cubic
    );
}

// ---------------------------------------------------------------------------------------------
// The generated WGSL blocks
// ---------------------------------------------------------------------------------------------

/// The `(name, literal)` of every `const NAME: f32 = literal;` line of a block, in order.
fn block_constants(block: &str) -> Vec<(String, String)> {
    block
        .lines()
        .filter(|l| !l.starts_with("//") && !l.trim().is_empty())
        .map(|l| {
            let rest = l
                .strip_prefix("const ")
                .unwrap_or_else(|| panic!("not a const line: {l}"));
            let (name, rest) = rest.split_once(": f32 = ").expect("a typed f32 const");
            let literal = rest.strip_suffix(';').expect("a semicolon");
            (name.to_string(), literal.to_string())
        })
        .collect()
}

/// The `f32` a WGSL compiler reads from the text `const X: f32 = {literal};`, read by naga itself.
fn naga_f32(literal: &str) -> f32 {
    let src = format!("const X: f32 = {literal};\n");
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("{literal}: {}", e.emit_to_string(&src)));
    let (_, constant) = module
        .constants
        .iter()
        .next()
        .unwrap_or_else(|| panic!("{literal}: no constant"));
    match module.global_expressions[constant.init] {
        naga::Expression::Literal(naga::Literal::F32(v)) => v,
        ref other => panic!("{literal}: expected an f32 literal, got {other:?}"),
    }
}

#[test]
fn the_block_of_each_package_is_the_registrys_device_constants() {
    for p in packages() {
        let Some((block, oracle)) = &p.blocks else {
            assert!(p.registry.iter().all(|c| c.wgsl.is_none()), "{}", p.name);
            continue;
        };
        // Deterministic text, and the same under the oracle constants until a fit lands.
        assert_eq!(*block, *oracle, "{}: DEFAULT and ORACLE differ", p.name);
        let generated = block_constants(block);
        let on_device: Vec<_> = p.registry.iter().filter(|c| c.wgsl.is_some()).collect();
        let mut names: Vec<&str> = generated.iter().map(|(n, _)| n.as_str()).collect();
        let mut want: Vec<&str> = on_device.iter().filter_map(|c| c.wgsl).collect();
        names.sort_unstable();
        want.sort_unstable();
        assert_eq!(
            names, want,
            "{}: the generated block and the registry name different constants",
            p.name
        );
        let distinct: BTreeSet<&str> = names.iter().copied().collect();
        assert_eq!(
            distinct.len(),
            names.len(),
            "{}: a constant is generated twice",
            p.name
        );
        for c in on_device {
            let wgsl = c.wgsl.expect("filtered");
            let (_, literal) = generated
                .iter()
                .find(|(n, _)| n == wgsl)
                .expect("named above");
            // WGSL wants a `.` or an exponent in a float literal.
            assert!(
                literal.contains('.') || literal.contains('e'),
                "{}: {wgsl} = {literal} is not a float literal",
                p.name
            );
            let read = naga_f32(literal);
            assert_eq!(
                read.to_bits(),
                (c.value as f32).to_bits(),
                "{}: {wgsl} = {literal} reads as {read}, the registry has {} ({})",
                p.name,
                c.value,
                c.name
            );
        }
    }
}

/// What the shaders wrote before the blocks existed: the literal of each device constant, as text.
/// The bits of the generated literals must be these (under `Tuning::ORACLE`, which does not move
/// when a fit changes `Tuning::DEFAULT`).
const ORIGINAL_LITERALS: &[(&str, &str)] = &[
    // adaptive_ratio.wgsl (the literals of the passes)
    ("SECOND_DIFF_WEIGHT", "1.6"),
    ("CA_RELAX_SLOPE", "2.5"),
    ("CA_RELAX_FLOOR_PX", "0.1"),
    ("CENTRE_WEIGHT_LINEAR", "18.0"),
    ("CENTRE_WEIGHT_CUBIC", "102.0"),
    ("CHROMA_FLOOR", "0.03"),
    ("SAT_DOMINANCE", "0.75"),
    ("SHADOW_AMOUNT", "0.75"),
    ("SHADOW_LAPLACIAN_SCALE", "5.0"),
    ("SHADOW_FADE_END", "4.0"),
    ("SHADOW_FADE_SPAN", "3.0"),
    ("MATCH_EXCESS_WEIGHT", "3.0"),
    // diffraction.wgsl (the clip penalty, the gate band and its constants block)
    ("CLIP_WEIGHT_LARGEST", "25.0"),
    ("CLIP_WEIGHT_MIDDLE", "5.0"),
    ("PULL_GAIN", "12.5"),
    ("PULL_BAND", "0.04"),
    ("RIDGE", "0.0005"),
    ("GATE", "0.5"),
    ("GATE_FADE_LO", "0.25"),
    ("GATE_FADE_HI", "0.5"),
    ("WEIGHT_MIN", "0.01"),
    ("CD_MIN", "-0.5"),
    ("RATIO_CAP", "1.5"),
    ("DIVIDE_GUARD", "1e-20"),
    ("MOMENT_FLOOR", "1e-15"),
    ("FLOOR_DN", "1.0"),
    // output_sharpen.wgsl (its constants block and the Scharr weights)
    ("GAIN_EPS", "1e-10"),
    ("LUMA_FLOOR", "1e-20"),
    ("RATIO_MAX", "16.0"),
    ("HALO_INV_N", "0.04"),
    ("HALO_INV_R", "0.5"),
    ("HALO_K_MIN_F", "6.0"),
    ("HALO_PULL_FADE", "0.25"),
    ("SCHARR_CENTRE", "0.3125"),
    ("SCHARR_CORNER", "0.09375"),
];

#[test]
fn every_generated_literal_is_the_f32_the_shader_had_before() {
    let before: BTreeMap<&str, (&str, f32)> = ORIGINAL_LITERALS
        .iter()
        .map(|(n, lit)| (*n, (*lit, naga_f32(lit))))
        .collect();
    let oracle_blocks = [
        adaptive_ratio::Tuning::ORACLE.wgsl_block(),
        diffraction::Tuning::ORACLE.wgsl_block(),
        output_sharpen::Tuning::ORACLE.wgsl_block(),
    ];
    let mut seen = BTreeSet::new();
    for block in &oracle_blocks {
        for (name, literal) in block_constants(block) {
            let (text, was) = before
                .get(name.as_str())
                .unwrap_or_else(|| panic!("{name} has no original literal in the table"));
            assert_eq!(
                naga_f32(&literal).to_bits(),
                was.to_bits(),
                "{name}: the block writes {literal}, the shader had {text}"
            );
            seen.insert(name);
        }
    }
    assert_eq!(seen.len(), ORIGINAL_LITERALS.len(), "{seen:?}");
}

#[test]
fn each_kernel_carries_its_block_once_and_the_shaders_do_not_declare_it() {
    let blocks: BTreeMap<&str, String> = packages()
        .into_iter()
        .filter_map(|p| p.blocks.map(|(b, _)| (p.kernel_prefix, b)))
        .collect();
    let mut checked = 0;
    for id in KernelId::ALL {
        let name = format!("{id:?}");
        for (prefix, block) in &blocks {
            if !name.starts_with(prefix) {
                continue;
            }
            for mode in [StorageMode::F32, StorageMode::F16Packed] {
                let source = shader_source(mode, id);
                assert_eq!(
                    source.matches(block.as_str()).count(),
                    1,
                    "{name} {mode:?}: the generated block is not in the source exactly once"
                );
            }
            checked += 1;
        }
    }
    assert_eq!(checked, 11 + 9 + 8, "the passes of the three kernels");
    // The shader text itself declares none of the constants of its block.
    for p in packages() {
        let Some((block, _)) = &p.blocks else {
            continue;
        };
        let file = p
            .sources
            .iter()
            .find(|s| s.ends_with(".wgsl"))
            .expect("a shader");
        let shader = std::fs::read_to_string(crate_dir().join(file)).expect("the shader");
        for (name, _) in block_constants(block) {
            assert!(
                !shader.contains(&format!("const {name}:")),
                "{}: the shader declares {name} itself",
                p.name
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// No stray copy of a registered value
// ---------------------------------------------------------------------------------------------

/// Where a registered value may appear as a literal besides its home: `(file, literal as written,
/// reason)`. A short list on purpose: each line is a number that is the same by accident or by
/// arithmetic, not a second copy of the constant.
const ALLOWED: &[(&str, &str, &str)] = &[(
    "src/cpu/output_sharpen.rs",
    "0.001",
    "MIN_AMOUNT: the least amount that makes a stage (0.1 %), not the halo dead zone",
)];

/// The text of `source` with comments, string literals and everything the tests and the tuning home
/// own removed: what is left is code that must not carry a registered value.
fn code_of(path: &str, source: &str) -> String {
    let mut text = source.to_string();
    // The home of the constants.
    while let (Some(a), Some(b)) = (text.find("// tuning:begin"), text.find("// tuning:end")) {
        assert!(a < b, "{path}: tuning markers out of order");
        text.replace_range(a..b + "// tuning:end".len(), "");
    }
    // Tests: everything from the first `#[cfg(test)]` on (they live at the foot of a file).
    if let Some(at) = text.find("#[cfg(test)]") {
        text.truncate(at);
    }
    // Comments and strings.
    let bytes = text.as_bytes();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'/' if bytes.get(i + 1) == Some(&b'/') => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if bytes.get(i + 1) == Some(&b'*') => {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i += 2;
            }
            b'"' => {
                i += 1;
                while i < bytes.len() && bytes[i] != b'"' {
                    if bytes[i] == b'\\' {
                        i += 1;
                    }
                    i += 1;
                }
                i += 1;
                out.push_str("\"\"");
            }
            c => {
                out.push(char::from(c));
                i += 1;
            }
        }
    }
    out
}

/// The numeric literals of `code`, as written (without a type suffix).
fn literals(code: &str) -> Vec<String> {
    let b = code.as_bytes();
    let ident = |c: u8| c.is_ascii_alphanumeric() || c == b'_' || c == b'.';
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let starts = b[i].is_ascii_digit() && (i == 0 || !ident(b[i - 1]));
        if !starts {
            i += 1;
            continue;
        }
        let begin = i;
        // Hexadecimal and binary literals are not constants of the registries.
        if b[i] == b'0' && matches!(b.get(i + 1), Some(b'x' | b'b')) {
            i += 2;
            while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
                i += 1;
            }
            continue;
        }
        while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
            i += 1;
        }
        if b.get(i) == Some(&b'.') && b.get(i + 1).is_some_and(u8::is_ascii_digit) {
            i += 1;
            while i < b.len() && (b[i].is_ascii_digit() || b[i] == b'_') {
                i += 1;
            }
        }
        if matches!(b.get(i), Some(b'e' | b'E'))
            && (b.get(i + 1).is_some_and(u8::is_ascii_digit)
                || (matches!(b.get(i + 1), Some(b'-' | b'+'))
                    && b.get(i + 2).is_some_and(u8::is_ascii_digit)))
        {
            i += 2;
            while i < b.len() && b[i].is_ascii_digit() {
                i += 1;
            }
        }
        out.push(code[begin..i].to_string());
        // A suffix (`f32`, `usize`, `u`) belongs to the literal.
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == b'_') {
            i += 1;
        }
    }
    out
}

fn value_of(literal: &str) -> Option<f64> {
    literal.replace('_', "").parse().ok()
}

#[test]
fn no_registered_value_is_written_again_outside_its_home() {
    let mut checked_files = 0;
    let mut offences = Vec::new();
    let mut excused = BTreeSet::new();
    for p in packages() {
        let distinct: Vec<&Constant> = p.registry.iter().filter(|c| c.distinct).collect();
        for file in p.sources {
            let source = std::fs::read_to_string(crate_dir().join(file)).expect("a source file");
            let code = code_of(file, &source);
            checked_files += 1;
            for literal in literals(&code) {
                let Some(v) = value_of(&literal) else {
                    continue;
                };
                for c in &distinct {
                    let same = v == c.value || f64::from(v as f32) == c.value;
                    if !same {
                        continue;
                    }
                    if ALLOWED
                        .iter()
                        .any(|(f, lit, _)| f == file && *lit == literal)
                    {
                        excused.insert((*file, literal.clone()));
                    } else {
                        offences.push(format!(
                            "{file}: the literal {literal} is the value of {} {} ({})",
                            c.id, c.name, p.name
                        ));
                    }
                }
            }
        }
    }
    assert_eq!(checked_files, 9, "the files of the three packages");
    offences.sort();
    offences.dedup();
    assert!(
        offences.is_empty(),
        "a registered value written outside its Tuning:\n{}",
        offences.join("\n")
    );
    // The allow-list is short and every entry excuses a real hit.
    assert!(ALLOWED.len() <= 4, "keep the allow-list short");
    for (file, literal, reason) in ALLOWED {
        assert!(!reason.is_empty());
        assert!(
            excused.contains(&(*file, literal.to_string())),
            "{file}: the allowed literal {literal} excuses nothing (remove it)"
        );
    }
}

#[test]
fn the_scan_finds_a_planted_copy() {
    // The proof of the proof: the scanner sees a registered value written into code, in a CPU file
    // and in a shader, and ignores it in a comment, a string and a test.
    let code = code_of(
        "x.rs",
        "fn f() { let a = 1.6 * x; } // 0.75\nconst S: &str = \"0.03\";\n#[cfg(test)]\nmod t { const Y: f64 = 18.0; }\n",
    );
    assert_eq!(literals(&code), vec!["1.6".to_string()]);
    let wgsl = code_of(
        "x.wgsl",
        "let w2 = ar_op(1.6 * rho); /* 0.03 */ let k = 1e-20; let n = 2u; let m = 0.5f;",
    );
    assert_eq!(
        literals(&wgsl),
        vec![
            "1.6".to_string(),
            "1e-20".to_string(),
            "2".to_string(),
            "0.5".to_string()
        ]
    );
    assert_eq!(value_of("1_000.0"), Some(1000.0));
}
