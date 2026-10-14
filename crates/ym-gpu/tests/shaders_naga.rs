//! Every generated WGSL module, in every storage mode, parses and validates with naga and
//! translates to MSL (Metal), SPIR-V (Vulkan) and HLSL (DirectX 12). No GPU needed: this runs on every
//! CI machine, including the macOS runners without a Metal device, and catches shader-language
//! problems that a lavapipe-only run would find later (or not at all).
//!
//! The validation baseline is `Capabilities::empty()`: a kernel that needs an optional capability
//! (push constants, 64-bit integers, subgroups, float atomics, binding arrays, ...) fails here, on
//! every machine, instead of on the one adapter that lacks it. Only the two f16 storage modes (legacy,
//! reached by `StoragePreference::Force`) add the capability their own declarations need
//! (`SHADER_FLOAT16` for `enable f16;`, `SHADER_FLOAT16_IN_FLOAT32` for `pack2x16float`). A source lint then checks the kernel contract of
//! `docs/gpu-plan.md` section 4.6 and the determinism rules that can be read off the source
//! (K1 to K3, D3, D6); the lint itself is proved on seeded bad shaders.

use naga::back::{hlsl, msl, spv};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use naga::{AddressSpace, ScalarKind, TypeInner};
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::{KernelId, KernelSet, shader_source};

/// The capability the storage mode's own declarations need; everything else must validate with none.
/// The default mode (f32) needs nothing.
fn baseline(mode: StorageMode) -> Capabilities {
    match mode {
        StorageMode::F16Native => Capabilities::SHADER_FLOAT16,
        StorageMode::F16Packed => Capabilities::SHADER_FLOAT16_IN_FLOAT32,
        StorageMode::F32 => Capabilities::empty(),
    }
}

fn parse(mode: StorageMode, id: KernelId, src: &str) -> naga::Module {
    naga::front::wgsl::parse_str(src)
        .unwrap_or_else(|e| panic!("{mode:?} {id:?}: parse error\n{}", e.emit_to_string(src)))
}

fn validate(caps: Capabilities, module: &naga::Module) -> Result<naga::valid::ModuleInfo, String> {
    Validator::new(ValidationFlags::all(), caps)
        .validate(module)
        .map_err(|e| format!("{e:?}"))
}

fn check(mode: StorageMode, id: KernelId) {
    let src = shader_source(mode, id);
    let module = parse(mode, id, &src);
    let info = validate(baseline(mode), &module)
        .unwrap_or_else(|e| panic!("{mode:?} {id:?}: validation error\n{e}"));
    msl::write_string(
        &module,
        &info,
        &msl::Options::default(),
        &msl::PipelineOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{mode:?} {id:?}: MSL translation failed: {e:?}"));
    spv::write_vec(&module, &info, &spv::Options::default(), None)
        .unwrap_or_else(|e| panic!("{mode:?} {id:?}: SPIR-V translation failed: {e:?}"));
    // DirectX 12: f16 needs shader model 6.2, the rest runs on the 5.1 floor.
    let options = hlsl::Options {
        shader_model: if mode == StorageMode::F16Native {
            hlsl::ShaderModel::V6_2
        } else {
            hlsl::ShaderModel::V5_1
        },
        ..hlsl::Options::default()
    };
    let mut out = String::new();
    hlsl::Writer::new(&mut out, &options, &hlsl::PipelineOptions::default())
        .write(&module, &info, None)
        .unwrap_or_else(|e| panic!("{mode:?} {id:?}: HLSL translation failed: {e:?}"));
    assert!(!out.is_empty(), "{mode:?} {id:?}: empty HLSL");
}

#[test]
fn all_kernels_validate_and_translate_in_all_storage_modes() {
    for mode in [
        StorageMode::F32,
        StorageMode::F16Packed,
        StorageMode::F16Native,
    ] {
        for id in KernelId::ALL {
            check(mode, id);
        }
    }
}

#[test]
fn f16_native_modules_enable_f16_and_the_others_do_not() {
    let native = shader_source(StorageMode::F16Native, KernelId::Color);
    assert!(native.starts_with("enable f16;"));
    assert!(native.contains("array<f16>"));
    let packed = shader_source(StorageMode::F16Packed, KernelId::Color);
    assert!(!packed.contains("enable f16"));
    assert!(packed.contains("unpack2x16float"));
    let plain = shader_source(StorageMode::F32, KernelId::Color);
    assert!(
        !plain.contains("array<f16>")
            && !plain.contains("2x16float")
            && !plain.contains("enable f16")
    );
}

#[test]
fn constants_are_generated_from_ym_color_not_typed() {
    let src = shader_source(StorageMode::F32, KernelId::Color);
    // LogC4 `a` as ym_color stores it, rounded to f32.
    let a = format!("{:?}", 2231.8263090676883_f64 as f32);
    assert!(
        src.contains(&format!("const LC4_A: f32 = {a};")),
        "LC4_A missing"
    );
    assert!(src.contains("const AWG4_TO_AP0_0: vec3<f32>"));
}

#[test]
fn the_adaptive_ratio_passes_share_one_source_and_each_has_one_entry_point() {
    // RA-09: eleven pipelines of `adaptive_ratio.wgsl`, f32 planes in every storage mode, so the text
    // of a pass differs between storage modes only by the generated prelude (and not at all in its
    // buffers: no f16 anywhere).
    let passes: Vec<KernelId> = KernelId::ALL
        .into_iter()
        .filter(|id| id.name().starts_with("adaptive_ratio"))
        .collect();
    assert_eq!(passes.len(), 11);
    for id in passes {
        for mode in [StorageMode::F32, StorageMode::F16Native] {
            let src = shader_source(mode, id);
            assert_eq!(src.matches("@compute").count(), 1, "{id:?}");
            assert!(src.contains("struct ArParams"), "{id:?}");
            assert!(
                !src.contains("array<f16>") && !src.contains("2x16float"),
                "{id:?}"
            );
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Placeholders and sets
// ---------------------------------------------------------------------------------------------

#[test]
fn placeholder_kernels_are_marked_in_name_and_text_and_write_nothing_but_a_nan() {
    let placeholders: Vec<KernelId> = KernelId::ALL
        .into_iter()
        .filter(|id| id.is_placeholder())
        .collect();
    // One per kernel family the plan still has to write (G-06 x4, G-07 x7, G-08 x1; G-04's sampler and
    // encode kernels are real).
    assert_eq!(placeholders.len(), 12, "{placeholders:?}");
    for id in &placeholders {
        let owner = id.placeholder_owner().expect("owner");
        let src = shader_source(StorageMode::F32, *id);
        assert!(
            src.contains(&format!("PLACEHOLDER kernel `{}`", id.name())),
            "{id:?} does not announce itself"
        );
        assert!(
            src.contains(owner),
            "{id:?} does not name its owner {owner}"
        );
        assert!(
            src.contains("0x7fc00000u"),
            "{id:?} must write a NaN, not look like a working kernel"
        );
        assert_eq!(src.matches("@compute").count(), 1);
        assert_eq!(
            src.matches("var<storage").count(),
            // Binding 0 of the stub, and nothing the shared prelude declares.
            1,
            "{id:?}"
        );
    }
    // The real kernels are not placeholders and say nothing of the sort.
    for id in KernelId::ALL.into_iter().filter(|id| !id.is_placeholder()) {
        assert!(
            !shader_source(StorageMode::F32, id).contains("PLACEHOLDER"),
            "{id:?}"
        );
    }
}

#[test]
fn every_kernel_is_in_exactly_one_warm_up_set_and_all_is_their_union() {
    for id in KernelId::ALL {
        let set = id.set();
        assert_ne!(set, KernelSet::All, "{id:?}");
        for s in [KernelSet::View, KernelSet::Raw, KernelSet::Tail] {
            assert_eq!(s.contains(id), s == set, "{id:?} in {s:?}");
        }
        assert!(KernelSet::All.contains(id));
    }
    let total: usize = [KernelSet::View, KernelSet::Raw, KernelSet::Tail]
        .iter()
        .map(|s| s.kernels().count())
        .sum();
    assert_eq!(total, KernelId::ALL.len());
    assert_eq!(KernelSet::All.kernels().count(), KernelId::ALL.len());
    // The colour chain and output sharpening are what a viewport frame needs; the adaptive ratio is
    // the raw chain; the resize is the export tail.
    assert_eq!(KernelId::Color.set(), KernelSet::View);
    assert_eq!(KernelId::OutputSharpenTransfer.set(), KernelSet::View);
    assert_eq!(KernelId::AdaptiveRatioGreen.set(), KernelSet::Raw);
    assert_eq!(KernelId::Resize.set(), KernelSet::Tail);
    // Names are unique (they are the pipeline labels).
    let mut names: Vec<_> = KernelId::ALL.iter().map(|k| k.name()).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), KernelId::ALL.len());
}

// ---------------------------------------------------------------------------------------------
// The lint
// ---------------------------------------------------------------------------------------------

/// Maximum storage buffers one stage binds (K3: the WebGPU default limit).
const MAX_STORAGE_BUFFERS: usize = 8;
/// Maximum uniform buffers one stage binds (K3).
const MAX_UNIFORM_BUFFERS: usize = 1;
/// Maximum workgroup memory in bytes (K2).
const MAX_WORKGROUP_BYTES: u32 = 16 * 1024;

/// `src` without its `//` comments, so a comment that mentions a forbidden word is not a violation.
fn strip_comments(src: &str) -> String {
    let mut out = String::with_capacity(src.len());
    for line in src.lines() {
        out.push_str(line.split_once("//").map_or(line, |(code, _)| code));
        out.push('\n');
    }
    out
}

/// The text-level part of the lint: directives and words the kernel contract forbids.
fn lint_text(mode: StorageMode, src: &str) -> Vec<String> {
    let code = strip_comments(src);
    let mut found = Vec::new();
    for line in code.lines() {
        let t = line.trim();
        if t.starts_with("enable ") || t.starts_with("requires ") || t.starts_with("diagnostic ") {
            // K1: `enable f16;` is the one directive, and only in the legacy f16-native mode.
            if !(t == "enable f16;" && mode == StorageMode::F16Native) {
                found.push(format!("K1: directive `{t}`"));
            }
        }
    }
    for (word, why) in [
        ("subgroup", "D3/K1: subgroup operations"),
        ("textureSample", "K1: texture sampling"),
        ("textureGather", "K1: texture sampling"),
        ("binding_array", "K1: binding arrays"),
        ("push_constant", "K1: push constants"),
        ("var<immediate", "K1: immediates"),
        ("atomic<f32>", "D1: float atomics"),
        ("fma(", "D6: fma() is never called"),
    ] {
        if code.contains(word) {
            found.push(format!("{why}: `{word}`"));
        }
    }
    found
}

/// The IR-level part: entry point shape, bindings, workgroup memory and scalar types.
fn lint_module(module: &naga::Module, info: &naga::valid::ModuleInfo) -> Vec<String> {
    let mut found = Vec::new();
    if module.entry_points.len() != 1 {
        found.push(format!(
            "K1: {} entry points (one `main`)",
            module.entry_points.len()
        ));
    }
    for (i, ep) in module.entry_points.iter().enumerate() {
        if ep.name != "main" || ep.stage != naga::ShaderStage::Compute {
            found.push(format!(
                "K1: entry point `{}` is not a compute `main`",
                ep.name
            ));
        }
        // K2.
        if !matches!(ep.workgroup_size, [16, 16, 1] | [256, 1, 1]) {
            found.push(format!(
                "K2: workgroup size {:?} (16 x 16 or 256)",
                ep.workgroup_size
            ));
        }
        let fi = info.get_entry_point(i);
        let (mut storage, mut uniform, mut workgroup) = (0usize, 0usize, 0u32);
        for (handle, var) in module.global_variables.iter() {
            if fi[handle].is_empty() {
                continue;
            }
            match var.space {
                AddressSpace::Storage { .. } => storage += 1,
                AddressSpace::Uniform => uniform += 1,
                AddressSpace::WorkGroup => {
                    workgroup += module.types[var.ty].inner.size(module.to_ctx());
                }
                _ => {}
            }
        }
        if storage > MAX_STORAGE_BUFFERS {
            found.push(format!(
                "K3: {storage} storage buffers (at most {MAX_STORAGE_BUFFERS})"
            ));
        }
        if uniform > MAX_UNIFORM_BUFFERS {
            found.push(format!(
                "K3: {uniform} uniform buffers (at most {MAX_UNIFORM_BUFFERS})"
            ));
        }
        if workgroup > MAX_WORKGROUP_BYTES {
            found.push(format!(
                "K2: {workgroup} bytes of workgroup memory (at most {MAX_WORKGROUP_BYTES})"
            ));
        }
    }
    for (_, ty) in module.types.iter() {
        match &ty.inner {
            TypeInner::Atomic(s) if s.kind == ScalarKind::Float => {
                found.push("D1: a float atomic".to_string());
            }
            TypeInner::Scalar(s) | TypeInner::Atomic(s) if s.width == 8 => {
                found.push(format!("K1: a 64-bit {:?}", s.kind));
            }
            TypeInner::Image { .. } | TypeInner::Sampler { .. } => {
                found.push("K1: a texture or sampler type".to_string());
            }
            TypeInner::BindingArray { .. } => found.push("K1: a binding array".to_string()),
            _ => {}
        }
    }
    found
}

/// The whole lint of one WGSL module.
fn lint(mode: StorageMode, src: &str) -> Vec<String> {
    let mut found = lint_text(mode, src);
    // A directive naga cannot parse is already reported by the text lint.
    if let Ok(module) = naga::front::wgsl::parse_str(src)
        && let Ok(info) = validate(baseline(mode), &module)
    {
        found.extend(lint_module(&module, &info));
    }
    found
}

/// Violations of the contract in kernels that predate it, each with the task that removes it. A
/// listed violation must still be there (`known_violations_are_still_needed`), so a task that fixes
/// one has to delete its line, and nothing new can hide behind this list.
const KNOWN_VIOLATIONS: &[(KernelId, &str, &str)] = &[];

fn known(id: KernelId, finding: &str) -> bool {
    KNOWN_VIOLATIONS
        .iter()
        .any(|(k, prefix, _)| *k == id && finding.starts_with(prefix))
}

#[test]
fn no_kernel_breaks_the_kernel_contract() {
    for mode in [
        StorageMode::F32,
        StorageMode::F16Packed,
        StorageMode::F16Native,
    ] {
        for id in KernelId::ALL {
            let src = shader_source(mode, id);
            let found: Vec<String> = lint(mode, &src)
                .into_iter()
                .filter(|f| !known(id, f))
                .collect();
            assert!(found.is_empty(), "{mode:?} {id:?}: {found:#?}");
        }
    }
}

#[test]
fn known_violations_are_still_needed() {
    for (id, prefix, owner) in KNOWN_VIOLATIONS {
        let found = lint(StorageMode::F32, &shader_source(StorageMode::F32, *id));
        assert!(
            found.iter().any(|f| f.starts_with(prefix)),
            "{id:?} no longer breaks `{prefix}`: delete its line from KNOWN_VIOLATIONS ({owner})"
        );
    }
}

#[test]
fn the_lint_fails_on_seeded_bad_shaders() {
    // Valid WGSL that breaks the IR-level rules: nine storage buffers, an 8 x 8 workgroup,
    // 20 KB of workgroup memory and a call to fma().
    let mut bad = String::new();
    for i in 0..9 {
        bad.push_str(&format!(
            "@group(0) @binding({i}) var<storage, read_write> b{i}: array<f32>;\n"
        ));
    }
    bad.push_str(
        "var<workgroup> big: array<u32, 5000>;\n\
         @compute @workgroup_size(8, 8)\n\
         fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n\
         \tbig[gid.x % 5000u] = gid.x;\n\
         \tlet x = fma(f32(gid.x), 2.0, 1.0);\n",
    );
    for i in 0..9 {
        bad.push_str(&format!("\tb{i}[gid.x] = x + f32(big[gid.x % 5000u]);\n"));
    }
    bad.push_str("}\n");
    let found = lint(StorageMode::F32, &bad);
    for rule in [
        "K3: 9 storage",
        "K2: workgroup size",
        "K2: 20000 bytes",
        "D6",
    ] {
        assert!(
            found.iter().any(|f| f.starts_with(rule)),
            "the lint missed {rule}: {found:#?}"
        );
    }

    // Text-level violations, none of which naga would even parse.
    let text_bad = "enable subgroups;\nenable f16;\n\
                    @compute @workgroup_size(16, 16)\nfn main() { let a = subgroupAdd(1.0); }\n";
    let found = lint(StorageMode::F32, text_bad);
    assert!(
        found
            .iter()
            .any(|f| f.contains("directive `enable subgroups;`"))
    );
    assert!(
        found.iter().any(|f| f.contains("directive `enable f16;`")),
        "`enable f16;` is allowed only in the f16-native mode"
    );
    assert!(found.iter().any(|f| f.contains("subgroup operations")));
    let found = lint(
        StorageMode::F16Native,
        "enable f16;\n@compute @workgroup_size(16, 16)\nfn main() { }\n",
    );
    assert!(found.is_empty(), "{found:#?}");

    // A comment that mentions a forbidden word is not a violation.
    let found = lint(
        StorageMode::F32,
        "// never use subgroupAdd or fma() here\n@compute @workgroup_size(256)\nfn main() { }\n",
    );
    assert!(found.is_empty(), "{found:#?}");

    // A 64-bit integer fails validation with the baseline capability set, so a kernel using one
    // cannot even reach the lint: that is the point of the baseline.
    let src = "@compute @workgroup_size(16, 16)\nfn main() { var x: i64 = 1li; x += 1li; }\n";
    let module = naga::front::wgsl::parse_str(src).expect("naga parses i64");
    assert!(
        validate(Capabilities::empty(), &module).is_err(),
        "a 64-bit integer must not validate without its capability"
    );
    assert!(
        validate(Capabilities::SHADER_INT64, &module).is_ok(),
        "the same shader validates once the capability is granted"
    );
}
