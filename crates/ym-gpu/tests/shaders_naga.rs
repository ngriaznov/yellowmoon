//! Every generated WGSL module, in every storage mode, parses and validates with naga and
//! translates to MSL (Metal) and SPIR-V (Vulkan). No GPU needed: this runs on every CI machine,
//! including the macOS runners without a Metal device, and catches shader-language problems that a
//! lavapipe-only run would find later (or not at all).

use naga::back::{msl, spv};
use naga::valid::{Capabilities, ValidationFlags, Validator};
use ym_gpu::device::StorageMode;
use ym_gpu::kernels::{KernelId, shader_source};

fn check(mode: StorageMode, id: KernelId) {
    let src = shader_source(mode, id);
    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("{mode:?} {id:?}: parse error\n{}", e.emit_to_string(&src)));
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .unwrap_or_else(|e| panic!("{mode:?} {id:?}: validation error\n{e:?}"));
    msl::write_string(
        &module,
        &info,
        &msl::Options::default(),
        &msl::PipelineOptions::default(),
    )
    .unwrap_or_else(|e| panic!("{mode:?} {id:?}: MSL translation failed: {e:?}"));
    spv::write_vec(&module, &info, &spv::Options::default(), None)
        .unwrap_or_else(|e| panic!("{mode:?} {id:?}: SPIR-V translation failed: {e:?}"));
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
