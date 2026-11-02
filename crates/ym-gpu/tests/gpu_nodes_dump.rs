//! A tool, not a gate: hashes the device's pictures of the default graph (Main alone) for every
//! configuration of `common::chains::cases()`, at every stop and in every output kind, and writes one
//! line per picture to `$YM_DUMP_DIR/default_graph.txt`.
//!
//! N-04 (`docs/23-nodes.md` section 5.4) moved the node parameters out of the uniform block and put the
//! node loop in the kernel; the default graph must give the same bytes on lavapipe as the kernel before
//! it. Run this test with `YM_DUMP_DIR=<dir>` on the commit before and on the commit after a change to
//! the colour kernel and compare the two files (`cmp`). Without the variable the test does nothing.
//!
//! ```text
//! WGPU_BACKEND=vulkan YM_REQUIRE_GPU=1 YM_DUMP_DIR=/tmp/after \
//!     cargo test -p ym-gpu --test gpu_nodes_dump -- --nocapture
//! ```

mod common;

use std::fmt::Write as _;

use common::chains::{self, camera_image};
use common::gpus;
use ym_color::display::ColorChain;
use ym_gpu::kernels::PixelData;
use ym_gpu::kernels::color::{ColorOutput, PreparedChain, Stop};
use ym_gpu::queue::{JobSpec, Priority};

/// FNV-1a over bytes: a fingerprint of a picture, not a hash anyone relies on.
fn fnv(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &b| {
        (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn hash_of(data: &PixelData) -> u64 {
    match data {
        PixelData::F32Planar(v) => fnv(bytemuck::cast_slice(v)),
        PixelData::Rgba8(v) => fnv(v),
        PixelData::U16Planar(v) => fnv(bytemuck::cast_slice(v)),
    }
}

#[test]
fn dump_the_default_graph_pictures() {
    let Some(dir) = std::env::var_os("YM_DUMP_DIR") else {
        return;
    };
    let Some(dev) = gpus().first() else { return };
    let stops = [
        Stop::C6,
        Stop::C7,
        Stop::C8,
        Stop::C9,
        Stop::C9b,
        Stop::C10,
        Stop::C11,
        Stop::DngLinearAwg4,
        Stop::C12,
        Stop::C13,
        Stop::Before,
    ];
    let mut outputs: Vec<(String, ColorOutput)> = stops
        .iter()
        .map(|&s| (format!("{s:?}"), ColorOutput::Stage(s)))
        .collect();
    outputs.push(("rgba8-dither".into(), ColorOutput::Rgba8 { dither: true }));
    outputs.push(("rgba8-round".into(), ColorOutput::Rgba8 { dither: false }));
    outputs.push((
        "before-rgba8".into(),
        ColorOutput::BeforeRgba8 { dither: true },
    ));
    outputs.push(("u16-10".into(), ColorOutput::U16 { bits: 10 }));
    let mut text = String::new();
    for (w, h, seed) in [(67usize, 41usize, 0xBEEFu64), (33, 17, 7)] {
        let img = camera_image(w, h, seed);
        for (name, params) in chains::cases() {
            let chain = ColorChain::new(&params).expect("chain");
            let img = img.clone();
            let outs: Vec<ColorOutput> = outputs.iter().map(|(_, o)| *o).collect();
            let hashes: Vec<u64> = dev
                .gpu
                .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                    let prepared = PreparedChain::new(ctx, &chain)?;
                    outs.iter()
                        .map(|o| {
                            prepared
                                .render(ctx, &img, (3, 5), *o)
                                .map(|r| hash_of(&r.data))
                        })
                        .collect()
                })
                .expect("GPU render");
            for ((what, _), hash) in outputs.iter().zip(hashes) {
                let _ = writeln!(text, "{w}x{h} {name} {what} {hash:016x}");
            }
        }
    }
    let dir = std::path::PathBuf::from(dir);
    std::fs::create_dir_all(&dir).expect("dir");
    std::fs::write(dir.join("default_graph.txt"), &text).expect("write");
    eprintln!("wrote {} lines to {}", text.lines().count(), dir.display());
}
