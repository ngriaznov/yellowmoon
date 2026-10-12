//! Fuzzes the Lossless M and S path: the tile payload of a valid synthetic file (4:2:0, 4:2:2 or
//! 4:4:4, selected by the first input byte) is replaced by the fuzz input, then the whole file is
//! decoded in both chroma modes. Seed with `gen_seeds`. Property: no panic; a successful decode is
//! a consistent RGB plane.

#![no_main]

use std::sync::OnceLock;

use libfuzzer_sys::fuzz_target;
use ym_raw::decode::{ChromaUpsampling, DecodeOptions, YcbcrOptions, decode_bytes};
use ym_raw::layout::ChromaSubsampling;
use ym_raw::metadata::probe_bytes;
use ym_raw::testgen::{ArwBuilder, LosslessMs, gradient_rgb};

/// `(file, offset of the first tile)` for each subsampling.
fn templates() -> &'static Vec<(Vec<u8>, usize)> {
    static TEMPLATES: OnceLock<Vec<(Vec<u8>, usize)>> = OnceLock::new();
    TEMPLATES.get_or_init(|| {
        [
            ChromaSubsampling::Yuv420,
            ChromaSubsampling::Yuv422,
            ChromaSubsampling::Yuv444,
        ]
        .into_iter()
        .map(|sub| {
            let rgb = gradient_rgb(64, 64, 1100, 9000, 1);
            let payload = LosslessMs::new(sub, rgb, 64, 64, [2.0, 1.0, 1.6]).tile(32, 32);
            let file = ArwBuilder::new("ILCE-7RM5", 64, 64)
                .lossless_ms(payload)
                .build(&[]);
            let meta = probe_bytes(&file).expect("the template is a valid file");
            let start = meta.storage.chunks().0[0] as usize;
            (file, start)
        })
        .collect()
    })
}

fuzz_target!(|data: &[u8]| {
    let Some((&select, payload)) = data.split_first() else {
        return;
    };
    let templates = templates();
    let (file, start) = &templates[usize::from(select) % templates.len()];
    let mut file = file.clone();
    let n = payload.len().min(file.len() - start);
    file[*start..*start + n].copy_from_slice(&payload[..n]);
    for chroma in [ChromaUpsampling::Replicate, ChromaUpsampling::Linear] {
        let options = DecodeOptions {
            ycbcr: YcbcrOptions {
                chroma,
                undo_white_balance: select & 0x80 != 0,
            },
        };
        if let Ok(image) = decode_bytes(&file, &options) {
            assert!(image.is_reduced());
            assert_eq!(image.data.data.len(), image.data.width * image.data.height);
            assert_eq!(image.data.width, 3 * image.pixel_width());
        }
    }
});
