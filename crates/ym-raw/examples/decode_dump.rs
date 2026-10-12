//! Decodes one ARW file and writes the samples as little-endian `u16`, for comparisons with other
//! decoders (`tools/oracle/rawpy_compare.py` writes LibRaw's pixels the same way).
//!
//! ```text
//! cargo run -p ym-raw --release --example decode_dump -- FILE.ARW OUT.bin [--replicate] [--keep-wb]
//! ```
//!
//! `--replicate` and `--keep-wb` select LibRaw's behaviour for Lossless M and S (replicated chroma,
//! white balance left in). The decoder's own summary is printed on stdout.

use std::time::Instant;

use ym_raw::decode::{ChromaUpsampling, DecodeOptions, YcbcrOptions, decode_with};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [file, out, flags @ ..] = &args[..] else {
        eprintln!("usage: decode_dump FILE.ARW OUT.bin [--replicate] [--keep-wb]");
        std::process::exit(2);
    };
    let options = DecodeOptions {
        ycbcr: YcbcrOptions {
            chroma: if flags.iter().any(|f| f == "--replicate") {
                ChromaUpsampling::Replicate
            } else {
                ChromaUpsampling::Linear
            },
            undo_white_balance: !flags.iter().any(|f| f == "--keep-wb"),
        },
    };
    let started = Instant::now();
    let image = match decode_with(file, &options) {
        Ok(image) => image,
        Err(error) => {
            println!(
                "{}",
                serde_json::json!({ "error": error.to_string(), "kind": error.kind() })
            );
            std::process::exit(1);
        }
    };
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    let bytes: Vec<u8> = image
        .data
        .data
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(out, bytes).expect("write the samples");
    println!(
        "{}",
        serde_json::json!({
            "decode": image.meta.decode.id(),
            "width": image.pixel_width(),
            "height": image.pixel_height(),
            "channels": image.channels(),
            "black": image.black,
            "white": image.white,
            "wb_as_shot": image.wb_as_shot,
            "tone_curve": image.tone_curve,
            "qstep": image.qstep.is_some(),
            "ms": ms,
        })
    );
}
