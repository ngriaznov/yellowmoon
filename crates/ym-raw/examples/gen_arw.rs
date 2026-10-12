//! Writes a synthetic ARW file for manual experiments (the LibRaw oracle script, ExifTool, the
//! `yellowmoon info` command).
//!
//! ```text
//! cargo run -p ym-raw --features testgen --example gen_arw -- ILCE-7RM4 128 96 /tmp/a.ARW \
//!     [iso=800] [black=512,512,512,512] [white=16300,16300,16300] [wb=2043,1024,1024,1755] \
//!     [sr2=modern|legacy|absent] [raw_ifd_levels=1] [bits=14] [raw_ifd_white=16383] \
//!     [sony_iso_word=2890] [header=none|sony_dsc]
//! ```

use ym_raw::testgen::{ArwBuilder, Sr2Variant, gradient_mosaic};

fn numbers<const N: usize>(text: &str) -> [u16; N] {
    let values: Vec<u16> = text
        .split(',')
        .map(|v| v.trim().parse().expect("a number"))
        .collect();
    values.try_into().expect("the right number of values")
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [model, width, height, out, options @ ..] = &args[..] else {
        eprintln!("usage: gen_arw MODEL WIDTH HEIGHT OUT.ARW [key=value ...]");
        std::process::exit(2);
    };
    let (width, height): (u32, u32) = (
        width.parse().expect("WIDTH is a number"),
        height.parse().expect("HEIGHT is a number"),
    );
    let mut builder = ArwBuilder::new(model, width, height);
    let mut bits = 14u16;
    for option in options {
        let (key, value) = option.split_once('=').expect("key=value");
        builder = match key {
            "iso" => builder.iso(value.parse().expect("iso")),
            "black" => builder.black(numbers::<4>(value)),
            "white" => builder.white(numbers::<3>(value)),
            "wb" => builder.wb(Some(numbers::<4>(value))),
            "raw_ifd_levels" => builder.raw_ifd_levels(value == "1"),
            "raw_ifd_white" => builder.raw_ifd_white_level(value.parse().expect("white level")),
            "sony_iso_word" => builder.sony_iso_word(value.parse().expect("an ISO word")),
            "header" => builder.maker_note_header(match value {
                "sony_dsc" => b"SONY DSC \0\0\0",
                _ => b"",
            }),
            "bits" => {
                bits = value.parse().expect("bits");
                builder.uncompressed(bits)
            }
            "sr2" => builder.sr2(match value {
                "modern" => Sr2Variant::Modern,
                "legacy" => Sr2Variant::Legacy,
                _ => Sr2Variant::Absent,
            }),
            other => panic!("unknown option {other}"),
        };
    }
    let white_for_content = (1u16 << bits) - 1000;
    let mosaic = gradient_mosaic(width, height, 600 >> (14 - bits), white_for_content, 1);
    builder
        .write_to(out.as_ref(), &mosaic)
        .expect("write the file");
    println!("wrote {out}");
}
