//! Fuzzes the lossless JPEG decoder (headers, Huffman tables, restart intervals, every predictor,
//! subsampled components). Seed the corpus with valid streams from
//! `cargo run -p ym-raw --features testgen --example gen_seeds -- <dir>`. Property: no panic, no
//! allocation beyond the sample budget; whatever decodes has consistent plane sizes.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ym_raw::decode::ljpeg::{Options, decode, parse_header};

fuzz_target!(|data: &[u8]| {
    let options = Options {
        max_samples: 1 << 20,
        sony_420_quirk: data.get(2).is_some_and(|b| b & 1 == 1),
    };
    let _ = parse_header(data);
    if let Ok(decoded) = decode(data, &options) {
        let mut samples = 0usize;
        for plane in &decoded.planes {
            assert_eq!(plane.data.len(), plane.width * plane.height);
            samples += plane.data.len();
        }
        assert!(samples <= options.max_samples);
        assert!(decoded.width > 0 && decoded.height > 0);
    }
});
