//! Fuzzes the TIFF/IFD walker: header, directory chains, SubIFD/EXIF/GPS pointers and every typed
//! value reader. Property: no panic, no unbounded allocation, for any bytes.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ym_raw::tiff::{ByteOrder, MemSource, Tiff};

fuzz_target!(|data: &[u8]| {
    let source = MemSource(data);

    // With a header: whatever the first bytes claim.
    if let Ok((tiff, first)) = Tiff::open(&source)
        && let Ok(nodes) = tiff.walk(first)
    {
        for node in &nodes {
            for entry in &node.ifd.entries {
                let _ = tiff.uints(entry, 64);
                let _ = tiff.ints(entry, 64);
                let _ = tiff.floats(entry, 64);
                let _ = tiff.ascii(entry);
                let _ = tiff.entry_location(entry);
                let _ = tiff.entry_prefix(entry, 16);
            }
        }
    }

    // Without a header: a directory at every plausible offset, in both byte orders, with and
    // without a value-offset bias (the SR2 case).
    for order in [ByteOrder::Little, ByteOrder::Big] {
        for bias in [0u64, 4096] {
            let tiff = Tiff::headerless(&source, order, bias);
            for offset in [0u64, 2, 8, 16] {
                if let Ok(ifd) = tiff.read_ifd(offset) {
                    for entry in &ifd.entries {
                        let _ = tiff.uints(entry, 16);
                        let _ = tiff.floats(entry, 16);
                    }
                }
            }
        }
    }
});
