//! Fuzz target for the ARRI Look File reader (`docs/08` section ARRI Look Files, `docs/13` "Fuzzing").
//! Properties checked:
//!
//! 1. `parse_file` and `parse` never panic, never abort on allocation, and return promptly.
//! 2. A look that is accepted is usable: a LUT of 2 to 129 points with finite entries, and a CDL
//!    that `Cdl::new` accepts.
//! 3. A container that is accepted is rebuilt by the writer into a file that reads back and is
//!    written again as the same bytes.
//!
//! The checksum (and the size field of `.alf4` and `.alf4c`) is repaired before the second read, so
//! the fuzzer gets past it to the tables, offsets and strings behind it.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ym_color::alf::{self, AlfKind};

fn check(bytes: &[u8]) {
    let Ok(file) = alf::parse_file(bytes) else {
        return;
    };
    if let Ok(look) = file.look() {
        assert!((2..=129).contains(&look.lut.size()));
        assert_eq!(look.lut.data().len(), look.lut.size().pow(3));
        assert!(look.lut.data().iter().flatten().all(|v| v.is_finite()));
        if let Some(cdl) = look.cdl {
            assert!(
                ym_color::cdl::Cdl::new(cdl.slope, cdl.offset, cdl.power, cdl.saturation).is_ok()
            );
        }
        assert!(alf::parse(bytes).is_ok());
    }
    // Compared as bytes (a parameter that is NaN is not equal to itself), and not for a file with a
    // string that was not UTF-8: it reads with replacement characters, so it is not what was in the file.
    if !format!("{file:?}").contains('\u{FFFD}')
        && let Ok(rebuilt) = file.to_bytes()
    {
        let again = alf::parse_file(&rebuilt).expect("a written file reads back");
        assert_eq!(again.to_bytes().as_ref(), Ok(&rebuilt));
    }
}

fuzz_target!(|data: &[u8]| {
    check(data);
    let Some(kind) = AlfKind::sniff(data) else {
        return;
    };
    if data.len() < 16 {
        return;
    }
    let mut fixed = data.to_vec();
    let (crc_at, from) = match kind {
        AlfKind::Aml => (8, 0x0C),
        AlfKind::Alf4 | AlfKind::Alf4c => {
            let len = u32::try_from(fixed.len()).unwrap_or(u32::MAX);
            fixed[8..12].copy_from_slice(&len.to_le_bytes());
            (0x0C, 0x10)
        }
    };
    let crc = alf::crc32(&fixed[from..]);
    fixed[crc_at..crc_at + 4].copy_from_slice(&crc.to_le_bytes());
    check(&fixed);
});
