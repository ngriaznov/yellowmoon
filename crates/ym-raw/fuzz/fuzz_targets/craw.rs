//! Fuzzes the cRAW block decoder and the quantisation map: any bytes as image data, any widths
//! (multiples of 32), heights and tone-curve knots. Property: no panic, and a correctly sized
//! buffer always decodes to `width * height` values that stay on the curve, each with a positive
//! `qstep`.

#![no_main]

use libfuzzer_sys::fuzz_target;
use ym_raw::decode::craw::{build_curve, decode_rows};

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }
    let width = 32 * (1 + usize::from(data[0] % 8));
    let height = 1 + usize::from(data[1] % 8);
    let knots = [
        u16::from_le_bytes([data[2], data[3]]),
        u16::from_le_bytes([data[4], data[5]]),
        u16::from_le_bytes([data[6], data[7]]),
        u16::from_le_bytes([data[8], data[9]]),
    ];
    let Ok(curve) = build_curve(knots) else {
        return;
    };
    let payload = &data[10..];
    // Too short is an error, never a panic.
    assert!(
        payload.len() >= width * height
            || decode_rows(payload, width, height, &curve, true).is_err()
    );
    let mut bytes = payload.to_vec();
    bytes.resize(width * height, 0);
    let (pixels, qstep) = decode_rows(&bytes, width, height, &curve, true)
        .expect("a buffer of the right size always decodes");
    assert_eq!(pixels.len(), width * height);
    assert_eq!(qstep.len(), pixels.len());
    let top = curve[4094];
    assert!(pixels.iter().all(|v| *v <= top));
    assert!(qstep.iter().all(|q| q.to_f32() > 0.0));
    let (again, _) = decode_rows(&bytes, width, height, &curve, false).expect("decodes");
    assert_eq!(again, pixels, "the qstep switch must not change the pixels");
});
