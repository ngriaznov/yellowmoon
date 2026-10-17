//! Fuzz target for the parametric look JSON loader and the curve spline behind it. Properties:
//! parsing never panics; an accepted look evaluates without a NaN on the unit cube (a CDL whose
//! output reaches `CDL_OUTPUT_LIMIT` at white is refused by the parser: `corpus/look_json/cdl-overflow.json`,
//! found by the nightly run of 2026-10-17) and bakes (or reports a typed error) without panicking.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ym_color::look::ParametricLook;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    let Ok(look) = ParametricLook::from_json(text) else {
        return;
    };
    for rgb in [
        [0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0],
        [0.9, 0.1, 0.4],
        [0.2, 0.7, 0.05],
        [0.5, 0.5, 0.5],
    ] {
        let out = look.evaluate(rgb);
        assert!(out.iter().all(|v| !v.is_nan()), "NaN from {rgb:?}");
    }
    let _ = look.bake(5);
    let _ = look.to_transform();
});
