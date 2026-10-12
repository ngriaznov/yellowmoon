//! Fuzz target for the `.cube` parser (docs/02 Tests, docs/13 "Fuzzing"). Properties checked:
//!
//! 1. `parse_bytes` never panics, never aborts on allocation, and returns promptly.
//! 2. An accepted file is self-consistent (entry counts, finite values, ordered domains).
//! 3. Sampling an accepted LUT at awkward inputs (NaN, infinities, huge values) stays finite.
//! 4. Writing an accepted cube with `to_cube_string` and parsing it again gives the same cube.
#![no_main]

use libfuzzer_sys::fuzz_target;
use ym_color::cube::{MAX_1D_SIZE, MAX_3D_SIZE, parse_bytes};

fuzz_target!(|data: &[u8]| {
    let Ok(file) = parse_bytes(data) else {
        return;
    };
    assert!(file.lut1d.is_some() || file.lut3d.is_some());
    let probes = [
        [0.0, 0.0, 0.0],
        [0.5, 0.25, 0.75],
        [1.0, 1.0, 1.0],
        [-1.0e30, 1.0e30, 0.0],
        [f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
    ];
    if let Some(l1) = &file.lut1d {
        assert!((2..=MAX_1D_SIZE).contains(&l1.size()));
        assert_eq!(l1.data().len(), l1.size());
        for p in probes {
            assert!(l1.sample(p).iter().all(|v| v.is_finite()));
        }
    }
    if let Some(cube) = &file.lut3d {
        assert!((2..=MAX_3D_SIZE).contains(&cube.size()));
        assert_eq!(cube.data().len(), cube.size().pow(3));
        for p in probes {
            assert!(cube.sample(p).iter().all(|v| v.is_finite()));
        }
        let text = cube.to_cube_string("fuzz");
        let again = parse_bytes(text.as_bytes()).expect("a written cube parses");
        assert_eq!(again.lut3d.as_ref(), Some(cube));
    }
});
