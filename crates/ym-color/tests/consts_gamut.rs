//! M1-01: every matrix printed in docs/02 section Constants, recomputed from primaries in f64.

use serde_json::Value;
use ym_color::consts::{
    ACES_WHITE_XY, AP0, AWG3, AWG4, D65_XY, DISPLAY_P3, Mat3, REC709, REC2020, cat02_adaptation,
    linalg, xy_to_xyz,
};
use ym_color::gamut::{self, LinearAwg3, LinearAwg4, matrices};

fn assert_mat(name: &str, got: &Mat3, want: &Mat3, tol: f64) {
    let worst = linalg::max_abs_diff(got, want);
    assert!(
        worst <= tol,
        "{name}: max abs diff {worst:e} exceeds {tol:e}\n got  {got:?}\n want {want:?}"
    );
}

// Printed in docs/02 section Gamuts (10 decimals).
const AWG3_TO_XYZ: Mat3 = [
    [0.6380076193, 0.2147038563, 0.0977444514],
    [0.2919537790, 0.8238410415, -0.1157948205],
    [0.0027982790, -0.0670342357, 1.1532937074],
];
const AWG4_TO_XYZ: Mat3 = [
    [0.7048583204, 0.1297602952, 0.1158373115],
    [0.2545241764, 0.7814777327, -0.0360019091],
    [0.0000000000, 0.0000000000, 1.0890577508],
];
const AWG3_TO_AWG4: Mat3 = [
    [0.8892557469, 0.1288981523, -0.0181538992],
    [0.0840832627, 1.0093919566, -0.0934752193],
    [0.0025694496, -0.0615525078, 1.0589830582],
];
const AWG4_TO_P3: Mat3 = [
    [1.5205023970, -0.4042986627, -0.1162037343],
    [-0.1360315837, 1.2698479799, -0.1338163962],
    [0.0058785171, -0.0548756605, 1.0489971435],
];
const AWG4_TO_REC709: Mat3 = [
    [1.8931234427, -0.7808815036, -0.1122419390],
    [-0.2057003583, 1.3402574894, -0.1345571311],
    [-0.0127057429, -0.1521848758, 1.1648906187],
];
// Printed in docs/02 (9 decimals; ARRI's IDTs print the same values rounded to 6 digits).
const AWG3_TO_AP0: Mat3 = [
    [0.680205505, 0.236136602, 0.083657893],
    [0.085414980, 1.017470879, -0.102885858],
    [0.002056522, -0.062562500, 1.060505979],
];

#[test]
fn printed_matrices_to_1e_9() {
    let m = matrices();
    assert_mat("AWG3 to XYZ", &m.awg3_to_xyz, &AWG3_TO_XYZ, 1e-9);
    assert_mat("AWG4 to XYZ", &m.awg4_to_xyz, &AWG4_TO_XYZ, 1e-9);
    assert_mat("AWG3 to AWG4", &m.awg3_to_awg4, &AWG3_TO_AWG4, 1e-9);
    assert_mat("AWG4 to Display P3", &m.awg4_to_p3, &AWG4_TO_P3, 1e-9);
    assert_mat("AWG4 to Rec.709", &m.awg4_to_rec709, &AWG4_TO_REC709, 1e-9);
    assert_mat("AWG3 to AP0 (CAT02)", &m.awg3_to_ap0, &AWG3_TO_AP0, 1e-9);
}

#[test]
fn printed_numbers_are_the_f64_values_rounded() {
    // The printed 10-decimal values are the correctly rounded f64 results (half a unit in the last place).
    let m = matrices();
    assert_mat("AWG3 to XYZ", &m.awg3_to_xyz, &AWG3_TO_XYZ, 5.1e-11);
    assert_mat("AWG4 to XYZ", &m.awg4_to_xyz, &AWG4_TO_XYZ, 5.1e-11);
    assert_mat("AWG3 to AWG4", &m.awg3_to_awg4, &AWG3_TO_AWG4, 5.1e-11);
    assert_mat("AWG4 to Display P3", &m.awg4_to_p3, &AWG4_TO_P3, 5.1e-11);
    assert_mat(
        "AWG4 to Rec.709",
        &m.awg4_to_rec709,
        &AWG4_TO_REC709,
        5.1e-11,
    );
    assert_mat("AWG3 to AP0", &m.awg3_to_ap0, &AWG3_TO_AP0, 5.1e-10);
}

#[test]
fn awg3_to_ap0_matches_the_json_to_1e_12() {
    let json: Value = serde_json::from_str(include_str!(
        "../../../data/arri/alexa_native_to_awg3_17cct.json"
    ))
    .unwrap();
    let want = mat_from_json(&json["awg3_to_ap0_cat02"]);
    assert_mat("AWG3 to AP0 vs JSON", &matrices().awg3_to_ap0, &want, 1e-12);
}

#[test]
fn awg_to_ap0_rows_sum_to_one() {
    // docs/02: rows of AWG3 to AP0 sum to 1 within 1e-15 .
    for (name, m) in [
        ("AWG3", matrices().awg3_to_ap0),
        ("AWG4", matrices().awg4_to_ap0),
    ] {
        for (row, s) in linalg::row_sums(&m).iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-15,
                "{name} to AP0 row {row} sums to {s:.17}"
            );
        }
    }
    // Same-white gamut conversions preserve white as well.
    for m in [
        matrices().awg3_to_awg4,
        matrices().awg4_to_awg3,
        matrices().awg4_to_p3,
        matrices().awg4_to_rec709,
        matrices().awg4_to_rec2020,
    ] {
        for s in linalg::row_sums(&m) {
            assert!((s - 1.0).abs() < 1e-15, "row sum {s:.17}");
        }
    }
}

#[test]
fn inverse_pairs_multiply_to_identity() {
    let m = matrices();
    for (a, b) in [
        (&m.awg3_to_awg4, &m.awg4_to_awg3),
        (&m.awg4_to_ap0, &m.ap0_to_awg4),
        (&m.awg3_to_ap0, &m.ap0_to_awg3),
        (&m.awg3_to_xyz, &m.xyz_to_awg3),
        (&m.cat02_d65_to_aces, &m.cat02_aces_to_d65),
    ] {
        assert_mat("A * A^-1", &linalg::mul(a, b), &linalg::IDENTITY, 1e-13);
    }
}

#[test]
fn white_points_map_to_equal_rgb() {
    // D65 white of each D65 gamut and the ACES white of AP0 are RGB (1,1,1) at Y = 1.
    let d65 = xy_to_xyz(D65_XY);
    let aces = xy_to_xyz(ACES_WHITE_XY);
    for (name, to_xyz) in [
        ("AWG3", matrices().xyz_to_awg3),
        ("AWG4", matrices().xyz_to_awg4),
        ("Rec.709", matrices().xyz_to_rec709),
        ("P3", matrices().xyz_to_p3),
        ("Rec.2020", matrices().xyz_to_rec2020),
    ] {
        let rgb = linalg::mul_vec(&to_xyz, d65);
        for c in rgb {
            assert!((c - 1.0).abs() < 1e-14, "{name} white {rgb:?}");
        }
    }
    let rgb = linalg::mul_vec(&matrices().xyz_to_ap0, aces);
    for c in rgb {
        assert!((c - 1.0).abs() < 1e-14, "AP0 white {rgb:?}");
    }
}

#[test]
fn ap0_is_derived_from_its_primaries() {
    // AP0 to XYZ: the canonical SMPTE ST 2065-1 matrix (published to 10 digits), up to that rounding.
    let want: Mat3 = [
        [0.9525523959, 0.0, 0.0000936786],
        [0.3439664498, 0.7281660966, -0.0721325464],
        [0.0, 0.0, 1.0088251844],
    ];
    assert_mat("AP0 to XYZ", &matrices().ap0_to_xyz, &want, 1e-9);
    assert_eq!(AP0.red, [0.7347, 0.2653]);
    assert_eq!(AP0.white, ACES_WHITE_XY);
}

#[test]
fn cat02_adapts_between_the_two_whites() {
    let d65 = xy_to_xyz(D65_XY);
    let aces = xy_to_xyz(ACES_WHITE_XY);
    let fwd = cat02_adaptation(d65, aces);
    let out = linalg::mul_vec(&fwd, d65);
    for k in 0..3 {
        assert!((out[k] - aces[k]).abs() < 1e-14);
    }
    // Adapting a white to itself is the identity.
    assert_mat(
        "CAT02 identity",
        &cat02_adaptation(d65, d65),
        &linalg::IDENTITY,
        1e-14,
    );
}

/// Cross-check against colour-science 0.4.7 (`matrix_RGB_to_RGB`, CAT02), values pasted from a local run.
/// colour-science stores AWG3 to 6 digits and AP0 to 10, so the AWG3 rows differ in the 7th digit;
/// the AWG4 matrices (stored to 12 digits) agree to 1e-9. Our values follow the documented
/// derivation from primaries; colour-science's are the rounded library numbers.
#[test]
fn cross_check_against_colour_science() {
    let m = matrices();
    let awg4_to_p3: Mat3 = [
        [1.520502397025, -0.404298662716, -0.116203734308],
        [-0.136031583652, 1.26984797986, -0.133816396208],
        [0.005878517056, -0.054875660547, 1.048997143491],
    ];
    let awg4_to_2020: Mat3 = [
        [1.119469059767, -0.055195632842, -0.064273426925],
        [-0.058484457047, 1.176734933498, -0.11825047645],
        [0.001547445177, -0.031135328767, 1.029587883589],
    ];
    let awg4_to_709: Mat3 = [
        [1.893123442653, -0.780881503617, -0.112241939036],
        [-0.205700358252, 1.340257489384, -0.134557131131],
        [-0.012705742927, -0.152184875796, 1.164890618723],
    ];
    let awg4_to_ap0: Mat3 = [
        [0.750957362826, 0.14442278671, 0.104619850507],
        [0.000821837059, 1.00739758487, -0.008219421923],
        [-0.000499952144, -0.000854177231, 1.001354129374],
    ];
    let awg3_to_ap0: Mat3 = [
        [0.680205916073, 0.236136749952, 0.083657407364],
        [0.085415069486, 1.017470771996, -0.102885854998],
        [0.002056264751, -0.062562283681, 1.060506248074],
    ];
    assert_mat("AWG4 to P3 vs colour", &m.awg4_to_p3, &awg4_to_p3, 1e-9);
    assert_mat(
        "AWG4 to 2020 vs colour",
        &m.awg4_to_rec2020,
        &awg4_to_2020,
        1e-9,
    );
    assert_mat(
        "AWG4 to 709 vs colour",
        &m.awg4_to_rec709,
        &awg4_to_709,
        1e-9,
    );
    assert_mat("AWG4 to AP0 vs colour", &m.awg4_to_ap0, &awg4_to_ap0, 1e-9);
    // Differences from colour-science's rounded AWG3 / AP0 constants stay below 1e-6.
    assert_mat(
        "AWG3 to AP0 vs colour (rounded library constants)",
        &m.awg3_to_ap0,
        &awg3_to_ap0,
        1e-6,
    );
}

#[test]
fn gamut_primaries_are_the_documented_ones() {
    assert_eq!(AWG3.green, [0.2210, 0.8480]);
    assert_eq!(AWG4.blue, [0.0991, -0.0308]);
    assert_eq!(REC709.red, [0.64, 0.33]);
    assert_eq!(DISPLAY_P3.green, [0.265, 0.690]);
    assert_eq!(REC2020.blue, [0.131, 0.046]);
}

#[test]
fn typed_spaces_convert_like_the_matrices() {
    let v = LinearAwg3([0.2, 0.4, 0.1]);
    let direct = linalg::mul_vec(&matrices().awg3_to_awg4, v.0);
    assert_eq!(v.to_awg4().0, direct);
    let w = LinearAwg4([0.3, 0.1, 0.05]);
    assert_eq!(w.to_awg3().to_awg4().0.len(), 3);
    let round = w.to_awg3().to_awg4().0;
    for (r, orig) in round.iter().zip(w.0) {
        assert!((r - orig).abs() < 1e-14);
    }
    assert!((gamut::awg4_luminance([1.0, 1.0, 1.0]) - 1.0).abs() < 1e-14);
}

fn mat_from_json(v: &Value) -> Mat3 {
    let mut m = [[0.0; 3]; 3];
    for (i, row) in v.as_array().unwrap().iter().enumerate() {
        for (j, x) in row.as_array().unwrap().iter().enumerate() {
            m[i][j] = x.as_f64().unwrap();
        }
    }
    m
}
