//! M1-01: the ARRI 17-CCT table: parsing, the derived native to AWG3 matrices, mired interpolation
//! between the bracketing pair, and the difference from ARRI's own `searchsorted` pairing.

use serde_json::Value;
use ym_color::arri::{ArriTable, CCT_MAX_K, CCT_MIN_K};
use ym_color::consts::{Mat3, linalg};
use ym_color::gamut::matrices;

const TABLE_CCTS: [f64; 17] = [
    2000.0, 2100.0, 2200.0, 2400.0, 2600.0, 2900.0, 3200.0, 3500.0, 3900.0, 4300.0, 4700.0, 5100.0,
    5600.0, 6500.0, 7500.0, 9000.0, 11000.0,
];

fn json_matrices() -> Vec<(f64, Mat3)> {
    let json: Value = serde_json::from_str(include_str!(
        "../../../data/arri/alexa_native_to_awg3_17cct.json"
    ))
    .unwrap();
    json["matrices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|entry| {
            let mut m = [[0.0; 3]; 3];
            for (i, row) in entry["native_to_awg3"]
                .as_array()
                .unwrap()
                .iter()
                .enumerate()
            {
                for (j, x) in row.as_array().unwrap().iter().enumerate() {
                    m[i][j] = x.as_f64().unwrap();
                }
            }
            (entry["cct"].as_f64().unwrap(), m)
        })
        .collect()
}

/// ARRI's pairing from `v3_IDT_maker.py::getIDTMatrix` (aces-dev v1.3.1), reimplemented from its
/// behaviour: `i = searchsorted(ccts, cct)` (left side), the last entry is returned as is, otherwise
/// the pair (i, i + 1) is interpolated in 1/CCT with weight `a`, which is negative (an extrapolation
/// backwards) for a CCT between two entries.
fn arri_pairing(table: &ArriTable, cct: f64) -> Mat3 {
    let ccts = table.ccts_k();
    let n = ccts.len();
    let i = ccts.partition_point(|&c| c < cct);
    assert!(i < n, "ARRI's script fails above the last entry");
    let m_i = *table.native_to_awg3_entry(i).unwrap();
    if i == n - 1 {
        return m_i;
    }
    let m_next = *table.native_to_awg3_entry(i + 1).unwrap();
    let a = (1.0 / cct - 1.0 / ccts[i]) / (1.0 / ccts[i + 1] - 1.0 / ccts[i]);
    linalg::lerp(&m_i, &m_next, a)
}

#[test]
fn embedded_table_equals_the_data_file() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../data/arri/Alexa_aces_matrix.txt"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let from_disk = ArriTable::parse(&text).unwrap();
    assert_eq!(&from_disk, ArriTable::builtin());
    assert_eq!(ArriTable::builtin().ccts_k(), &TABLE_CCTS);
    assert_eq!(CCT_MIN_K, TABLE_CCTS[0]);
    assert_eq!(CCT_MAX_K, TABLE_CCTS[16]);
}

#[test]
fn derived_native_to_awg3_matches_the_json_to_1e_12() {
    let table = ArriTable::builtin();
    let reference = json_matrices();
    assert_eq!(reference.len(), 17);
    let mut worst = 0.0_f64;
    for (i, (cct, want)) in reference.iter().enumerate() {
        assert_eq!(table.ccts_k()[i], *cct);
        let got = table.native_to_awg3_entry(i).unwrap();
        worst = worst.max(linalg::max_abs_diff(got, want));
    }
    println!("max |derived - JSON| over the 17 matrices: {worst:e}");
    assert!(worst < 1e-12, "{worst:e}");
}

#[test]
fn rows_sum_to_one() {
    let table = ArriTable::builtin();
    // docs/02: derived rows sum to 1 within 1e-12 (the source table within ~5e-15).
    assert!(
        table.max_row_sum_error_awg3() < 1e-12,
        "{:e}",
        table.max_row_sum_error_awg3()
    );
    assert!(
        table.max_row_sum_error_ap0() < 1e-12,
        "{:e}",
        table.max_row_sum_error_ap0()
    );
    println!(
        "row sum error: table native->AP0 {:e}, derived native->AWG3 {:e}",
        table.max_row_sum_error_ap0(),
        table.max_row_sum_error_awg3()
    );
    // Interpolated matrices keep it, anywhere in the range.
    let mut cct = 1800.0;
    while cct < 12000.0 {
        for s in linalg::row_sums(&table.native_to_awg3(cct)) {
            assert!((s - 1.0).abs() < 1e-12, "cct {cct}: {s}");
        }
        for s in linalg::row_sums(&table.native_to_awg4(cct)) {
            assert!((s - 1.0).abs() < 1e-12, "cct {cct}: {s}");
        }
        cct += 37.0;
    }
}

#[test]
fn interpolation_equals_table_values_at_the_17_ccts() {
    let table = ArriTable::builtin();
    for (i, &cct) in TABLE_CCTS.iter().enumerate() {
        // Exactly (bit for bit) the table entry, not just close to it.
        assert_eq!(
            &table.native_to_awg3(cct),
            table.native_to_awg3_entry(i).unwrap(),
            "AWG3 at {cct} K"
        );
        assert_eq!(
            &table.native_to_ap0(cct),
            table.native_to_ap0_entry(i).unwrap(),
            "AP0 at {cct} K"
        );
    }
    for (i, (cct, want)) in json_matrices().iter().enumerate() {
        assert!(
            linalg::max_abs_diff(&table.native_to_awg3(*cct), want) < 1e-12,
            "{i}"
        );
    }
}

#[test]
fn interpolation_is_linear_in_mired_between_the_bracketing_pair() {
    let table = ArriTable::builtin();
    // 3333 K sits between 3200 K and 3500 K.
    let cct = 3333.0;
    let w = (1.0e6 / cct - 1.0e6 / 3200.0) / (1.0e6 / 3500.0 - 1.0e6 / 3200.0);
    let a = table.native_to_awg3_entry(6).unwrap();
    let b = table.native_to_awg3_entry(7).unwrap();
    let want = linalg::lerp(a, b, w);
    assert!(linalg::max_abs_diff(&table.native_to_awg3(cct), &want) < 1e-15);
    let (i, weight) = table.bracket(cct);
    assert_eq!(i, 6);
    assert!((weight - w).abs() < 1e-15);
    // Linear in mired, not in kelvin: the kelvin midpoint is not the half-way matrix.
    let (_, w_mid) = table.bracket(3350.0);
    assert!((w_mid - 0.5).abs() > 0.01);
    // The mired midpoint is.
    let mid = 1.0e6 / (0.5 * (1.0e6 / 3200.0 + 1.0e6 / 3500.0));
    let (_, w_half) = table.bracket(mid);
    assert!((w_half - 0.5).abs() < 1e-12);
}

#[test]
fn interpolation_is_continuous_across_table_entries() {
    let table = ArriTable::builtin();
    for &cct in &TABLE_CCTS[1..16] {
        let below = table.native_to_awg3(cct - 1e-7);
        let at = table.native_to_awg3(cct);
        let above = table.native_to_awg3(cct + 1e-7);
        assert!(linalg::max_abs_diff(&below, &at) < 1e-9, "below {cct}");
        assert!(linalg::max_abs_diff(&above, &at) < 1e-9, "above {cct}");
    }
}

#[test]
fn clamps_to_2000_and_11000() {
    let table = ArriTable::builtin();
    assert_eq!(table.native_to_awg3(1500.0), table.native_to_awg3(2000.0));
    assert_eq!(table.native_to_awg3(0.0), table.native_to_awg3(2000.0));
    assert_eq!(table.native_to_awg3(-5.0), table.native_to_awg3(2000.0));
    assert_eq!(
        table.native_to_awg3(25_000.0),
        table.native_to_awg3(11000.0)
    );
    assert_eq!(
        table.native_to_awg3(f64::INFINITY),
        table.native_to_awg3(11000.0)
    );
}

#[test]
fn arri_searchsorted_pairing_agrees_at_table_ccts_and_differs_between() {
    let table = ArriTable::builtin();
    // Both agree exactly at the 17 table CCTs (docs/02 asks for this test).
    for &cct in &TABLE_CCTS {
        let ours = table.native_to_awg3(cct);
        let theirs = arri_pairing(table, cct);
        assert!(linalg::max_abs_diff(&ours, &theirs) < 1e-15, "{cct} K");
    }
    // Between entries ARRI's pairing extrapolates from the next pair; ours stays inside the bracket.
    let mut worst = 0.0_f64;
    let mut worst_at = 0.0;
    for cct in [
        2050.0, 2150.0, 2300.0, 2500.0, 2750.0, 3000.0, 3350.0, 3700.0, 4100.0, 4500.0, 4900.0,
        5350.0, 6000.0, 7000.0, 8000.0,
    ] {
        let ours = table.native_to_awg3(cct);
        let theirs = arri_pairing(table, cct);
        let d = linalg::max_abs_diff(&ours, &theirs);
        assert!(d > 1e-6, "pairings should differ at {cct} K, diff {d:e}");
        if d > worst {
            worst = d;
            worst_at = cct;
        }
    }
    println!("largest |ours - ARRI pairing| between entries: {worst:e} at {worst_at} K");
    // Between 9000 and 11000 K ARRI's script returns the 11000 K matrix unchanged.
    let theirs = arri_pairing(table, 10_000.0);
    assert_eq!(&theirs, table.native_to_awg3_entry(16).unwrap());
    let ours = table.native_to_awg3(10_000.0);
    assert!(linalg::max_abs_diff(&ours, &theirs) > 1e-4);
}

#[test]
fn native_to_awg4_is_awg3_then_the_exact_gamut_matrix() {
    let table = ArriTable::builtin();
    let want = linalg::mul(&matrices().awg3_to_awg4, &table.native_to_awg3(4500.0));
    assert_eq!(table.native_to_awg4(4500.0), want);
}

#[test]
fn white_balanced_native_white_maps_to_white() {
    // Rows sum to 1, so native (1,1,1) is AWG3 (1,1,1): the D65 white. (Balanced input, docs/02.)
    let table = ArriTable::builtin();
    for cct in [2000.0, 3200.0, 5600.0, 6500.0, 11000.0, 4321.0] {
        let out = linalg::mul_vec(&table.native_to_awg3(cct), [1.0, 1.0, 1.0]);
        for c in out {
            assert!((c - 1.0).abs() < 1e-12);
        }
    }
}
