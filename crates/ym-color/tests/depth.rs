//! The Depth mask's maths (`docs/algorithms/06` section 5.8, D-04 and the gradient of 2026-11-08): the rank against an
//! independent f64 count, the gradient against the formula, the far limit's bounds, the plane and its cache, Pick depth,
//! the Focus plane's median and the Show depth plane.

use std::sync::Arc;

use ym_color::mask_depth::{
    DEPTH_FAR_MIN, DEPTH_PICK_MARGIN, DEPTH_RULES_VERSION, DepthMap, DepthMapError, DepthSource,
    depth_far, depth_gradient, depth_plane, depth_ranks, depth_s0, pick_depth, pick_far,
};

use ym_color::mask_raster::RasterSource;

/// A small deterministic generator (xorshift64*), so the tests need no crate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The mid-rank by counting, from the definition: `2 less + eq - 1`, and 0 for the smallest value's group.
fn count_rank(d: &[f32], i: usize) -> u32 {
    let v = f64::from(d[i]);
    let less = d.iter().filter(|&&x| f64::from(x) < v).count();
    let eq = d.iter().filter(|&&x| f64::from(x) == v).count();
    if less == 0 {
        // The tie group of the smallest value is the far end (D-H2).
        return 0;
    }
    (2 * less + eq - 1) as u32
}

fn map_of(key: u64, w: usize, h: usize, d: Vec<f32>) -> Arc<DepthMap> {
    Arc::new(DepthMap::new(key, w, h, d).expect("a map"))
}

#[test]
fn the_rank_matches_an_independent_count_on_random_maps() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for n in [2usize, 3, 17, 200, 700] {
        let d: Vec<f32> = (0..n).map(|_| rng.unit() * 9.0).collect();
        let k = depth_ranks(&d);
        for (i, &got) in k.iter().enumerate() {
            assert_eq!(got, count_rank(&d, i), "n {n} cell {i}");
        }
    }
}

#[test]
fn the_rank_matches_the_count_on_maps_with_many_ties() {
    let mut rng = Rng(77);
    // A quarter of the cells are equal, the rest random; plus a few more repeated values.
    let n = 400;
    let mut d: Vec<f32> = (0..n).map(|_| rng.unit()).collect();
    for v in d.iter_mut().step_by(4) {
        *v = 0.25;
    }
    for v in d.iter_mut().skip(1).step_by(37) {
        *v = 0.75;
    }
    let k = depth_ranks(&d);
    for (i, &got) in k.iter().enumerate() {
        assert_eq!(got, count_rank(&d, i), "cell {i}");
    }
    // Equal values have equal ranks, whatever the order they are listed in.
    let tie: Vec<u32> = (0..n).filter(|&i| d[i] == 0.25).map(|i| k[i]).collect();
    assert!(tie.len() >= 90 && tie.iter().all(|&v| v == tie[0]));

    // The same values listed in the opposite order give the same rank for each value.
    let mut reversed = d.clone();
    reversed.reverse();
    let kr = depth_ranks(&reversed);
    for i in 0..n {
        assert_eq!(k[i], kr[n - 1 - i]);
    }
}

#[test]
fn a_map_of_one_value_or_one_cell_is_all_far_end() {
    // D-H2: the tie group of the smallest value is the far end, and a map of one value is that group whole: nothing
    // in it is nearer than anything else, so it has no mask at any limit.
    let flat = map_of(1, 4, 3, vec![0.7; 12]);
    for i in 0..12 {
        assert_eq!(flat.k[i], 0);
        assert_eq!(flat.r(i), 0.0);
        assert_eq!(flat.z(i), 1.0);
    }
    let one = map_of(2, 1, 1, vec![3.0]);
    assert_eq!(one.k, vec![0]);
    assert_eq!(one.r(0), 0.0);
    assert_eq!(one.z(0), 1.0);
    // -0.0 and +0.0 are equal.
    let zeros = depth_ranks(&[0.0, -0.0, 1.0]);
    assert_eq!(zeros[0], zeros[1]);
}

#[test]
fn the_ends_of_the_rank_and_r_plus_z_is_one() {
    let mut rng = Rng(5);
    let d: Vec<f32> = (0..300).map(|_| rng.unit()).collect();
    let m = map_of(3, 20, 15, d.clone());
    let nearest = (0..300)
        .max_by(|&a, &b| d[a].total_cmp(&d[b]))
        .expect("a cell");
    let farthest = (0..300)
        .min_by(|&a, &b| d[a].total_cmp(&d[b]))
        .expect("a cell");
    assert_eq!(m.r(nearest), 1.0);
    assert_eq!(m.z(nearest), 0.0);
    assert_eq!(m.r(farthest), 0.0);
    assert_eq!(m.z(farthest), 1.0);
    for i in 0..300 {
        assert!((m.r(i) + m.z(i) - 1.0).abs() <= 1e-15);
        // z is the share that is nearer, ties as half, from its own definition.
        let greater = d.iter().filter(|&&x| x > d[i]).count();
        let eq = d.iter().filter(|&&x| x == d[i]).count();
        let want = (2 * greater + eq - 1) as f64 / (2.0 * 299.0);
        assert_eq!(m.z(i), want);
    }
    // A group of equal smallest values is the far end whole (r = 0, z = 1), and the other cells keep their
    // mid-ranks: 1.0 has less 6, eq 1: k = 12.
    let mut d2 = vec![0.0f32; 6];
    d2.extend([1.0, 2.0, 3.0, 4.0]);
    let m2 = map_of(4, 5, 2, d2);
    for i in 0..6 {
        assert_eq!((m2.r(i), m2.z(i)), (0.0, 1.0));
    }
    assert_eq!(m2.k[6], 12);
    assert_eq!(m2.r(6), 12.0 / 18.0);
    assert_eq!(m2.z(9), 0.0);
    // A tie group that is not the smallest keeps its mid-rank: two cells at 2.0 over six below, k = 2 * 6 + 2 - 1.
    let m3 = map_of(
        5,
        5,
        2,
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 3.0, 4.0],
    );
    assert_eq!(m3.k[6], 13);
    assert_eq!(m3.k[0], 0);
}

#[test]
fn a_map_that_is_not_a_map_is_refused() {
    assert_eq!(
        DepthMap::new(1, 2, 2, vec![0.0; 3]).unwrap_err(),
        DepthMapError::Size {
            width: 2,
            height: 2,
            found: 3
        }
    );
    assert!(matches!(
        DepthMap::new(1, 0, 2, vec![]),
        Err(DepthMapError::Size { .. })
    ));
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        assert_eq!(
            DepthMap::new(1, 2, 1, vec![0.0, bad]).unwrap_err(),
            DepthMapError::NotFinite
        );
    }
}

/// The gradient of section 5.8 step 2 in f64 from the raw setting, as a code: written out here, not read from the code
/// it checks.
fn reference_gradient(z: f64, far: f64) -> u16 {
    let f = if far.is_nan() {
        1.0
    } else {
        far.clamp(0.02, 1.0)
    };
    let s0 = (1.0 - z / f).clamp(0.0, 1.0);
    (s0 * 65535.0).round() as u16
}

#[test]
fn the_gradient_is_one_at_the_nearest_zero_at_the_limit_and_linear_between() {
    for far in [0.02, 0.1, 0.35, 0.5, 0.8, 1.0] {
        assert_eq!(depth_gradient(0.0, far), 1.0, "far {far}");
        // 0 at the limit and beyond.
        assert_eq!(depth_gradient(far, far), 0.0, "far {far}");
        if far < 1.0 {
            assert_eq!(
                depth_gradient(far + 0.05_f64.min(1.0 - far), far),
                0.0,
                "far {far}"
            );
        }
        assert_eq!(depth_gradient(1.0, far), 0.0, "far {far}");
        // Linear in between: the quarter points of [0, far].
        for q in [0.25, 0.5, 0.75] {
            let got = depth_gradient(q * far, far);
            assert!((got - (1.0 - q)).abs() < 1e-12, "far {far} q {q}: {got}");
        }
    }
    // Monotone and bounded over a fine grid.
    let mut last = 1.0;
    for i in 0..=1000 {
        let s = depth_gradient(f64::from(i) / 1000.0, 0.37);
        assert!((0.0..=1.0).contains(&s) && s <= last);
        last = s;
    }
}

#[test]
fn a_far_limit_of_one_is_the_show_depth_picture_exactly() {
    let mut rng = Rng(7);
    let d: Vec<f32> = (0..3000)
        .map(|_| if rng.unit() < 0.3 { 0.0 } else { rng.unit() })
        .collect();
    let m = map_of(50, 60, 50, d);
    for i in 0..m.len() {
        // 1 - z, in f64, is the formula; r is z's complement from the same integers.
        assert_eq!(depth_gradient(m.z(i), 1.0), 1.0 - m.z(i));
        assert!((depth_gradient(m.z(i), 1.0) - m.r(i)).abs() < 1e-15);
    }
    // The codes are the Show depth plane's, cell for cell.
    assert_eq!(depth_s0(&m, 1.0), m.show_plane(1).data);
}

#[test]
fn the_plane_matches_the_formula_over_z_and_far() {
    let n = 400;
    let m = map_of(51, n, 1, (0..n).map(|i| i as f32 + 1.0).collect());
    for far in [0.02, 0.2, 0.5, 0.8, 1.0, 1.7, -0.3, f64::NAN] {
        let plane = depth_s0(&m, far);
        for (i, &code) in plane.iter().enumerate() {
            assert_eq!(code, reference_gradient(m.z(i), far), "far {far} cell {i}");
        }
    }
}

#[test]
fn the_far_limit_is_held_to_its_least_and_nan_reads_as_the_whole_depth() {
    assert_eq!(depth_far(1.0), 1.0);
    assert_eq!(depth_far(0.35), 0.35);
    assert_eq!(depth_far(5.0), 1.0);
    assert_eq!(depth_far(0.0), DEPTH_FAR_MIN);
    assert_eq!(depth_far(-1.0), DEPTH_FAR_MIN);
    assert_eq!(depth_far(f64::NAN), 1.0);
    assert!((DEPTH_FAR_MIN - 0.02).abs() < 1e-15);
    // The division is safe at the least limit: every cell is a finite code, and only the nearest 2 % have any mask.
    let m = map_of(52, 1000, 1, (0..1000).map(|i| (1000 - i) as f32).collect());
    let plane = depth_s0(&m, 0.0);
    assert!(plane.iter().filter(|&&c| c > 0).count() <= 21);
    assert_eq!(plane[0], 65535);
    assert_eq!(depth_s0(&m, 0.0), depth_s0(&m, DEPTH_FAR_MIN));
}

#[test]
fn the_nearest_cell_is_full_and_the_far_end_is_none_at_any_limit() {
    // One nearest cell, the rest a ramp, a tie group of the smallest value (the sky) at the far end.
    let d = vec![9.0, 5.0, 4.0, 3.0, 0.0, 0.0];
    let m = map_of(53, 6, 1, d);
    for far in [0.1, 0.5, 1.0] {
        let plane = depth_s0(&m, far);
        assert_eq!(plane[0], 65535, "far {far}");
        assert_eq!(plane[4], 0, "far {far}");
        assert_eq!(plane[5], 0, "far {far}");
    }
}

#[test]
fn a_sky_of_ties_at_the_far_end_gets_no_mask_and_a_pick_on_it_takes_the_whole_depth() {
    // 57 % zeros (the sky) and the rest a ramp (the D-H2 tie rule: the sky is z = 1 whole).
    let n = 1000;
    let sky = 570;
    let d: Vec<f32> = (0..n)
        .map(|i| if i < sky { 0.0 } else { (i - sky + 1) as f32 })
        .collect();
    let m = map_of(17, n, 1, d);
    assert!((0..sky).all(|i| m.z(i) == 1.0));
    assert!((sky..n).all(|i| m.z(i) < 0.5));
    // Whatever the limit, the sky has none of the mask, and the nearest cell has all of it.
    for far in [0.3, 0.7, 1.0] {
        let plane = depth_s0(&m, far);
        assert!((0..sky).all(|i| plane[i] == 0), "far {far}");
        assert_eq!(plane[n - 1], 65535);
    }
    // A pick on the sky gives the whole depth.
    assert_eq!(pick_depth(&m, 0.2, 0.5), 1.0);
}

#[test]
fn the_ranks_and_the_plane_are_the_same_every_time() {
    let mut rng = Rng(2026);
    let d: Vec<f32> = (0..2000)
        .map(|_| if rng.unit() < 0.4 { 0.0 } else { rng.unit() })
        .collect();
    let a = map_of(18, 50, 40, d.clone());
    let b = map_of(18, 50, 40, d);
    assert_eq!(a.k, b.k);
    assert_eq!(depth_s0(&a, 0.6), depth_s0(&b, 0.6));
    assert_eq!(depth_s0(&a, 0.6), depth_s0(&a, 0.6));
}

#[test]
fn the_plane_is_the_maps_cells_cut_once_and_the_cache_keeps_the_limit_and_nothing_else() {
    let m = map_of(
        9_100,
        8,
        8,
        (0..64).map(|i| ((i * 37) % 64) as f32).collect(),
    );
    let source = |far: f64| DepthSource {
        map: Arc::clone(&m),
        far,
    };
    let base = depth_plane(&source(0.5));
    assert_eq!((base.width, base.height), (8, 8));
    assert_eq!(base.data, depth_s0(&m, 0.5));
    assert_eq!(base.key, source(0.5).key());
    // The same limit is the same plane, Arc for Arc (Feather, Grow and Refine edges are not in the key).
    assert!(Arc::ptr_eq(&base, &depth_plane(&source(0.5))));
    // Another limit is a new plane with a new key.
    let p = depth_plane(&source(0.6));
    assert!(!Arc::ptr_eq(&base, &p));
    assert_ne!(p.key, base.key);
    // The key is of the limit after the clamping: above 1 is 1, under the least is the least, NaN is 1.
    assert_eq!(source(1.4).key(), source(1.0).key());
    assert_eq!(source(f64::NAN).key(), source(1.0).key());
    assert_eq!(source(-1.0).key(), source(DEPTH_FAR_MIN).key());
    // The rules' version is in the key: it differs from the version-2 key of the same numbers by construction.
    assert_eq!(DEPTH_RULES_VERSION, 3);
    // A new map (another photo) is a miss with the same limit, and the same codes.
    let again = map_of(9_101, 8, 8, m.d.clone());
    let p = depth_plane(&DepthSource {
        map: again,
        far: 0.5,
    });
    assert!(!Arc::ptr_eq(&base, &p));
    assert_ne!(p.key, base.key);
    assert_eq!(p.data, base.data);
    // The raster source of a row gives the same plane at any aspect.
    let row = RasterSource::Depth(Arc::new(source(0.5)));
    assert!(Arc::ptr_eq(&row.plane([1.0, 0.6]), &base));
    assert_eq!(row.key(), base.key);
}

#[test]
fn a_depth_plane_does_not_share_a_key_with_the_same_numbers_in_another_domain() {
    // The tag in the key keeps a Depth plane from being read for a map whose key happens to be equal elsewhere.
    let m = map_of(1, 2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    let s = DepthSource { map: m, far: 0.5 };
    assert_ne!(s.key(), 1);
}

#[test]
fn pick_depth_on_a_ramp_map_gives_the_known_limit() {
    // 100 columns, d falls to the right: the left is near. Cell i has r = (99 - i) / 99.
    let m = map_of(20, 100, 1, (0..100).map(|i| (100 - i) as f32).collect());
    // The middle of the photo: z = 0.5 gives a limit of 0.6.
    assert!((pick_depth(&m, 0.5, 0.5) - 0.6).abs() < 1e-12);
    // The left end: z = 0, the limit is the margin alone.
    assert!((pick_depth(&m, 0.0, 0.5) - DEPTH_PICK_MARGIN).abs() < 1e-12);
    // The right end: z = 1, the limit is held to 1.
    assert_eq!(pick_depth(&m, 1.0, 0.5), 1.0);
    // A quarter across: x = 0.25 is at u = 24.5, between cells 24 and 25, r = (99 - 24.5) / 99.
    let z: f64 = 1.0 - (99.0 - 24.5) / 99.0;
    assert!((pick_depth(&m, 0.25, 0.5) - (z + 0.1)).abs() < 1e-12);
    // A map with all values equal is all far end (D-H2): the pick is the whole depth.
    let flat = map_of(21, 5, 5, vec![2.0; 25]);
    assert_eq!(pick_depth(&flat, 0.3, 0.7), 1.0);
}

#[test]
fn pick_far_is_the_depth_plus_the_margin_held_to_one() {
    assert!((DEPTH_PICK_MARGIN - 0.1).abs() < 1e-15);
    assert_eq!(pick_far(0.5), 0.5 + 0.1);
    assert_eq!(pick_far(0.0), 0.1);
    assert_eq!(pick_far(0.9), 0.9 + 0.1);
    assert_eq!(pick_far(0.95), 1.0);
    assert_eq!(pick_far(1.0), 1.0);
    // NaN reads as the middle of the photo.
    assert_eq!(pick_far(f64::NAN), 0.6);
    // Not rounded: the sum is kept whole, and always inside the limit's range.
    assert_eq!(pick_far(0.3345), 0.3345 + 0.1);
    for i in 0..=1000 {
        let z = f64::from(i) / 1000.0;
        assert_eq!(pick_far(z), (z + 0.1).min(1.0), "z {z}");
        // The subject stays in the mask: its own cell keeps a share of the mask (at 0.1 of the limit at the least).
        assert!(depth_gradient(z, pick_far(z)) > 0.0 || z >= 1.0);
    }
}

#[test]
fn invert_is_the_rows_complement_and_the_plane_is_not_inverted() {
    // Invert belongs to the row (section 3, step 1) and acts after the plane: 1 - S0, here from the plane's codes.
    let m = map_of(54, 5, 1, vec![5.0, 4.0, 3.0, 2.0, 1.0]);
    let plane = depth_s0(&m, 0.5);
    let inverted: Vec<u16> = plane.iter().map(|&c| 65535 - c).collect();
    assert_eq!(plane[0], 65535);
    assert_eq!(inverted[0], 0);
    assert_eq!(inverted[4], 65535);
    for (c, i) in plane.iter().zip(&inverted) {
        assert_eq!(u32::from(*c) + u32::from(*i), 65535);
    }
}

#[test]
fn the_focus_median_takes_the_side_that_holds_most_of_the_box() {
    // 20 x 10 cells: the left 12 columns are near (d = 10 + noise), the right 8 columns far (d = noise).
    let mut rng = Rng(9);
    let d: Vec<f32> = (0..200)
        .map(|i| if i % 20 < 12 { 10.0 } else { 0.0 } + rng.unit())
        .collect();
    let m = map_of(30, 20, 10, d);
    // A box of 8 x 4 cells centred on x = 0.5 (cell column 10): it covers columns 6..=13, 6 near and 2 far.
    let z_box = m.focus_z(0.5, 0.5, 8.0 / 20.0, 4.0 / 10.0);
    // The median lies on the near side: z of the near cells is below 0.6 (they are the nearest 60 %).
    assert!(z_box < 0.6, "z {z_box}");
    // A box centred in the far part: the median is far (z above 0.4).
    let z_far = m.focus_z(0.9, 0.5, 0.2, 0.4);
    assert!(z_far > 0.4, "z {z_far}");
}

#[test]
fn the_focus_median_is_computed_by_the_definition_odd_and_even_counts() {
    // A ramp of 100 x 1 distinct cells; k = 2 rank, r = rank / 99, z = 1 - r.
    let m = map_of(31, 100, 1, (0..100).map(|i| i as f32).collect());
    // Cells 40..=44 (five centres inside): the middle rank is 42. x in [0.40, 0.45] holds centres 0.405 .. 0.445.
    let z = m.focus_z(0.425, 0.5, 0.05, 1.0);
    let want = (198.0 - 2.0 * 42.0) / 198.0;
    assert!((z - want).abs() < 1e-15, "{z} {want}");
    // Six centres 40..=45: the median is (k_a + k_b) / 2 of ranks 42 and 43, k = 84 and 86.
    let z = m.focus_z(0.43, 0.5, 0.06, 1.0);
    let want = (198.0 - 85.0) / 198.0;
    assert!((z - want).abs() < 1e-15, "{z} {want}");
}

#[test]
fn the_focus_box_has_a_least_size_is_cut_to_the_photo_and_falls_back_to_the_cell_that_holds_the_point()
 {
    let m = map_of(
        32,
        100,
        100,
        (0..10_000).map(|i| (i % 100) as f32).collect(),
    );
    // A zero-size box is made 3 cells wide: the column holding x = 0.505 (cell 50) and its two neighbours; the
    // median of three cells of one row is the middle one, the column of cell 50 (d = 50 when it is in the row).
    let z = m.focus_z(0.505, 0.505, 0.0, 0.0);
    let col50 = (0..10_000)
        .filter(|i| i % 100 == 50)
        .map(|i| m.z(i))
        .sum::<f64>()
        / 100.0;
    assert!((z - col50).abs() < 0.05, "{z} vs {col50}");
    // A box at the edge of the photo is cut to it and still answers.
    let edge = m.focus_z(0.0, 0.0, 0.1, 0.1);
    assert!((0.0..=1.0).contains(&edge));
    let corner = m.focus_z(1.0, 1.0, 0.1, 0.1);
    assert!((0.0..=1.0).contains(&corner));
    // A box too small to hold a centre of a coarse map: 2 x 2 cells and a point between the centres.
    let coarse = map_of(33, 2, 2, vec![4.0, 3.0, 2.0, 1.0]);
    let z = coarse.focus_z(0.5, 0.5, 0.0, 0.0);
    assert!((0.0..=1.0).contains(&z));
    // A point that is not a number reads as the middle; a map of one cell is the far end.
    let one = map_of(34, 1, 1, vec![1.0]);
    assert_eq!(one.focus_z(f64::NAN, 0.3, 0.1, 0.1), 1.0);
}

#[test]
fn the_show_depth_plane_is_white_where_the_map_is_nearest() {
    let m = map_of(40, 4, 1, vec![0.1, 0.9, 0.5, 0.3]);
    let plane = m.show_plane(77);
    assert_eq!((plane.width, plane.height, plane.key), (4, 1, 77));
    assert_eq!(plane.data[1], 65535, "the largest d is the nearest: white");
    assert_eq!(plane.data[0], 0, "the smallest d is the farthest: black");
    assert!(plane.data[2] > plane.data[3]);
    // A flat map is all far end, black (D-H2).
    let flat = map_of(41, 3, 3, vec![1.0; 9]);
    assert!(flat.show_plane(1).data.iter().all(|&c| c == 0));
}
