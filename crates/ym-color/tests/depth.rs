//! The Depth mask's maths (`docs/algorithms/06` section 5.8, D-04): the rank against an independent f64 count, the band
//! against the formula, `near`/`far` order and the minimum ramp, the plane and its cache, Pick depth, the Focus plane's
//! median and the Show depth plane.

use std::sync::Arc;

use ym_color::mask_depth::{
    DEPTH_PICK_WIDTH, DEPTH_RAMP_MIN, DepthMap, DepthMapError, DepthSource, depth_band, depth_cut,
    depth_plane, depth_ramp, depth_ranks, depth_s0, pick_band, pick_depth,
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

/// The smoothstep written out here, so the test does not lean on the code it checks.
fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The mid-rank by counting, from the definition: `2 less + eq - 1`.
fn count_rank(d: &[f32], i: usize) -> u32 {
    let v = f64::from(d[i]);
    let less = d.iter().filter(|&&x| f64::from(x) < v).count();
    let eq = d.iter().filter(|&&x| f64::from(x) == v).count();
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
fn a_map_of_one_value_is_flat_and_a_map_of_one_cell_is_in_the_middle() {
    let flat = map_of(1, 4, 3, vec![0.7; 12]);
    for i in 0..12 {
        // less 0, eq 12: k = 11, the middle of 0..=22.
        assert_eq!(flat.k[i], 11);
        assert_eq!(flat.r(i), 0.5);
        assert_eq!(flat.z(i), 0.5);
    }
    let one = map_of(2, 1, 1, vec![3.0]);
    assert_eq!(one.k, vec![0]);
    assert_eq!(one.r(0), 0.5);
    assert_eq!(one.z(0), 0.5);
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
    // A group of m equal smallest values has r = (m - 1) / (2 (n - 1)).
    let mut d2 = vec![0.0f32; 6];
    d2.extend([1.0, 2.0, 3.0, 4.0]);
    let m2 = map_of(4, 5, 2, d2);
    assert_eq!(m2.r(0), 5.0 / 18.0);
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

/// The band of section 5.8 step 2 in f64, from the raw settings, as a code.
fn reference_band(z: f64, near: f64, far: f64, softness: f64) -> u16 {
    let (mut nu, mut phi) = (near.clamp(0.0, 1.0), far.clamp(0.0, 1.0));
    if nu > phi {
        let mean = (nu + phi) / 2.0;
        nu = mean;
        phi = mean;
    }
    let q = (0.25 * softness.clamp(0.0, 1.0)).max(0.02);
    let s = smooth((z - (nu - q)) / q) * (1.0 - smooth((z - phi) / q));
    (s * 65535.0 + 0.5).floor() as u16
}

#[test]
fn the_band_matches_the_formula_over_z_near_far_and_softness() {
    // A ramp map: 1001 distinct values, so z walks 0..=1 in steps of 0.001.
    let m = map_of(10, 1001, 1, (0..1001).map(|i| (1000 - i) as f32).collect());
    for near in [0.0, 0.1, 0.3, 0.5, 0.8, 1.0] {
        for far in [0.0, 0.2, 0.5, 0.9, 1.0] {
            for softness in [0.0, 0.04, 0.5, 1.0] {
                let got = depth_s0(&m, near, far, softness);
                for (i, code) in got.iter().enumerate() {
                    let want = reference_band(m.z(i), near, far, softness);
                    assert_eq!(
                        *code, want,
                        "near {near} far {far} softness {softness} cell {i}"
                    );
                }
            }
        }
    }
}

#[test]
fn near_and_far_are_in_depth_order_and_invert_is_the_complement() {
    // z = 0 is the nearest cell. The default band (0, 0.5) holds the nearest half.
    let m = map_of(11, 100, 1, (0..100).map(|i| (100 - i) as f32).collect());
    let band = depth_s0(&m, 0.0, 0.5, 0.0);
    // Cell 0 has the largest d, so it is the nearest: z = 0, in the band.
    assert_eq!(band[0], 65535);
    // Cell 99 is the farthest: out.
    assert_eq!(band[99], 0);
    // The nearest cells are in and the farthest are out, monotonically along the ramp.
    assert!(band.windows(2).all(|p| p[0] >= p[1]));
    // The far half is the other band; Invert (1 - S0) is that band's complement of the near one.
    let far_half = depth_s0(&m, 0.5, 1.0, 0.0);
    assert_eq!(far_half[99], 65535);
    assert_eq!(far_half[0], 0);
    for i in 0..100 {
        // Away from the ramp the two halves and the inverse agree.
        let z = m.z(i);
        if (z - 0.5).abs() > 0.05 {
            assert_eq!(65535 - band[i], far_half[i], "cell {i}");
        }
    }
}

#[test]
fn near_above_far_is_both_at_their_mean_and_nan_reads_as_the_default() {
    assert_eq!(depth_band(0.8, 0.2, 0.5), (0.5, 0.5, 0.5));
    assert_eq!(depth_band(1.5, -2.0, 7.0), (0.5, 0.5, 1.0));
    assert_eq!(depth_band(f64::NAN, f64::NAN, f64::NAN), (0.0, 0.5, 0.5));
    let m = map_of(12, 50, 1, (0..50).map(|i| i as f32).collect());
    assert_eq!(depth_s0(&m, 0.8, 0.2, 0.3), depth_s0(&m, 0.5, 0.5, 0.3));
    assert_eq!(
        depth_s0(&m, f64::NAN, f64::NAN, f64::NAN),
        depth_s0(&m, 0.0, 0.5, 0.5)
    );
}

#[test]
fn the_ramp_has_a_minimum_and_the_full_band_is_one_everywhere() {
    assert_eq!(depth_ramp(0.0), DEPTH_RAMP_MIN);
    assert_eq!(depth_ramp(0.04), DEPTH_RAMP_MIN);
    assert_eq!(depth_ramp(0.2), 0.05);
    assert_eq!(depth_ramp(1.0), 0.25);
    // A one-cell difference of rank is not a step even at softness 0: the smoothstep's slope is at most 1.5 / q = 75, and
    // a cell is 0.001 of z here, so a neighbour differs by at most 0.075 (a step would be 1).
    let m = map_of(13, 1001, 1, (0..1001).map(|i| (1000 - i) as f32).collect());
    let band = depth_s0(&m, 0.3, 0.6, 0.0);
    let biggest = band
        .windows(2)
        .map(|p| p[0].abs_diff(p[1]))
        .max()
        .expect("pairs");
    assert!(f64::from(biggest) / 65535.0 <= 0.0751, "step {biggest}");
    // near 0, far 1: S0 = 1 everywhere, at any softness.
    for softness in [0.0, 0.5, 1.0] {
        assert!(depth_s0(&m, 0.0, 1.0, softness).iter().all(|&c| c == 65535));
    }
    // The ramps lie outside the band: it rises over [nu - q, nu] and falls over [phi, phi + q], one half in the middle.
    let q = 0.1;
    assert!((depth_cut(0.75, 0.2, 0.7, q) - 0.5).abs() < 1e-12);
    assert!((depth_cut(0.15, 0.2, 0.7, q) - 0.5).abs() < 1e-12);
    assert_eq!(depth_cut(0.7, 0.2, 0.7, q), 1.0);
    assert_eq!(depth_cut(0.8, 0.2, 0.7, q), 0.0);
    assert_eq!(depth_cut(0.4, 0.2, 0.7, q), 1.0);
}

#[test]
fn the_plane_is_the_maps_cells_cut_once_and_the_cache_keeps_the_band_and_nothing_else() {
    let m = map_of(
        9_100,
        8,
        8,
        (0..64).map(|i| ((i * 37) % 64) as f32).collect(),
    );
    let source = |near: f64, far: f64, softness: f64| DepthSource {
        map: Arc::clone(&m),
        near,
        far,
        softness,
    };
    let base = depth_plane(&source(0.0, 0.5, 0.5));
    assert_eq!((base.width, base.height), (8, 8));
    assert_eq!(base.data, depth_s0(&m, 0.0, 0.5, 0.5));
    assert_eq!(base.key, source(0.0, 0.5, 0.5).key());
    // The same band is the same plane, Arc for Arc (Feather, Grow and Refine edges are not in the key).
    assert!(Arc::ptr_eq(&base, &depth_plane(&source(0.0, 0.5, 0.5))));
    // Each of near, far and softness is a new plane with a new key.
    for other in [
        source(0.1, 0.5, 0.5),
        source(0.0, 0.6, 0.5),
        source(0.0, 0.5, 0.6),
    ] {
        let p = depth_plane(&other);
        assert!(!Arc::ptr_eq(&base, &p));
        assert_ne!(p.key, base.key);
    }
    // The key is of the band after the clamping and the swap: near above far is its mean.
    assert_eq!(source(0.8, 0.2, 0.5).key(), source(0.5, 0.5, 0.5).key());
    assert_eq!(source(-1.0, 0.5, 0.5).key(), source(0.0, 0.5, 0.5).key());
    // A new map (another photo, other parameters) is a miss with the same numbers, and the same codes.
    let again = map_of(9_101, 8, 8, m.d.clone());
    let p = depth_plane(&DepthSource {
        map: again,
        near: 0.0,
        far: 0.5,
        softness: 0.5,
    });
    assert!(!Arc::ptr_eq(&base, &p));
    assert_ne!(p.key, base.key);
    assert_eq!(p.data, base.data);
    // The raster source of a row gives the same plane at any aspect.
    let row = RasterSource::Depth(Arc::new(source(0.0, 0.5, 0.5)));
    assert!(Arc::ptr_eq(&row.plane([1.0, 0.6]), &base));
    assert_eq!(row.key(), base.key);
}

#[test]
fn a_depth_plane_does_not_share_a_key_with_the_same_numbers_in_another_domain() {
    // The tag in the key keeps a Depth plane from being read for a map whose key happens to be equal elsewhere.
    let m = map_of(1, 2, 2, vec![0.0, 1.0, 2.0, 3.0]);
    let s = DepthSource {
        map: m,
        near: 0.0,
        far: 0.5,
        softness: 0.5,
    };
    assert_ne!(s.key(), 1);
}

#[test]
fn pick_depth_on_a_ramp_map_gives_the_known_band() {
    // 100 columns, d falls to the right: the left is near. Cell i has r = (99 - i) / 99.
    let m = map_of(20, 100, 1, (0..100).map(|i| (100 - i) as f32).collect());
    // The middle of the photo: z = 0.5 gives 0.4 to 0.6.
    let (near, far) = pick_depth(&m, 0.5, 0.5);
    assert_eq!((near, far), (0.4, 0.6));
    // The left end: z = 0, the pair is moved to start at 0.
    assert_eq!(pick_depth(&m, 0.0, 0.5), (0.0, 0.2));
    assert_eq!(pick_depth(&m, 0.001, 0.5), (0.0, 0.2));
    // The right end: z = 1, the pair is moved to end at 1.
    assert_eq!(pick_depth(&m, 1.0, 0.5), (0.8, 1.0));
    // A quarter across: x = 0.25 is at u = 24.5, between cells 24 and 25, r = (99 - 24.5) / 99.
    let z: f64 = 1.0 - (99.0 - 24.5) / 99.0;
    let (n, f) = pick_depth(&m, 0.25, 0.5);
    assert_eq!(n, ((z - 0.1) * 100.0).round() / 100.0);
    assert_eq!(f, ((z + 0.1) * 100.0).round() / 100.0);
    // A map with all values equal gives 0.4 to 0.6.
    let flat = map_of(21, 5, 5, vec![2.0; 25]);
    assert_eq!(pick_depth(&flat, 0.3, 0.7), (0.4, 0.6));
}

#[test]
fn pick_band_moves_the_pair_as_a_whole_and_rounds_to_two_decimals() {
    assert!((DEPTH_PICK_WIDTH - 0.2).abs() < 1e-15);
    assert_eq!(pick_band(0.5), (0.4, 0.6));
    assert_eq!(pick_band(0.05), (0.0, 0.2));
    assert_eq!(pick_band(0.0), (0.0, 0.2));
    assert_eq!(pick_band(0.95), (0.8, 1.0));
    assert_eq!(pick_band(1.0), (0.8, 1.0));
    // Rounding: z - 0.1 = 0.2345 rounds to 0.23, z + 0.1 = 0.4345 to 0.43.
    let (n, f) = pick_band(0.3345);
    assert_eq!((n, f), (0.23, 0.43));
    // NaN reads as the middle.
    assert_eq!(pick_band(f64::NAN), (0.4, 0.6));
    // The pair is always inside [0, 1] and 0.2 wide (to the rounding).
    for i in 0..=1000 {
        let (n, f) = pick_band(f64::from(i) / 1000.0);
        assert!((0.0..=1.0).contains(&n) && (0.0..=1.0).contains(&f));
        assert!((f - n - 0.2).abs() <= 0.0101, "z {i}");
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
    // A point that is not a number reads as the middle; a map of one cell answers one half.
    let one = map_of(34, 1, 1, vec![1.0]);
    assert_eq!(one.focus_z(f64::NAN, 0.3, 0.1, 0.1), 0.5);
}

#[test]
fn the_show_depth_plane_is_white_where_the_map_is_nearest() {
    let m = map_of(40, 4, 1, vec![0.1, 0.9, 0.5, 0.3]);
    let plane = m.show_plane(77);
    assert_eq!((plane.width, plane.height, plane.key), (4, 1, 77));
    assert_eq!(plane.data[1], 65535, "the largest d is the nearest: white");
    assert_eq!(plane.data[0], 0, "the smallest d is the farthest: black");
    assert!(plane.data[2] > plane.data[3]);
    // A flat map is mid grey.
    let flat = map_of(41, 3, 3, vec![1.0; 9]);
    assert!(flat.show_plane(1).data.iter().all(|&c| c == 32768));
}
