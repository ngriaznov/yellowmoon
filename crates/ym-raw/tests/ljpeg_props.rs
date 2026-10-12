//! Property tests for the lossless JPEG decoder: random shapes of stream, produced by the
//! test-only encoder, decode to exactly the planes that went in.

use proptest::prelude::*;
use ym_raw::decode::ljpeg::{Options, decode};
use ym_raw::testgen::SplitMix64;
use ym_raw::testgen::ljpeg_enc::{EncodeParams, HuffmanChoice, encode};

fn sampling() -> impl Strategy<Value = Vec<(u8, u8)>> {
    prop_oneof![
        Just(vec![(1, 1)]),
        Just(vec![(1, 1); 2]),
        Just(vec![(1, 1); 3]),
        Just(vec![(1, 1); 4]),
        Just(vec![(2, 2), (1, 1), (1, 1)]),
        Just(vec![(2, 1), (1, 1), (1, 1)]),
        Just(vec![(3, 2), (1, 2), (2, 1)]),
    ]
}

fn params() -> impl Strategy<Value = (EncodeParams, u64, Vec<u8>)> {
    (
        sampling(),
        1u8..=7,
        2u8..=16,
        0u8..3,
        1usize..70,
        1usize..40,
        0usize..3,
        any::<bool>(),
        any::<bool>(),
        any::<u64>(),
    )
        .prop_map(
            |(
                sampling,
                predictor,
                precision,
                pt,
                width,
                height,
                restart_rows,
                skewed,
                quirk,
                seed,
            )| {
                let pt = pt.min(precision - 1);
                let mut p = EncodeParams {
                    precision,
                    predictor,
                    point_transform: pt,
                    restart_interval: 0,
                    sony_420_quirk: quirk,
                    huffman: if skewed {
                        HuffmanChoice::Skewed
                    } else {
                        HuffmanChoice::Optimal
                    },
                    width,
                    height,
                    sampling,
                };
                p.restart_interval = restart_rows * p.mcus_x();
                (p, seed, Vec::new())
            },
        )
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(300))]

    #[test]
    fn random_streams_round_trip((p, seed, _unused) in params()) {
        let mut rng = SplitMix64(seed);
        let max = (1u64 << p.precision) - 1;
        let planes: Vec<Vec<u16>> = (0..p.sampling.len())
            .map(|c| {
                let (w, h) = p.plane_size(c);
                // Smooth ramps with occasional jumps, in whole point-transform units.
                let mut level = rng.below(max + 1);
                (0..w * h)
                    .map(|_| {
                        if rng.below(11) == 0 {
                            level = rng.below(max + 1);
                        } else {
                            level = (level as i64 + rng.below(9) as i64 - 4).clamp(0, max as i64) as u64;
                        }
                        ((level >> p.point_transform) << p.point_transform) as u16
                    })
                    .collect()
            })
            .collect();
        let bytes = encode(&planes, &p);
        let decoded = decode(
            &bytes,
            &Options { sony_420_quirk: p.sony_420_quirk, ..Options::default() },
        ).expect("decodes");
        prop_assert_eq!(decoded.planes.len(), planes.len());
        for (got, want) in decoded.planes.iter().zip(&planes) {
            prop_assert_eq!(&got.data, want);
        }
    }

    /// Whatever the bytes, the decoder returns without panicking, and what it accepts is the
    /// size it promised.
    #[test]
    fn mutated_streams_never_panic((p, seed, _unused) in params(), flips in proptest::collection::vec((any::<u16>(), any::<u8>()), 1..6)) {
        let mut rng = SplitMix64(seed);
        let planes: Vec<Vec<u16>> = (0..p.sampling.len())
            .map(|c| {
                let (w, h) = p.plane_size(c);
                (0..w * h).map(|_| (rng.next_u64() & ((1 << p.precision) - 1) & !((1 << p.point_transform) - 1)) as u16).collect()
            })
            .collect();
        let mut bytes = encode(&planes, &p);
        for (at, value) in flips {
            let i = usize::from(at) % bytes.len();
            bytes[i] = value;
        }
        if let Ok(decoded) = decode(&bytes, &Options::default()) {
            for plane in &decoded.planes {
                prop_assert_eq!(plane.data.len(), plane.width * plane.height);
            }
        }
    }
}
