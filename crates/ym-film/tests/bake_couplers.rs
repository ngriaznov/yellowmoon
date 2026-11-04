//! The baker takes the coupler matrix from the chain it bakes (F-04e), not from a file of its own.
mod bake_common;
use bake_common::*;
use ym_film::Output;
use ym_film::bake::{BakeSettings, DRAFT_SIZE};

#[test]
fn a_baked_chain_and_the_direct_chain_use_the_same_coupler_matrix() {
    let chain = chain(EIGHT[0], Output::DisplayP3);
    let m = chain.couplers();
    assert!(
        m.iter().flatten().any(|v| *v != 0.0),
        "the matrix is the coupler matrix, not zeros"
    );
    let mut bs = BakeSettings::of(&chain);
    bs.t3a_size = DRAFT_SIZE;
    bs.t3b_size = DRAFT_SIZE;
    let baked = baker(2).bake(&chain, &bs).expect("bake");
    let amount = chain.params().coupler_amount;
    assert_eq!(baked.coupler_amount(), amount);
    // The inhibitor is `amount * M * d` with the chain's own matrix, to the bit.
    for d in [[0.3, 0.9, 1.7], [1.1, 0.2, 0.6], [2.0, 2.2, 1.9]] {
        let q = baked.coupler_inhibitor(d);
        for i in 0..3 {
            let inh: f64 = (0..3).map(|j| m[i][j] * d[j]).sum();
            assert_eq!(q[i], amount * inh, "row {i}");
        }
    }
    // Stage 5 on the baked curves agrees with the direct chain's stage 5.
    let mut worst = 0.0_f64;
    for log_e in [[-1.0, -0.8, -1.2], [0.0, 0.1, -0.1], [0.9, 1.1, 0.7]] {
        let direct = chain.develop_stage().couple(log_e, amount);
        let via = baked.film_density(log_e);
        for i in 0..3 {
            worst = worst.max((direct[i] - via[i]).abs());
        }
    }
    assert!(worst < 2e-3, "stage 5 baked against direct: {worst:.3e}");
}
