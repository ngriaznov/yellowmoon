//! The planner cache of the diffraction stage under contention.
//!
//! The geometry of a plan (the PSF kernel, `sigma_G`, the radii) is a pure function of the effective
//! pixel pitch and the f-number, and is cached by those two numbers. Planning takes a few
//! milliseconds and happens outside the lock, so threads that ask for the same new key at the same
//! time all plan it. Whatever the order they get back in, they must end up with the one cached
//! geometry: two plans of the same pitch and f-number share their kernel. The test sees that
//! through the public API (the address of the kernel of each plan), in a process of its own, so
//! that no other test fills the cache with other keys meanwhile.

use std::sync::{Arc, Barrier};

use ym_gpu::cpu::NoiseDn;
use ym_gpu::cpu::sharpen::diffraction::{DiffractionParams, DiffractionPlan, SPEC_LUMA};

/// Threads that ask for the same key together.
const THREADS: usize = 16;

/// Distinct keys, one per round: fewer than the cache holds, so no round evicts another's entry.
const ROUNDS: usize = 6;

#[test]
fn threads_that_plan_the_same_geometry_at_once_share_one_kernel() {
    for round in 0..ROUNDS {
        // A pitch no other test uses, a new one every round.
        let pitch_um = 4.1 + 0.07 * round as f64;
        let barrier = Arc::new(Barrier::new(THREADS));
        let handles: Vec<_> = (0..THREADS)
            .map(|_| {
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    let plan = DiffractionPlan::new(
                        &DiffractionParams::new(pitch_um, 11.0),
                        [2.0, 1.0, 1.5],
                        &NoiseDn::new(1.0, 5.0),
                        SPEC_LUMA,
                    )
                    .expect("an active plan");
                    // The address of the kernel identifies the geometry the plan holds. The plan
                    // is kept so that the address stays valid until it is compared.
                    (plan.kernel().as_ptr() as usize, plan)
                })
            })
            .collect();
        let results: Vec<_> = handles
            .into_iter()
            .map(|h| h.join().expect("a planning thread"))
            .collect();
        let first = results[0].0;
        for (i, (address, plan)) in results.iter().enumerate() {
            assert_eq!(
                *address, first,
                "round {round}: the plan of thread {i} holds its own copy of the geometry"
            );
            assert_eq!(plan.kernel(), results[0].1.kernel());
        }
        // A plan made afterwards hits the same entry.
        let later = DiffractionPlan::new(
            &DiffractionParams::new(pitch_um, 11.0),
            [2.0, 1.0, 1.5],
            &NoiseDn::new(1.0, 5.0),
            SPEC_LUMA,
        )
        .expect("an active plan");
        assert_eq!(later.kernel().as_ptr() as usize, first, "round {round}");
    }
}
