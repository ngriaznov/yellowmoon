//! SCRATCH (never committed): constants overridable by env YM_RC="i=v,i=v" or `set`.
use std::sync::Once;
use std::sync::atomic::{AtomicU64, Ordering};
const NAN: u64 = 0x7ff8_0000_0000_0000;
static T: [AtomicU64; 128] = [const { AtomicU64::new(NAN) }; 128];
static INIT: Once = Once::new();
fn init() {
    INIT.call_once(|| {
        if let Ok(s) = std::env::var("YM_RC") {
            for p in s.split(',') {
                if let Some((a, b)) = p.split_once('=') {
                    let i: usize = a.trim().parse().unwrap();
                    let v: f64 = b.trim().parse().unwrap();
                    T[i].store(v.to_bits(), Ordering::Relaxed);
                }
            }
        }
    });
}
pub fn set(i: usize, v: f64) {
    init();
    T[i].store(v.to_bits(), Ordering::Relaxed);
}
pub fn reset() {
    init();
    for t in &T {
        t.store(NAN, Ordering::Relaxed);
    }
}
#[inline]
pub fn rc(i: usize, d: f64) -> f64 {
    init();
    let v = f64::from_bits(T[i].load(Ordering::Relaxed));
    if v.is_nan() { d } else { v }
}
