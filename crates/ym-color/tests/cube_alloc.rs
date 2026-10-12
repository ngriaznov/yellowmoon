//! M1-10: the `.cube` parser's memory use on hostile text is bounded by what the format needs, not by
//! how the text is split up. A counting global allocator measures the peak number of live bytes
//! during `parse`; the input itself is allocated before the measurement starts.
//!
//! This is its own test binary (a global allocator is per binary) and holds a single test so no other
//! test thread allocates while the counters are running.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use ym_color::cube::{CubeError, parse};

struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

fn add(bytes: usize) {
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
}

// SAFETY: every method forwards to the system allocator unchanged and only updates counters.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        add(layout.size());
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        add(new_size);
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Peak live bytes allocated while `parse` runs on `text` (the text itself is not counted).
fn peak_during_parse(text: &str) -> (usize, Result<(), CubeError>) {
    let baseline = LIVE.load(Ordering::Relaxed);
    PEAK.store(baseline, Ordering::Relaxed);
    let result = parse(text).map(|_| ());
    (
        PEAK.load(Ordering::Relaxed).saturating_sub(baseline),
        result,
    )
}

#[test]
fn hostile_text_does_not_amplify_into_large_allocations() {
    const MIB: usize = 1024 * 1024;
    // Each of these inputs is 16 MiB (built one at a time to keep the test's own memory small). A
    // parser that collects the words of a header line, upper-cases a keyword or buffers a title
    // would allocate several times that.
    type Build = fn() -> String;
    let cases: [(&str, Build); 8] = [
        ("many words after a keyword", || {
            "LUT_3D_SIZE 2 ".to_string() + &"1 ".repeat(8 * 1024 * 1024)
        }),
        ("many words after an unknown keyword", || {
            "FOO ".repeat(4 * 1024 * 1024)
        }),
        ("one giant keyword", || "A".repeat(16 * 1024 * 1024)),
        ("giant title", || {
            "TITLE ".to_string() + &"x".repeat(16 * 1024 * 1024)
        }),
        ("giant number", || {
            "LUT_3D_SIZE 2\n".to_string() + &"1".repeat(16 * 1024 * 1024)
        }),
        ("giant DOMAIN_MIN line", || {
            "DOMAIN_MIN ".to_string() + &"0 ".repeat(8 * 1024 * 1024)
        }),
        ("blank lines", || "\n".repeat(16 * 1024 * 1024)),
        ("comment lines", || "#\n".repeat(8 * 1024 * 1024)),
    ];
    for (name, build) in cases {
        let text = build();
        let (peak, result) = peak_during_parse(&text);
        assert!(result.is_err(), "{name}: hostile text must not parse");
        assert!(
            peak < MIB,
            "{name}: parsing {} MiB of text allocated {} KiB at its peak",
            text.len() / MIB,
            peak / 1024
        );
    }
}
