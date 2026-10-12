//! Hostile inputs for the whole decode path (M0-05 review): damaged and deliberately malformed
//! synthetic files of every decodable variant must end in a typed error or in a consistent image,
//! never in a panic, a hang or an allocation far beyond what the declared image needs.
//!
//! The container-level mutations patch real TIFF tags (image size, tile size, strip counts,
//! compression, bit depth), because random byte flips rarely hit those and they are what decide
//! how much a decoder allocates. A counting global allocator measures the peak number of live
//! bytes of every attempt. Everything is seeded and deterministic.
//!
//! The bound is 1.5 GiB: a valid 120 MP Lossless M or S image needs about 1.1 GiB of planes
//! (docs/04: images are capped at 120 MP), nothing a damaged small file can ask for should exceed
//! that. This test is the only one in its binary, so nothing else allocates in parallel.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ym_raw::decode::{DecodeOptions, decode_bytes};
use ym_raw::error::RawError;
use ym_raw::layout::ChromaSubsampling;
use ym_raw::testgen::{
    Arw1, ArwBuilder, CrawEncoder, LosslessL, LosslessMs, Packed12, SplitMix64, gradient_mosaic,
    gradient_rgb,
};

/// Counts live heap bytes and remembers the peak.
struct Counting;

static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

// SAFETY: forwards every call to the system allocator and only adds bookkeeping.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = LIVE
                .fetch_add(layout.size(), Ordering::Relaxed)
                .wrapping_add(layout.size());
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc_zeroed(layout) };
        if !ptr.is_null() {
            let live = LIVE
                .fetch_add(layout.size(), Ordering::Relaxed)
                .wrapping_add(layout.size());
            PEAK.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) };
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let new = unsafe { System.realloc(ptr, layout, new_size) };
        if !new.is_null() {
            if new_size >= layout.size() {
                let grown = new_size - layout.size();
                let live = LIVE.fetch_add(grown, Ordering::Relaxed).wrapping_add(grown);
                PEAK.fetch_max(live, Ordering::Relaxed);
            } else {
                LIVE.fetch_sub(layout.size() - new_size, Ordering::Relaxed);
            }
        }
        new
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

static OK: AtomicUsize = AtomicUsize::new(0);
static CORRUPT: AtomicUsize = AtomicUsize::new(0);
static OTHER: AtomicUsize = AtomicUsize::new(0);
static WORST_PEAK: AtomicUsize = AtomicUsize::new(0);

const PEAK_LIMIT: usize = 1536 << 20;
const TIME_LIMIT: Duration = Duration::from_secs(20);

/// The small valid files, one per decodable variant.
fn seeds() -> Vec<(&'static str, Vec<u8>)> {
    let m = gradient_mosaic(96, 64, 512, 16000, 1);
    let m12 = gradient_mosaic(96, 64, 128, 4000, 2);
    let encoder = CrawEncoder::new();
    let mc = gradient_mosaic(96, 64, 512, encoder.max_value(), 3);
    let ms = |sub, tile: u32| {
        let rgb = gradient_rgb(70, 50, 1100, 9000, 4);
        ArwBuilder::new("ILCE-7RM5", 70, 50)
            .lossless_ms(LosslessMs::new(sub, rgb, 70, 50, [2.0, 1.0, 1.6]).tile(tile, tile))
            .build(&[])
    };
    vec![
        ("unc14", ArwBuilder::new("ILCE-7RM4", 96, 64).build(&m)),
        (
            "unc12",
            ArwBuilder::new("ILCE-7M2", 96, 64)
                .uncompressed(12)
                .build(&m12),
        ),
        (
            "packed12",
            ArwBuilder::new("DSLR-A900", 96, 64)
                .payload(Packed12)
                .build(&m12),
        ),
        (
            "arw1",
            ArwBuilder::new("DSLR-A300", 96, 64)
                .payload(Arw1)
                .black([0; 4])
                .white([3840; 3])
                .build(&m12),
        ),
        (
            "craw",
            ArwBuilder::new("ILCE-7M3", 96, 64).craw(encoder).build(&mc),
        ),
        (
            "lossless_l",
            ArwBuilder::new("ILCE-1", 96, 64)
                .payload(LosslessL::new().tile(32, 32))
                .build(&m),
        ),
        (
            "lossless_l_restarts",
            ArwBuilder::new("ILCE-1", 96, 64)
                .payload(LosslessL::new().tile(64, 64).restart_rows(2).predictor(4))
                .build(&m),
        ),
        ("ms420", ms(ChromaSubsampling::Yuv420, 32)),
        ("ms422", ms(ChromaSubsampling::Yuv422, 32)),
        ("ms444", ms(ChromaSubsampling::Yuv444, 64)),
    ]
}

/// `(entry offset, field type, count)` of every directory entry with `tag` in a little-endian
/// TIFF, following next-IFD links, SubIFDs and the EXIF pointer.
fn entries(file: &[u8], tag: u16) -> Vec<(usize, u16, u32)> {
    fn u16at(f: &[u8], at: usize) -> Option<u16> {
        Some(u16::from_le_bytes(f.get(at..at + 2)?.try_into().ok()?))
    }
    fn u32at(f: &[u8], at: usize) -> Option<u32> {
        Some(u32::from_le_bytes(f.get(at..at + 4)?.try_into().ok()?))
    }
    let mut found = Vec::new();
    let mut seen = Vec::new();
    let mut pending = vec![u32at(file, 4).unwrap_or(0) as usize];
    while let Some(at) = pending.pop() {
        if at < 8 || seen.contains(&at) {
            continue;
        }
        seen.push(at);
        let Some(count) = u16at(file, at) else {
            continue;
        };
        for i in 0..usize::from(count) {
            let e = at + 2 + 12 * i;
            let (Some(t), Some(ty), Some(n)) =
                (u16at(file, e), u16at(file, e + 2), u32at(file, e + 4))
            else {
                break;
            };
            if t == tag {
                found.push((e, ty, n));
            }
            if (t == 0x014A || t == 0x8769) && n == 1 {
                pending.push(u32at(file, e + 8).unwrap_or(0) as usize);
            } else if t == 0x014A && n < 16 {
                let table = u32at(file, e + 8).unwrap_or(0) as usize;
                for k in 0..n as usize {
                    pending.push(u32at(file, table + 4 * k).unwrap_or(0) as usize);
                }
            }
        }
        pending.push(u32at(file, at + 2 + 12 * usize::from(count)).unwrap_or(0) as usize);
    }
    found
}

/// Writes `value` into the first value slot of every entry with `tag` (SHORT or LONG).
fn patch(file: &mut [u8], tag: u16, value: u32) {
    for (e, ty, _) in entries(file, tag) {
        match ty {
            3 => file[e + 8..e + 10].copy_from_slice(&(value as u16).to_le_bytes()),
            4 => file[e + 8..e + 12].copy_from_slice(&value.to_le_bytes()),
            _ => {}
        }
    }
}

/// Decodes `file`, checks the outcome and returns the peak of live heap bytes.
fn attempt(name: &str, what: &str, file: &[u8]) {
    let before = LIVE.load(Ordering::Relaxed);
    PEAK.store(before, Ordering::Relaxed);
    let started = Instant::now();
    let result = std::panic::catch_unwind(|| decode_bytes(file, &DecodeOptions::default()));
    let elapsed = started.elapsed();
    let peak = PEAK.load(Ordering::Relaxed).saturating_sub(before);
    let result = match result {
        Ok(result) => result,
        Err(_) => panic!("{name} / {what}: the decoder panicked"),
    };
    assert!(
        peak <= PEAK_LIMIT,
        "{name} / {what}: peak heap {} MiB",
        peak >> 20
    );
    assert!(
        elapsed <= TIME_LIMIT,
        "{name} / {what}: took {elapsed:?} ({peak} bytes)"
    );
    WORST_PEAK.fetch_max(peak, Ordering::Relaxed);
    match &result {
        Ok(_) => OK.fetch_add(1, Ordering::Relaxed),
        Err(RawError::Corrupt { .. }) => CORRUPT.fetch_add(1, Ordering::Relaxed),
        Err(_) => OTHER.fetch_add(1, Ordering::Relaxed),
    };
    match result {
        Ok(image) => {
            assert_eq!(
                image.data.data.len(),
                image.data.width * image.data.height,
                "{name} / {what}"
            );
            assert_eq!(
                image.pixel_width(),
                image.meta.width as usize,
                "{name} / {what}"
            );
            assert_eq!(
                image.pixel_height(),
                image.meta.height as usize,
                "{name} / {what}"
            );
            if let Some(q) = &image.qstep {
                assert_eq!(q.data.len(), image.data.data.len(), "{name} / {what}");
                assert!(
                    q.data.iter().all(|v| v.to_f32().is_finite()),
                    "{name} / {what}"
                );
            }
        }
        Err(RawError::Io(e)) => {
            panic!("{name} / {what}: an in-memory decode cannot fail with I/O: {e}")
        }
        Err(_) => {}
    }
}

#[test]
fn damaged_and_malformed_files_never_panic_hang_or_balloon() {
    let seeds = seeds();
    let mut rng = SplitMix64(0x5EED);
    for (name, file) in &seeds {
        // The pristine file decodes.
        decode_bytes(file, &DecodeOptions::default()).unwrap_or_else(|e| panic!("{name}: {e}"));

        // 1. Random byte damage anywhere, and in the container area in particular.
        for round in 0..120 {
            let mut f = file.clone();
            let limit = if round % 2 == 0 {
                f.len().min(2048)
            } else {
                f.len()
            };
            for _ in 0..=rng.below(6) {
                let at = rng.below(limit as u64) as usize;
                f[at] = rng.next_u64() as u8;
            }
            attempt(name, &format!("random damage {round}"), &f);
        }

        // 2. Semantic damage to the tags that size buffers.
        let extremes = [
            0u32,
            1,
            2,
            3,
            31,
            33,
            255,
            256,
            4096,
            16384,
            65535,
            65536,
            0x7FFF_FFFF,
            u32::MAX,
        ];
        for tag in [
            0x0100u16, // ImageWidth
            0x0101,    // ImageLength
            0x0102,    // BitsPerSample
            0x0103,    // Compression
            0x0115,    // SamplesPerPixel
            0x0116,    // RowsPerStrip
            0x0117,    // StripByteCounts
            0x0111,    // StripOffsets
            0x0142,    // TileWidth
            0x0143,    // TileLength
            0x0144,    // TileOffsets
            0x0145,    // TileByteCounts
            0x0212,    // YCbCrSubSampling
            0x7010,    // the cRAW tone curve
        ] {
            for value in extremes {
                let mut f = file.clone();
                patch(&mut f, tag, value);
                attempt(name, &format!("tag {tag:#06x} = {value}"), &f);
            }
        }
        // Two sizes at once: a large image with a small file behind it.
        for (w, h) in [
            (65535u32, 1833u32),
            (10000, 12000),
            (12000, 10000),
            (65535, 65535),
        ] {
            let mut f = file.clone();
            patch(&mut f, 0x0100, w);
            patch(&mut f, 0x0101, h);
            attempt(name, &format!("size {w} x {h}"), &f);
        }

        // 3. Truncation at many points.
        for keep in (0..file.len()).step_by(file.len() / 40 + 1) {
            attempt(name, &format!("cut to {keep}"), &file[..keep]);
        }
    }

    eprintln!(
        "hostile inputs: {} decoded, {} corrupt, {} other errors, worst peak heap {} MiB",
        OK.load(Ordering::Relaxed),
        CORRUPT.load(Ordering::Relaxed),
        OTHER.load(Ordering::Relaxed),
        WORST_PEAK.load(Ordering::Relaxed) >> 20
    );
}
