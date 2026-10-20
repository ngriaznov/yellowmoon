//! Shared helpers for the integration tests: deterministic synthetic scenes, a metadata record, a
//! small TIFF structure walker, SHA-256 and Python helpers.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};
use ym_export::exif::{CaptureTime, ExifInfo, Gps};
use ym_export::request::Metadata;
use ym_export::xmp::{XmpInfo, YmInfo};

/// xorshift64*: deterministic noise with no dependencies.
pub struct Rng(pub u64);

impl Rng {
    pub fn next_u32(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }

    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f32 {
        self.next_u32() as f32 / 4_294_967_296.0
    }
}

/// A display-linear photo-like scene in [0, 1]: smooth colour gradients, a few hard-edged patches
/// and mild noise. Deterministic.
pub fn display_scene(w: u32, h: u32) -> Vec<f32> {
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15 ^ u64::from(w) << 20 ^ u64::from(h));
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let mut p = [
                0.05 + 0.9 * fx * fx,
                0.05 + 0.9 * fy,
                0.1 + 0.8 * (1.0 - fx) * (0.3 + 0.7 * fy),
            ];
            // A saturated patch and a dark patch with hard edges.
            if (0.6..0.8).contains(&fx) && (0.2..0.45).contains(&fy) {
                p = [0.9, 0.05, 0.02];
            }
            if (0.1..0.25).contains(&fx) && (0.55..0.8).contains(&fy) {
                p = [0.004, 0.006, 0.01];
            }
            for c in p {
                v.push((c + (rng.unit() - 0.5) * 0.01).clamp(0.0, 1.0));
            }
        }
    }
    v
}

/// A scene-linear AWG4 scene: grey-card level values, highlights to ~6, a few out-of-gamut
/// negative samples. Deterministic.
pub fn awg4_scene(w: u32, h: u32) -> Vec<f32> {
    let mut rng = Rng(0xD1B5_4A32_D192_ED03 ^ u64::from(w) << 24 ^ u64::from(h));
    let mut v = Vec::with_capacity((w * h * 3) as usize);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 / w as f32, y as f32 / h as f32);
            let base = 0.18 * (0.2 + 4.0 * fx * fx);
            let mut p = [base * (1.0 + 0.5 * fy), base, base * (1.5 - fy)];
            if (0.7..0.8).contains(&fx) && (0.1..0.3).contains(&fy) {
                p = [6.0, 4.5, 3.0]; // a highlight
            }
            if (0.2..0.25).contains(&fx) && (0.6..0.7).contains(&fy) {
                p = [0.1, -0.03, 0.2]; // out of gamut
            }
            for c in p {
                v.push(c * (1.0 + (rng.unit() - 0.5) * 0.02));
            }
        }
    }
    v
}

pub fn capture_time() -> CaptureTime {
    CaptureTime {
        year: 2026,
        month: 10,
        day: 4,
        hour: 14,
        minute: 22,
        second: 31,
        millisecond: Some(120),
        utc_offset_minutes: Some(120),
    }
}

/// A fully populated metadata record.
pub fn metadata() -> Metadata {
    Metadata {
        exif: ExifInfo {
            make: Some("SONY".into()),
            model: Some("ILCE-1".into()),
            lens_make: Some("Sony".into()),
            lens_model: Some("FE 24-70mm F2.8 GM II".into()),
            exposure_time_s: Some(1.0 / 250.0),
            f_number: Some(2.8),
            iso: Some(400),
            focal_length_mm: Some(35.0),
            focal_length_35mm: Some(35),
            exposure_bias_ev: Some(-0.7),
            capture_time: Some(capture_time()),
            gps: Some(Gps {
                latitude_deg: 38.7223,
                longitude_deg: -9.1393,
                altitude_m: Some(12.5),
            }),
        },
        xmp: XmpInfo {
            app_version: "0.0.1".into(),
            capture_time: Some(capture_time()),
            ym: YmInfo {
                pipeline_version: Some("1.0.0".into()),
                profile: Some("sony-ilce-7rm4".into()),
                profile_version: Some("1.0.0".into()),
                profile_match: Some("default".into()),
                look: Some("builtin:neutral".into()),
                look_sha256: Some("0".repeat(64)),
                look_intensity_pct: Some(100),
                display_transform: Some("aces2-sdr100-p3lim".into()),
                models: Some("jdd 1.2.0 (sha256:abc)".into()),
                // Left out, so the golden files keep their bytes; `src/xmp.rs` tests them.
                auto: None,
                apple_auto: None,
                settings: Some(serde_json::json!({"exposure": "auto", "contrast": 0})),
            },
        },
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---- TIFF structure walker -----------------------------------------------------------------

/// One IFD entry with its raw value bytes.
#[derive(Debug, Clone)]
pub struct IfdEntry {
    pub tag: u16,
    pub ty: u16,
    pub count: u32,
    pub data: Vec<u8>,
    /// Where the value lives when it does not fit in the entry.
    pub offset: Option<u32>,
}

impl IfdEntry {
    pub fn u32s(&self) -> Vec<u32> {
        match self.ty {
            3 => self
                .data
                .chunks_exact(2)
                .map(|b| u32::from(u16::from_le_bytes([b[0], b[1]])))
                .collect(),
            4 | 13 => self
                .data
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect(),
            1 | 7 => self.data.iter().map(|&b| u32::from(b)).collect(),
            other => panic!("tag {} type {other} isn't an integer", self.tag),
        }
    }

    pub fn f64s(&self) -> Vec<f64> {
        match self.ty {
            5 => self
                .data
                .chunks_exact(8)
                .map(|b| {
                    f64::from(u32::from_le_bytes(b[0..4].try_into().unwrap()))
                        / f64::from(u32::from_le_bytes(b[4..8].try_into().unwrap()))
                })
                .collect(),
            10 => self
                .data
                .chunks_exact(8)
                .map(|b| {
                    f64::from(i32::from_le_bytes(b[0..4].try_into().unwrap()))
                        / f64::from(i32::from_le_bytes(b[4..8].try_into().unwrap()))
                })
                .collect(),
            11 => self
                .data
                .chunks_exact(4)
                .map(|b| f64::from(f32::from_le_bytes(b.try_into().unwrap())))
                .collect(),
            _ => self.u32s().into_iter().map(f64::from).collect(),
        }
    }

    pub fn ascii(&self) -> String {
        assert_eq!(self.ty, 2, "tag {}", self.tag);
        String::from_utf8(self.data.iter().copied().take_while(|&b| b != 0).collect()).unwrap()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Ifd {
    pub entries: BTreeMap<u16, IfdEntry>,
    pub next: u32,
}

impl Ifd {
    pub fn get(&self, tag: u16) -> &IfdEntry {
        self.entries.get(&tag).unwrap_or_else(|| {
            panic!(
                "tag {tag} missing; have {:?}",
                self.entries.keys().collect::<Vec<_>>()
            )
        })
    }

    pub fn has(&self, tag: u16) -> bool {
        self.entries.contains_key(&tag)
    }

    pub fn one(&self, tag: u16) -> u32 {
        self.get(tag).u32s()[0]
    }
}

pub fn type_size(ty: u16) -> usize {
    match ty {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 => 8,
        other => panic!("unknown type {other}"),
    }
}

/// Reads the IFD at `offset` of a little-endian TIFF.
pub fn read_ifd(file: &[u8], offset: usize) -> Ifd {
    let n = u16::from_le_bytes([file[offset], file[offset + 1]]) as usize;
    let mut ifd = Ifd::default();
    let mut previous = None;
    for i in 0..n {
        let e = &file[offset + 2 + 12 * i..offset + 14 + 12 * i];
        let tag = u16::from_le_bytes([e[0], e[1]]);
        let ty = u16::from_le_bytes([e[2], e[3]]);
        let count = u32::from_le_bytes(e[4..8].try_into().unwrap());
        let len = type_size(ty) * count as usize;
        let (data, data_offset) = if len <= 4 {
            (e[8..8 + len].to_vec(), None)
        } else {
            let off = u32::from_le_bytes(e[8..12].try_into().unwrap()) as usize;
            assert!(off + len <= file.len(), "tag {tag}: data past the end");
            (file[off..off + len].to_vec(), Some(off as u32))
        };
        if let Some(p) = previous {
            assert!(tag > p, "tags must be ascending: {p} then {tag}");
        }
        previous = Some(tag);
        ifd.entries.insert(
            tag,
            IfdEntry {
                tag,
                ty,
                count,
                data,
                offset: data_offset,
            },
        );
    }
    ifd.next = u32::from_le_bytes(
        file[offset + 2 + 12 * n..offset + 6 + 12 * n]
            .try_into()
            .unwrap(),
    );
    ifd
}

pub fn ifd0(file: &[u8]) -> Ifd {
    assert_eq!(&file[0..4], b"II*\0");
    read_ifd(
        file,
        u32::from_le_bytes(file[4..8].try_into().unwrap()) as usize,
    )
}

// ---- Python helpers --------------------------------------------------------------------------

pub fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap()
}

pub fn tests_py(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/py")
        .join(name)
}

/// Whether `python3 -c "import <module>"` works.
pub fn python_has(module: &str) -> bool {
    Command::new("python3")
        .args(["-I", "-c", &format!("import {module}")])
        .output()
        .is_ok_and(|o| o.status.success())
}

pub fn tool_exists(name: &str) -> bool {
    Command::new(name)
        .arg("-ver")
        .output()
        .is_ok_and(|o| o.status.success())
}

/// Runs `python3 -I script args...` and returns stdout; panics with stderr on failure.
pub fn run_python(script: &Path, args: &[&str]) -> String {
    let out = Command::new("python3")
        .arg("-I")
        .arg(script)
        .args(args)
        .output()
        .expect("python3 should start");
    assert!(
        out.status.success(),
        "python failed ({}):\n{}\n{}",
        script.display(),
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

/// Skips (returns false) when the Python module is missing, unless `YM_REQUIRE_EXTERNAL_TOOLS` is
/// set, in which case it fails. Prints why it skipped.
pub fn require_python_modules(modules: &[&str]) -> bool {
    let missing: Vec<&str> = modules.iter().copied().filter(|m| !python_has(m)).collect();
    if missing.is_empty() {
        return true;
    }
    assert!(
        std::env::var_os("YM_REQUIRE_EXTERNAL_TOOLS").is_none(),
        "missing Python modules: {missing:?}"
    );
    eprintln!("SKIPPED: python modules missing: {missing:?}");
    false
}

/// Reads IFD0 and every directory reachable from it through SubIFDs (330), the Exif IFD (34665) and
/// the GPS IFD (34853): `(name, ifd)`.
pub fn all_ifds(file: &[u8]) -> Vec<(String, Ifd)> {
    let first = u32::from_le_bytes(file[4..8].try_into().unwrap()) as usize;
    let mut out = Vec::new();
    let mut next = Some(first);
    let mut n = 0;
    while let Some(off) = next.take() {
        let ifd = read_ifd(file, off);
        if ifd.next != 0 {
            next = Some(ifd.next as usize);
        }
        for (tag, name) in [(330_u16, "sub"), (34665, "exif"), (34853, "gps")] {
            if let Some(e) = ifd.entries.get(&tag) {
                for (i, off) in e.u32s().into_iter().enumerate() {
                    out.push((format!("{name}{i}"), read_ifd(file, off as usize)));
                }
            }
        }
        out.push((format!("ifd{n}"), ifd));
        n += 1;
    }
    out
}

/// Asserts the TIFF alignment rule: every value that does not fit in its entry starts on an even offset.
pub fn assert_values_word_aligned(file: &[u8]) {
    for (name, ifd) in all_ifds(file) {
        for e in ifd.entries.values() {
            if let Some(off) = e.offset {
                assert_eq!(
                    off % 2,
                    0,
                    "{name} tag {} has its value at odd offset {off}",
                    e.tag
                );
            }
        }
    }
}
