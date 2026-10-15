//! Helpers shared by the integration tests: synthetic `.cube` files, zips and package lists.
//!
//! Everything here is generated from our own formulas. No ARRI data is used anywhere; only
//! ARRI-like **file names** are, so the recognition rules have something to read.

#![allow(dead_code)]

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use ym_color::lut::Lut3d;
use ym_looks::library::Library;
use ym_looks::recognize::PackageDb;
use zip::CompressionMethod;
use zip::write::SimpleFileOptions;

/// A deterministic synthetic LUT: a gentle, seed-dependent colour change on a `size`-point cube.
pub fn cube_text(size: usize, seed: u32, title: &str) -> String {
    let s = f64::from(seed % 97) / 97.0;
    Lut3d::bake(size, [0.0; 3], [1.0; 3], |c| {
        [
            (c[0] * (0.90 + 0.1 * s) + 0.02 * s).clamp(0.0, 1.0),
            c[1].powf(0.9 + 0.2 * s),
            (c[2] * 0.95 + 0.05 * c[0] * s).clamp(0.0, 1.0),
        ]
    })
    .expect("bake")
    .to_cube_string(title)
}

/// Writes `entries` (path inside the zip, bytes) to a zip file.
pub fn write_zip(path: &Path, entries: &[(String, Vec<u8>)]) {
    let file = File::create(path).expect("create zip");
    let mut writer = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    for (name, bytes) in entries {
        writer.start_file(name.as_str(), options).expect("start");
        writer.write_all(bytes).expect("write");
    }
    writer.finish().expect("finish");
}

/// The five package kinds of `docs/08-looks.md`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    LookLogC4,
    DisplayLogC4,
    LookLogC3,
    LookLogC3Rec709,
    DisplayLogC3,
}

impl Kind {
    pub const ALL: [Kind; 5] = [
        Kind::LookLogC4,
        Kind::DisplayLogC4,
        Kind::LookLogC3,
        Kind::LookLogC3Rec709,
        Kind::DisplayLogC3,
    ];

    /// The package id in `data/looks/arri-packages.json`.
    pub fn package_id(self) -> &'static str {
        match self {
            Kind::LookLogC4 => "look-library-logc4",
            Kind::DisplayLogC4 => "display-luts-logc4",
            Kind::LookLogC3 => "look-library-logc3",
            Kind::LookLogC3Rec709 => "look-library-logc3-rec709",
            Kind::DisplayLogC3 => "display-luts-logc3",
        }
    }

    /// An ARRI-like name for the downloaded zip.
    pub fn zip_name(self) -> &'static str {
        match self {
            Kind::LookLogC4 => "ARRI_Look_Library_LogC4.zip",
            Kind::DisplayLogC4 => "ARRI_Display_LUTs_LogC4.zip",
            Kind::LookLogC3 => "ARRI_Look_Library_LogC3.zip",
            Kind::LookLogC3Rec709 => "ARRI_Look_Library_LogC3_to_Rec709.zip",
            Kind::DisplayLogC3 => "ARRI_Display_LUTs_LogC3.zip",
        }
    }

    /// The (small) LUT size the test list uses for this kind.
    pub fn lut_size(self) -> usize {
        match self {
            Kind::LookLogC4 | Kind::DisplayLogC4 => 5,
            _ => 3,
        }
    }

    pub fn is_display(self) -> bool {
        matches!(self, Kind::DisplayLogC4 | Kind::DisplayLogC3)
    }
}

const GROUPS: [&str; 3] = ["1000 Colour", "3000 Film", "4000 Mono"];
const DISPLAYS: [&str; 5] = [
    "Gamma24_Rec709",
    "P3_D65",
    "Rec2020_PQ",
    "HLG",
    "Rec2020_HLG",
];

/// `n` synthetic files in the layout of package `kind`: (path inside the zip, `.cube` text).
pub fn package_files(kind: Kind, n: usize) -> Vec<(String, String)> {
    let enc = match kind {
        Kind::LookLogC4 | Kind::DisplayLogC4 => "LogC4",
        _ => "LogC3",
    };
    (0..n)
        .map(|i| {
            let seed = (i as u32 + 1) * 7 + kind as u32;
            let (path, title) = if kind.is_display() {
                let d = DISPLAYS[i % DISPLAYS.len()];
                let d = if i >= DISPLAYS.len() {
                    format!("{d}_v{}", i / DISPLAYS.len() + 1)
                } else {
                    d.to_string()
                };
                (format!("{enc}_to_{d}.cube"), format!("{enc} to {d}"))
            } else {
                let group = GROUPS[i % GROUPS.len()];
                let name = format!("Look_{:02}", i + 1);
                let file = if kind == Kind::LookLogC3Rec709 {
                    format!("{name}_{enc}_to_Rec709.cube")
                } else {
                    format!("{name}_{enc}.cube")
                };
                (format!("{group}/{file}"), format!("{enc} {name}"))
            };
            (path, cube_text(kind.lut_size(), seed, &title))
        })
        .collect()
}

/// Writes a package zip (plus the junk a real one carries: ALF4 files, a PDF, `__MACOSX`) and
/// returns its path.
pub fn write_package_zip(dir: &Path, kind: Kind, n: usize) -> PathBuf {
    let mut entries: Vec<(String, Vec<u8>)> = package_files(kind, n)
        .into_iter()
        .flat_map(|(path, text)| {
            let junk = format!("__MACOSX/{}", path.replace('/', "/._"));
            vec![
                (path.replace(".cube", ".alf4"), b"binary look file".to_vec()),
                (junk, b"apple double".to_vec()),
                (path, text.into_bytes()),
            ]
        })
        .collect();
    entries.push(("README.pdf".to_string(), b"%PDF-1.4 fake".to_vec()));
    let path = dir.join(kind.zip_name());
    write_zip(&path, &entries);
    path
}

/// The real `data/looks/arri-packages.json` with the numbers shrunk so synthetic packages stay small:
/// `n` files per package and the test LUT sizes. Patterns, markers and names are the real ones; the
/// fingerprints (hashes of ARRI's real files) are left out, since synthetic files cannot match them.
pub fn small_db(n: usize) -> PackageDb {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../data/looks/arri-packages.json");
    let mut db = PackageDb::load(&path).expect("arri-packages.json");
    for package in &mut db.packages {
        package.fingerprints.clear();
    }
    for kind in Kind::ALL {
        let package = db
            .packages
            .iter_mut()
            .find(|p| p.id == kind.package_id())
            .expect("package");
        package.expected.cube_files = Some(n);
        package.expected.lut_sizes = vec![kind.lut_size()];
    }
    db
}

/// A library under a fresh temporary app folder.
pub fn open_library(root: &Path) -> Library {
    Library::open(root, None).expect("open")
}

/// Lists a folder's files recursively as sorted `/`-separated relative paths.
pub fn tree(dir: &Path) -> Vec<String> {
    fn walk(base: &Path, dir: &Path, out: &mut Vec<String>) {
        let Ok(entries) = fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(base, &p, out);
            } else {
                out.push(
                    p.strip_prefix(base)
                        .unwrap()
                        .to_string_lossy()
                        .replace('\\', "/"),
                );
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

pub fn no_progress() -> impl FnMut(ym_looks::import::Progress<'_>) {
    |_| {}
}
