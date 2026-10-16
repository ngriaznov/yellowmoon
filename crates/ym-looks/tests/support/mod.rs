//! Helpers shared by the integration tests: synthetic `.cube` files, zips and package lists.
//!
//! Everything here is generated from our own formulas. No ARRI data is used anywhere; only
//! ARRI-like **file names** are, so the recognition rules have something to read.

#![allow(dead_code)]

use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

use ym_color::alf::AlfFile;
use ym_color::cdl::Cdl;
use ym_color::lut::Lut3d;
use ym_looks::library::Library;
use ym_looks::recognize::PackageDb;
use zip::CompressionMethod;
use zip::write::SimpleFileOptions;

/// A deterministic synthetic LUT: a gentle, seed-dependent colour change on a `size`-point cube.
pub fn look_lut(size: usize, seed: u32) -> Lut3d {
    let s = f64::from(seed % 97) / 97.0;
    Lut3d::bake(size, [0.0; 3], [1.0; 3], |c| {
        [
            (c[0] * (0.90 + 0.1 * s) + 0.02 * s).clamp(0.0, 1.0),
            c[1].powf(0.9 + 0.2 * s),
            (c[2] * 0.95 + 0.05 * c[0] * s).clamp(0.0, 1.0),
        ]
    })
    .expect("bake")
}

/// [`look_lut`] as the text of a `.cube` file.
pub fn cube_text(size: usize, seed: u32, title: &str) -> String {
    look_lut(size, seed).to_cube_string(title)
}

/// A CDL that changes every channel, for tests of look files.
pub fn some_cdl() -> Cdl {
    Cdl::new([1.1, 1.2, 1.3], [0.01, 0.02, 0.03], [0.9, 1.1, 1.3], 1.5).expect("cdl")
}

/// An `.alf4` written by our own writer from [`look_lut`].
pub fn alf4_bytes(size: usize, seed: u32, cdl: Option<&Cdl>) -> Vec<u8> {
    AlfFile::alf4(&look_lut(size, seed), cdl, "", "look.cube")
        .to_bytes()
        .expect("write")
}

/// An `.alf4c` with a CMT from [`look_lut`] and display transforms of its own.
pub fn alf4c_bytes(size: usize, seed: u32, cdl: Option<&Cdl>) -> Vec<u8> {
    AlfFile::alf4c(
        Some(&look_lut(size, seed)),
        cdl,
        Some(&look_lut(3, seed + 1)),
        Some(&look_lut(3, seed + 2)),
        "custom",
    )
    .to_bytes()
    .expect("write")
}

/// An `.aml` written by our own writer from [`look_lut`].
pub fn aml_bytes(size: usize, seed: u32, cdl: Option<&Cdl>) -> Vec<u8> {
    AlfFile::aml(&look_lut(size, seed), cdl, "look.cube")
        .to_bytes()
        .expect("write")
}

/// The folder and zip names of ARRI's two look-file libraries (names are facts).
pub const ALF4_ROOT: &str = "ARRI Look Library LogC4 (log-to-log) ALF4";
pub const ALF4_ZIP: &str = "arri-look-library-logc4-log-to-log-alf4-data.zip";
pub const ALF2_ROOT: &str = "ARRI Look Library LogC3 to Rec709 ALF2";
pub const ALF2_ZIP: &str = "ARRI Look Library LogC3 (log-to-Rec709) ALF2_data.zip";
/// The LUT size of the synthetic look-file libraries (ARRI's have 33).
pub const ALF_LUT_SIZE: usize = 3;

/// The group folders and numbers the synthetic look-file libraries use.
const ALF_GROUPS: [(&str, u32); 3] = [
    ("1000 Black-and-White", 1000),
    ("3000 Film", 3000),
    ("4000 Tinted", 4000),
];

/// `n` synthetic looks in the layout of ARRI's ALF4 zip: (path inside the zip, bytes). Look `i` is
/// called `Look 01`, `Look 02`, ..., and has the id the `.cube` package from [`package_files`]
/// (`Kind::LookLogC4`) gives the same look.
pub fn alf4_package_files(n: usize) -> Vec<(String, Vec<u8>)> {
    (0..n)
        .map(|i| {
            let (folder, number) = ALF_GROUPS[i % 3];
            let code = number + 10 * (i as u32 / 3 + 1);
            (
                format!("{ALF4_ROOT}/{folder}/LL-{code}-Look-{:02}.alf4", i + 1),
                alf4_bytes(ALF_LUT_SIZE, i as u32 + 1, None),
            )
        })
        .collect()
}

/// `n` synthetic looks in the layout of ARRI's ALF2 zip, three strengths each: `3 * n` files.
pub fn aml_package_files(n: usize) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();
    for i in 0..n {
        let (folder, number) = ALF_GROUPS[i % 3];
        let code = number + 10 * (i as u32 / 3 + 1);
        for (k, strength) in ["Low", "Medium", "High"].into_iter().enumerate() {
            out.push((
                format!(
                    "{ALF2_ROOT}/{folder}/LL-{code}-Look-{:02}-{strength}.aml",
                    i + 1
                ),
                aml_bytes(ALF_LUT_SIZE, 3 * i as u32 + k as u32 + 1, None),
            ));
        }
    }
    out
}

/// Writes `files` to a zip with the junk a Mac adds (`__MACOSX/._...`, `.DS_Store`).
pub fn write_mac_zip(path: &Path, files: &[(String, Vec<u8>)]) {
    let mut entries: Vec<(String, Vec<u8>)> = Vec::new();
    for (name, bytes) in files {
        entries.push((name.clone(), bytes.clone()));
        let (folder, file) = name.rsplit_once('/').unwrap_or(("", name));
        entries.push((
            format!("__MACOSX/{folder}/._{file}"),
            b"\x00\x05\x16\x07 apple double".to_vec(),
        ));
    }
    entries.push((format!("{ALF4_ROOT}/.DS_Store"), b"junk".to_vec()));
    write_zip(path, &entries);
}

/// Writes the synthetic ALF4 library as ARRI's zip is named and laid out.
pub fn write_alf4_package_zip(dir: &Path, n: usize) -> PathBuf {
    let path = dir.join(ALF4_ZIP);
    write_mac_zip(&path, &alf4_package_files(n));
    path
}

/// Writes the synthetic ALF2 library as ARRI's zip is named and laid out.
pub fn write_aml_package_zip(dir: &Path, n: usize) -> PathBuf {
    let path = dir.join(ALF2_ZIP);
    write_mac_zip(&path, &aml_package_files(n));
    path
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

/// Writes a package zip (plus the junk a real one carries: `.xml` files, a PDF, `__MACOSX`) and
/// returns its path.
pub fn write_package_zip(dir: &Path, kind: Kind, n: usize) -> PathBuf {
    let mut entries: Vec<(String, Vec<u8>)> = package_files(kind, n)
        .into_iter()
        .flat_map(|(path, text)| {
            let junk = format!("__MACOSX/{}", path.replace('/', "/._"));
            vec![
                (path.replace(".cube", ".xml"), b"<adicam/>".to_vec()),
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
    // The look-file libraries: `n` looks, and three strengths of each for the LogC3 one.
    for (id, files) in [
        ("look-library-logc4-alf4", n),
        ("look-library-logc3-rec709-alf2", 3 * n),
    ] {
        let package = db
            .packages
            .iter_mut()
            .find(|p| p.id == id)
            .expect("look-file package");
        package.expected.cube_files = Some(files);
        package.expected.lut_sizes = vec![ALF_LUT_SIZE];
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
