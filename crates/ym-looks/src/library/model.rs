//! The look model: [`Look`], its group, input, LUT reference and status (`docs/08-looks.md` section Look model).

use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use ym_color::alf::{self, AlfKind};
use ym_color::cdl::Cdl;
use ym_color::cube::{self, CubeError, MAX_FILE_BYTES};
use ym_color::look::LookInput;
use ym_color::lut::Lut;

use super::ids::LookId;
use crate::manifest::{Source, sha256_hex};

/// What the user is told next to a look whose file is gone or changed.
pub const DISABLED_MESSAGE: &str = "File missing or changed. Import again.";

/// The encoding an ARRI display LUT expects as input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DisplayInput {
    /// LogC4 / ARRI Wide Gamut 4.
    #[serde(rename = "logc4")]
    LogC4,
    /// LogC3 EI 800 / ARRI Wide Gamut 3.
    #[serde(rename = "logc3")]
    LogC3,
}

impl DisplayInput {
    /// The text used in ids and the manifest: `logc4` or `logc3`.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::LogC4 => "logc4",
            Self::LogC3 => "logc3",
        }
    }

    /// The inverse of [`DisplayInput::slug`].
    #[must_use]
    pub fn from_slug(slug: &str) -> Option<Self> {
        match slug {
            "logc4" => Some(Self::LogC4),
            "logc3" => Some(Self::LogC3),
            _ => None,
        }
    }
}

/// Where a look is listed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookGroup {
    /// Shipped with the app ("Built in").
    BuiltIn,
    /// One of ARRI's groups, from the package's folder names ("ARRI looks · imported").
    Arri {
        /// The four-digit group number, for example 3000.
        group_no: u32,
        /// The group's name, for example "Film".
        group_name: String,
    },
    /// Imported by the user ("Your LUTs").
    User,
}

/// Whether a look's file can be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LookStatus {
    /// The file is there. [`Library::verify`](super::Library::verify) and loading also check that
    /// it still has the imported checksum.
    Ok,
    /// The file is gone.
    Missing,
    /// The file is there but its checksum differs from the one recorded at import.
    Changed,
}

impl LookStatus {
    /// Whether the look can be used.
    #[must_use]
    pub fn is_ok(self) -> bool {
        self == Self::Ok
    }
}

/// The kind of file a look is kept as in the library: the original file, byte for byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LookFormat {
    /// A `.cube` LUT. The only kind before ARRI Look Files were read, so a manifest entry without
    /// a `format` is this.
    #[default]
    Cube,
    /// An ARRI Look File for ALEXA 35 (`.alf4`): a CDL and a LogC4 LUT.
    Alf4,
    /// An ARRI custom colour management file (`.alf4c`): its CDL and creative LUT are used.
    Alf4c,
    /// An ARRI Look File for LogC3 cameras (`.aml`): a CDL and a LogC3-to-Rec.709 LUT.
    Aml,
}

impl LookFormat {
    /// The file extension without the dot: `cube`, `alf4`, `alf4c` or `aml`.
    #[must_use]
    pub fn extension(self) -> &'static str {
        match self {
            Self::Cube => "cube",
            Self::Alf4 => "alf4",
            Self::Alf4c => "alf4c",
            Self::Aml => "aml",
        }
    }

    /// The extension with its dot, for messages: `.cube`.
    #[must_use]
    pub fn noun(self) -> &'static str {
        match self {
            Self::Cube => ".cube",
            Self::Alf4 => ".alf4",
            Self::Alf4c => ".alf4c",
            Self::Aml => ".aml",
        }
    }

    /// Whether this is a plain `.cube` LUT (serde skips the field then).
    #[must_use]
    pub fn is_cube(&self) -> bool {
        *self == Self::Cube
    }

    /// Whether this is one of the ARRI Look File containers.
    #[must_use]
    pub fn is_look_file(self) -> bool {
        !self.is_cube()
    }

    /// The format of a parsed container.
    #[must_use]
    pub fn from_kind(kind: AlfKind) -> Self {
        match kind {
            AlfKind::Alf4 => Self::Alf4,
            AlfKind::Alf4c => Self::Alf4c,
            AlfKind::Aml => Self::Aml,
        }
    }

    /// The format a file name's extension says (case ignored); `None` for any other.
    #[must_use]
    pub fn from_extension(extension: &str) -> Option<Self> {
        [Self::Cube, Self::Alf4, Self::Alf4c, Self::Aml]
            .into_iter()
            .find(|f| f.extension().eq_ignore_ascii_case(extension))
    }
}

/// Why a LUT file could not be loaded.
#[derive(Debug, thiserror::Error)]
pub enum LutLoadError {
    /// The file is not there.
    #[error("{DISABLED_MESSAGE}")]
    Missing,
    /// The file's checksum differs from the one recorded at import.
    #[error("{DISABLED_MESSAGE}")]
    Changed {
        /// The checksum recorded in the manifest.
        expected: String,
        /// The checksum of the file now.
        actual: String,
    },
    /// The file could not be read.
    #[error("The LUT file could not be read: {0}")]
    Unreadable(#[source] io::Error),
    /// The file no longer parses as a 3D `.cube` LUT.
    #[error("The LUT file is not a usable .cube file: {0}")]
    Invalid(String),
    /// The file no longer reads as an ARRI Look File, or is not the look it was imported as.
    #[error("The look file is not usable: {0}")]
    InvalidLookFile(String),
}

/// What reading a look's file gives: the LUT, and for a look file its CDL and the encoding it
/// expects.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedLook {
    /// The LUT.
    pub lut: Lut,
    /// The CDL stored in an ARRI Look File, applied before the LUT; `None` for a `.cube` and for a
    /// look file whose CDL is the identity.
    pub cdl: Option<Cdl>,
    /// The encoding the file says it is made for (look files only); `None` for a `.cube`.
    pub input: Option<LookInput>,
}

/// A LUT file in the library: path, SHA-256 and size. Loaded lazily and checked against its hash
/// every time it is read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LutRef {
    /// Absolute path of the copied file (`.cube`, or an ARRI Look File).
    pub path: PathBuf,
    /// SHA-256 recorded at import, lower-case hex.
    pub sha256: String,
    /// Points per axis of the 3D table.
    pub lut_size: usize,
    /// What kind of file it is.
    pub format: LookFormat,
}

impl LutRef {
    /// Reads the file (at most `limit` bytes) and checks its SHA-256.
    fn read_checked(&self, limit: usize) -> Result<Vec<u8>, LutLoadError> {
        let file = match File::open(&self.path) {
            Ok(f) => f,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Err(LutLoadError::Missing),
            Err(e) => return Err(LutLoadError::Unreadable(e)),
        };
        let mut bytes = Vec::new();
        file.take(limit as u64 + 1)
            .read_to_end(&mut bytes)
            .map_err(LutLoadError::Unreadable)?;
        let actual = sha256_hex(&bytes);
        if actual != self.sha256 {
            return Err(LutLoadError::Changed {
                expected: self.sha256.clone(),
                actual,
            });
        }
        Ok(bytes)
    }

    /// Reads a `.cube` file, checks its SHA-256 and parses it. (A look file goes through
    /// [`LutRef::load_look`], which also gives its CDL.)
    ///
    /// # Errors
    ///
    /// [`LutLoadError::Missing`] or [`LutLoadError::Changed`] (the look is "disabled"),
    /// [`LutLoadError::Unreadable`], or [`LutLoadError::Invalid`] (also for a ref that is not a
    /// `.cube`).
    pub fn load(&self) -> Result<Lut, LutLoadError> {
        if self.format.is_look_file() {
            return Err(LutLoadError::Invalid(format!(
                "it is a {} look file, not a .cube",
                self.format.noun()
            )));
        }
        let bytes = self.read_checked(MAX_FILE_BYTES)?;
        cube::parse_bytes(&bytes)
            .and_then(cube::CubeFile::into_lut)
            .map_err(|e| LutLoadError::Invalid(cube_error_text(&e)))
    }

    /// Reads the file, checks its SHA-256 and reads it as what it is: a `.cube`, or an ARRI Look
    /// File with its CDL (`docs/08-looks.md` section ARRI Look Files).
    ///
    /// # Errors
    ///
    /// As [`LutRef::load`], with [`LutLoadError::InvalidLookFile`] for a look file that no longer
    /// reads, or that is not the kind of look it was imported as.
    pub fn load_look(&self) -> Result<LoadedLook, LutLoadError> {
        if self.format.is_cube() {
            return Ok(LoadedLook {
                lut: self.load()?,
                cdl: None,
                input: None,
            });
        }
        let bytes = self.read_checked(alf::MAX_FILE_BYTES)?;
        let look = alf::parse(&bytes).map_err(|e| LutLoadError::InvalidLookFile(e.to_string()))?;
        if LookFormat::from_kind(look.kind) != self.format {
            return Err(LutLoadError::InvalidLookFile(format!(
                "it was imported as a {} file but is a {} file. Import it again",
                self.format.noun(),
                look.kind
            )));
        }
        Ok(LoadedLook {
            lut: Lut::from_cube(look.lut),
            cdl: look.cdl,
            input: Some(look.input),
        })
    }
}

/// A look in the library (`docs/08-looks.md` section Look model).
#[derive(Debug, Clone, PartialEq)]
pub struct Look {
    /// `builtin:neutral`, `arri:<group>:<slug>`, `user:<slug>`.
    pub id: LookId,
    /// Display name.
    pub name: String,
    /// Where it is listed.
    pub group: LookGroup,
    /// The encoding it expects; `LogC3ToRec709` looks include their own display rendering.
    pub input: LookInput,
    /// A CDL applied before the LUT. Imported looks have none; built-in parametric looks carry
    /// theirs inside the baked cube.
    pub cdl: Option<Cdl>,
    /// The LUT file of an imported look. `None` for built-in looks, whose cube is baked on load.
    pub lut: Option<LutRef>,
    /// Where an imported look came from.
    pub source: Option<Source>,
    /// Whether its file is usable. Disabled looks stay in the list.
    pub status: LookStatus,
}

impl Look {
    /// Whether the look can be applied.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.status.is_ok()
    }

    /// "File missing or changed. Import again." for a disabled look.
    #[must_use]
    pub fn disabled_reason(&self) -> Option<&'static str> {
        (!self.status.is_ok()).then_some(DISABLED_MESSAGE)
    }

    /// Whether the look replaces the display transform (a LogC3-to-Rec.709 look).
    #[must_use]
    pub fn includes_display_rendering(&self) -> bool {
        self.input == LookInput::LogC3ToRec709
    }
}

/// An imported ARRI display LUT: ARRI's own display rendering, used instead of ACES 2.
#[derive(Debug, Clone, PartialEq)]
pub struct DisplayLut {
    /// `arri-display:<input>:<target>`.
    pub id: LookId,
    /// The encoding it expects.
    pub input: DisplayInput,
    /// The display it renders to (`rec709-g24`, …).
    pub target: String,
    /// The file.
    pub lut: LutRef,
    /// Whether its file is usable.
    pub status: LookStatus,
}

impl DisplayLut {
    /// "File missing or changed. Import again." for a disabled LUT.
    #[must_use]
    pub fn disabled_reason(&self) -> Option<&'static str> {
        (!self.status.is_ok()).then_some(DISABLED_MESSAGE)
    }
}

/// A `.cube` parse error as a short sentence for the person importing the file.
#[must_use]
pub fn cube_error_text(error: &CubeError) -> String {
    let text = match error {
        CubeError::NoThreeDimensionalLut => {
            "It is a 1D LUT. Yellowmoon needs a 3D LUT (a 1D shaper in front of a 3D table is fine)"
                .to_string()
        }
        CubeError::NotUtf8 => "It is not a text file, so it is not a .cube LUT".to_string(),
        CubeError::TooLarge { limit, .. } => {
            format!("It is larger than the {} MB limit", limit / (1024 * 1024))
        }
        CubeError::MissingSize => {
            "It has no LUT_3D_SIZE line, so it is not a .cube LUT".to_string()
        }
        other => {
            let text = other.to_string();
            let mut chars = text.chars();
            match chars.next() {
                Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
                None => text,
            }
        }
    };
    format!("{text}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_input_slugs_round_trip() {
        for input in [DisplayInput::LogC4, DisplayInput::LogC3] {
            assert_eq!(DisplayInput::from_slug(input.slug()), Some(input));
        }
        assert_eq!(DisplayInput::from_slug("logc5"), None);
        assert_eq!(
            serde_json::to_string(&DisplayInput::LogC3).unwrap(),
            "\"logc3\""
        );
    }

    #[test]
    fn cube_errors_read_as_sentences() {
        let e = cube::parse("LUT_3D_SIZE 2\n0 0 0\nbad row here\n").unwrap_err();
        let text = cube_error_text(&e);
        assert!(text.starts_with("Line 3"), "{text}");
        assert!(text.ends_with('.'));
        let e = cube::parse("LUT_1D_SIZE 2\n0 0 0\n1 1 1\n")
            .and_then(cube::CubeFile::into_lut)
            .unwrap_err();
        assert!(cube_error_text(&e).contains("1D LUT"));
    }

    #[test]
    fn lut_ref_detects_missing_and_changed_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.cube");
        let text = ym_color::lut::Lut3d::identity(3)
            .unwrap()
            .to_cube_string("a");
        std::fs::write(&path, &text).unwrap();
        let mut lut_ref = LutRef {
            path: path.clone(),
            sha256: sha256_hex(text.as_bytes()),
            lut_size: 3,
            format: LookFormat::Cube,
        };
        assert_eq!(lut_ref.load().unwrap().cube.size(), 3);
        std::fs::write(&path, text.replace("0.5", "0.25")).unwrap();
        assert!(matches!(lut_ref.load(), Err(LutLoadError::Changed { .. })));
        std::fs::remove_file(&path).unwrap();
        assert!(matches!(lut_ref.load(), Err(LutLoadError::Missing)));
        // A matching hash over text that is not a LUT is reported as invalid, not as changed.
        std::fs::write(&path, "nonsense").unwrap();
        lut_ref.sha256 = sha256_hex(b"nonsense");
        assert!(matches!(lut_ref.load(), Err(LutLoadError::Invalid(_))));
    }
}
