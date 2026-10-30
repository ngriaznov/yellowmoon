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

/// Where a look is listed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookGroup {
    /// Shipped with the app ("Built in").
    BuiltIn,
    /// One of ARRI's groups, from the package's folder names ("Film"). The section it is shown in
    /// is its [`LookLibrary`].
    Arri {
        /// The four-digit group number, for example 3000.
        group_no: u32,
        /// The group's name, for example "Film".
        group_name: String,
    },
    /// Imported by the user ("Your LUTs").
    User,
}

/// The library a look belongs to: the sections of the Look tab and the LIBRARY column of
/// `ym looks list` (`docs/08-looks.md` section Look model).
///
/// The variants are in display order. [`LookGroup`] says where a look is listed (built in, one of
/// ARRI's numbered groups, the person's own); the library also says which of ARRI's packages it
/// came from, because the same title ("Film") is in several of them. It is worked out from the
/// group, the input and the file format by [`LookLibrary::classify`] (and so by [`Look::library`]),
/// never stored, so the front ends cannot disagree about it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum LookLibrary {
    /// Shipped with the app.
    #[serde(rename = "built_in")]
    BuiltIn,
    /// ARRI's LogC4 looks: the 65-point `.cube` library, or the `.alf4` version of it.
    #[serde(rename = "arri_logc4")]
    ArriLogC4,
    /// ARRI's LogC3 log-to-log `.cube` library (input LogC3 EI 800).
    #[serde(rename = "arri_logc3")]
    ArriLogC3,
    /// ARRI's LogC3 to Rec.709 `.cube` library (the look includes the display rendering).
    #[serde(rename = "arri_logc3_to_rec709")]
    ArriLogC3ToRec709,
    /// ARRI's ALF2 look files (`.aml`): each look in three strengths, LogC3 to Rec.709.
    #[serde(rename = "arri_alf2")]
    ArriAlf2,
    /// The person's own `.cube`, `.alf4`, `.alf4c` and `.aml` files ("Your LUTs").
    #[serde(rename = "user")]
    User,
}

impl LookLibrary {
    /// Every library, in display order.
    pub const ALL: [Self; 6] = [
        Self::BuiltIn,
        Self::ArriLogC4,
        Self::ArriLogC3,
        Self::ArriLogC3ToRec709,
        Self::ArriAlf2,
        Self::User,
    ];

    /// The library of a look with this group, input and file format (`None` for a look that has
    /// no file, which is what a built-in look is).
    ///
    /// * built-in group: [`LookLibrary::BuiltIn`];
    /// * ARRI group, input LogC4: [`LookLibrary::ArriLogC4`], whatever the file is (`.cube`,
    ///   `.alf4` or `.alf4c`);
    /// * ARRI group, input LogC3 EI 800: [`LookLibrary::ArriLogC3`];
    /// * ARRI group, input LogC3 to Rec.709: [`LookLibrary::ArriAlf2`] for an `.aml`, otherwise
    ///   [`LookLibrary::ArriLogC3ToRec709`];
    /// * user group: [`LookLibrary::User`], whatever the input and the file are.
    #[must_use]
    pub fn classify(group: &LookGroup, input: LookInput, format: Option<LookFormat>) -> Self {
        match group {
            LookGroup::BuiltIn => Self::BuiltIn,
            LookGroup::User => Self::User,
            LookGroup::Arri { .. } => match input {
                LookInput::LogC4 => Self::ArriLogC4,
                LookInput::LogC3Ei800 => Self::ArriLogC3,
                LookInput::LogC3ToRec709 if format == Some(LookFormat::Aml) => Self::ArriAlf2,
                LookInput::LogC3ToRec709 => Self::ArriLogC3ToRec709,
            },
        }
    }

    /// The heading of the library's section: "Built in", "ARRI looks · LogC4", "ARRI looks ·
    /// LogC3", "ARRI looks · LogC3 to Rec.709", "ARRI looks · LogC3 to Rec.709 · ALF2" or "Your
    /// LUTs". The Look tab mirrors these words (`src/state/looks.ts`).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::BuiltIn => "Built in",
            Self::ArriLogC4 => "ARRI looks · LogC4",
            Self::ArriLogC3 => "ARRI looks · LogC3",
            Self::ArriLogC3ToRec709 => "ARRI looks · LogC3 to Rec.709",
            Self::ArriAlf2 => "ARRI looks · LogC3 to Rec.709 · ALF2",
            Self::User => "Your LUTs",
        }
    }

    /// The short word that tells two looks of the same name apart, and fills the LIBRARY column
    /// of `ym looks list`: "LogC4", "LogC3", "LogC3 to Rec.709", "ALF2" or "Your LUTs". `None`
    /// for the built-in looks, whose names need none.
    #[must_use]
    pub fn qualifier(self) -> Option<&'static str> {
        match self {
            Self::BuiltIn => None,
            Self::ArriLogC4 => Some("LogC4"),
            Self::ArriLogC3 => Some("LogC3"),
            Self::ArriLogC3ToRec709 => Some("LogC3 to Rec.709"),
            Self::ArriAlf2 => Some("ALF2"),
            Self::User => Some("Your LUTs"),
        }
    }

    /// The text used in JSON: `built_in`, `arri_logc4`, `arri_logc3`, `arri_logc3_to_rec709`,
    /// `arri_alf2` or `user`.
    #[must_use]
    pub fn slug(self) -> &'static str {
        match self {
            Self::BuiltIn => "built_in",
            Self::ArriLogC4 => "arri_logc4",
            Self::ArriLogC3 => "arri_logc3",
            Self::ArriLogC3ToRec709 => "arri_logc3_to_rec709",
            Self::ArriAlf2 => "arri_alf2",
            Self::User => "user",
        }
    }

    /// The inverse of [`LookLibrary::slug`].
    #[must_use]
    pub fn from_slug(slug: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|l| l.slug() == slug)
    }
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

    /// The kind of file the look is kept as; `None` for a built-in look, which has no file.
    #[must_use]
    pub fn format(&self) -> Option<LookFormat> {
        self.lut.as_ref().map(|l| l.format)
    }

    /// Points per axis of the look's 3D table: 65 for ARRI's LogC4 `.cube`, 33 for an `.alf4`;
    /// `None` for a built-in look.
    #[must_use]
    pub fn lut_size(&self) -> Option<usize> {
        self.lut.as_ref().map(|l| l.lut_size)
    }

    /// The library the look is listed under: its section of the Look tab and its LIBRARY column in
    /// `ym looks list` (see [`LookLibrary`]).
    #[must_use]
    pub fn library(&self) -> LookLibrary {
        LookLibrary::classify(&self.group, self.input, self.format())
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

    const INPUTS: [LookInput; 3] = [
        LookInput::LogC4,
        LookInput::LogC3Ei800,
        LookInput::LogC3ToRec709,
    ];
    const FORMATS: [Option<LookFormat>; 5] = [
        None,
        Some(LookFormat::Cube),
        Some(LookFormat::Alf4),
        Some(LookFormat::Alf4c),
        Some(LookFormat::Aml),
    ];

    fn arri() -> LookGroup {
        LookGroup::Arri {
            group_no: 3000,
            group_name: "Film".into(),
        }
    }

    #[test]
    fn a_built_in_look_is_always_in_the_built_in_library() {
        for input in INPUTS {
            for format in FORMATS {
                assert_eq!(
                    LookLibrary::classify(&LookGroup::BuiltIn, input, format),
                    LookLibrary::BuiltIn,
                    "{input:?} {format:?}"
                );
            }
        }
    }

    #[test]
    fn the_persons_own_look_is_always_in_your_luts() {
        for input in INPUTS {
            for format in FORMATS {
                assert_eq!(
                    LookLibrary::classify(&LookGroup::User, input, format),
                    LookLibrary::User,
                    "{input:?} {format:?}"
                );
            }
        }
    }

    #[test]
    fn an_arri_look_is_sorted_by_input_and_file_format() {
        use LookFormat::{Alf4, Alf4c, Aml, Cube};
        use LookInput::{LogC3Ei800, LogC3ToRec709, LogC4};
        use LookLibrary::{ArriAlf2, ArriLogC3, ArriLogC3ToRec709, ArriLogC4};
        // Every combination of input and format an ARRI look can have (a missing format reads
        // as a cube), with the library it is listed under.
        let table = [
            (LogC4, None, ArriLogC4),
            (LogC4, Some(Cube), ArriLogC4),
            (LogC4, Some(Alf4), ArriLogC4),
            (LogC4, Some(Alf4c), ArriLogC4),
            (LogC4, Some(Aml), ArriLogC4),
            (LogC3Ei800, None, ArriLogC3),
            (LogC3Ei800, Some(Cube), ArriLogC3),
            (LogC3Ei800, Some(Alf4), ArriLogC3),
            (LogC3Ei800, Some(Alf4c), ArriLogC3),
            (LogC3Ei800, Some(Aml), ArriLogC3),
            (LogC3ToRec709, None, ArriLogC3ToRec709),
            (LogC3ToRec709, Some(Cube), ArriLogC3ToRec709),
            (LogC3ToRec709, Some(Alf4), ArriLogC3ToRec709),
            (LogC3ToRec709, Some(Alf4c), ArriLogC3ToRec709),
            (LogC3ToRec709, Some(Aml), ArriAlf2),
        ];
        assert_eq!(table.len(), INPUTS.len() * FORMATS.len());
        for (input, format, library) in table {
            assert_eq!(
                LookLibrary::classify(&arri(), input, format),
                library,
                "{input:?} {format:?}"
            );
        }
    }

    #[test]
    fn a_look_reads_its_library_from_its_file() {
        let lut = |format, lut_size| {
            Some(LutRef {
                path: PathBuf::from("/looks/a"),
                sha256: "0".repeat(64),
                lut_size,
                format,
            })
        };
        let look = |group, input, lut| Look {
            id: LookId::builtin("x"),
            name: "Film".into(),
            group,
            input,
            cdl: None,
            lut,
            source: None,
            status: LookStatus::Ok,
        };
        let neutral = look(LookGroup::BuiltIn, LookInput::LogC4, None);
        assert_eq!(neutral.library(), LookLibrary::BuiltIn);
        assert_eq!((neutral.format(), neutral.lut_size()), (None, None));

        let cube = look(arri(), LookInput::LogC4, lut(LookFormat::Cube, 65));
        assert_eq!(cube.library(), LookLibrary::ArriLogC4);
        assert_eq!(
            (cube.format(), cube.lut_size()),
            (Some(LookFormat::Cube), Some(65))
        );

        let alf4 = look(arri(), LookInput::LogC4, lut(LookFormat::Alf4, 33));
        assert_eq!(alf4.library(), LookLibrary::ArriLogC4);
        assert_eq!(
            (alf4.format(), alf4.lut_size()),
            (Some(LookFormat::Alf4), Some(33))
        );

        let logc3 = look(arri(), LookInput::LogC3Ei800, lut(LookFormat::Cube, 33));
        assert_eq!(logc3.library(), LookLibrary::ArriLogC3);
        let rec709 = look(arri(), LookInput::LogC3ToRec709, lut(LookFormat::Cube, 33));
        assert_eq!(rec709.library(), LookLibrary::ArriLogC3ToRec709);
        let aml = look(arri(), LookInput::LogC3ToRec709, lut(LookFormat::Aml, 33));
        assert_eq!(aml.library(), LookLibrary::ArriAlf2);

        // The person's own look file of the same kinds stays in Your LUTs.
        let own_aml = look(
            LookGroup::User,
            LookInput::LogC3ToRec709,
            lut(LookFormat::Aml, 33),
        );
        assert_eq!(own_aml.library(), LookLibrary::User);
        let own_alf4c = look(
            LookGroup::User,
            LookInput::LogC4,
            lut(LookFormat::Alf4c, 33),
        );
        assert_eq!(own_alf4c.library(), LookLibrary::User);
    }

    #[test]
    fn libraries_come_in_display_order_with_their_words() {
        let words: Vec<(&str, Option<&str>)> = LookLibrary::ALL
            .iter()
            .map(|l| (l.label(), l.qualifier()))
            .collect();
        assert_eq!(
            words,
            vec![
                ("Built in", None),
                ("ARRI looks · LogC4", Some("LogC4")),
                ("ARRI looks · LogC3", Some("LogC3")),
                ("ARRI looks · LogC3 to Rec.709", Some("LogC3 to Rec.709")),
                ("ARRI looks · LogC3 to Rec.709 · ALF2", Some("ALF2")),
                ("Your LUTs", Some("Your LUTs")),
            ]
        );
        let mut sorted = LookLibrary::ALL;
        sorted.sort();
        assert_eq!(
            sorted,
            LookLibrary::ALL,
            "the variants are in display order"
        );
        let mut labels: Vec<&str> = words.iter().map(|w| w.0).collect();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(labels.len(), LookLibrary::ALL.len(), "no two share a label");
    }

    #[test]
    fn library_slugs_are_the_json_words() {
        let slugs = [
            "built_in",
            "arri_logc4",
            "arri_logc3",
            "arri_logc3_to_rec709",
            "arri_alf2",
            "user",
        ];
        for (library, slug) in LookLibrary::ALL.into_iter().zip(slugs) {
            assert_eq!(library.slug(), slug);
            assert_eq!(LookLibrary::from_slug(slug), Some(library));
            assert_eq!(
                serde_json::to_string(&library).unwrap(),
                format!("\"{slug}\"")
            );
            assert_eq!(
                serde_json::from_str::<LookLibrary>(&format!("\"{slug}\"")).unwrap(),
                library
            );
        }
        assert_eq!(LookLibrary::from_slug("arri"), None);
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
