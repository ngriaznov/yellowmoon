//! Albums: the filmstrip saved to a file the person names (`docs/31-albums.md`).
//!
//! An album is `<name>.ymalbum`: the photos in strip order (a relative and an absolute path, a fingerprint, the whole
//! settings graph as stored, picked or not), the current photo, the versions that wrote it, and the Auto provider's answer a
//! photo uses. It is written only when the person chooses Save, to a file they name: **no recent list, no autosave, no copy
//! in the application folder** (hard rule 4). It holds settings, never renders.
//!
//! * [`Album::capture`] reads a [`Session`] as it is; [`Album::to_json`] and [`Album::from_json`] are the file;
//!   [`write_album`] writes it atomically.
//! * [`Session::open_album`] replaces the session with the album's photos: finds each file, keeps the ones it cannot find as
//!   failed entries, applies the stored settings and the provider's answers, and asks no provider.
//! * [`Session::album_state`] and [`Session::set_album`] keep the edited state: a counter that every change bumps and the
//!   counter at the last save.
//!
//! [`Session`]: crate::session::Session
//! [`Session::open_album`]: crate::session::Session::open_album
//! [`Session::album_state`]: crate::session::Session::album_state
//! [`Session::set_album`]: crate::session::Session::set_album

use std::io::{Read, Seek, SeekFrom};
use std::path::{Component, Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use ym_looks::library::LookId;
use ym_looks::manifest::atomic_write;

use crate::apple_auto::{
    AdvisorAnswer, AdvisorInfo, AppleAuto, AutoSteps, InstructionsId, InstructionsKind,
};
use crate::auto_provider::{AutoProvider, ToolInfo};
use crate::presets::write_canonical;
use crate::session::{PhotoId, Session};
use crate::settings::{DevelopSettings, SettingsError};
use crate::version::{PIPELINE_VERSION, app_version};

/// The version of the file format (`yellowmoon_album`).
pub const FORMAT_VERSION: u64 = 1;

/// The largest album file that is read or written: 512 MiB.
pub const MAX_ALBUM_BYTES: u64 = 512 * 1024 * 1024;

/// The most photos an album holds (a sanity bound).
pub const MAX_ALBUM_PHOTOS: usize = 100_000;

/// How much of the start and of the end of a file a [`Fingerprint`] hashes.
pub const FINGERPRINT_BYTES: u64 = 64 * 1024;

/// The extension of an album file, without the dot.
pub const FILE_EXTENSION: &str = "ymalbum";

/// "This file isn't a Yellowmoon album."
pub const NOT_AN_ALBUM_SENTENCE: &str = "This file isn't a Yellowmoon album.";

/// "This album was made by a newer version of Yellowmoon. Update to open it."
pub const TOO_NEW_SENTENCE: &str =
    "This album was made by a newer version of Yellowmoon. Update to open it.";

/// "This file is too large to be an album."
pub const TOO_LARGE_SENTENCE: &str = "This file is too large to be an album.";

/// "Yellowmoon couldn't read this file. Check that it is still there, then try again."
pub const COULDNT_READ_SENTENCE: &str =
    "Yellowmoon couldn't read this file. Check that it is still there, then try again.";

/// The note of a photo whose file is not the one the album was made from: "This file has changed since the album was saved."
pub const CHANGED_SENTENCE: &str = "This file has changed since the album was saved.";

/// The reason a failed entry's cell shows: "Not found: /Users/anna/Pictures/Lisbon/DSC04206.ARW".
#[must_use]
pub fn not_found_sentence(absolute: &str) -> String {
    format!("Not found: {absolute}")
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Why an album could not be read, saved or opened. The text is plain words with a next step (`docs/31` sections 2, 4 and 5).
#[derive(Debug, thiserror::Error)]
pub enum AlbumError {
    /// A file that is not an album.
    #[error("This file isn't a Yellowmoon album.")]
    NotAnAlbum,
    /// An album of a newer version, or with a field this version does not know.
    #[error("This album was made by a newer version of Yellowmoon. Update to open it.")]
    TooNew,
    /// A file above 512 MiB, or with more than 100,000 photos.
    #[error("This file is too large to be an album.")]
    TooLarge,
    /// The settings inside the file cannot be used (the source error is kept for the log).
    #[error("This file isn't a Yellowmoon album.")]
    Settings(#[source] SettingsError),
    /// The file could not be read.
    #[error("Yellowmoon couldn't read this file. Check that it is still there, then try again.")]
    Io(#[source] std::io::Error),
    /// The album could not be written; nothing was changed.
    #[error(
        "Yellowmoon couldn't save the album to {folder}. Check that you can save there and that the disk has free space, then try again. Nothing was changed."
    )]
    Write {
        /// The folder the album was to be written in.
        folder: String,
        /// What the system said.
        #[source]
        source: std::io::Error,
    },
    /// A photo whose path is not valid UTF-8.
    #[error(
        "Yellowmoon can't save the path of {file} in an album. Rename the file or its folder, then try again."
    )]
    PathNotUtf8 {
        /// The file's name, as far as it can be shown.
        file: String,
    },
}

// ---------------------------------------------------------------------------------------------
// The pieces of the file
// ---------------------------------------------------------------------------------------------

/// The versions that wrote the file, for diagnosis only: opening an album never refuses or warns because the pipeline differs.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct WrittenBy {
    /// The Yellowmoon version.
    pub version: String,
    /// `PIPELINE_VERSION`.
    pub pipeline: String,
}

impl WrittenBy {
    /// This build's versions.
    #[must_use]
    pub fn current() -> Self {
        Self {
            version: app_version().to_owned(),
            pipeline: PIPELINE_VERSION.to_owned(),
        }
    }
}

/// Whether a file is the one the album was made from: its size and the SHA-256 of its first and last 64 KiB (the whole file when
/// it is smaller; the two may overlap). It reads at most 128 KiB, so saving 500 photos stays fast.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Fingerprint {
    /// The file's size in bytes.
    pub size: u64,
    /// The SHA-256 of the first 64 KiB, 64 lower-case hex digits.
    pub head_sha256: String,
    /// The SHA-256 of the last 64 KiB.
    pub tail_sha256: String,
}

fn hex(digest: &[u8]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

impl Fingerprint {
    /// The fingerprint of the file at `path`.
    ///
    /// # Errors
    ///
    /// The I/O error of opening or reading the file.
    pub fn of(path: &Path) -> std::io::Result<Self> {
        let mut file = std::fs::File::open(path)?;
        let size = file.metadata()?.len();
        let n = size.min(FINGERPRINT_BYTES);
        let mut chunk = |at: u64| -> std::io::Result<String> {
            file.seek(SeekFrom::Start(at))?;
            let mut buf = vec![0u8; usize::try_from(n).unwrap_or(0)];
            file.read_exact(&mut buf)?;
            Ok(hex(&Sha256::digest(&buf)))
        };
        let head_sha256 = chunk(0)?;
        let tail_sha256 = chunk(size - n)?;
        Ok(Self {
            size,
            head_sha256,
            tail_sha256,
        })
    }

    /// A fingerprint that says nothing (a file that could not be read when the album was saved and had none kept).
    #[must_use]
    pub fn unknown() -> Self {
        Self {
            size: 0,
            head_sha256: String::new(),
            tail_sha256: String::new(),
        }
    }
}

/// The Auto provider's answer a photo uses (`docs/31` section 2): the JSON of `ym:AutoAdvisor` (`docs/22` section 14.2), kept so
/// that opening the album gives the estimates it gave and asks no provider. It is the album's one deliberate copy of a provider's
/// answer; Re-run asks again.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AutoAnswer {
    /// Whose answer: `apple`, `claude` or `agy`.
    pub provider: AutoProvider,
    /// The model that answered.
    pub model: String,
    /// The macOS version, for Apple Intelligence.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub os: String,
    /// The program's version, for Claude Code and Antigravity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_version: Option<String>,
    /// Where the program's requests went.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api: Option<String>,
    /// The schema version of a program's answer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<u32>,
    /// How the picture was encoded for a program.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image: Option<String>,
    /// Default or edited instructions.
    pub instructions: InstructionsKind,
    /// The first 12 hex digits of the SHA-256 of the instructions.
    pub instructions_sha256: String,
    /// The version of the prompt.
    pub prompt: u32,
    /// The version of the table that turns steps into values (for diagnosis: the steps are what is kept).
    pub table: u32,
    /// Whether the provider gives the same answer again.
    pub reproducible: bool,
    /// `answered`: only answers are kept.
    pub result: String,
    /// What the photo shows.
    pub scene: String,
    /// The look the model gave the photo and how.
    pub plan: String,
    /// The steps; a step the model left empty is `null`.
    pub steps: AutoSteps,
}

impl AutoAnswer {
    /// The answer a photo's provider state holds, or `None` for anything but an answer (a failure, pending, off).
    #[must_use]
    pub fn from_applied(state: &AppleAuto) -> Option<Self> {
        let json = state.advisor_json()?;
        let answer: Self = serde_json::from_str(&json).ok()?;
        (answer.result == "answered").then_some(answer)
    }

    /// The provider state this answer is: [`AppleAuto::Answered`], as the provider's own answer would have made it.
    #[must_use]
    pub fn to_applied(&self) -> AppleAuto {
        AppleAuto::Answered {
            answer: AdvisorAnswer {
                steps: self.steps,
                scene: self.scene.clone(),
                plan: self.plan.clone(),
                served_model: None,
            },
            info: AdvisorInfo {
                provider: self.provider,
                model: self.model.clone(),
                os: self.os.clone(),
                tool: self.tool_version.as_ref().map(|version| ToolInfo {
                    version: version.clone(),
                    api: self.api.clone(),
                    auth: None,
                }),
                prompt: self.prompt,
                instructions: InstructionsId {
                    instructions: self.instructions,
                    instructions_sha256: self.instructions_sha256.clone(),
                },
            },
        }
    }
}

/// One photo of the strip.
#[derive(Debug, Clone, PartialEq)]
pub struct AlbumPhoto {
    /// The path relative to the folder of the album file (`../Lisbon/DSC04201.ARW`), or `None` when there is none.
    pub path: Option<String>,
    /// The absolute path.
    pub absolute: String,
    /// Whether a file is the one the album was made from.
    pub fingerprint: Fingerprint,
    /// Picked for the batch actions.
    pub picked: bool,
    /// The photo's whole settings graph as stored: Auto stays Auto, and the crop, the turn, the white pick, the Object clicks, the
    /// brush strokes and every node are kept (this is the photo itself, not a recipe).
    pub settings: DevelopSettings,
    /// The provider's answer the photo uses, where one answered.
    pub auto: Option<AutoAnswer>,
}

/// An album: the strip as it is saved.
#[derive(Debug, Clone, PartialEq)]
pub struct Album {
    /// The versions that wrote it.
    pub written_by: WrittenBy,
    /// The index of the current photo.
    pub current: Option<usize>,
    /// The photos in strip order.
    pub photos: Vec<AlbumPhoto>,
}

/// An open album's name, file and whether it has changes that are not saved (`docs/31` section 6).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlbumState {
    /// The album file.
    pub path: PathBuf,
    /// The album's name: the file's name without its extension ("Lisbon").
    pub name: String,
    /// Something changed since the album was opened or last saved.
    pub edited: bool,
}

impl AlbumState {
    /// The state of an album file with no changes.
    #[must_use]
    pub fn new(path: &Path, edited: bool) -> Self {
        Self {
            path: path.to_path_buf(),
            name: path
                .file_stem()
                .map_or_else(String::new, |s| s.to_string_lossy().into_owned()),
            edited,
        }
    }
}

/// Where a photo of an opened album stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlbumPhotoState {
    /// The file was found and is the one the album was made from.
    Found,
    /// The file was found and has changed since the album was saved ([`CHANGED_SENTENCE`]); its settings still apply.
    Changed,
    /// Neither path is a file: a failed entry, kept so that saving again does not drop it ([`not_found_sentence`]).
    Missing {
        /// The album's absolute path of the photo.
        absolute: String,
    },
}

/// What [`Session::open_album`](crate::session::Session::open_album) did.
#[derive(Debug, Clone, PartialEq)]
pub struct AlbumOutcome {
    /// Each photo of the strip, in strip order (a file listed twice is one photo), and where it stands.
    pub photos: Vec<(PhotoId, AlbumPhotoState)>,
    /// The album as the session now keeps it: not edited.
    pub album: AlbumState,
}

// ---------------------------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------------------------

/// The path of `target` relative to the folder `dir` (`../Lisbon/DSC04201.ARW`, `/` between the parts), or `None` when it cannot be
/// made (a relative path, another root).
#[must_use]
pub fn relative_path(dir: &Path, target: &Path) -> Option<String> {
    if !dir.is_absolute() || !target.is_absolute() {
        return None;
    }
    let d: Vec<Component<'_>> = dir.components().collect();
    let t: Vec<Component<'_>> = target.components().collect();
    let common = d.iter().zip(&t).take_while(|(a, b)| a == b).count();
    if common == 0 {
        return None;
    }
    let mut parts: Vec<&str> = vec![".."; d.len() - common];
    for c in &t[common..] {
        parts.push(c.as_os_str().to_str()?);
    }
    Some(parts.join("/"))
}

/// `rel` (an album's relative path, `/` between the parts) against the folder `dir`, with the `.` and `..` parts taken out where they
/// can be (`Albums/../Shoot/DSC04201.ARW` is `Shoot/DSC04201.ARW`), so that a photo found by its relative path has the path it
/// would have if it had been opened by hand.
#[must_use]
pub fn resolve_relative(dir: &Path, rel: &str) -> PathBuf {
    let mut out = dir.to_path_buf();
    for part in rel.split('/').filter(|p| !p.is_empty()) {
        match part {
            "." => {}
            ".." => {
                if !out.pop() {
                    out.push("..");
                }
            }
            other => out.push(other),
        }
    }
    out
}

/// The folder an album file is in (the path itself when it is a folder), absolute.
#[must_use]
pub fn album_dir(album_path: &Path) -> PathBuf {
    let dir = if album_path.is_dir() {
        album_path.to_path_buf()
    } else {
        album_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_default()
    };
    if dir.is_absolute() {
        dir
    } else {
        std::env::current_dir().map_or(dir.clone(), |cwd| cwd.join(&dir))
    }
}

// ---------------------------------------------------------------------------------------------
// The file
// ---------------------------------------------------------------------------------------------

fn classify(e: &serde_json::Error) -> AlbumError {
    if e.to_string().contains("unknown field") {
        AlbumError::TooNew
    } else {
        AlbumError::NotAnAlbum
    }
}

impl Album {
    /// The strip of `session` as it is now (`docs/31` section 5): the photos in order with their picks and the current photo, each
    /// photo's settings as stored (`Session::capture_photos`: Auto kept, nothing pinned, a failed photo's too), the provider's
    /// answers, and paths relative to the folder of `album_path`. Failed entries are written as they were.
    ///
    /// # Errors
    ///
    /// [`AlbumError::PathNotUtf8`] for a photo whose path is not UTF-8 (the save is refused), [`AlbumError::TooLarge`] above
    /// 100,000 photos.
    pub fn capture(session: &Session, album_path: &Path) -> Result<Self, AlbumError> {
        let dir = album_dir(album_path);
        let captured = session.capture_photos();
        if captured.len() > MAX_ALBUM_PHOTOS {
            return Err(AlbumError::TooLarge);
        }
        let mut photos = Vec::with_capacity(captured.len());
        let mut current = None;
        for (n, c) in captured.into_iter().enumerate() {
            let absolute = c
                .path
                .to_str()
                .ok_or_else(|| AlbumError::PathNotUtf8 {
                    file: c
                        .path
                        .file_name()
                        .map_or_else(String::new, |n| n.to_string_lossy().into_owned()),
                })?
                .to_owned();
            if c.current {
                current = Some(n);
            }
            let fingerprint = Fingerprint::of(&c.path)
                .ok()
                .or(c.fingerprint)
                .unwrap_or_else(Fingerprint::unknown);
            photos.push(AlbumPhoto {
                path: relative_path(&dir, &c.path),
                absolute,
                fingerprint,
                picked: c.picked,
                settings: c.settings,
                auto: c.answer.as_ref().and_then(AutoAnswer::from_applied),
            });
        }
        Ok(Self {
            written_by: WrittenBy::current(),
            current,
            photos,
        })
    }

    /// The file: UTF-8 JSON, compact, keys sorted, one newline at the end, so the same photos and settings are the same bytes.
    ///
    /// # Errors
    ///
    /// [`AlbumError::Settings`] for settings that cannot be written, [`AlbumError::TooLarge`] for a file above 512 MiB.
    pub fn to_json(&self) -> Result<String, AlbumError> {
        let mut doc = Map::new();
        doc.insert("yellowmoon_album".to_owned(), Value::from(FORMAT_VERSION));
        doc.insert(
            "written_by".to_owned(),
            serde_json::to_value(&self.written_by)
                .map_err(|e| AlbumError::Settings(SettingsError::Parse(e.to_string())))?,
        );
        doc.insert(
            "current".to_owned(),
            self.current.map_or(Value::Null, |c| Value::from(c as u64)),
        );
        let mut photos = Vec::with_capacity(self.photos.len());
        for p in &self.photos {
            let mut m = Map::new();
            m.insert(
                "path".to_owned(),
                p.path.clone().map_or(Value::Null, Value::String),
            );
            m.insert("absolute".to_owned(), Value::String(p.absolute.clone()));
            let to_value = |r: Result<Value, serde_json::Error>| {
                r.map_err(|e| AlbumError::Settings(SettingsError::Parse(e.to_string())))
            };
            m.insert(
                "fingerprint".to_owned(),
                to_value(serde_json::to_value(&p.fingerprint))?,
            );
            m.insert("picked".to_owned(), Value::Bool(p.picked));
            m.insert(
                "settings".to_owned(),
                to_value(serde_json::to_value(&p.settings))?,
            );
            if let Some(a) = &p.auto {
                m.insert("auto".to_owned(), to_value(serde_json::to_value(a))?);
            }
            photos.push(Value::Object(m));
        }
        doc.insert("photos".to_owned(), Value::Array(photos));
        let mut text = String::new();
        write_canonical(&Value::Object(doc), &mut text);
        text.push('\n');
        if text.len() as u64 > MAX_ALBUM_BYTES {
            return Err(AlbumError::TooLarge);
        }
        Ok(text)
    }

    /// Reads an album (`docs/31` section 2).
    ///
    /// * A `yellowmoon_album` above 1, an unknown top-level or photo field, and settings with a field this version does not know are
    ///   [`AlbumError::TooNew`], decided before anything else is looked at.
    /// * A file that is not JSON, lacks `yellowmoon_album` or `photos`, has a photo that is not valid, or holds the flat settings of
    ///   Yellowmoon 0.1 is [`AlbumError::NotAnAlbum`].
    /// * An older album always reads: what is missing takes its default, and numbers out of range are clamped like any settings
    ///   document.
    ///
    /// # Errors
    ///
    /// [`AlbumError`] as above; [`AlbumError::TooLarge`] for text above 512 MiB or more than 100,000 photos.
    pub fn from_json(text: &str) -> Result<Self, AlbumError> {
        if text.len() as u64 > MAX_ALBUM_BYTES {
            return Err(AlbumError::TooLarge);
        }
        let Ok(Value::Object(mut doc)) = serde_json::from_str::<Value>(text) else {
            return Err(AlbumError::NotAnAlbum);
        };
        match doc.remove("yellowmoon_album").and_then(|v| v.as_u64()) {
            Some(FORMAT_VERSION) => {}
            Some(v) if v > FORMAT_VERSION => return Err(AlbumError::TooNew),
            _ => return Err(AlbumError::NotAnAlbum),
        }
        let written_by = match doc.remove("written_by") {
            None | Some(Value::Null) => WrittenBy::default(),
            Some(v) => serde_json::from_value(v).map_err(|e| classify(&e))?,
        };
        let current = doc.remove("current");
        let photos = doc.remove("photos");
        if !doc.is_empty() {
            return Err(AlbumError::TooNew);
        }
        let Some(Value::Array(items)) = photos else {
            return Err(AlbumError::NotAnAlbum);
        };
        if items.len() > MAX_ALBUM_PHOTOS {
            return Err(AlbumError::TooLarge);
        }
        let mut photos = Vec::with_capacity(items.len());
        for item in items {
            photos.push(read_photo(item)?);
        }
        let current = current
            .and_then(|v| v.as_u64())
            .and_then(|c| usize::try_from(c).ok())
            .filter(|c| *c < photos.len());
        Ok(Self {
            written_by,
            current,
            photos,
        })
    }

    /// Reads the album file at `path` (the size is looked at before the file is read).
    ///
    /// # Errors
    ///
    /// [`AlbumError::TooLarge`], [`AlbumError::Io`], and the errors of [`Album::from_json`] ([`AlbumError::NotAnAlbum`] for a file
    /// that is not UTF-8).
    pub fn read_file(path: &Path) -> Result<Self, AlbumError> {
        let meta = std::fs::metadata(path).map_err(AlbumError::Io)?;
        if meta.len() > MAX_ALBUM_BYTES {
            return Err(AlbumError::TooLarge);
        }
        let bytes = std::fs::read(path).map_err(AlbumError::Io)?;
        let text = std::str::from_utf8(&bytes).map_err(|_| AlbumError::NotAnAlbum)?;
        Self::from_json(text)
    }
}

fn read_photo(item: Value) -> Result<AlbumPhoto, AlbumError> {
    let Value::Object(mut m) = item else {
        return Err(AlbumError::NotAnAlbum);
    };
    let path = match m.remove("path") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(s),
        Some(_) => return Err(AlbumError::NotAnAlbum),
    };
    let absolute = match m.remove("absolute") {
        Some(Value::String(s)) => s,
        _ => return Err(AlbumError::NotAnAlbum),
    };
    let fingerprint = match m.remove("fingerprint") {
        Some(v) => serde_json::from_value(v).map_err(|e| classify(&e))?,
        None => Fingerprint::unknown(),
    };
    let picked = match m.remove("picked") {
        None => false,
        Some(Value::Bool(b)) => b,
        Some(_) => return Err(AlbumError::NotAnAlbum),
    };
    let settings = match m.remove("settings") {
        Some(v @ Value::Object(_)) => match DevelopSettings::from_value(v, &LookId::neutral()) {
            Ok((s, _)) => s,
            Err(SettingsError::OldForm) => return Err(AlbumError::NotAnAlbum),
            Err(SettingsError::Parse(_)) => return Err(AlbumError::TooNew),
            // A node kind this version does not have is a newer version's (`docs/31` section 2), not a broken file.
            Err(SettingsError::Graph(crate::settings::GraphError::UnknownKind(_))) => {
                return Err(AlbumError::TooNew);
            }
            Err(e) => return Err(AlbumError::Settings(e)),
        },
        _ => return Err(AlbumError::NotAnAlbum),
    };
    let auto = match m.remove("auto") {
        None | Some(Value::Null) => None,
        Some(v) => Some(serde_json::from_value(v).map_err(|e| classify(&e))?),
    };
    if !m.is_empty() {
        return Err(AlbumError::TooNew);
    }
    Ok(AlbumPhoto {
        path,
        absolute,
        fingerprint,
        picked,
        settings,
        auto,
    })
}

/// Writes the album to `path`, atomically: a temporary file in the album's folder, `fsync`, rename, `fsync` of the folder. It writes
/// only the album file, and never touches an original (hard rule 5). A write that fails changes nothing.
///
/// # Errors
///
/// [`AlbumError::Write`] with the sentence of `docs/31` section 5, and the errors of [`Album::to_json`].
pub fn write_album(path: &Path, album: &Album) -> Result<(), AlbumError> {
    let text = album.to_json()?;
    atomic_write(path, text.as_bytes()).map_err(|source| AlbumError::Write {
        folder: path
            .parent()
            .map_or_else(|| path.display().to_string(), |p| p.display().to_string()),
        source,
    })
}
