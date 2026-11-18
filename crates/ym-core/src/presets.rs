//! Presets: a name and some of a photo's settings, saved as a file and applied to other photos
//! (`docs/30-presets.md`).
//!
//! * **The groups** ([`PresetGroup`], section 2): Look, Lens, Light, Colour, Detail, Curve, RAW and Nodes. A preset
//!   holds the groups the person ticked, each whole.
//! * **Saving** ([`Preset::from_photo`], section 4.1): Copy settings' patch ([`SettingsPatch::copied_from`]: Auto stays
//!   Auto, a picked white becomes the manual Kelvin and tint it resolved to), with the Object clicks and the Brush
//!   strokes taken out, cut to the groups. [`changed_groups`] says which groups to tick for the person.
//! * **Applying** ([`Preset::patch_for`], section 4.2): a node-level [`SettingsPatch`] that [`Session::paste_settings`]
//!   applies (and `ym render --preset` applies with [`DevelopSettings::apply_patch`]), so the app and the CLI give the
//!   same bytes. [`Preset::matches`] says whether applying would change nothing (the ring on a tile).
//! * **The file** ([`Preset::to_json`], [`Preset::from_json`], section 5): `presets/<safe name>.ympreset`, compact JSON
//!   with sorted keys, described by `schema/preset.schema.json`.
//! * **The store and the service** ([`PresetStore`], [`PresetService`], sections 6 and 7): the folder with its lock,
//!   and the store tied to the look library and the session, one method of `Core` for each IPC command.
//!
//! A preset is about no photo: no path, file name, camera, date or location. It is one of the three things that
//! persist on their own (`docs/01` *Persistence*).
//!
//! [`Session::paste_settings`]: crate::session::Session::paste_settings

mod service;
mod store;

use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use ym_looks::library::LookId;

pub use self::service::{
    DamagedPreset, ImportExists, ImportOutcome, ImportRefused, PresetApplyReport, PresetEntry,
    PresetInspect, PresetList, PresetService, PresetSource,
};
pub use self::store::{PresetListing, PresetStore, StoredPreset};
use crate::session::SessionError;
use crate::settings::{
    DevelopSettings, NodeId, ResolvedSettings, SettingsError, SettingsField, SettingsPatch,
};

/// The version of the file format (`yellowmoon_preset`).
pub const FORMAT_VERSION: u64 = 1;

/// The largest preset file that is read or written: 4 MiB.
pub const MAX_PRESET_BYTES: usize = 4 * 1024 * 1024;

/// The longest name, in characters.
pub const MAX_NAME_CHARS: usize = 64;

/// The extension of a preset file, without the dot.
pub const FILE_EXTENSION: &str = "ympreset";

/// "This file isn't a Yellowmoon preset."
pub const NOT_A_PRESET_SENTENCE: &str = "This file isn't a Yellowmoon preset.";

/// "This preset was made by a newer version of Yellowmoon. Update to use it."
pub const TOO_NEW_SENTENCE: &str =
    "This preset was made by a newer version of Yellowmoon. Update to use it.";

/// "This file is too large to be a preset."
pub const TOO_LARGE_SENTENCE: &str = "This file is too large to be a preset.";

/// "Yellowmoon couldn't read this file." (an I/O error).
pub const COULDNT_READ_SENTENCE: &str = "Yellowmoon couldn't read this file.";

/// "Another preset has this name." (two files with one name).
pub const DUPLICATE_NAME_SENTENCE: &str = "Another preset has this name.";

/// "Give the preset a name."
pub const EMPTY_NAME_SENTENCE: &str = "Give the preset a name.";

/// "Use 64 characters or fewer."
pub const LONG_NAME_SENTENCE: &str = "Use 64 characters or fewer.";

/// What the notice says when a preset's look is not in the library: "Warm evening uses the look Film warm, which isn't
/// imported. Everything else was applied." (`docs/30` section 9.5).
#[must_use]
pub fn missing_look_sentence(preset: &str, look: &str) -> String {
    format!("{preset} uses the look {look}, which isn't imported. Everything else was applied.")
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Why a name was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NameProblem {
    /// Nothing but white space.
    Empty,
    /// More than 64 characters.
    TooLong,
    /// A control character.
    Control,
}

/// Why a preset command failed. The text is plain words with a next step (`docs/30` section 9.9).
#[derive(Debug, thiserror::Error)]
pub enum PresetError {
    /// A file that is not a preset.
    #[error("This file isn't a Yellowmoon preset.")]
    NotAPreset,
    /// A preset of a newer version, or with a field this version does not know.
    #[error("This preset was made by a newer version of Yellowmoon. Update to use it.")]
    TooNew,
    /// A file above 4 MiB.
    #[error("This file is too large to be a preset.")]
    TooLarge,
    /// The settings inside the file cannot be used (the source error is kept for the log).
    #[error("This file isn't a Yellowmoon preset.")]
    Settings(#[source] SettingsError),
    /// The settings of a photo could not be cut into a preset (a bug: the source error says why).
    #[error(
        "Yellowmoon couldn't make a preset from these settings. Nothing was saved. Try again with fewer groups ticked, and report the problem on GitHub if it keeps happening."
    )]
    CouldNotMake(#[source] SettingsError),
    /// The text of a preset would not read back, so it was not written (a bug).
    #[error(
        "Yellowmoon couldn't save this preset: the file it would write can't be read back. Nothing was saved. Try again with fewer groups ticked, and report the problem on GitHub if it keeps happening."
    )]
    WouldNotReadBack(#[source] Box<PresetError>),
    /// A photo or file that holds no Yellowmoon settings (an export without `ym:Settings`, or not an export).
    #[error("This file has no Yellowmoon settings. Choose a photo that Yellowmoon exported.")]
    NoSettings,
    /// A name that cannot be used.
    #[error("{}", name_sentence(*.0))]
    InvalidName(NameProblem),
    /// A preset needs at least one group.
    #[error("Tick at least one group.")]
    NoGroups,
    /// A name that is taken (ignoring case).
    #[error("A preset called {0} already exists.")]
    Exists(String),
    /// No preset has this name.
    #[error("There is no preset called {0}.")]
    NotFound(String),
    /// A path that is not a preset file.
    #[error("There is no preset file at {0}.")]
    PathNotFound(String),
    /// A query that matches several presets by prefix.
    #[error("{query} matches more than one preset: {}. Use more of the name.", names.join(", "))]
    Ambiguous {
        /// What was asked for.
        query: String,
        /// The presets it matches, in the list's order.
        names: Vec<String>,
    },
    /// A file the settings could not be read from (a settings file, not a preset).
    #[error("Yellowmoon couldn't read settings from this file.")]
    SettingsFile(String),
    /// A file could not be written.
    #[error(
        "Yellowmoon couldn't save the preset to {folder}. Check that you can save there and that the disk has free space, then try again. Nothing was changed."
    )]
    Write {
        /// The folder.
        folder: String,
        /// What the system said.
        #[source]
        source: std::io::Error,
    },
    /// A file could not be read.
    #[error("Yellowmoon couldn't read {file}. Check that it is still there, then try again.")]
    Read {
        /// The file's name.
        file: String,
        /// What the system said.
        #[source]
        source: std::io::Error,
    },
    /// Another process holds the presets folder.
    #[error(
        "Another Yellowmoon is still changing the presets. Wait for it to finish, then try again."
    )]
    Busy,
    /// The file could not be moved to the Trash.
    #[error(transparent)]
    Trash(#[from] ym_looks::import::trash::TrashError),
    /// A command that names a photo failed.
    #[error(transparent)]
    Session(#[from] SessionError),
}

fn name_sentence(problem: NameProblem) -> &'static str {
    match problem {
        NameProblem::Empty => EMPTY_NAME_SENTENCE,
        NameProblem::TooLong => LONG_NAME_SENTENCE,
        NameProblem::Control => "Use a name without control characters.",
    }
}

impl PresetError {
    /// The CLI exit code (`docs/30` section 11): 3 for a preset that is missing, 2 for a usage mistake or a file that is
    /// refused, 1 otherwise.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        match self {
            Self::NotFound(_) | Self::PathNotFound(_) => 3,
            Self::NotAPreset
            | Self::TooNew
            | Self::TooLarge
            | Self::Settings(_)
            | Self::InvalidName(_)
            | Self::NoGroups
            | Self::NoSettings
            | Self::Exists(_)
            | Self::Ambiguous { .. }
            | Self::SettingsFile(_) => 2,
            _ => 1,
        }
    }
}

/// Trims a name and checks it: 1 to 64 characters, no control characters.
///
/// # Errors
///
/// [`PresetError::InvalidName`].
pub fn normalise_name(name: &str) -> Result<String, PresetError> {
    let name = name.trim();
    if name.is_empty() {
        return Err(PresetError::InvalidName(NameProblem::Empty));
    }
    if name.chars().count() > MAX_NAME_CHARS {
        return Err(PresetError::InvalidName(NameProblem::TooLong));
    }
    if name.chars().any(char::is_control) {
        return Err(PresetError::InvalidName(NameProblem::Control));
    }
    Ok(name.to_owned())
}

/// Whether two names are one name: equal ignoring case (`to_lowercase`).
#[must_use]
pub fn same_name(a: &str, b: &str) -> bool {
    a == b || a.to_lowercase() == b.to_lowercase()
}

// ---------------------------------------------------------------------------------------------
// Groups
// ---------------------------------------------------------------------------------------------

/// One of the eight groups of fields a preset holds (`docs/30` section 2). The ids are the JSON names; the order is the order
/// of the dialog, the file and the CLI's help.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PresetGroup {
    /// `look` and `look_intensity`.
    Look,
    /// `lens`.
    Lens,
    /// `exposure`, `contrast`, `brightness`, `highlights`, `shadows`, `white` and `black`.
    Light,
    /// `light_source`, `white_balance`, `color_balance`, `saturation` and `color_warp`.
    Color,
    /// `sharpen`, `noise_reduction` and `output_sharpening`.
    Detail,
    /// `curve`.
    Curve,
    /// `decode` and `rolloff`.
    Raw,
    /// Every node but the Input, Main and the Output, the links, Main's `name`, `bypass`, `trim` and `masks`, and the three
    /// nodes' `pos`.
    Nodes,
}

impl PresetGroup {
    /// The eight groups, in the order of `docs/30` section 2.
    pub const ALL: [PresetGroup; 8] = [
        Self::Look,
        Self::Lens,
        Self::Light,
        Self::Color,
        Self::Detail,
        Self::Curve,
        Self::Raw,
        Self::Nodes,
    ];

    /// The id (`look`, `lens`, `light`, `color`, `detail`, `curve`, `raw`, `nodes`).
    #[must_use]
    pub fn id(self) -> &'static str {
        match self {
            Self::Look => "look",
            Self::Lens => "lens",
            Self::Light => "light",
            Self::Color => "color",
            Self::Detail => "detail",
            Self::Curve => "curve",
            Self::Raw => "raw",
            Self::Nodes => "nodes",
        }
    }

    /// The label the person reads (British spelling).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Look => "Look",
            Self::Lens => "Lens",
            Self::Light => "Light",
            Self::Color => "Colour",
            Self::Detail => "Detail",
            Self::Curve => "Curve",
            Self::Raw => "RAW",
            Self::Nodes => "Nodes",
        }
    }

    /// The group with this id.
    #[must_use]
    pub fn from_id(id: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|g| g.id() == id)
    }

    /// The fields of the group that live in the Input, Main and the Output. [`PresetGroup::Nodes`] also holds every other
    /// node whole and the links, which are not listed here.
    #[must_use]
    pub fn fields(self) -> Vec<(NodeId, SettingsField)> {
        use SettingsField as F;
        let (input, main, output) = (NodeId::Input, NodeId::MAIN, NodeId::Output);
        match self {
            Self::Look => vec![(output, F::Look), (output, F::LookIntensity)],
            Self::Lens => vec![(input, F::Lens)],
            Self::Light => vec![
                (input, F::Exposure),
                (main, F::Contrast),
                (main, F::Brightness),
                (main, F::Highlights),
                (main, F::Shadows),
                (main, F::White),
                (main, F::Black),
            ],
            Self::Color => vec![
                (input, F::LightSource),
                (input, F::WhiteBalance),
                (main, F::ColorBalance),
                (main, F::Saturation),
                (main, F::ColorWarp),
            ],
            Self::Detail => vec![
                (input, F::Sharpen),
                (input, F::NoiseReduction),
                (output, F::OutputSharpening),
            ],
            Self::Curve => vec![(main, F::Curve)],
            Self::Raw => vec![(input, F::Decode), (input, F::Rolloff)],
            Self::Nodes => vec![
                (main, F::Name),
                (main, F::Bypass),
                (main, F::Trim),
                (main, F::Masks),
                (input, F::Pos),
                (main, F::Pos),
                (output, F::Pos),
            ],
        }
    }
}

/// The groups as a sorted set: in the order of `docs/30` section 2, each once.
fn sorted_groups(groups: &[PresetGroup]) -> Vec<PresetGroup> {
    groups
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// The patch that holds the fields of `groups` as `settings` has them: the Input's, Main's and the Output's fields of each
/// group (`SettingsPatch::of_fields`) and, with [`PresetGroup::Nodes`], every other node whole and the links.
fn group_patch(settings: &DevelopSettings, groups: &[PresetGroup]) -> SettingsPatch {
    let mut by_node: BTreeMap<NodeId, Vec<SettingsField>> = BTreeMap::new();
    for g in groups {
        for (node, field) in g.fields() {
            by_node.entry(node).or_default().push(field);
        }
    }
    let mut patch = SettingsPatch::default();
    for (node, fields) in by_node {
        patch
            .nodes
            .extend(SettingsPatch::of_fields(settings, node, &fields).nodes);
    }
    if groups.contains(&PresetGroup::Nodes) {
        let g = &settings.nodes;
        for (id, c) in &g.extra {
            patch.nodes.insert(
                NodeId::Corrector(*id),
                crate::settings::NodePatch::whole_corrector(c),
            );
        }
        for (id, f) in &g.films {
            patch.nodes.insert(
                NodeId::Corrector(*id),
                crate::settings::NodePatch::whole_film(f),
            );
        }
        for (id, f) in &g.faces {
            patch.nodes.insert(
                NodeId::Corrector(*id),
                crate::settings::NodePatch::whole_face_refinement(f),
            );
        }
        for (id, u) in &g.uniforms {
            patch.nodes.insert(
                NodeId::Corrector(*id),
                crate::settings::NodePatch::whole_color_uniform(u),
            );
        }
        for (id, m) in &g.mixers {
            patch.nodes.insert(
                NodeId::Corrector(*id),
                crate::settings::NodePatch::whole_mixer(m),
            );
        }
        patch.links = Some(g.links.clone());
    }
    patch
}

/// Empties what a preset never holds (`docs/30` section 3): the Object rows' clicks and the Brush rows' strokes, in every node
/// that has masks (Face Refinement's too), and the Protect strokes of Face Refinement nodes.
fn empty_rows_for_preset(settings: &mut DevelopSettings) {
    let g = &mut settings.nodes;
    let masks = std::iter::once(&mut g.main.masks)
        .chain(g.extra.values_mut().map(|c| &mut c.masks))
        .chain(g.films.values_mut().map(|f| &mut f.masks))
        .chain(g.uniforms.values_mut().map(|u| &mut u.masks))
        .chain(g.faces.values_mut().map(|f| &mut f.masks));
    for m in masks {
        for row in &mut m.items {
            *row = row.emptied_for_preset();
        }
    }
    // Protect strokes are painted over one face's moles and scars: like a Brush row's, they are not saved.
    for face in g.faces.values_mut() {
        face.protect.strokes.clear();
    }
}

/// A photo's settings as Copy settings takes them, with the rows emptied: the document a preset is cut from.
fn prepared_from_resolved(source: &ResolvedSettings) -> Result<DevelopSettings, PresetError> {
    let mut s = DevelopSettings::default();
    s.apply_patch(&SettingsPatch::copied_from(source), None)
        .map_err(PresetError::CouldNotMake)?;
    empty_rows_for_preset(&mut s);
    Ok(s)
}

/// A settings document (no photo, so no picked white to resolve) as a preset is cut from it: the Input's crop, orientation
/// and white pick left out, the rows emptied.
fn prepared_from_settings(settings: &DevelopSettings) -> DevelopSettings {
    let mut s = settings.clone();
    let default = crate::settings::InputNode::default();
    let input = s.input_mut();
    input.crop = default.crop;
    input.orientation = default.orientation;
    input.white_pick = default.white_pick;
    empty_rows_for_preset(&mut s);
    s
}

/// The groups in which `prepared` differs from `defaults`, per group over the group's fields.
fn changed_in(prepared: &DevelopSettings, defaults: &DevelopSettings) -> Vec<PresetGroup> {
    PresetGroup::ALL
        .into_iter()
        .filter(|g| group_patch(prepared, &[*g]) != group_patch(defaults, &[*g]))
        .collect()
}

/// The groups the Save dialog ticks for a photo (`docs/30` section 4.1): those in which the photo's stored settings differ
/// from `defaults`, the default graph with the effective default look. Auto equals Auto, so a photo that Apple Auto graded
/// and the person never touched ticks nothing, and a picked white (saved as a manual white balance) ticks Colour.
#[must_use]
pub fn changed_groups(source: &ResolvedSettings, defaults: &DevelopSettings) -> Vec<PresetGroup> {
    match prepared_from_resolved(source) {
        Ok(prepared) => changed_in(&prepared, defaults),
        Err(_) => Vec::new(),
    }
}

/// [`changed_groups`] for settings that came from a file: no photo, so a picked white stays as it is.
#[must_use]
pub fn changed_groups_of_settings(
    settings: &DevelopSettings,
    defaults: &DevelopSettings,
) -> Vec<PresetGroup> {
    changed_in(&prepared_from_settings(settings), defaults)
}

// ---------------------------------------------------------------------------------------------
// The preset
// ---------------------------------------------------------------------------------------------

/// The look of a preset that has the `look` group: its id, and its name so that a missing look can be named.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PresetLook {
    /// The look's id (`builtin:signature`, `user:film-warm`).
    pub id: LookId,
    /// The look's name when the preset was saved ("Film warm").
    pub name: String,
}

/// A saved recipe: a name and the settings of some groups.
#[derive(Debug, Clone, PartialEq)]
pub struct Preset {
    /// The name as typed, trimmed: 1 to 64 characters.
    pub name: String,
    /// The groups the preset holds: sorted, each once, at least one.
    pub groups: Vec<PresetGroup>,
    /// The look, present only with the `look` group.
    pub look: Option<PresetLook>,
    /// The whole settings document read over the default graph. The groups say which fields count; the rest are ignored.
    pub settings: DevelopSettings,
}

/// What applying a preset to a photo is: the patch, and the look that was left out because it is not in the library.
#[derive(Debug, Clone, PartialEq)]
pub struct PresetPatch {
    /// The node-level patch (`docs/23` section 3.7) to give to [`Session::paste_settings`] or
    /// [`DevelopSettings::apply_patch`].
    ///
    /// [`Session::paste_settings`]: crate::session::Session::paste_settings
    pub patch: SettingsPatch,
    /// The name of the look the preset has and the library has not (its fields are left out of the patch), if any.
    pub missing_look: Option<String>,
}

impl Preset {
    /// Makes a preset from a photo's settings (`docs/30` section 4.1): Copy settings' patch ([`SettingsPatch::copied_from`]),
    /// the Object and Brush rows emptied ([`crate::settings::MaskItem::emptied_for_preset`]), cut to `groups` over the default
    /// graph. `look_name` is the look's name when the `look` group is chosen (its id stands in when it is `None`).
    ///
    /// # Errors
    ///
    /// [`PresetError::InvalidName`] and [`PresetError::NoGroups`].
    pub fn from_photo(
        name: &str,
        source: &ResolvedSettings,
        groups: &[PresetGroup],
        look_name: Option<&str>,
    ) -> Result<Self, PresetError> {
        Self::cut(name, &prepared_from_resolved(source)?, groups, look_name)
    }

    /// [`Preset::from_photo`] for a settings document that is not a photo's (`ym presets save --from settings.json`): a
    /// picked white has no estimates to resolve it, so the white pick is left out like the crop and the orientation.
    ///
    /// # Errors
    ///
    /// [`PresetError::InvalidName`], [`PresetError::NoGroups`].
    pub fn from_settings(
        name: &str,
        settings: &DevelopSettings,
        groups: &[PresetGroup],
        look_name: Option<&str>,
    ) -> Result<Self, PresetError> {
        Self::cut(name, &prepared_from_settings(settings), groups, look_name)
    }

    fn cut(
        name: &str,
        prepared: &DevelopSettings,
        groups: &[PresetGroup],
        look_name: Option<&str>,
    ) -> Result<Self, PresetError> {
        let name = normalise_name(name)?;
        let groups = sorted_groups(groups);
        if groups.is_empty() {
            return Err(PresetError::NoGroups);
        }
        let mut settings = DevelopSettings::default();
        settings
            .apply_patch(&group_patch(prepared, &groups), None)
            .map_err(PresetError::CouldNotMake)?;
        let look = groups.contains(&PresetGroup::Look).then(|| {
            let id = settings.output().look.clone();
            PresetLook {
                name: look_name.map_or_else(|| id.as_str().to_owned(), str::to_owned),
                id,
            }
        });
        Ok(Self {
            name,
            groups,
            look,
            settings,
        })
    }

    /// Whether the preset holds this group.
    #[must_use]
    pub fn has(&self, group: PresetGroup) -> bool {
        self.groups.contains(&group)
    }

    /// The patch that applies the preset to `target` (`docs/30` section 4.2): the fields of its groups, and with `nodes` every
    /// other node whole, the links, and `remove` for the target's other nodes the preset does not have. Without `nodes` the
    /// target's nodes and links stay. The Input's crop, orientation and white pick are never touched. When the `look` group's
    /// look is not usable (`look_usable` false) the look fields are left out and the look is named in
    /// [`PresetPatch::missing_look`]; the rest applies.
    #[must_use]
    pub fn patch_for(&self, target: &DevelopSettings, look_usable: bool) -> PresetPatch {
        let mut patch = group_patch(&self.settings, &self.groups);
        let mut missing_look = None;
        if self.has(PresetGroup::Look) && !look_usable {
            if let Some(output) = patch.nodes.get_mut(&NodeId::Output) {
                output.look = None;
                output.look_intensity = None;
                if *output == crate::settings::NodePatch::default() {
                    patch.nodes.remove(&NodeId::Output);
                }
            }
            missing_look = Some(self.look.as_ref().map_or_else(
                || self.settings.output().look.as_str().to_owned(),
                |l| l.name.clone(),
            ));
        }
        PresetPatch {
            patch: patch.pasted_onto(target),
            missing_look,
        }
    }

    /// Whether applying the preset to `target` would change nothing (`docs/30` section 4.3): the ring on a tile. A preset with a
    /// missing look matches when everything else matches.
    #[must_use]
    pub fn matches(&self, target: &DevelopSettings, look_usable: bool) -> bool {
        let patch = self.patch_for(target, look_usable).patch;
        let mut copy = target.clone();
        copy.apply_patch(&patch, None).is_ok() && copy == *target
    }

    /// The settings after the preset is applied to `settings`, or the error that stops it.
    ///
    /// # Errors
    ///
    /// [`PresetError::Settings`] when the patch cannot be applied.
    pub fn applied_to(
        &self,
        settings: &DevelopSettings,
        look_usable: bool,
    ) -> Result<DevelopSettings, PresetError> {
        let patch = self.patch_for(settings, look_usable).patch;
        let mut next = settings.clone();
        next.apply_patch(&patch, None)
            .map_err(PresetError::Settings)?;
        Ok(next)
    }

    // --- The file ---------------------------------------------------------------------------

    /// The file: UTF-8 JSON, compact, keys sorted, one newline at the end (`docs/30` section 5). The same preset is the same
    /// bytes. The `settings` hold only the fields of the groups.
    ///
    /// # Errors
    ///
    /// [`PresetError::TooLarge`] for a preset above 4 MiB, which could not be read back.
    pub fn to_json(&self) -> Result<String, PresetError> {
        let mut doc = Map::new();
        doc.insert("yellowmoon_preset".to_owned(), Value::from(FORMAT_VERSION));
        doc.insert("name".to_owned(), Value::String(self.name.clone()));
        doc.insert(
            "groups".to_owned(),
            Value::Array(
                self.groups
                    .iter()
                    .map(|g| Value::String(g.id().to_owned()))
                    .collect(),
            ),
        );
        if let (true, Some(look)) = (self.has(PresetGroup::Look), &self.look) {
            let mut l = Map::new();
            l.insert("id".to_owned(), Value::String(look.id.as_str().to_owned()));
            l.insert("name".to_owned(), Value::String(look.name.clone()));
            doc.insert("look".to_owned(), Value::Object(l));
        }
        doc.insert(
            "settings".to_owned(),
            pruned_settings(&self.settings, &self.groups)?,
        );
        let mut text = String::new();
        write_canonical(&Value::Object(doc), &mut text);
        text.push('\n');
        if text.len() > MAX_PRESET_BYTES {
            return Err(PresetError::TooLarge);
        }
        Ok(text)
    }

    /// Reads a preset file (`docs/30` section 5).
    ///
    /// * A newer version, an unknown top-level field and settings with a field this version does not know are
    ///   [`PresetError::TooNew`], decided before anything else is looked at.
    /// * A file that is not JSON, has no `yellowmoon_preset` or no `settings`, has a name or a group that is not valid, or holds
    ///   the flat settings of Yellowmoon 0.1 is [`PresetError::NotAPreset`].
    /// * An older preset (fields missing) reads with the defaults; numbers out of range are clamped.
    ///
    /// # Errors
    ///
    /// [`PresetError`] as above, and [`PresetError::TooLarge`] for text above 4 MiB.
    pub fn from_json(text: &str) -> Result<Self, PresetError> {
        if text.len() > MAX_PRESET_BYTES {
            return Err(PresetError::TooLarge);
        }
        let Ok(Value::Object(mut doc)) = serde_json::from_str::<Value>(text) else {
            return Err(PresetError::NotAPreset);
        };
        // The version first: nothing is guessed from a file made by a newer version.
        match doc.remove("yellowmoon_preset").and_then(|v| v.as_u64()) {
            Some(FORMAT_VERSION) => {}
            Some(v) if v > FORMAT_VERSION => return Err(PresetError::TooNew),
            _ => return Err(PresetError::NotAPreset),
        }
        let name = match doc.remove("name") {
            Some(Value::String(s)) => normalise_name(&s).map_err(|_| PresetError::NotAPreset)?,
            _ => return Err(PresetError::NotAPreset),
        };
        let groups = match doc.remove("groups") {
            Some(Value::Array(items)) => {
                let mut groups = Vec::new();
                for item in items {
                    let group = item
                        .as_str()
                        .and_then(PresetGroup::from_id)
                        .ok_or(PresetError::NotAPreset)?;
                    if groups.contains(&group) {
                        return Err(PresetError::NotAPreset);
                    }
                    groups.push(group);
                }
                groups
            }
            _ => return Err(PresetError::NotAPreset),
        };
        if groups.is_empty() {
            return Err(PresetError::NotAPreset);
        }
        let groups = sorted_groups(&groups);
        let look = doc.remove("look");
        let settings = doc.remove("settings");
        if !doc.is_empty() {
            return Err(PresetError::TooNew);
        }
        let Some(settings @ Value::Object(_)) = settings else {
            return Err(PresetError::NotAPreset);
        };
        let settings = match DevelopSettings::from_value(settings, &LookId::neutral()) {
            Ok((s, _)) => s,
            Err(SettingsError::OldForm) => return Err(PresetError::NotAPreset),
            Err(SettingsError::Parse(_)) => return Err(PresetError::TooNew),
            Err(e) => return Err(PresetError::Settings(e)),
        };
        let has_look = groups.contains(&PresetGroup::Look);
        let look = match (has_look, look) {
            (true, Some(Value::Object(l))) => {
                let name = match l.get("name") {
                    Some(Value::String(s)) => s.clone(),
                    _ => return Err(PresetError::NotAPreset),
                };
                match l.get("id") {
                    Some(Value::String(s)) if !s.is_empty() && l.len() == 2 => {
                        LookId::parse(s).map_err(|_| PresetError::NotAPreset)?;
                    }
                    _ => return Err(PresetError::NotAPreset),
                }
                // The id is the settings' own look.
                Some(PresetLook {
                    id: settings.output().look.clone(),
                    name,
                })
            }
            (false, None) => None,
            _ => return Err(PresetError::NotAPreset),
        };
        Ok(Self {
            name,
            groups,
            look,
            settings,
        })
    }
}

/// The settings document with only the fields of `groups` (`docs/30` section 5): the Input, Main and the Output hold the fields of
/// their groups (and are left out when they hold none), the other nodes and the links are there with `nodes`.
fn pruned_settings(
    settings: &DevelopSettings,
    groups: &[PresetGroup],
) -> Result<Value, PresetError> {
    let full = serde_json::to_value(settings)
        .map_err(|e| PresetError::Settings(SettingsError::Parse(e.to_string())))?;
    let mut allowed: BTreeMap<NodeId, HashSet<SettingsField>> = BTreeMap::new();
    for g in groups {
        for (node, field) in g.fields() {
            allowed.entry(node).or_default().insert(field);
        }
    }
    let with_nodes = groups.contains(&PresetGroup::Nodes);
    let mut nodes = Map::new();
    let items = full
        .pointer("/nodes/items")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut kept = Vec::new();
    for item in items {
        let Value::Object(map) = item else { continue };
        let id = match map.get("id") {
            Some(Value::String(s)) => NodeId::from_str(s).ok(),
            Some(Value::Number(n)) => n
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .map(NodeId::Corrector),
            _ => None,
        };
        let Some(id) = id else { continue };
        if matches!(id, NodeId::Input | NodeId::Output | NodeId::MAIN) {
            let allow = allowed.get(&id).cloned().unwrap_or_default();
            if let Some(item) = prune_node(map, &allow) {
                kept.push(item);
            }
        } else if with_nodes {
            kept.push(Value::Object(map));
        }
    }
    nodes.insert("items".to_owned(), Value::Array(kept));
    if with_nodes && let Some(links) = full.pointer("/nodes/links") {
        nodes.insert("links".to_owned(), links.clone());
    }
    let mut doc = Map::new();
    doc.insert("nodes".to_owned(), Value::Object(nodes));
    Ok(Value::Object(doc))
}

/// One of the fixed nodes with only the allowed fields; `None` when none is left.
fn prune_node(map: Map<String, Value>, allow: &HashSet<SettingsField>) -> Option<Value> {
    let field_of = |name: &str| SettingsField::ALL.into_iter().find(|f| f.name() == name);
    let mut out = Map::new();
    let mut any = false;
    for (key, value) in map {
        match key.as_str() {
            "id" | "kind" => {
                out.insert(key, value);
            }
            "grade" => {
                let Value::Object(grade) = value else {
                    continue;
                };
                let kept: Map<String, Value> = grade
                    .into_iter()
                    .filter(|(k, _)| field_of(k).is_some_and(|f| allow.contains(&f)))
                    .collect();
                if !kept.is_empty() {
                    any = true;
                    out.insert(key, Value::Object(kept));
                }
            }
            other => {
                if field_of(other).is_some_and(|f| allow.contains(&f)) {
                    any = true;
                    out.insert(key, value);
                }
            }
        }
    }
    any.then_some(Value::Object(out))
}

/// Writes `value` as compact JSON with the keys of every object sorted, so that a preset saved twice is the same bytes whatever
/// order the JSON library keeps its maps in.
pub(crate) fn write_canonical(value: &Value, out: &mut String) {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push('{');
            for (i, key) in keys.into_iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&Value::String(key.clone()).to_string());
                out.push(':');
                if let Some(v) = map.get(key) {
                    write_canonical(v, out);
                }
            }
            out.push('}');
        }
        Value::Array(items) => {
            out.push('[');
            for (i, v) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_canonical(v, out);
            }
            out.push(']');
        }
        scalar => out.push_str(&scalar.to_string()),
    }
}
