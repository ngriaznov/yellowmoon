//! From options and a settings file to the `DevelopSettings` of each photo.
//!
//! A photo starts as the session opens it: automatic everywhere, with the default look. Then, in
//! this order,
//!
//! 1. `--settings` replaces the settings wholesale (the document is a graph of nodes read over the
//!    default graph, so it may hold only what it changes, and its numbers are clamped like a
//!    slider's; a document in the flat form of Yellowmoon 0.1 is refused);
//! 2. the options of the command line are applied as a node patch on top (each sets one field of
//!    the Input, Main or the Output, `docs/10` *Where the flags land*), so an option given alongside
//!    the file wins. `--exposure-offset` is the one option that depends on the photo: it becomes
//!    `Manual(EV_auto + offset)` with that photo's own automatic exposure.
//!
//! Nothing here is saved anywhere: the settings live with the photo for this run (`docs/10`).

use std::collections::BTreeMap;
use std::path::Path;

use serde_json::Value;
use ym_color::wb::LightSource;
use ym_core::settings::{
    AutoOr, DecodePatch, DevelopSettings, ExposureSetting, FieldRef, GradePatch, LensKind,
    LensPatch, NodeId, NodePatch, NoiseReductionManualPatch, PercentSetting, RolloffPatch,
    SettingsError, SettingsPatch, SharpenManualPatch, WhiteBalanceManualPatch,
};
use ym_looks::library::LookId;

use crate::cli::{
    ExposureArg, LightArg, NrArg, PercentArg, RenderArgs, RolloffArg, SharpenArg, WbArg,
};

/// `--contrast`, `--saturation`, `--highlights`, `--shadows` or `--black` as a setting: `auto` is
/// Auto (0, or Apple Auto's value for the photo, `docs/19` section 5), a number is a manual value.
fn percent_setting(arg: PercentArg) -> PercentSetting {
    match arg {
        PercentArg::Auto => AutoOr::Auto,
        PercentArg::Value(value) => PercentSetting::manual_value(value),
    }
}

/// `--lens` resolved against the lens library: the lens and the Flares and Edges the Lens card moves
/// to when it is chosen (`LensInfo`, `docs/21` section 12, decision 6).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LensChoice {
    /// The lens.
    pub kind: LensKind,
    /// Its starting Flares, used unless `--flares` is given.
    pub start_flares: f64,
    /// Its starting Edges, used unless `--edges` is given.
    pub start_edges: f64,
}

/// The options of `ym render` that change a photo's settings.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Overrides {
    /// `--look`, already resolved to an id.
    pub look: Option<LookId>,
    /// `--intensity`.
    pub intensity: Option<f64>,
    /// `--exposure`.
    pub exposure: Option<ExposureArg>,
    /// `--exposure-offset`.
    pub exposure_offset: Option<f64>,
    /// `--contrast`.
    pub contrast: Option<PercentArg>,
    /// `--brightness`.
    pub brightness: Option<f64>,
    /// `--highlights`.
    pub highlights: Option<PercentArg>,
    /// `--shadows`.
    pub shadows: Option<PercentArg>,
    /// `--white`.
    pub white: Option<f64>,
    /// `--black`.
    pub black: Option<PercentArg>,
    /// `--saturation`.
    pub saturation: Option<PercentArg>,
    /// `--wb`.
    pub wb: Option<WbArg>,
    /// `--light`.
    pub light: Option<LightArg>,
    /// `--nr`.
    pub nr: Option<NrArg>,
    /// `--sharpen`.
    pub sharpen: Option<SharpenArg>,
    /// `--output-sharpening`.
    pub output_sharpening: Option<f64>,
    /// `--no-neural`.
    pub no_neural: bool,
    /// `--no-highlight-recovery`.
    pub no_highlight_recovery: bool,
    /// `--no-lens`.
    pub no_lens: bool,
    /// `--no-ca`.
    pub no_ca: bool,
    /// `--rolloff`.
    pub rolloff: Option<RolloffArg>,
    /// `--lens`, resolved.
    pub lens: Option<LensChoice>,
    /// `--flares`.
    pub flares: Option<f64>,
    /// `--edges`.
    pub edges: Option<f64>,
    /// `--anamorphic`.
    pub anamorphic: Option<f64>,
}

impl Overrides {
    /// The overrides of a `render` command; `look` is the id `--look` resolved to and `lens` the
    /// lens `--lens` resolved to.
    pub fn from_args(args: &RenderArgs, look: Option<LookId>, lens: Option<LensChoice>) -> Self {
        Self {
            look,
            intensity: args.intensity,
            exposure: args.exposure,
            exposure_offset: args.exposure_offset,
            contrast: args.contrast,
            brightness: args.brightness,
            highlights: args.highlights,
            shadows: args.shadows,
            white: args.white,
            black: args.black,
            saturation: args.saturation,
            wb: args.wb,
            light: args.light,
            nr: args.nr,
            sharpen: args.sharpen,
            output_sharpening: args.output_sharpening,
            no_neural: args.no_neural,
            no_highlight_recovery: args.no_highlight_recovery,
            no_lens: args.no_lens,
            no_ca: args.no_ca,
            rolloff: args.rolloff,
            lens,
            flares: args.flares,
            edges: args.edges,
            anamorphic: args.anamorphic,
        }
    }

    /// The `lens` part of the patch. A chosen lens brings its starting Flares and Edges, as the card
    /// sends them, unless an amount is given; an amount alone changes that amount only.
    fn lens_patch(&self) -> Option<LensPatch> {
        if self.lens.is_none()
            && self.flares.is_none()
            && self.edges.is_none()
            && self.anamorphic.is_none()
        {
            return None;
        }
        Some(LensPatch {
            kind: self.lens.map(|l| l.kind),
            flares: self.flares.or(self.lens.map(|l| l.start_flares)),
            edges: self.edges.or(self.lens.map(|l| l.start_edges)),
            anamorphic: self.anamorphic,
        })
    }

    /// The patch for one photo whose automatic exposure is `auto_exposure_ev`. Each option sets a
    /// field of one node (`docs/10` *Where the flags land*): the Input, Main (node 1) or the
    /// Output.
    #[must_use]
    pub fn patch(&self, auto_exposure_ev: f64) -> SettingsPatch {
        // The Output: the look and the output sharpening.
        let output = NodePatch {
            look: self.look.clone(),
            look_intensity: self.intensity,
            output_sharpening: self.output_sharpening,
            ..NodePatch::default()
        };
        // Main: the grade. `auto` is Auto, a number a manual value (`docs/19` section 8).
        let grade = GradePatch {
            contrast: self.contrast.map(percent_setting),
            brightness: self.brightness,
            highlights: self.highlights.map(percent_setting),
            shadows: self.shadows.map(percent_setting),
            white: self.white,
            black: self.black.map(percent_setting),
            saturation: self.saturation.map(percent_setting),
            ..GradePatch::default()
        };
        // The Input: the raw, light, lens and detail settings.
        let mut input = NodePatch {
            white_balance: self.wb.map(|wb| match wb {
                WbArg::Auto => AutoOr::Auto,
                WbArg::Manual { kelvin, tint } => AutoOr::Manual(WhiteBalanceManualPatch {
                    kelvin: Some(kelvin),
                    tint,
                }),
            }),
            light_source: self.light.map(|l| match l {
                LightArg::Auto => LightSource::Auto,
                LightArg::Day => LightSource::Day,
                LightArg::Shade => LightSource::Shade,
                LightArg::Tungsten => LightSource::Tungsten,
                LightArg::Led => LightSource::Led,
                LightArg::Fluorescent => LightSource::Fluorescent,
            }),
            noise_reduction: self.nr.map(|nr| match nr {
                NrArg::Auto => AutoOr::Auto,
                NrArg::Manual {
                    luminance,
                    color,
                    detail,
                } => AutoOr::Manual(NoiseReductionManualPatch {
                    luminance: Some(luminance),
                    color: Some(color),
                    detail: Some(detail),
                }),
            }),
            sharpen: self.sharpen.map(|s| match s {
                SharpenArg::Auto => AutoOr::Auto,
                SharpenArg::Manual { amount, radius_px } => AutoOr::Manual(SharpenManualPatch {
                    amount: Some(amount),
                    radius_px: Some(radius_px),
                }),
            }),
            rolloff: self.rolloff.map(|r| RolloffPatch {
                highlights: Some(r.highlights),
                low_light: Some(r.low_light),
            }),
            lens: self.lens_patch(),
            ..NodePatch::default()
        };
        input.exposure = match (self.exposure, self.exposure_offset) {
            (Some(ExposureArg::Auto), _) => Some(AutoOr::Auto),
            (Some(ExposureArg::Ev(ev)), _) => Some(ExposureSetting::manual_ev(ev)),
            (None, Some(offset)) => Some(ExposureSetting::manual_ev(auto_exposure_ev + offset)),
            (None, None) => None,
        };
        let decode = DecodePatch {
            neural_demosaic: self.no_neural.then_some(false),
            highlight_recovery: self.no_highlight_recovery.then_some(false),
            lens_corrections: self.no_lens.then_some(false),
            chromatic_aberration: self.no_ca.then_some(false),
        };
        if decode != DecodePatch::default() {
            input.decode = Some(decode);
        }
        let mut patch = SettingsPatch::default();
        if input != NodePatch::default() {
            patch.nodes.insert(NodeId::Input, input);
        }
        if grade != GradePatch::default() {
            patch.nodes.insert(
                NodeId::MAIN,
                NodePatch {
                    grade: Some(grade),
                    ..NodePatch::default()
                },
            );
        }
        if output != NodePatch::default() {
            patch.nodes.insert(NodeId::Output, output);
        }
        patch
    }

    /// True when no option changes a photo's settings.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// A `--settings` document, checked.
#[derive(Debug, Clone, PartialEq)]
pub enum SettingsFile {
    /// One document for every photo.
    Global(Box<DevelopSettings>),
    /// One document per file name.
    PerFile(BTreeMap<String, DevelopSettings>),
}

/// A settings document read, with the fields that were out of range and were limited to it.
#[derive(Debug, Clone, PartialEq)]
pub struct LoadedSettings {
    /// The settings.
    pub file: SettingsFile,
    /// `(entry, field names)` for every document that had values out of range; the entry is empty
    /// for a global document. A field is named `<node>.<field>`, for example `1.grade.contrast`.
    pub clamped: Vec<(String, Vec<String>)>,
}

/// Whether a key of a per-file settings object can name a photo: photos are ARW files.
fn is_file_key(key: &str) -> bool {
    key.to_ascii_lowercase().ends_with(".arw")
}

impl SettingsFile {
    /// Reads and checks a settings file. `default_look` is the look a document that names none
    /// gets (the user's default look).
    ///
    /// A JSON object whose only key is `nodes` (or that is empty) is one document for every photo;
    /// an object whose keys are all names of ARW files is a map from file names to documents. A
    /// document in the flat form of Yellowmoon 0.1 is refused with its own sentence
    /// ([`SettingsError::OldForm`]).
    ///
    /// # Errors
    ///
    /// A sentence saying what is wrong with the file, naming the entry when there are several.
    pub fn parse(text: &str, default_look: &LookId) -> Result<LoadedSettings, String> {
        let value: Value = serde_json::from_str(text)
            .map_err(|e| format!("The settings file is not valid JSON: {e}."))?;
        let Value::Object(map) = value else {
            return Err(
                "The settings file must hold a JSON object: one settings document, or an object keyed by file name."
                    .to_owned(),
            );
        };
        let describe = |fields: Vec<FieldRef>| -> Vec<String> {
            fields.into_iter().map(|f| f.to_string()).collect()
        };
        // The form of Yellowmoon 0.1: every key is a field of it. Nothing was saved in 0.1, so the
        // file is refused, not read.
        let (graph, others): (Vec<&String>, Vec<&String>) =
            map.keys().partition(|k| k.as_str() == "nodes");
        // A key that is not `nodes` is a photo's file name when it looks like one. A misspelt
        // `nodes` is neither, and must be an error as it is in a document (docs/01: "a typo is
        // not a silent no-op"): taken for a file name it would only earn the warning that no
        // photo has that name.
        let (names, unknown): (Vec<&String>, Vec<&String>) =
            others.into_iter().partition(|k| is_file_key(k));
        if graph.is_empty()
            && !map.is_empty()
            && map
                .keys()
                .all(|k| ym_core::settings::OLD_FORM_KEYS.contains(&k.as_str()))
        {
            return Err(SettingsError::OldForm.to_string());
        }
        if !unknown.is_empty() {
            return Err(format!(
                "The settings file has {} that {} not `nodes` or the name of an ARW file: {}. Check the spelling, or name the photo's file (`DSC00001.ARW`) to give that photo its own settings.",
                if unknown.len() == 1 { "a key" } else { "keys" },
                if unknown.len() == 1 { "is" } else { "are" },
                join(&unknown)
            ));
        }
        if !graph.is_empty() && !names.is_empty() {
            return Err(format!(
                "The settings file mixes `nodes` with file names ({}). Use one document, or an object keyed by file name.",
                join(&names)
            ));
        }
        if names.is_empty() {
            let (settings, clamped) = DevelopSettings::from_value(Value::Object(map), default_look)
                .map_err(|e| format!("The settings file can't be used: {e}"))?;
            let clamped = if clamped.is_empty() {
                Vec::new()
            } else {
                vec![(String::new(), describe(clamped))]
            };
            return Ok(LoadedSettings {
                file: Self::Global(Box::new(settings)),
                clamped,
            });
        }
        let mut per_file = BTreeMap::new();
        let mut all_clamped = Vec::new();
        for (name, doc) in map {
            let (settings, clamped) = DevelopSettings::from_value(doc, default_look)
                .map_err(|e| format!("The settings for {name} can't be used: {e}"))?;
            if !clamped.is_empty() {
                all_clamped.push((name.clone(), describe(clamped)));
            }
            per_file.insert(name, settings);
        }
        Ok(LoadedSettings {
            file: Self::PerFile(per_file),
            clamped: all_clamped,
        })
    }

    /// The settings document for a photo's file, if the file has one: the global document for
    /// every photo, else the entry named like the file (exactly, else without regard to case, else
    /// the full path).
    #[must_use]
    pub fn for_file(&self, path: &Path) -> Option<&DevelopSettings> {
        match self {
            Self::Global(s) => Some(s.as_ref()),
            Self::PerFile(map) => {
                let name = path.file_name()?.to_string_lossy();
                map.get(name.as_ref())
                    .or_else(|| {
                        map.iter()
                            .find(|(k, _)| k.eq_ignore_ascii_case(&name))
                            .map(|(_, v)| v)
                    })
                    .or_else(|| map.get(path.to_string_lossy().as_ref()))
            }
        }
    }

    /// The entries of a per-file map that name none of `files`.
    #[must_use]
    pub fn unused_entries(&self, files: &[std::path::PathBuf]) -> Vec<String> {
        match self {
            Self::Global(_) => Vec::new(),
            Self::PerFile(map) => map
                .keys()
                .filter(|k| {
                    !files.iter().any(|f| {
                        f.file_name()
                            .is_some_and(|n| n.to_string_lossy().eq_ignore_ascii_case(k))
                            || f.to_string_lossy() == k.as_str()
                    })
                })
                .cloned()
                .collect(),
        }
    }
}

fn join(items: &[&String]) -> String {
    items
        .iter()
        .map(|s| s.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use ym_core::settings::AutoOr;

    fn neutral() -> LookId {
        LookId::builtin("neutral")
    }

    fn vintage() -> LensChoice {
        LensChoice {
            kind: LensKind::Vintage,
            start_flares: 45.0,
            start_edges: 40.0,
        }
    }

    /// The Input's part of a patch (empty when the patch has none).
    fn input(p: &SettingsPatch) -> NodePatch {
        p.input_patch().cloned().unwrap_or_default()
    }

    /// Main's grade part of a patch (empty when the patch has none).
    fn grade(p: &SettingsPatch) -> GradePatch {
        p.main_patch()
            .and_then(|n| n.grade.clone())
            .unwrap_or_default()
    }

    /// The Output's part of a patch (empty when the patch has none).
    fn output(p: &SettingsPatch) -> NodePatch {
        p.output_patch().cloned().unwrap_or_default()
    }

    #[test]
    fn a_lens_brings_its_starting_amounts_and_an_amount_given_beats_them() {
        // No lens option, no lens in the patch.
        assert_eq!(input(&Overrides::default().patch(0.0)).lens, None);
        assert!(Overrides::default().is_empty());
        // The lens alone: the kind with its starting Flares and Edges, as the card sends them.
        let only_lens = Overrides {
            lens: Some(vintage()),
            ..Overrides::default()
        };
        assert!(!only_lens.is_empty());
        assert_eq!(
            input(&only_lens.patch(0.0)).lens,
            Some(LensPatch {
                kind: Some(LensKind::Vintage),
                flares: Some(45.0),
                edges: Some(40.0),
                anamorphic: None,
            })
        );
        // An amount beats the lens's start, for that amount only; Anamorphic is never a start.
        let overridden = Overrides {
            lens: Some(vintage()),
            flares: Some(10.0),
            anamorphic: Some(50.0),
            ..Overrides::default()
        };
        assert_eq!(
            input(&overridden.patch(0.0)).lens,
            Some(LensPatch {
                kind: Some(LensKind::Vintage),
                flares: Some(10.0),
                edges: Some(40.0),
                anamorphic: Some(50.0),
            })
        );
        // An amount alone changes that amount only and leaves the lens as it is.
        let amount = Overrides {
            edges: Some(0.0),
            ..Overrides::default()
        };
        assert_eq!(
            input(&amount.patch(0.0)).lens,
            Some(LensPatch {
                kind: None,
                flares: None,
                edges: Some(0.0),
                anamorphic: None,
            })
        );
    }

    #[test]
    fn a_settings_file_may_carry_a_lens() {
        let loaded = SettingsFile::parse(
            r#"{"nodes":{"items":[{"id":"input","lens":{"kind":"petzval","flares":140,"anamorphic":20}}]}}"#,
            &neutral(),
        )
        .expect("parses");
        let SettingsFile::Global(s) = &loaded.file else {
            panic!("global");
        };
        let lens = s.input().lens;
        assert_eq!(lens.kind, LensKind::Petzval);
        assert_eq!(
            (lens.flares, lens.edges, lens.anamorphic),
            (100.0, 0.0, 20.0)
        );
        assert_eq!(
            loaded.clamped,
            vec![(String::new(), vec!["input.lens".to_owned()])]
        );
        assert!(
            SettingsFile::parse(
                r#"{"nodes":{"items":[{"id":"input","lens":{"kind":"leica"}}]}}"#,
                &neutral()
            )
            .is_err()
        );
    }

    #[test]
    fn a_document_of_nodes_is_global_and_may_name_only_what_it_changes() {
        let loaded = SettingsFile::parse(
            r#"{"nodes":{"items":[
                {"id":1,"grade":{"contrast":{"kind":"manual","value":20}}},
                {"id":"input","exposure":{"kind":"manual","ev":0.5}}]}}"#,
            &neutral(),
        )
        .expect("parses");
        let SettingsFile::Global(s) = &loaded.file else {
            panic!("global");
        };
        assert_eq!(s.main_grade().contrast, PercentSetting::manual_value(20.0));
        assert_eq!(s.input().exposure, ExposureSetting::manual_ev(0.5));
        assert!(s.input().sharpen.is_auto());
        assert_eq!(s.output().look, neutral());
        assert_eq!(s.nodes.links, ym_core::settings::NodeGraph::default_links());
        assert!(loaded.clamped.is_empty());
        // An empty file is the default graph, for every photo.
        assert!(matches!(
            SettingsFile::parse("{}", &neutral()).expect("empty").file,
            SettingsFile::Global(_)
        ));
    }

    #[test]
    fn a_document_without_a_look_gets_the_default_look() {
        let sig = LookId::builtin("signature");
        let SettingsFile::Global(s) = SettingsFile::parse(
            r#"{"nodes":{"items":[{"id":1,"grade":{"saturation":{"kind":"manual","value":5}}}]}}"#,
            &sig,
        )
        .expect("parses")
        .file
        else {
            panic!("global");
        };
        assert_eq!(s.output().look, sig);
    }

    #[test]
    fn an_object_keyed_by_file_name_is_a_map_and_matches_ignoring_case() {
        let text = r#"{
            "DSC00001.ARW": {"nodes":{"items":[{"id":1,"grade":{"contrast":{"kind":"manual","value":10}}}]}},
            "dsc00002.arw": {"nodes":{"items":[{"id":1,"grade":{"saturation":{"kind":"manual","value":-5}}}]}}}"#;
        let loaded = SettingsFile::parse(text, &neutral()).expect("parses");
        let file = &loaded.file;
        let one = file.for_file(Path::new("/x/DSC00001.ARW")).expect("entry");
        assert_eq!(
            one.main_grade().contrast,
            PercentSetting::manual_value(10.0)
        );
        let two = file.for_file(Path::new("/x/DSC00002.ARW")).expect("entry");
        assert_eq!(
            two.main_grade().saturation,
            PercentSetting::manual_value(-5.0)
        );
        assert!(file.for_file(Path::new("/x/DSC00003.ARW")).is_none());
        let unused = file.unused_entries(&[Path::new("/x/DSC00001.ARW").to_path_buf()]);
        assert_eq!(unused, ["dsc00002.arw"]);
    }

    #[test]
    fn mistakes_in_the_file_are_named() {
        let mixed =
            SettingsFile::parse(r#"{"nodes": {}, "a.ARW": {}}"#, &neutral()).expect_err("mixed");
        assert!(
            mixed.contains("mixes") && mixed.contains("a.ARW"),
            "{mixed}"
        );
        let typo = SettingsFile::parse(
            r#"{"a.ARW": {"nodes":{"items":[{"id":1,"grade":{"contrsat":1}}]}}}"#,
            &neutral(),
        )
        .expect_err("typo");
        assert!(
            typo.contains("a.ARW") && typo.contains("contrsat"),
            "{typo}"
        );
        let not_json = SettingsFile::parse("contrast: 1", &neutral()).expect_err("not json");
        assert!(not_json.contains("not valid JSON"), "{not_json}");
        let array = SettingsFile::parse("[]", &neutral()).expect_err("array");
        assert!(array.contains("JSON object"), "{array}");
        // A rule of the graph is named with its plain sentence, and the entry that broke it.
        let cycle = SettingsFile::parse(
            r#"{"nodes":{"items":[{"id":2,"kind":"corrector"}],"links":[
                {"from":{"node":"input","port":"picture"},"to":{"node":1,"port":"picture"}},
                {"from":{"node":1,"port":"picture"},"to":{"node":2,"port":"picture"}},
                {"from":{"node":2,"port":"picture"},"to":{"node":1,"port":"picture"}},
                {"from":{"node":2,"port":"picture"},"to":{"node":"output","port":"picture"}}]}}"#,
            &neutral(),
        )
        .expect_err("two links on one input");
        assert!(cycle.contains("An input can have one link"), "{cycle}");
    }

    #[test]
    fn a_misspelt_nodes_key_is_an_error_also_when_its_value_is_an_object() {
        // `{"nodez": {}}` is a valid settings document for a "file" called nodez; taken for one it
        // would be a warning that no photo has that name, and the typo would change nothing.
        for text in [
            r#"{"nodez": {"items": []}}"#,
            r#"{"node": {}}"#,
            r#"{"DSC00001": {"nodes": {}}}"#,
        ] {
            let error = SettingsFile::parse(text, &neutral()).expect_err(text);
            assert!(
                error.contains("not `nodes` or the name of an ARW file"),
                "{text}: {error}"
            );
        }
        let error = SettingsFile::parse(r#"{"nodes": {}, "nodez": {}}"#, &neutral())
            .expect_err("typo next to nodes");
        assert!(
            error.contains("nodez") && !error.contains("mixes"),
            "{error}"
        );
        // A name with any case of the extension is a file name.
        assert!(SettingsFile::parse(r#"{"x.arw": {}, "Y.ARW": {}}"#, &neutral()).is_ok());
    }

    #[test]
    fn values_out_of_range_are_limited_and_reported_with_their_node() {
        let loaded = SettingsFile::parse(
            r#"{"nodes":{"items":[{"id":1,"grade":{"contrast":{"kind":"manual","value":500}}}]}}"#,
            &neutral(),
        )
        .expect("parses");
        let SettingsFile::Global(s) = &loaded.file else {
            panic!("global");
        };
        assert_eq!(s.main_grade().contrast, PercentSetting::manual_value(100.0));
        assert_eq!(
            loaded.clamped,
            [(String::new(), vec!["1.grade.contrast".to_owned()])]
        );
    }

    #[test]
    fn the_exposure_offset_is_relative_to_the_photos_own_automatic_exposure() {
        let o = Overrides {
            exposure_offset: Some(-0.5),
            ..Overrides::default()
        };
        assert_eq!(
            input(&o.patch(1.25)).exposure,
            Some(ExposureSetting::manual_ev(0.75))
        );
        assert_eq!(
            input(&o.patch(-2.0)).exposure,
            Some(ExposureSetting::manual_ev(-2.5))
        );
        let absolute = Overrides {
            exposure: Some(ExposureArg::Ev(1.0)),
            ..Overrides::default()
        };
        assert_eq!(
            input(&absolute.patch(9.0)).exposure,
            Some(ExposureSetting::manual_ev(1.0))
        );
        let auto = Overrides {
            exposure: Some(ExposureArg::Auto),
            ..Overrides::default()
        };
        assert_eq!(input(&auto.patch(9.0)).exposure, Some(AutoOr::Auto));
        assert!(Overrides::default().patch(1.0).is_empty());
    }

    #[test]
    fn each_option_changes_its_own_field_of_its_own_node_and_nothing_else() {
        // Every flag alone: the patch holds exactly the field it names, on the node `docs/10`
        // says (a swapped pair, such as `--no-neural` feeding `highlight_recovery`, or a grade
        // field landing on the Input, would otherwise go unnoticed).
        let decode = |d: DecodePatch| {
            SettingsPatch::input(NodePatch {
                decode: Some(d),
                ..NodePatch::default()
            })
        };
        let main = |g: GradePatch| SettingsPatch::main_grade(g);
        let out = |n: NodePatch| SettingsPatch::output(n);
        let cases: Vec<(&str, Overrides, SettingsPatch)> = vec![
            (
                "--intensity",
                Overrides {
                    intensity: Some(80.0),
                    ..Overrides::default()
                },
                out(NodePatch {
                    look_intensity: Some(80.0),
                    ..NodePatch::default()
                }),
            ),
            (
                "--look",
                Overrides {
                    look: Some(LookId::builtin("signature")),
                    ..Overrides::default()
                },
                out(NodePatch {
                    look: Some(LookId::builtin("signature")),
                    ..NodePatch::default()
                }),
            ),
            (
                "--contrast",
                Overrides {
                    contrast: Some(PercentArg::Value(21.0)),
                    ..Overrides::default()
                },
                main(GradePatch {
                    contrast: Some(PercentSetting::manual_value(21.0)),
                    ..GradePatch::default()
                }),
            ),
            (
                "--brightness",
                Overrides {
                    brightness: Some(11.0),
                    ..Overrides::default()
                },
                main(GradePatch {
                    brightness: Some(11.0),
                    ..GradePatch::default()
                }),
            ),
            (
                "--highlights",
                Overrides {
                    highlights: Some(PercentArg::Value(-12.0)),
                    ..Overrides::default()
                },
                main(GradePatch {
                    highlights: Some(PercentSetting::manual_value(-12.0)),
                    ..GradePatch::default()
                }),
            ),
            (
                "--shadows",
                Overrides {
                    shadows: Some(PercentArg::Value(22.0)),
                    ..Overrides::default()
                },
                main(GradePatch {
                    shadows: Some(PercentSetting::manual_value(22.0)),
                    ..GradePatch::default()
                }),
            ),
            (
                "--white",
                Overrides {
                    white: Some(33.0),
                    ..Overrides::default()
                },
                main(GradePatch {
                    white: Some(33.0),
                    ..GradePatch::default()
                }),
            ),
            (
                "--black",
                Overrides {
                    black: Some(PercentArg::Auto),
                    ..Overrides::default()
                },
                main(GradePatch {
                    black: Some(AutoOr::Auto),
                    ..GradePatch::default()
                }),
            ),
            (
                "--saturation",
                Overrides {
                    saturation: Some(PercentArg::Value(-15.0)),
                    ..Overrides::default()
                },
                main(GradePatch {
                    saturation: Some(PercentSetting::manual_value(-15.0)),
                    ..GradePatch::default()
                }),
            ),
            (
                "--output-sharpening",
                Overrides {
                    output_sharpening: Some(35.0),
                    ..Overrides::default()
                },
                out(NodePatch {
                    output_sharpening: Some(35.0),
                    ..NodePatch::default()
                }),
            ),
            (
                "--rolloff",
                Overrides {
                    rolloff: Some(RolloffArg {
                        highlights: 0.7,
                        low_light: 0.2,
                    }),
                    ..Overrides::default()
                },
                SettingsPatch::input(NodePatch {
                    rolloff: Some(RolloffPatch {
                        highlights: Some(0.7),
                        low_light: Some(0.2),
                    }),
                    ..NodePatch::default()
                }),
            ),
            (
                "--light",
                Overrides {
                    light: Some(LightArg::Tungsten),
                    ..Overrides::default()
                },
                SettingsPatch::input(NodePatch {
                    light_source: Some(LightSource::Tungsten),
                    ..NodePatch::default()
                }),
            ),
            (
                "--no-neural",
                Overrides {
                    no_neural: true,
                    ..Overrides::default()
                },
                decode(DecodePatch {
                    neural_demosaic: Some(false),
                    ..DecodePatch::default()
                }),
            ),
            (
                "--no-highlight-recovery",
                Overrides {
                    no_highlight_recovery: true,
                    ..Overrides::default()
                },
                decode(DecodePatch {
                    highlight_recovery: Some(false),
                    ..DecodePatch::default()
                }),
            ),
            (
                "--no-lens",
                Overrides {
                    no_lens: true,
                    ..Overrides::default()
                },
                decode(DecodePatch {
                    lens_corrections: Some(false),
                    ..DecodePatch::default()
                }),
            ),
            (
                "--no-ca",
                Overrides {
                    no_ca: true,
                    ..Overrides::default()
                },
                decode(DecodePatch {
                    chromatic_aberration: Some(false),
                    ..DecodePatch::default()
                }),
            ),
        ];
        for (flag, overrides, expected) in cases {
            assert_eq!(overrides.patch(0.0), expected, "{flag}");
        }
        // Together they add up, node by node and field by field.
        let all = Overrides {
            brightness: Some(11.0),
            shadows: Some(PercentArg::Value(22.0)),
            white: Some(33.0),
            intensity: Some(50.0),
            no_neural: true,
            no_highlight_recovery: true,
            no_lens: true,
            ..Overrides::default()
        };
        let p = all.patch(0.0);
        assert_eq!(p.nodes.len(), 3, "the Input, Main and the Output");
        let g = grade(&p);
        assert_eq!(
            (g.brightness, g.shadows, g.white),
            (
                Some(11.0),
                Some(PercentSetting::manual_value(22.0)),
                Some(33.0)
            )
        );
        assert_eq!(output(&p).look_intensity, Some(50.0));
        assert_eq!(
            input(&p).decode,
            Some(DecodePatch {
                neural_demosaic: Some(false),
                highlight_recovery: Some(false),
                lens_corrections: Some(false),
                chromatic_aberration: None,
            })
        );
    }

    #[test]
    fn auto_contrast_and_saturation_are_auto_in_the_patch_and_win_over_a_file() {
        let o = Overrides {
            contrast: Some(PercentArg::Auto),
            saturation: Some(PercentArg::Auto),
            ..Overrides::default()
        };
        let p = o.patch(0.0);
        assert_eq!(grade(&p).contrast, Some(AutoOr::Auto));
        assert_eq!(grade(&p).saturation, Some(AutoOr::Auto));
        let loaded = SettingsFile::parse(
            r#"{"nodes":{"items":[{"id":1,"grade":{
                "contrast":{"kind":"manual","value":20},"saturation":{"kind":"manual","value":-5}}}]}}"#,
            &neutral(),
        )
        .expect("file");
        let SettingsFile::Global(mut s) = loaded.file else {
            panic!("global");
        };
        assert_eq!(s.main_grade().contrast, PercentSetting::manual_value(20.0));
        s.apply_patch(&p, None).expect("applies");
        assert!(s.main_grade().contrast.is_auto() && s.main_grade().saturation.is_auto());
    }

    #[test]
    fn a_wb_without_a_tint_leaves_the_tint_out_of_the_patch() {
        let o = Overrides {
            wb: Some(WbArg::Manual {
                kelvin: 5200.0,
                tint: None,
            }),
            no_ca: true,
            ..Overrides::default()
        };
        let p = input(&o.patch(0.0));
        assert_eq!(
            p.white_balance,
            Some(AutoOr::Manual(WhiteBalanceManualPatch {
                kelvin: Some(5200.0),
                tint: None
            }))
        );
        let decode = p.decode.expect("decode patch");
        assert_eq!(decode.chromatic_aberration, Some(false));
        assert_eq!(decode.neural_demosaic, None);
    }

    #[test]
    fn the_output_sharpening_option_is_a_plain_number_patch() {
        let o = Overrides {
            output_sharpening: Some(35.0),
            ..Overrides::default()
        };
        assert!(!o.is_empty());
        let p = o.patch(0.0);
        assert_eq!(output(&p).output_sharpening, Some(35.0));
        let mut s = DevelopSettings::default();
        s.apply_patch(&p, None).expect("applies");
        assert_eq!(s.output().output_sharpening, 35.0);
        // Without the option the patch says nothing and the default 100 stands.
        let none = Overrides::default().patch(0.0);
        assert_eq!(output(&none).output_sharpening, None);
        let mut d = DevelopSettings::default();
        d.apply_patch(&none, None).expect("applies");
        assert_eq!(d.output().output_sharpening, 100.0);
        // A settings file may carry it too, and the option wins over it.
        let loaded = SettingsFile::parse(
            r#"{"nodes":{"items":[{"id":"output","output_sharpening":60}]}}"#,
            &neutral(),
        )
        .expect("file");
        let SettingsFile::Global(mut from_file) = loaded.file else {
            panic!("global");
        };
        assert_eq!(from_file.output().output_sharpening, 60.0);
        from_file.apply_patch(&p, None).expect("applies");
        assert_eq!(from_file.output().output_sharpening, 35.0);
    }
}
