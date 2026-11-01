//! The patch of the node graph (`docs/23-nodes.md` section 3.7): a node-level patch, what Copy
//! settings takes, what a paste applies, the patch that undoes one and Reset.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Deserializer, Serialize};
use ym_color::wb::LightSource;
use ym_looks::library::LookId;

use super::graph::Sanitiser;
use super::graph::{
    CorrectorNode, DEFAULT_NAME, GraphError, InputNode, MAIN_ID, MAIN_NAME, NodeGraph, NodeId,
    NodeKind, NodeLink, NodePos, OutputNode, node_fields,
};
use super::masks::{MaskItem, MaskShape, MasksPatch};
use super::{
    AutoOr, AutoSeeds, BalanceSetting, ColorBalancePatch, ColorBalancePoint, ColorWarp,
    ContrastSetting, Crop, CurvesPatch, DecodePatch, DevelopSettings, ExposureSetting, FieldRef,
    LensPatch, NoiseReductionManualPatch, NoiseReductionValue, Orientation, PercentSetting,
    ResolvedSettings, RolloffPatch, SaturationSetting, SettingsError, SettingsField,
    SharpenManualPatch, SharpenValue, StageGroup, WhiteBalanceManualPatch, WhiteBalanceValue,
    double_option, percent_patch,
};

// ---------------------------------------------------------------------------------------------
// The patch types
// ---------------------------------------------------------------------------------------------

/// A corrector's trims to change.
#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrimPatch {
    /// New exposure trim in stops.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure_ev: Option<f64>,
    /// New temperature trim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// New tint trim.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tint: Option<f64>,
}

/// A corrector's grade to change: the old patch's grade fields, under the same names.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GradePatch {
    /// New contrast: Auto (Reset), or Manual with its value. A plain number reads as Manual with
    /// that value, and 0 as Auto.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub contrast: Option<ContrastSetting>,
    /// New brightness.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub brightness: Option<f64>,
    /// New highlights, read like `contrast`.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub highlights: Option<PercentSetting>,
    /// New shadows, read like `contrast`.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub shadows: Option<PercentSetting>,
    /// New white.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub white: Option<f64>,
    /// New black, read like `contrast`.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub black: Option<PercentSetting>,
    /// Colour-balance wheels to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_balance: Option<ColorBalancePatch>,
    /// New saturation, read like `contrast`.
    #[serde(
        default,
        deserialize_with = "percent_patch",
        skip_serializing_if = "Option::is_none"
    )]
    pub saturation: Option<SaturationSetting>,
    /// New colour warp: the whole list of pins.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub color_warp: Option<ColorWarp>,
    /// Curves to replace.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub curve: Option<CurvesPatch>,
}

impl GradePatch {
    /// The names of the fields the patch carries.
    fn names(&self) -> Vec<SettingsField> {
        let mut v = Vec::new();
        macro_rules! has {
            ($($f:ident => $n:expr),* $(,)?) => { $(if self.$f.is_some() { v.push($n); })* };
        }
        has!(
            contrast => SettingsField::Contrast, brightness => SettingsField::Brightness,
            highlights => SettingsField::Highlights, shadows => SettingsField::Shadows,
            white => SettingsField::White, black => SettingsField::Black,
            color_balance => SettingsField::ColorBalance, saturation => SettingsField::Saturation,
            color_warp => SettingsField::ColorWarp, curve => SettingsField::Curve,
        );
        v
    }

    /// Takes over the fields `other` sets.
    fn absorb(&mut self, other: Self) {
        macro_rules! take {
            ($($field:ident),* $(,)?) => {
                $(if other.$field.is_some() { self.$field = other.$field; })*
            };
        }
        take!(
            contrast,
            brightness,
            highlights,
            shadows,
            white,
            black,
            color_balance,
            saturation,
            color_warp,
            curve,
        );
    }

    /// Applies the patch to a grade.
    fn apply(&self, g: &mut super::graph::NodeGrade) {
        if let Some(v) = self.contrast {
            g.contrast = v;
        }
        if let Some(v) = self.brightness {
            g.brightness = v;
        }
        if let Some(v) = self.highlights {
            g.highlights = v;
        }
        if let Some(v) = self.shadows {
            g.shadows = v;
        }
        if let Some(v) = self.white {
            g.white = v;
        }
        if let Some(v) = self.black {
            g.black = v;
        }
        if let Some(cb) = &self.color_balance {
            if let Some(v) = cb.shadows {
                g.color_balance.shadows = v;
            }
            if let Some(v) = cb.midtones {
                g.color_balance.midtones = v;
            }
            if let Some(v) = cb.highlights {
                g.color_balance.highlights = v;
            }
        }
        if let Some(v) = self.saturation {
            g.saturation = v;
        }
        if let Some(v) = &self.color_warp {
            g.color_warp = v.clone();
        }
        if let Some(c) = &self.curve {
            if let Some(v) = &c.luma {
                g.curve.luma = v.clone();
            }
            if let Some(v) = &c.red {
                g.curve.red = v.clone();
            }
            if let Some(v) = &c.green {
                g.curve.green = v.clone();
            }
            if let Some(v) = &c.blue {
                g.curve.blue = v.clone();
            }
        }
    }
}

/// A partial of one node's fields (`docs/23` section 3.7). It is one flat object whatever the kind:
/// the fields of every kind are here, and a field the node's kind does not have is refused when the
/// patch is applied ("A corrector has no field called “exposure”."). A node the graph does not have
/// is added when `kind` is given.
///
/// Nested objects (`decode`, `rolloff`, `lens`, `trim`, `grade`, `color_balance` and `curve`) may
/// carry only the parts to change, and a manual `white_balance`, `sharpen` or `noise_reduction` may
/// leave out a part, which keeps its current resolved value. `crop: null` and `white_pick: null`
/// clear the Input's field.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NodePatch {
    /// The node's kind: needed to add a node, and refused when it is not the node's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind: Option<NodeKind>,
    /// The card's place.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pos: Option<NodePos>,

    // The Input.
    /// New exposure (Manual to set it, Auto to reset it).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exposure: Option<ExposureSetting>,
    /// New light preset.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub light_source: Option<LightSource>,
    /// New white balance; a manual one may leave out Kelvin or tint.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub white_balance: Option<AutoOr<WhiteBalanceManualPatch>>,
    /// New white pick; `Some(None)` (JSON `null`) clears it. A pick also sets `light_source` and
    /// `white_balance` to Auto unless the same patch says otherwise: the picked spot then renders
    /// neutral, and the Kelvin and tint rows show the picked light (`docs/11` section Pick white).
    #[serde(
        default,
        deserialize_with = "double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub white_pick: Option<Option<[f64; 2]>>,
    /// RAW › Range amounts to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rolloff: Option<RolloffPatch>,
    /// Decode switches to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decode: Option<DecodePatch>,
    /// New noise reduction; a manual one may leave out a part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub noise_reduction: Option<AutoOr<NoiseReductionManualPatch>>,
    /// New sharpening; a manual one may leave out a part.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sharpen: Option<AutoOr<SharpenManualPatch>>,
    /// Lens parts to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lens: Option<LensPatch>,
    /// New crop; `Some(None)` (JSON `null`) removes it.
    #[serde(
        default,
        deserialize_with = "double_option",
        skip_serializing_if = "Option::is_none"
    )]
    pub crop: Option<Option<Crop>>,
    /// New orientation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orientation: Option<Orientation>,

    // The Output.
    /// New look.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look: Option<LookId>,
    /// New look intensity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub look_intensity: Option<f64>,
    /// New output sharpening amount, percent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_sharpening: Option<f64>,

    // A corrector.
    /// New name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Bypass on or off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bypass: Option<bool>,
    /// Trims to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trim: Option<TrimPatch>,
    /// Grade fields to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grade: Option<GradePatch>,
    /// Masks to change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub masks: Option<MasksPatch>,
}

/// A node-level patch of a photo's settings (`update_settings(photo_id, patch)`), `docs/23`
/// section 3.7: `nodes` maps ids to node patches, `remove` lists nodes to delete (their links go
/// with them) and `links`, when present, replaces the links whole. A patch is atomic.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SettingsPatch {
    /// Node patches by id (`"input"`, `"output"`, `"1"`, `"2"`, …).
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub nodes: BTreeMap<NodeId, NodePatch>,
    /// The corrector nodes to delete.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub remove: Vec<NodeId>,
    /// The links, replacing the photo's whole.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub links: Option<Vec<NodeLink>>,
}

// ---------------------------------------------------------------------------------------------
// Which fields a node patch names
// ---------------------------------------------------------------------------------------------

impl NodePatch {
    /// The fields the patch carries, as the names the unknown-field rule uses (`kind` and `pos`
    /// are common to every kind and are not listed).
    fn field_names(&self) -> Vec<&'static str> {
        let mut v = Vec::new();
        macro_rules! has {
            ($($f:ident => $n:expr),* $(,)?) => { $(if self.$f.is_some() { v.push($n); })* };
        }
        has!(
            exposure => "exposure", light_source => "light_source",
            white_balance => "white_balance", white_pick => "white_pick", rolloff => "rolloff",
            decode => "decode", noise_reduction => "noise_reduction", sharpen => "sharpen",
            lens => "lens", crop => "crop", orientation => "orientation", look => "look",
            look_intensity => "look_intensity", output_sharpening => "output_sharpening",
            name => "name", bypass => "bypass", trim => "trim", grade => "grade", masks => "masks",
        );
        v
    }

    /// Whether the patch names this field of its node (`grade.*` fields are named through
    /// `grade`).
    fn touches(&self, field: SettingsField) -> bool {
        match field {
            SettingsField::Exposure => self.exposure.is_some(),
            SettingsField::LightSource => self.light_source.is_some(),
            SettingsField::WhiteBalance => self.white_balance.is_some(),
            SettingsField::WhitePick => self.white_pick.is_some(),
            SettingsField::Rolloff => self.rolloff.is_some(),
            SettingsField::Decode => self.decode.is_some(),
            SettingsField::NoiseReduction => self.noise_reduction.is_some(),
            SettingsField::Sharpen => self.sharpen.is_some(),
            SettingsField::Lens => self.lens.is_some(),
            SettingsField::Crop => self.crop.is_some(),
            SettingsField::Orientation => self.orientation.is_some(),
            SettingsField::Look => self.look.is_some(),
            SettingsField::LookIntensity => self.look_intensity.is_some(),
            SettingsField::OutputSharpening => self.output_sharpening.is_some(),
            SettingsField::Name => self.name.is_some(),
            SettingsField::Bypass => self.bypass.is_some(),
            SettingsField::Pos => self.pos.is_some(),
            SettingsField::Trim => self.trim.is_some(),
            SettingsField::Masks => self.masks.is_some(),
            grade_field => self
                .grade
                .as_ref()
                .is_some_and(|g| g.names().contains(&grade_field)),
        }
    }

    /// Takes over the fields `other` sets (grade fields merge one by one).
    fn absorb(&mut self, other: Self) {
        macro_rules! take {
            ($($field:ident),* $(,)?) => {
                $(if other.$field.is_some() { self.$field = other.$field; })*
            };
        }
        take!(
            kind,
            pos,
            exposure,
            light_source,
            white_balance,
            white_pick,
            rolloff,
            decode,
            noise_reduction,
            sharpen,
            lens,
            crop,
            orientation,
            look,
            look_intensity,
            output_sharpening,
            name,
            bypass,
            trim,
            masks,
        );
        if let Some(g) = other.grade {
            match &mut self.grade {
                Some(mine) => mine.absorb(g),
                None => self.grade = Some(g),
            }
        }
    }

    /// The patch that carries one field of an Input.
    fn of_input_field(n: &InputNode, field: SettingsField) -> Option<Self> {
        let mut p = Self::default();
        match field {
            SettingsField::Exposure => p.exposure = Some(n.exposure),
            SettingsField::LightSource => p.light_source = Some(n.light_source),
            SettingsField::WhiteBalance => {
                p.white_balance = Some(match n.white_balance {
                    AutoOr::Auto => AutoOr::Auto,
                    AutoOr::Manual(m) => AutoOr::Manual(WhiteBalanceManualPatch {
                        kelvin: Some(m.kelvin),
                        tint: Some(m.tint),
                    }),
                });
            }
            SettingsField::WhitePick => p.white_pick = Some(n.white_pick),
            SettingsField::Rolloff => {
                p.rolloff = Some(RolloffPatch {
                    highlights: Some(n.rolloff.highlights),
                    low_light: Some(n.rolloff.low_light),
                });
            }
            SettingsField::Decode => {
                p.decode = Some(DecodePatch {
                    neural_demosaic: Some(n.decode.neural_demosaic),
                    highlight_recovery: Some(n.decode.highlight_recovery),
                    lens_corrections: Some(n.decode.lens_corrections),
                    chromatic_aberration: Some(n.decode.chromatic_aberration),
                });
            }
            SettingsField::NoiseReduction => {
                p.noise_reduction = Some(match n.noise_reduction {
                    AutoOr::Auto => AutoOr::Auto,
                    AutoOr::Manual(m) => AutoOr::Manual(NoiseReductionManualPatch {
                        luminance: Some(m.luminance),
                        color: Some(m.color),
                        detail: Some(m.detail),
                    }),
                });
            }
            SettingsField::Sharpen => {
                p.sharpen = Some(match n.sharpen {
                    AutoOr::Auto => AutoOr::Auto,
                    AutoOr::Manual(m) => AutoOr::Manual(SharpenManualPatch {
                        amount: Some(m.amount),
                        radius_px: Some(m.radius_px),
                    }),
                });
            }
            SettingsField::Lens => {
                p.lens = Some(LensPatch {
                    kind: Some(n.lens.kind),
                    flares: Some(n.lens.flares),
                    edges: Some(n.lens.edges),
                    anamorphic: Some(n.lens.anamorphic),
                });
            }
            SettingsField::Crop => p.crop = Some(n.crop),
            SettingsField::Orientation => p.orientation = Some(n.orientation),
            SettingsField::Pos => p.pos = Some(n.pos),
            _ => return None,
        }
        Some(p)
    }

    /// The patch that carries one field of an Output.
    fn of_output_field(n: &OutputNode, field: SettingsField) -> Option<Self> {
        let mut p = Self::default();
        match field {
            SettingsField::Look => p.look = Some(n.look.clone()),
            SettingsField::LookIntensity => p.look_intensity = Some(n.look_intensity),
            SettingsField::OutputSharpening => p.output_sharpening = Some(n.output_sharpening),
            SettingsField::Pos => p.pos = Some(n.pos),
            _ => return None,
        }
        Some(p)
    }

    /// The patch that carries one field of a corrector.
    fn of_corrector_field(n: &CorrectorNode, field: SettingsField) -> Option<Self> {
        let mut p = Self::default();
        let mut g = GradePatch::default();
        let g_ = &n.grade;
        match field {
            SettingsField::Name => p.name = Some(n.name.clone()),
            SettingsField::Bypass => p.bypass = Some(n.bypass),
            SettingsField::Pos => p.pos = Some(n.pos),
            SettingsField::Trim => {
                p.trim = Some(TrimPatch {
                    exposure_ev: Some(n.trim.exposure_ev),
                    temperature: Some(n.trim.temperature),
                    tint: Some(n.trim.tint),
                });
            }
            SettingsField::Masks => {
                p.masks = Some(MasksPatch {
                    invert: Some(n.masks.invert),
                    items: Some(n.masks.items.clone()),
                    edit: None,
                });
            }
            SettingsField::Contrast => g.contrast = Some(g_.contrast),
            SettingsField::Brightness => g.brightness = Some(g_.brightness),
            SettingsField::Highlights => g.highlights = Some(g_.highlights),
            SettingsField::Shadows => g.shadows = Some(g_.shadows),
            SettingsField::White => g.white = Some(g_.white),
            SettingsField::Black => g.black = Some(g_.black),
            SettingsField::ColorBalance => {
                g.color_balance = Some(ColorBalancePatch {
                    shadows: Some(g_.color_balance.shadows),
                    midtones: Some(g_.color_balance.midtones),
                    highlights: Some(g_.color_balance.highlights),
                });
            }
            SettingsField::Saturation => g.saturation = Some(g_.saturation),
            SettingsField::ColorWarp => g.color_warp = Some(g_.color_warp.clone()),
            SettingsField::Curve => {
                g.curve = Some(CurvesPatch {
                    luma: Some(g_.curve.luma.clone()),
                    red: Some(g_.curve.red.clone()),
                    green: Some(g_.curve.green.clone()),
                    blue: Some(g_.curve.blue.clone()),
                });
            }
            _ => return None,
        }
        if g != GradePatch::default() {
            p.grade = Some(g);
        }
        Some(p)
    }

    /// A patch that holds every field of an Input.
    #[must_use]
    pub fn whole_input(n: &InputNode) -> Self {
        let mut p = Self::default();
        for f in SettingsField::ALL {
            if f.belongs_to(NodeKind::Input)
                && let Some(one) = Self::of_input_field(n, f)
            {
                p.absorb(one);
            }
        }
        p
    }

    /// A patch that holds every field of an Output.
    #[must_use]
    pub fn whole_output(n: &OutputNode) -> Self {
        let mut p = Self::default();
        for f in SettingsField::ALL {
            if f.belongs_to(NodeKind::Output)
                && let Some(one) = Self::of_output_field(n, f)
            {
                p.absorb(one);
            }
        }
        p
    }

    /// A patch that holds every field of a corrector, with its `kind`, so that it adds the node when
    /// the photo does not have it.
    #[must_use]
    pub fn whole_corrector(n: &CorrectorNode) -> Self {
        let mut p = Self {
            kind: Some(NodeKind::Corrector),
            ..Self::default()
        };
        for f in SettingsField::ALL {
            if f.belongs_to(NodeKind::Corrector)
                && let Some(one) = Self::of_corrector_field(n, f)
            {
                p.absorb(one);
            }
        }
        p
    }

    /// Applies the Input's fields. `base` is the Input as it was before the patch (a partial manual
    /// value is filled from it, else from the seeds).
    fn apply_input(
        &self,
        next: &mut InputNode,
        base: &InputNode,
        seeds: Option<&AutoSeeds>,
    ) -> Result<(), SettingsError> {
        if let Some(v) = self.exposure {
            next.exposure = v;
        }
        if let Some(v) = self.light_source {
            next.light_source = v;
        }
        if let Some(wb) = &self.white_balance {
            next.white_balance = match wb {
                AutoOr::Auto => AutoOr::Auto,
                AutoOr::Manual(p) => {
                    let b = match base.white_balance {
                        AutoOr::Manual(m) => m,
                        AutoOr::Auto => match (seeds, p.kelvin, p.tint) {
                            (_, Some(k), Some(t)) => WhiteBalanceValue { kelvin: k, tint: t },
                            (Some(s), _, _) => WhiteBalanceValue {
                                kelvin: s.kelvin,
                                tint: s.tint,
                            },
                            (None, _, _) => {
                                return Err(SettingsError::NeedsEstimates("white_balance"));
                            }
                        },
                    };
                    AutoOr::Manual(WhiteBalanceValue {
                        kelvin: p.kelvin.unwrap_or(b.kelvin),
                        tint: p.tint.unwrap_or(b.tint),
                    })
                }
            };
        }
        if let Some(r) = &self.rolloff {
            if let Some(v) = r.highlights {
                next.rolloff.highlights = v;
            }
            if let Some(v) = r.low_light {
                next.rolloff.low_light = v;
            }
        }
        if let Some(d) = &self.decode {
            if let Some(v) = d.neural_demosaic {
                next.decode.neural_demosaic = v;
            }
            if let Some(v) = d.highlight_recovery {
                next.decode.highlight_recovery = v;
            }
            if let Some(v) = d.lens_corrections {
                next.decode.lens_corrections = v;
            }
            if let Some(v) = d.chromatic_aberration {
                next.decode.chromatic_aberration = v;
            }
        }
        if let Some(n) = &self.noise_reduction {
            next.noise_reduction = match n {
                AutoOr::Auto => AutoOr::Auto,
                AutoOr::Manual(p) => {
                    let b = match base.noise_reduction {
                        AutoOr::Manual(m) => m,
                        AutoOr::Auto => match (seeds, p.luminance, p.color, p.detail) {
                            (_, Some(l), Some(c), Some(d)) => NoiseReductionValue {
                                luminance: l,
                                color: c,
                                detail: d,
                            },
                            (Some(s), _, _, _) => s.noise_reduction,
                            (None, ..) => {
                                return Err(SettingsError::NeedsEstimates("noise_reduction"));
                            }
                        },
                    };
                    AutoOr::Manual(NoiseReductionValue {
                        luminance: p.luminance.unwrap_or(b.luminance),
                        color: p.color.unwrap_or(b.color),
                        detail: p.detail.unwrap_or(b.detail),
                    })
                }
            };
        }
        if let Some(s) = &self.sharpen {
            next.sharpen = match s {
                AutoOr::Auto => AutoOr::Auto,
                AutoOr::Manual(p) => {
                    let b = match base.sharpen {
                        AutoOr::Manual(m) => m,
                        AutoOr::Auto => match (seeds, p.amount, p.radius_px) {
                            (_, Some(a), Some(r)) => SharpenValue {
                                amount: a,
                                radius_px: r,
                            },
                            (Some(s), _, _) => s.sharpen,
                            (None, _, _) => return Err(SettingsError::NeedsEstimates("sharpen")),
                        },
                    };
                    AutoOr::Manual(SharpenValue {
                        amount: p.amount.unwrap_or(b.amount),
                        radius_px: p.radius_px.unwrap_or(b.radius_px),
                    })
                }
            };
        }
        if let Some(l) = &self.lens {
            if let Some(v) = l.kind {
                next.lens.kind = v;
            }
            if let Some(v) = l.flares {
                next.lens.flares = v;
            }
            if let Some(v) = l.edges {
                next.lens.edges = v;
            }
            if let Some(v) = l.anamorphic {
                next.lens.anamorphic = v;
            }
        }
        if let Some(v) = self.crop {
            next.crop = v;
        }
        if let Some(v) = self.orientation {
            next.orientation = v;
        }
        if let Some(v) = self.white_pick {
            next.white_pick = v;
            if v.is_some() {
                // A picked white sets the white balance: the spot renders neutral and the Kelvin
                // and tint rows show the picked light.
                if self.light_source.is_none() {
                    next.light_source = LightSource::Auto;
                }
                if self.white_balance.is_none() {
                    next.white_balance = AutoOr::Auto;
                }
            }
        }
        Ok(())
    }

    fn apply_output(&self, next: &mut OutputNode) {
        if let Some(v) = &self.look {
            next.look = v.clone();
        }
        if let Some(v) = self.look_intensity {
            next.look_intensity = v;
        }
        if let Some(v) = self.output_sharpening {
            next.output_sharpening = v;
        }
    }

    fn apply_corrector(&self, next: &mut CorrectorNode, id: u32) -> Result<(), SettingsError> {
        if let Some(v) = &self.name {
            next.name.clone_from(v);
        }
        if let Some(v) = self.bypass {
            next.bypass = v;
        }
        if let Some(t) = &self.trim {
            if let Some(v) = t.exposure_ev {
                next.trim.exposure_ev = v;
            }
            if let Some(v) = t.temperature {
                next.trim.temperature = v;
            }
            if let Some(v) = t.tint {
                next.trim.tint = v;
            }
        }
        if let Some(g) = &self.grade {
            g.apply(&mut next.grade);
        }
        if let Some(m) = &self.masks {
            m.apply(&mut next.masks, &NodeId::Corrector(id).to_string())?;
        }
        Ok(())
    }

    /// Checks that every field the patch carries belongs to `kind`.
    fn check_kind(&self, kind: NodeKind) -> Result<(), GraphError> {
        let own = node_fields(kind);
        for name in self.field_names() {
            if !own.contains(&name) {
                return Err(GraphError::UnknownField {
                    kind: kind.sentence(),
                    field: name.to_owned(),
                });
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Fields
// ---------------------------------------------------------------------------------------------

impl DevelopSettings {
    /// The fields that differ from `other`, node by node (nodes that only one of the two has are
    /// not listed: [`DevelopSettings::diff_nodes`]).
    #[must_use]
    pub fn diff(&self, other: &DevelopSettings) -> Vec<FieldRef> {
        let mut out = Vec::new();
        for id in self.nodes.ids() {
            if !other.nodes.has(id) {
                continue;
            }
            for field in SettingsField::ALL {
                if field.belongs_to(id.kind()) && !self.field_equals(other, id, field) {
                    out.push(FieldRef { node: id, field });
                }
            }
        }
        out
    }

    /// The nodes `other` has that `self` has not, and the nodes `self` has that `other` has not.
    #[must_use]
    pub fn diff_nodes(&self, other: &DevelopSettings) -> (Vec<NodeId>, Vec<NodeId>) {
        let added = other
            .nodes
            .ids()
            .into_iter()
            .filter(|id| !self.nodes.has(*id))
            .collect();
        let removed = self
            .nodes
            .ids()
            .into_iter()
            .filter(|id| !other.nodes.has(*id))
            .collect();
        (added, removed)
    }

    fn field_equals(&self, o: &DevelopSettings, node: NodeId, field: SettingsField) -> bool {
        let (a, b) = (&self.nodes, &o.nodes);
        match node {
            NodeId::Input => {
                let (a, b) = (&a.input, &b.input);
                match field {
                    SettingsField::Exposure => a.exposure == b.exposure,
                    SettingsField::LightSource => a.light_source == b.light_source,
                    SettingsField::WhiteBalance => a.white_balance == b.white_balance,
                    SettingsField::WhitePick => a.white_pick == b.white_pick,
                    SettingsField::Rolloff => a.rolloff == b.rolloff,
                    SettingsField::Decode => a.decode == b.decode,
                    SettingsField::NoiseReduction => a.noise_reduction == b.noise_reduction,
                    SettingsField::Sharpen => a.sharpen == b.sharpen,
                    SettingsField::Lens => a.lens == b.lens,
                    SettingsField::Crop => a.crop == b.crop,
                    SettingsField::Orientation => a.orientation == b.orientation,
                    SettingsField::Pos => a.pos == b.pos,
                    _ => true,
                }
            }
            NodeId::Output => {
                let (a, b) = (&a.output, &b.output);
                match field {
                    SettingsField::Look => a.look == b.look,
                    SettingsField::LookIntensity => a.look_intensity == b.look_intensity,
                    SettingsField::OutputSharpening => a.output_sharpening == b.output_sharpening,
                    SettingsField::Pos => a.pos == b.pos,
                    _ => true,
                }
            }
            NodeId::Corrector(n) => {
                let (Some(a), Some(b)) = (a.corrector(n), b.corrector(n)) else {
                    return true;
                };
                let (ga, gb) = (&a.grade, &b.grade);
                match field {
                    SettingsField::Name => a.name == b.name,
                    SettingsField::Bypass => a.bypass == b.bypass,
                    SettingsField::Pos => a.pos == b.pos,
                    SettingsField::Trim => a.trim == b.trim,
                    SettingsField::Masks => a.masks == b.masks,
                    SettingsField::Contrast => ga.contrast == gb.contrast,
                    SettingsField::Brightness => ga.brightness == gb.brightness,
                    SettingsField::Highlights => ga.highlights == gb.highlights,
                    SettingsField::Shadows => ga.shadows == gb.shadows,
                    SettingsField::White => ga.white == gb.white,
                    SettingsField::Black => ga.black == gb.black,
                    SettingsField::ColorBalance => ga.color_balance == gb.color_balance,
                    SettingsField::Saturation => ga.saturation == gb.saturation,
                    SettingsField::ColorWarp => ga.color_warp == gb.color_warp,
                    SettingsField::Curve => ga.curve == gb.curve,
                    _ => true,
                }
            }
        }
    }
}

/// What [`DevelopSettings::apply_patch`] did.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PatchReport {
    /// The fields whose value changed, by node (including `light_source` and `white_balance` when
    /// a white pick reset them, and every field of a node the patch added).
    pub changed: Vec<FieldRef>,
    /// The fields in which a number was out of range and was clamped.
    pub clamped: Vec<FieldRef>,
    /// The nodes the patch added.
    pub added: Vec<NodeId>,
    /// The nodes the patch removed.
    pub removed: Vec<NodeId>,
    /// The links changed.
    pub links_changed: bool,
}

impl PatchReport {
    /// The first render stage group any change affects, if anything that renders changed. `name` and
    /// `pos` re-run nothing; nodes added or removed and links are the nodes' group.
    #[must_use]
    pub fn first_group(&self) -> Option<StageGroup> {
        let structural = (!self.added.is_empty() || !self.removed.is_empty() || self.links_changed)
            .then_some(StageGroup::Grade);
        self.changed
            .iter()
            .filter_map(|f| f.field.group())
            .chain(structural)
            .min()
    }

    /// True when nothing changed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.changed.is_empty()
            && self.added.is_empty()
            && self.removed.is_empty()
            && !self.links_changed
    }
}

// ---------------------------------------------------------------------------------------------
// Applying a patch
// ---------------------------------------------------------------------------------------------

impl DevelopSettings {
    /// Applies a patch with the rules of `docs/23` section 3.7.
    ///
    /// The patch is applied atomically: on `Err` the settings are untouched. Numbers out of range
    /// are clamped and reported in [`PatchReport::clamped`]. `seeds` are the photo's resolved
    /// automatic values; they fill the missing part of a partial manual `white_balance`,
    /// `sharpen` or `noise_reduction` of the Input while the setting is `Auto` (a patch with only
    /// `kelvin` on an Auto white balance is Manual at that Kelvin and the automatic tint).
    ///
    /// # Errors
    ///
    /// [`SettingsError`] for a value that is not a number, an unusable curve or crop, a partial
    /// manual patch that needs `seeds` when there are none, and any rule of the graph
    /// ([`GraphError`]) the result breaks.
    pub fn apply_patch(
        &mut self,
        patch: &SettingsPatch,
        seeds: Option<&AutoSeeds>,
    ) -> Result<PatchReport, SettingsError> {
        let mut next = self.clone();
        let mut removed = Vec::new();
        for id in &patch.remove {
            match id {
                NodeId::Input | NodeId::Output => return Err(GraphError::CannotRemove.into()),
                NodeId::Corrector(n) if *n == MAIN_ID => {
                    return Err(GraphError::CannotRemove.into());
                }
                NodeId::Corrector(n) => {
                    if next.nodes.extra.remove(n).is_none() {
                        return Err(GraphError::PatchUnknownNode(id.to_string()).into());
                    }
                    next.nodes
                        .links
                        .retain(|l| l.from.node != *id && l.to.node != *id);
                    removed.push(*id);
                }
            }
        }
        let mut added = Vec::new();
        for (id, np) in &patch.nodes {
            let exists = next.nodes.has(*id);
            if let Some(k) = np.kind
                && k != id.kind()
            {
                return Err(GraphError::PatchWrongKind {
                    node: id.sentence(),
                    actual: id.kind().with_article(),
                    given: k.with_article(),
                }
                .into());
            }
            if !exists {
                // Only a corrector can be new, and it must say what it is.
                let NodeId::Corrector(n) = *id else {
                    return Err(GraphError::PatchUnknownNode(id.to_string()).into());
                };
                if np.kind.is_none() {
                    return Err(GraphError::PatchUnknownNode(id.to_string()).into());
                }
                let mut node = CorrectorNode::default();
                node.pos = [next.nodes.corrector_count() as f64, 0.0];
                next.nodes.extra.insert(n, node);
                added.push(*id);
            }
            np.check_kind(id.kind())?;
            match *id {
                NodeId::Input => {
                    let base = self.nodes.input.clone();
                    np.apply_input(&mut next.nodes.input, &base, seeds)?;
                    if let Some(p) = np.pos {
                        next.nodes.input.pos = p;
                    }
                }
                NodeId::Output => {
                    np.apply_output(&mut next.nodes.output);
                    if let Some(p) = np.pos {
                        next.nodes.output.pos = p;
                    }
                }
                NodeId::Corrector(n) => {
                    if let Some(c) = next.nodes.corrector_mut(n) {
                        np.apply_corrector(c, n)?;
                        if let Some(p) = np.pos {
                            c.pos = p;
                        }
                    }
                }
            }
        }
        if let Some(links) = &patch.links {
            next.nodes.links.clone_from(links);
        }
        let mut s = Sanitiser::default();
        next.nodes.sanitise(&mut s)?;
        let mut changed = self.diff(&next);
        // The fields of a node the patch added are all new.
        for id in &added {
            for field in SettingsField::ALL {
                if field.belongs_to(NodeKind::Corrector) {
                    let r = FieldRef { node: *id, field };
                    if !changed.contains(&r) {
                        changed.push(r);
                    }
                }
            }
        }
        // Only report clamping in fields the patch touched: a stored value is always in range.
        let mut clamped = s.clamped;
        clamped.retain(|r| {
            changed.contains(r)
                || patch
                    .nodes
                    .get(&r.node)
                    .is_some_and(|np| np.touches(r.field))
        });
        let links_changed = self.nodes.links != next.nodes.links;
        *self = next;
        Ok(PatchReport {
            changed,
            clamped,
            added,
            removed,
            links_changed,
        })
    }
}

// ---------------------------------------------------------------------------------------------
// Building patches
// ---------------------------------------------------------------------------------------------

impl SettingsPatch {
    /// True when the patch carries no change.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        *self == SettingsPatch::default()
    }

    /// A patch of one node.
    #[must_use]
    pub fn node(id: NodeId, patch: NodePatch) -> Self {
        Self {
            nodes: BTreeMap::from([(id, patch)]),
            ..Self::default()
        }
    }

    /// A patch of the Input.
    #[must_use]
    pub fn input(patch: NodePatch) -> Self {
        Self::node(NodeId::Input, patch)
    }

    /// A patch of Main (node 1).
    #[must_use]
    pub fn main(patch: NodePatch) -> Self {
        Self::node(NodeId::MAIN, patch)
    }

    /// A patch of Main's grade.
    #[must_use]
    pub fn main_grade(grade: GradePatch) -> Self {
        Self::main(NodePatch {
            grade: Some(grade),
            ..NodePatch::default()
        })
    }

    /// A patch of the Output.
    #[must_use]
    pub fn output(patch: NodePatch) -> Self {
        Self::node(NodeId::Output, patch)
    }

    /// This patch with `patch` added for the node `id`: fields the node already has in this patch
    /// are replaced.
    #[must_use]
    pub fn and(mut self, id: NodeId, patch: NodePatch) -> Self {
        match self.nodes.get_mut(&id) {
            Some(mine) => mine.absorb(patch),
            None => {
                self.nodes.insert(id, patch);
            }
        }
        self
    }

    /// The Input's patch, if the patch has one.
    #[must_use]
    pub fn input_patch(&self) -> Option<&NodePatch> {
        self.nodes.get(&NodeId::Input)
    }

    /// Main's patch, if the patch has one.
    #[must_use]
    pub fn main_patch(&self) -> Option<&NodePatch> {
        self.nodes.get(&NodeId::MAIN)
    }

    /// The Output's patch, if the patch has one.
    #[must_use]
    pub fn output_patch(&self) -> Option<&NodePatch> {
        self.nodes.get(&NodeId::Output)
    }

    /// The look the patch sets on the Output, if any.
    #[must_use]
    pub fn look(&self) -> Option<&LookId> {
        self.output_patch().and_then(|p| p.look.as_ref())
    }

    /// A patch that moves the exposure slider: the Input's exposure, Manual at `ev`.
    #[must_use]
    pub fn exposure_ev(ev: f64) -> Self {
        Self::input(NodePatch {
            exposure: Some(ExposureSetting::manual_ev(ev)),
            ..NodePatch::default()
        })
    }

    /// A patch that sets the Input's Kelvin and tint by hand.
    #[must_use]
    pub fn white_balance(kelvin: f64, tint: f64) -> Self {
        Self::input(NodePatch {
            white_balance: Some(AutoOr::Manual(WhiteBalanceManualPatch {
                kelvin: Some(kelvin),
                tint: Some(tint),
            })),
            ..NodePatch::default()
        })
    }

    /// A patch that moves only the Kelvin slider (the tint keeps its current value).
    #[must_use]
    pub fn kelvin(kelvin: f64) -> Self {
        Self::input(NodePatch {
            white_balance: Some(AutoOr::Manual(WhiteBalanceManualPatch {
                kelvin: Some(kelvin),
                tint: None,
            })),
            ..NodePatch::default()
        })
    }

    /// A patch that sets the Input's sharpening by hand.
    #[must_use]
    pub fn sharpen(amount: f64, radius_px: f64) -> Self {
        Self::input(NodePatch {
            sharpen: Some(AutoOr::Manual(SharpenManualPatch {
                amount: Some(amount),
                radius_px: Some(radius_px),
            })),
            ..NodePatch::default()
        })
    }

    /// A patch that sets the Input's noise reduction by hand.
    #[must_use]
    pub fn noise_reduction(luminance: f64, color: f64, detail: f64) -> Self {
        Self::input(NodePatch {
            noise_reduction: Some(AutoOr::Manual(NoiseReductionManualPatch {
                luminance: Some(luminance),
                color: Some(color),
                detail: Some(detail),
            })),
            ..NodePatch::default()
        })
    }

    /// A patch that sets Main's contrast by hand.
    #[must_use]
    pub fn contrast(value: f64) -> Self {
        Self::main_grade(GradePatch {
            contrast: Some(PercentSetting::manual_value(value)),
            ..GradePatch::default()
        })
    }

    /// A patch that sets the Output's look and intensity.
    #[must_use]
    pub fn look_at(look: LookId, intensity: f64) -> Self {
        Self::output(NodePatch {
            look: Some(look),
            look_intensity: Some(intensity),
            ..NodePatch::default()
        })
    }

    /// The patch that takes `field` of `node` in `from` (every part of it).
    #[must_use]
    pub fn of_field(from: &DevelopSettings, node: NodeId, field: SettingsField) -> Self {
        Self::of_fields(from, node, &[field])
    }

    /// The patch that takes the fields of `node` in `from` (every part of each). A field the node's
    /// kind does not have, or a node `from` does not have, is left out.
    #[must_use]
    pub fn of_fields(from: &DevelopSettings, node: NodeId, fields: &[SettingsField]) -> Self {
        let mut np = NodePatch::default();
        for field in fields {
            let one = match node {
                NodeId::Input => NodePatch::of_input_field(&from.nodes.input, *field),
                NodeId::Output => NodePatch::of_output_field(&from.nodes.output, *field),
                NodeId::Corrector(n) => from
                    .nodes
                    .corrector(n)
                    .and_then(|c| NodePatch::of_corrector_field(c, *field)),
            };
            if let Some(one) = one {
                np.absorb(one);
            }
        }
        if np == NodePatch::default() {
            Self::default()
        } else {
            Self::node(node, np)
        }
    }

    /// The patch that turns `from` into `to`: per node the fields that differ, a node `to` has and
    /// `from` has not whole and with its kind, `remove` for the correctors only `from` has, and
    /// `links` when they differ. This is what an undo entry holds (`docs/23` section 3.7); applying
    /// it to `from` gives `to`.
    #[must_use]
    pub fn between(from: &DevelopSettings, to: &DevelopSettings) -> Self {
        let mut patch = Self::default();
        let (added, removed) = from.diff_nodes(to);
        patch.remove = removed;
        for id in added {
            if let NodeId::Corrector(n) = id
                && let Some(c) = to.nodes.corrector(n)
            {
                patch.nodes.insert(id, NodePatch::whole_corrector(c));
            }
        }
        let mut by_node: BTreeMap<NodeId, Vec<SettingsField>> = BTreeMap::new();
        for r in from.diff(to) {
            by_node.entry(r.node).or_default().push(r.field);
        }
        for (id, fields) in by_node {
            for (id, np) in Self::of_fields(to, id, &fields).nodes {
                patch.nodes.insert(id, np);
            }
        }
        if from.nodes.links != to.nodes.links {
            patch.links = Some(to.nodes.links.clone());
        }
        patch
    }

    /// The Reset patch for a field of a node: restores its default (Auto for the automatic ones of
    /// the Input and Main, a manual 0 in another corrector, `default_look` on the Output).
    #[must_use]
    pub fn reset(node: NodeId, field: SettingsField, default_look: &LookId) -> Self {
        Self::of_field(&defaults_for(node, default_look), node, field)
    }

    /// The Reset patch of a node as its head's Reset has it: a corrector's trim and grade (not its
    /// name, bypass, position or masks), every field of the Input and of the Output.
    #[must_use]
    pub fn reset_node(node: NodeId, default_look: &LookId) -> Self {
        let fields: Vec<SettingsField> = SettingsField::ALL
            .into_iter()
            .filter(|f| {
                f.belongs_to(node.kind())
                    && !matches!(
                        f,
                        SettingsField::Name
                            | SettingsField::Bypass
                            | SettingsField::Pos
                            | SettingsField::Masks
                    )
            })
            .collect();
        Self::of_fields(&defaults_for(node, default_look), node, &fields)
    }

    /// What Copy settings takes from a photo, to paste onto others (`docs/23` section 3.8): the
    /// whole graph, every node and every link, but [`COPY_LEAVES_OUT`] (the Input's `crop`,
    /// `orientation` and `white_pick`, which belong to the photo's own geometry and pixels).
    ///
    /// * Auto stays Auto in the Input and Main: each photo gets its own automatic values (with Apple
    ///   Auto, its own answer). Manual values are taken as they are.
    /// * A white balance that came from a picked white (a white picked, the light preset and the
    ///   white balance both Auto) becomes a manual white balance at the Kelvin and tint the pick
    ///   resolved to: the point is not meaningful on another photo, the light it found is.
    /// * Object masks arrive empty (their clicks are not copied); the other rows are copied as they
    ///   are, and Subject, People and Background are worked out again on the new photo.
    ///
    /// Every part of every manual value is in the patch, so it applies to a photo that has no
    /// automatic estimates yet.
    #[must_use]
    pub fn copied_from(source: &ResolvedSettings) -> Self {
        let g = &source.settings.nodes;
        let mut input = NodePatch::whole_input(&g.input);
        input.crop = None;
        input.orientation = None;
        input.white_pick = None;
        let white_came_from_pick = g.input.white_pick.is_some()
            && g.input.light_source == LightSource::Auto
            && g.input.white_balance.is_auto();
        if white_came_from_pick {
            input.white_balance = Some(AutoOr::Manual(WhiteBalanceManualPatch {
                kelvin: Some(source.values.input.kelvin),
                tint: Some(source.values.input.tint),
            }));
        }
        let mut patch = Self::input(input);
        for (id, c) in g.correctors() {
            let mut c = c.clone();
            for row in &mut c.masks.items {
                *row = row.emptied();
            }
            patch
                .nodes
                .insert(NodeId::Corrector(id), NodePatch::whole_corrector(&c));
        }
        patch
            .nodes
            .insert(NodeId::Output, NodePatch::whole_output(&g.output));
        patch.links = Some(g.links.clone());
        patch
    }

    /// The patch a paste applies to `target`: this patch without the fields of
    /// [`COPY_LEAVES_OUT`] (a paste never touches the photo's own geometry, pick or sample), and,
    /// when it carries the links, a graph that replaces the target's: `remove` lists the target's
    /// correctors that the patch does not give.
    #[must_use]
    pub fn pasted_onto(&self, target: &DevelopSettings) -> Self {
        let mut patch = self.clone();
        if let Some(input) = patch.nodes.get_mut(&NodeId::Input) {
            input.crop = None;
            input.orientation = None;
            input.white_pick = None;
        }
        if patch.links.is_some() {
            let given: BTreeSet<NodeId> = patch.nodes.keys().copied().collect();
            for (id, _) in target.nodes.correctors() {
                let node = NodeId::Corrector(id);
                if id != MAIN_ID && !given.contains(&node) && !patch.remove.contains(&node) {
                    patch.remove.push(node);
                }
            }
        }
        patch
    }
}

/// The settings a node's defaults are taken from: the default graph, with the node added as a new
/// corrector when it is one (a manual 0 everywhere) and `default_look` on the Output.
fn defaults_for(node: NodeId, default_look: &LookId) -> DevelopSettings {
    let mut d = DevelopSettings::with_look(default_look.clone());
    if let NodeId::Corrector(n) = node
        && n != MAIN_ID
    {
        d.nodes.extra.insert(n, CorrectorNode::default());
    }
    d
}

/// The fields Copy settings leaves out: the Input's crop and orientation (the photo's own geometry)
/// and the Pick white point (a spot of that photo's pixels). See [`SettingsPatch::copied_from`].
pub const COPY_LEAVES_OUT: [FieldRef; 3] = [
    FieldRef {
        node: NodeId::Input,
        field: SettingsField::Crop,
    },
    FieldRef {
        node: NodeId::Input,
        field: SettingsField::Orientation,
    },
    FieldRef {
        node: NodeId::Input,
        field: SettingsField::WhitePick,
    },
];
