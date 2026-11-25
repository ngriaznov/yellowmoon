//! Presets in the core (`docs/30` sections 2 to 7): the groups, saving from a photo, applying through the core's Paste path,
//! matching, stacking, the missing look, the pre-ticked groups, the events and the wire types.

#![allow(clippy::field_reassign_with_default)]

mod common;

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use common::rich::{every_mask, rich};
use common::{Layout, Scene, estimates_of_file, light, wb_levels};
use serde_json::{Value, json};
use ym_color::look::LookInput;
use ym_core::app::{Core, CoreError, CoreOptions};
use ym_core::apple_auto::{AdvisorAnswer, AdvisorInfo, AppleAuto, AutoSteps};
use ym_core::auto::tables_of;
use ym_core::auto_provider::AutoProvider;
use ym_core::presets::{
    DamagedPreset, ImportOutcome, PresetApplyReport, PresetEntry, PresetGroup, PresetSource,
    changed_groups, missing_look_sentence,
};
use ym_core::session::{OpenOptions, PhotoId};
use ym_core::settings::{
    AutoOr, COPY_LEAVES_OUT, CorrectorNode, DevelopSettings, FilmNode, MaskShape, NodeId, NodeKind,
    NodeLink, NodePatch, PercentSetting, ResolvedSettings, SettingsError, SettingsField,
    SettingsPatch, WhiteBalanceManualPatch, WhiteBalanceValue,
};
use ym_looks::library::LookId;

const MODEL: &str = "ILCE-7RM4";
const PROFILE: &str = "sony-ilce-7rm4";

struct Rig {
    core: Core,
    app: tempfile::TempDir,
    _root: tempfile::TempDir,
    folder: PathBuf,
    ids: Vec<PhotoId>,
}

fn rig(n: usize) -> Rig {
    let root = tempfile::tempdir().expect("root");
    let folder = root.path().join("Shoot");
    std::fs::create_dir(&folder).expect("folder");
    for k in 0..n {
        let scene = Scene::new(
            300,
            200,
            light(PROFILE, 4800.0 + 400.0 * k as f64),
            Layout::Patchwork,
            0.35,
        )
        .noise(2e-4, 1e-5)
        .seed(k as u64 + 1);
        scene
            .builder(MODEL)
            .capture_time(&format!("2026:10:04 12:{:02}:00", 3 * k), "07", None)
            .write_to(
                &folder.join(format!("DSC{:05}.ARW", k + 1)),
                &scene.mosaic(),
            )
            .expect("write");
    }
    open(root, folder)
}

fn open(root: tempfile::TempDir, folder: PathBuf) -> Rig {
    let app = tempfile::tempdir().expect("app root");
    let mut options = CoreOptions::new(common::data_root(), app.path().to_path_buf());
    options.decode_threads = 2;
    options.thumbnails = false;
    options.home = Some(root.path().to_path_buf());
    let core = Core::open(options).expect("core");
    let out = core
        .session
        .open(std::slice::from_ref(&folder), OpenOptions::default());
    assert!(
        core.session.wait_idle(Duration::from_secs(120)),
        "developed"
    );
    Rig {
        core,
        app,
        _root: root,
        folder,
        ids: out.added,
    }
}

impl Rig {
    fn settings(&self, id: PhotoId) -> DevelopSettings {
        self.core.session.settings(id).expect("settings").settings
    }

    fn resolved(&self, id: PhotoId) -> ResolvedSettings {
        self.core.session.settings(id).expect("settings")
    }

    fn set(&self, id: PhotoId, s: DevelopSettings) {
        self.core.session.replace_settings(id, s).expect("replaced");
    }

    fn patch(&self, id: PhotoId, p: &SettingsPatch) {
        self.core.session.update_settings(id, p).expect("patched");
    }

    fn save(&self, name: &str, groups: &[PresetGroup], from: PhotoId) -> PresetEntry {
        self.core
            .preset_save(name, groups, &PresetSource::Photo { photo_id: from }, false)
            .expect("saved")
    }

    fn apply(&self, name: &str, to: &[PhotoId]) -> PresetApplyReport {
        self.core.preset_apply(name, to).expect("applied")
    }
}

/// Every Object row's clicks and every Brush row's strokes taken out, as a preset does.
fn emptied(s: &DevelopSettings) -> DevelopSettings {
    let mut s = s.clone();
    let g = &mut s.nodes;
    let masks = std::iter::once(&mut g.main.masks)
        .chain(g.extra.values_mut().map(|c| &mut c.masks))
        .chain(g.films.values_mut().map(|f| &mut f.masks))
        .chain(g.uniforms.values_mut().map(|u| &mut u.masks))
        .chain(g.blurs.values_mut().map(|b| &mut b.masks))
        .chain(g.references.values_mut().map(|r| &mut r.masks));
    for m in masks {
        for row in &mut m.items {
            *row = row.emptied_for_preset();
        }
    }
    s
}

/// What the app does with the source in the rich tests: a look that is in the library, and everything else.
fn rich_with_a_usable_look() -> DevelopSettings {
    let mut s = rich();
    s.output_mut().look = LookId::neutral();
    s
}

fn fixed_fields() -> Vec<(NodeId, SettingsField)> {
    let mut v = Vec::new();
    for node in [NodeId::Input, NodeId::MAIN, NodeId::Output] {
        for f in SettingsField::ALL {
            if f.belongs_to(node.kind()) {
                v.push((node, f));
            }
        }
    }
    v
}

fn same(a: &DevelopSettings, b: &DevelopSettings, node: NodeId, field: SettingsField) -> bool {
    SettingsPatch::of_field(a, node, field) == SettingsPatch::of_field(b, node, field)
}

fn never_saved(node: NodeId, field: SettingsField) -> bool {
    COPY_LEAVES_OUT
        .iter()
        .any(|r| r.node == node && r.field == field)
}

// ---------------------------------------------------------------------------------------------
// The groups
// ---------------------------------------------------------------------------------------------

#[test]
fn every_field_of_the_settings_is_in_exactly_one_group_besides_the_three_never_saved() {
    let groups_of = |node: NodeId, field: SettingsField| -> Vec<PresetGroup> {
        PresetGroup::ALL
            .into_iter()
            .filter(|g| g.fields().contains(&(node, field)))
            .collect()
    };
    let mut seen = 0;
    for (node, field) in fixed_fields() {
        let groups = groups_of(node, field);
        if never_saved(node, field) {
            assert!(
                groups.is_empty(),
                "{node}.{field:?} is never saved: {groups:?}"
            );
        } else {
            assert_eq!(groups.len(), 1, "{node}.{field:?} is in {groups:?}");
        }
        seen += 1;
    }
    assert!(
        seen >= 30,
        "the Input, Main and the Output have {seen} fields"
    );
    // `pos` of the three fixed nodes goes with nodes, and Main's own fields too.
    for node in [NodeId::Input, NodeId::MAIN, NodeId::Output] {
        assert_eq!(groups_of(node, SettingsField::Pos), [PresetGroup::Nodes]);
    }
    for field in [
        SettingsField::Name,
        SettingsField::Bypass,
        SettingsField::Trim,
        SettingsField::Masks,
    ] {
        assert_eq!(groups_of(NodeId::MAIN, field), [PresetGroup::Nodes]);
    }
    // The never-saved three are the Input's crop, orientation and white pick (what Copy settings leaves out).
    assert_eq!(COPY_LEAVES_OUT.len(), 3);
    assert!(COPY_LEAVES_OUT.iter().all(|r| r.node == NodeId::Input));
    // The table of docs/30 section 2.
    use SettingsField as F;
    for (group, fields) in [
        (PresetGroup::Look, vec![F::Look, F::LookIntensity]),
        (PresetGroup::Lens, vec![F::Lens]),
        (
            PresetGroup::Light,
            vec![
                F::Exposure,
                F::Contrast,
                F::Brightness,
                F::Highlights,
                F::Shadows,
                F::White,
                F::Black,
            ],
        ),
        (
            PresetGroup::Color,
            vec![
                F::LightSource,
                F::WhiteBalance,
                F::ColorBalance,
                F::Saturation,
                F::ColorWarp,
            ],
        ),
        (
            PresetGroup::Detail,
            vec![F::Sharpen, F::NoiseReduction, F::OutputSharpening],
        ),
        (PresetGroup::Curve, vec![F::Curve]),
        (PresetGroup::Raw, vec![F::Decode, F::Rolloff]),
    ] {
        let got: BTreeSet<String> = group
            .fields()
            .iter()
            .map(|(_, f)| f.name().to_owned())
            .collect();
        let want: BTreeSet<String> = fields.iter().map(|f| f.name().to_owned()).collect();
        assert_eq!(got, want, "{}", group.id());
    }
    // The ids, labels and order.
    let ids: Vec<_> = PresetGroup::ALL.iter().map(|g| g.id()).collect();
    assert_eq!(
        ids,
        [
            "look", "lens", "light", "color", "detail", "curve", "raw", "nodes"
        ]
    );
    let labels: Vec<_> = PresetGroup::ALL.iter().map(|g| g.label()).collect();
    assert_eq!(
        labels,
        [
            "Look", "Lens", "Light", "Colour", "Detail", "Curve", "RAW", "Nodes"
        ]
    );
    assert_eq!(
        serde_json::to_string(&PresetGroup::Color).unwrap(),
        "\"color\""
    );
    assert_eq!(PresetGroup::from_id("raw"), Some(PresetGroup::Raw));
    assert!(PresetGroup::ALL.windows(2).all(|w| w[0] < w[1]));
    // A field that exists in no group would fail the first loop; the other kinds of node are in `nodes` whole.
    assert!(
        SettingsField::ALL
            .iter()
            .all(|f| NodeKind::ALL.iter().any(|k| f.belongs_to(*k)))
    );
}

#[test]
fn a_preset_of_one_group_holds_that_groups_fields_and_nothing_else_for_every_group() {
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    r.set(source, rich_with_a_usable_look());
    let initial = r.settings(target);
    let want = emptied(&r.settings(source));
    for group in PresetGroup::ALL {
        r.set(target, initial.clone());
        let name = format!("Only {}", group.label());
        let entry = r.save(&name, &[group], source);
        assert_eq!(entry.groups, [group]);
        let report = r.apply(&name, &[target]);
        assert_eq!(report.pasted.len(), 1);
        assert!(report.missing_look.is_none(), "{group:?}");
        let after = r.settings(target);
        for (node, field) in fixed_fields() {
            let fields = group.fields();
            let expect = if never_saved(node, field) || !fields.contains(&(node, field)) {
                &initial
            } else {
                &want
            };
            assert!(
                same(&after, expect, node, field),
                "{group:?}: {node}.{field:?} is {}",
                if std::ptr::eq(expect, &initial) {
                    "not the target's own"
                } else {
                    "not the source's"
                }
            );
        }
        if group == PresetGroup::Nodes {
            assert_eq!(after.nodes.extra, want.nodes.extra);
            assert_eq!(after.nodes.links, want.nodes.links);
        } else {
            assert_eq!(after.nodes.extra, initial.nodes.extra, "{group:?}");
            assert_eq!(after.nodes.links, initial.nodes.links, "{group:?}");
        }
        // The group changes the photo and nothing else does: every other group's patch is the target's own.
        for other in PresetGroup::ALL
            .into_iter()
            .filter(|g| *g != group && *g != PresetGroup::Nodes)
        {
            for (node, field) in other.fields() {
                assert!(
                    same(&after, &initial, node, field),
                    "{group:?} moved {node}.{field:?}"
                );
            }
        }
    }
}

#[test]
fn film_nodes_and_the_mixers_travel_with_nodes() {
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    let mut s = DevelopSettings::with_look(r.core.looks.effective_default());
    s.nodes.films.insert(4, FilmNode::default());
    let layer = ym_core::settings::MixerNode::new_of(NodeKind::Layer).expect("a layer");
    s.nodes.mixers.insert(5, layer);
    r.set(source, s.clone());
    r.save("Nodes", &[PresetGroup::Nodes], source);
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    own.nodes.extra.insert(9, CorrectorNode::default());
    r.set(target, own);
    r.apply("Nodes", &[target]);
    let after = r.settings(target);
    assert_eq!(after.nodes.films, s.nodes.films);
    assert_eq!(after.nodes.mixers, s.nodes.mixers);
    assert!(
        after.nodes.extra.is_empty(),
        "the preset's nodes replace the photo's other nodes"
    );
}

/// CU-07: a Colour Uniform node (target, amounts, masks with the seeded Colour range row) is in the `nodes` group, is saved from
/// a photo that has it, applies to another photo, and arrives with its Brush and Object rows emptied. (Settings and round trips
/// only: the node renders as the identity until CU-04.)
#[test]
fn a_color_uniform_node_travels_with_the_nodes_group() {
    use ym_core::settings::{ColorUniformNode, NodeId};
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    let doc = json!({ "nodes": {
        "items": [
            { "id": 2, "kind": "color_uniform", "name": "Skin", "pos": [1, 0],
              "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": -0.03 },
              "compress": { "hue": 60, "saturation": 40, "lightness": 10 },
              "masks": { "invert": false, "items": [
                { "kind": "color_range", "mode": "add", "invert": false, "enabled": true, "opacity": 1,
                  "center": [0.0875, 0.0423], "width": 0.05, "softness": 0.5 },
                { "kind": "brush", "mode": "add", "strokes": [ { "size": 0.02, "points": [[0.1, 0.2], [0.3, 0.4]] } ] }
              ] } } ],
        "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    let (mut s, _) = DevelopSettings::from_value(doc, &r.core.looks.effective_default())
        .expect("a graph with the node");
    s.nodes.extra.clear();
    r.set(source, s.clone());
    let want = emptied(&r.settings(source));
    let node: &ColorUniformNode = want.nodes.color_uniform(2).expect("the node");
    assert_eq!(node.name, "Skin");
    assert!(node.target.is_some());
    assert_eq!(node.masks.items.len(), 2, "the rows stay");
    // The group's own table has nothing for it: it is carried whole with the other nodes.
    assert!(
        PresetGroup::Nodes
            .fields()
            .iter()
            .all(|(n, _)| !matches!(n, NodeId::Corrector(2)))
    );
    r.save("Uniform", &[PresetGroup::Nodes], source);
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    own.nodes.extra.insert(9, CorrectorNode::default());
    r.set(target, own);
    r.apply("Uniform", &[target]);
    let after = r.settings(target);
    assert_eq!(after.nodes.uniforms, want.nodes.uniforms);
    assert_eq!(after.nodes.links, want.nodes.links);
    assert!(
        after.nodes.extra.is_empty(),
        "the preset's nodes replace the photo's own"
    );
    let json = serde_json::to_string(&after.nodes.uniforms).unwrap();
    assert!(
        !json.contains("\"points\":[["),
        "no Brush stroke arrives: {json}"
    );
    // Another group leaves the target's own nodes alone.
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    own.nodes.extra.insert(9, CorrectorNode::default());
    r.set(target, own.clone());
    r.save("Light only", &[PresetGroup::Light], source);
    r.apply("Light only", &[target]);
    assert!(r.settings(target).nodes.uniforms.is_empty());
    assert_eq!(r.settings(target).nodes.extra, own.nodes.extra);
}

/// RG-02: a Reference Grade node (fitted-style curves, saturation, hue curves, element grades, Strength, Skin protection and a mask
/// row) is in the `nodes` group like any node, is saved from a photo that has it, applies the same numbers to another photo (it does
/// not re-fit: there is nothing to fit from) and arrives with its Brush strokes emptied. The field test of `docs/30` section 14
/// (every field is in one group) holds for the new fields: they are in the group that carries every other node whole.
#[test]
fn a_reference_grade_node_travels_with_the_nodes_group_and_applies_the_same_numbers() {
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    let doc = json!({ "nodes": {
        "items": [
            { "id": 2, "kind": "reference_grade", "name": "Match", "pos": [1, 0],
              "grade": { "saturation": { "kind": "manual", "value": -12 },
                         "curve": { "luma": [[0, 0], [0.2784, 0.27], [1, 1]],
                                    "blue": [[0, 0], [0.2784, 0.27], [1, 1]] } },
              "hue_curves": { "hue_vs_hue": [ { "hue_deg": 0, "value": -4 }, { "hue_deg": 120, "value": 6 } ],
                              "hue_vs_sat": [ { "hue_deg": 240, "value": 1.2 } ] },
              "elements": { "sky": { "exposure_ev": -0.3, "hue_deg": 8, "chroma": 1.1 },
                            "skin": { "hue_deg": 3 }, "subject": { "exposure_ev": 0.2 } },
              "strength": 65, "skin_protection": false,
              "masks": { "invert": false, "items": [
                { "kind": "brush", "mode": "add", "strokes": [ { "size": 0.02, "points": [[0.1, 0.2], [0.3, 0.4]] } ] },
                { "kind": "radial", "mode": "add" }
              ] } } ],
        "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    let (mut s, _) = DevelopSettings::from_value(doc, &r.core.looks.effective_default())
        .expect("a graph with the node");
    s.nodes.extra.clear();
    r.set(source, s.clone());
    let want = emptied(&r.settings(source));
    let node = want.nodes.reference_grade(2).expect("the node");
    assert_eq!(node.strength, 65.0);
    assert!(!node.skin_protection);
    assert_eq!(node.masks.items.len(), 2, "the rows stay");
    // The group's own table has nothing for it: it is carried whole with the other nodes.
    assert!(
        PresetGroup::Nodes
            .fields()
            .iter()
            .all(|(n, _)| !matches!(n, NodeId::Corrector(2)))
    );
    r.save("Match", &[PresetGroup::Nodes], source);
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    own.nodes.extra.insert(9, CorrectorNode::default());
    r.set(target, own);
    r.apply("Match", &[target]);
    let after = r.settings(target);
    assert_eq!(after.nodes.references, want.nodes.references);
    assert_eq!(after.nodes.links, want.nodes.links);
    assert!(after.nodes.extra.is_empty());
    let json = serde_json::to_string(&after.nodes.references).unwrap();
    assert!(
        !json.contains("\"points\":[["),
        "no Brush stroke arrives: {json}"
    );
    // Another group leaves the target's own nodes alone.
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    own.nodes.extra.insert(9, CorrectorNode::default());
    r.set(target, own.clone());
    r.save("Light only", &[PresetGroup::Light], source);
    r.apply("Light only", &[target]);
    assert!(r.settings(target).nodes.references.is_empty());
    assert_eq!(r.settings(target).nodes.extra, own.nodes.extra);
}

/// BL-11: a Blur node (type, every group, the Zoom centre, the switch and masks: a Subject row, a Radial row and a Brush row) is
/// in the `nodes` group, is saved from a photo that has it, applies to another photo and arrives with its Brush strokes
/// emptied, its Subject row kept (it is worked out again there) and its Zoom centre carried. (Settings and round trips only: the
/// node renders as the identity until BL-04.)
#[test]
fn a_blur_node_travels_with_the_nodes_group() {
    use ym_core::settings::{BlurNode, BlurType};
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    let doc = json!({ "nodes": {
        "items": [
            { "id": 2, "kind": "blur", "name": "Background", "pos": [1, 0], "type": "zoom",
              "soft": { "radius": 33 }, "lens": { "radius": 50, "blades": 6, "rotation_deg": 15, "highlights": 60 },
              "motion": { "length": 40, "angle_deg": 30 },
              "zoom": { "amount": 35, "center": [0.25, 0.75] }, "respect_mask_edges": false,
              "masks": { "invert": false, "items": [
                { "kind": "subject", "mode": "add", "invert": true, "enabled": true, "opacity": 1, "feather": 0.08 },
                { "kind": "radial" },
                { "kind": "brush", "mode": "add", "strokes": [ { "size": 0.02, "points": [[0.1, 0.2], [0.3, 0.4]] } ] }
              ] } } ],
        "links": [
            { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
            { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
            { "from": { "node": 2, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } });
    let (mut s, _) = DevelopSettings::from_value(doc, &r.core.looks.effective_default())
        .expect("a graph with the node");
    s.nodes.extra.clear();
    r.set(source, s.clone());
    let want = emptied(&r.settings(source));
    let node: &BlurNode = want.nodes.blur(2).expect("the node");
    assert_eq!(node.name, "Background");
    assert_eq!(node.blur_type, BlurType::Zoom);
    assert_eq!(node.zoom.center, [0.25, 0.75]);
    assert_eq!(node.masks.items.len(), 3, "the rows stay");
    // The group's own table has nothing for it: it is carried whole with the other nodes.
    assert!(
        PresetGroup::Nodes
            .fields()
            .iter()
            .all(|(n, _)| !matches!(n, NodeId::Corrector(2)))
    );
    r.save("Blurred", &[PresetGroup::Nodes], source);
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    own.nodes.extra.insert(9, CorrectorNode::default());
    r.set(target, own);
    r.apply("Blurred", &[target]);
    let after = r.settings(target);
    assert_eq!(after.nodes.blurs, want.nodes.blurs);
    assert_eq!(after.nodes.links, want.nodes.links);
    assert!(
        after.nodes.extra.is_empty(),
        "the preset's nodes replace the photo's own"
    );
    let arrived = after.nodes.blur(2).expect("the node arrived");
    assert_eq!(arrived.zoom.center, [0.25, 0.75], "the centre is carried");
    assert!(!arrived.respect_mask_edges);
    assert_eq!(
        arrived.masks.items[0].shape.kind_name(),
        "subject",
        "the Subject row is kept, to be worked out again here"
    );
    let json = serde_json::to_string(&after.nodes.blurs).unwrap();
    assert!(
        !json.contains("\"points\":[["),
        "no Brush stroke arrives: {json}"
    );
    // Another group leaves the target's own nodes alone.
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    own.nodes.extra.insert(9, CorrectorNode::default());
    r.set(target, own.clone());
    r.save("Light only", &[PresetGroup::Light], source);
    r.apply("Light only", &[target]);
    assert!(r.settings(target).nodes.blurs.is_empty());
    assert_eq!(r.settings(target).nodes.extra, own.nodes.extra);
}

/// S-19: a Face Refinement node's Teeth Desaturation and Gamma travel with the Nodes group, per face.
#[test]
fn a_face_refinement_nodes_teeth_controls_travel_with_the_nodes_group() {
    use ym_core::settings::{FaceRefinementNode, FaceSettings};
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    let mut s = DevelopSettings::with_look(r.core.looks.effective_default());
    let node = FaceRefinementNode {
        same_for_all: false,
        all: FaceSettings {
            teeth_desaturation: 40.0,
            teeth_gamma: -30.0,
            ..FaceSettings::default()
        },
        faces: vec![
            FaceSettings {
                teeth_desaturation: 70.0,
                teeth_gamma: 55.0,
                ..FaceSettings::default()
            },
            FaceSettings::default(),
        ],
        ..FaceRefinementNode::default()
    };
    s.nodes.faces.insert(6, node.clone());
    r.set(source, s);
    r.save("Faces", &[PresetGroup::Nodes], source);
    r.set(
        target,
        DevelopSettings::with_look(r.core.looks.effective_default()),
    );
    r.apply("Faces", &[target]);
    let after = r.settings(target);
    let got = after.nodes.faces.get(&6).expect("the Face Refinement node");
    assert_eq!(got.all, node.all);
    assert_eq!(got.faces, node.faces);
    assert_eq!(got.faces[0].teeth_gamma, 55.0);
}

// ---------------------------------------------------------------------------------------------
// Auto stays Auto
// ---------------------------------------------------------------------------------------------

#[test]
fn auto_stays_auto_and_a_manual_value_is_carried() {
    let r = rig(3);
    let (auto_photo, manual_photo, target) = (r.ids[0], r.ids[1], r.ids[2]);
    r.patch(
        manual_photo,
        &SettingsPatch::exposure_ev(0.7).and(
            NodeId::MAIN,
            NodePatch {
                grade: Some(ym_core::settings::GradePatch {
                    contrast: Some(PercentSetting::manual_value(22.0)),
                    ..Default::default()
                }),
                ..NodePatch::default()
            },
        ),
    );
    // A preset of Light from a photo with Auto exposure sets the target's exposure to Auto.
    r.patch(target, &SettingsPatch::exposure_ev(-1.1));
    r.save("Auto light", &[PresetGroup::Light], auto_photo);
    assert!(r.settings(target).input().exposure != AutoOr::Auto);
    r.apply("Auto light", &[target]);
    let after = r.settings(target);
    assert!(after.input().exposure.is_auto(), "Auto stays Auto");
    assert!(after.main_grade().contrast.is_auto());
    assert!(r.resolved(target).auto.input.exposure);
    // Each photo keeps its own estimate: the resolved exposure is the target's, not the source's.
    assert_eq!(
        r.resolved(target).values.input.exposure_ev,
        r.core
            .session
            .estimates(target)
            .expect("estimates")
            .exposure
            .ev
    );
    // A manual exposure is carried to a photo that is Auto.
    r.save("Manual light", &[PresetGroup::Light], manual_photo);
    r.apply("Manual light", &[target]);
    let after = r.settings(target);
    assert_eq!(
        after.input().exposure,
        ym_core::settings::ExposureSetting::manual_ev(0.7)
    );
    assert_eq!(
        after.main_grade().contrast,
        PercentSetting::manual_value(22.0)
    );
    // The file says Auto too.
    let text = std::fs::read_to_string(r.app.path().join("presets/Auto light.ympreset")).unwrap();
    assert!(text.contains(r#""exposure":{"kind":"auto"}"#), "{text}");
}

// ---------------------------------------------------------------------------------------------
// A picked white
// ---------------------------------------------------------------------------------------------

fn pick_rig() -> Rig {
    let root = tempfile::tempdir().expect("root");
    let folder = root.path().join("Pick");
    std::fs::create_dir(&folder).expect("folder");
    // Lit at 3000 K, but the camera's white balance says daylight.
    let s = Scene::new(512, 384, light(PROFILE, 3000.0), Layout::Patchwork, 0.5).noise(1e-4, 1e-6);
    s.builder(MODEL)
        .wb(Some(wb_levels(light(PROFILE, 6500.0))))
        .write_to(&folder.join("a.ARW"), &s.mosaic())
        .expect("write");
    let plain =
        Scene::new(384, 288, light(PROFILE, 5200.0), Layout::Patchwork, 0.3).noise(2e-4, 1e-6);
    plain
        .builder(MODEL)
        .write_to(&folder.join("b.ARW"), &plain.mosaic())
        .expect("write");
    open(root, folder)
}

#[test]
fn a_picked_white_is_saved_as_the_manual_kelvin_and_tint_it_resolved_to() {
    let r = pick_rig();
    let (a, b) = (r.ids[0], r.ids[1]);
    let pick = SettingsPatch::default().and(
        NodeId::Input,
        NodePatch {
            white_pick: Some(Some([0.4167, 0.125])),
            ..NodePatch::default()
        },
    );
    r.patch(a, &pick);
    let picked = r.resolved(a);
    assert!(picked.auto.input.white_balance && picked.auto.input.light_source);
    assert!((picked.values.input.kelvin - 3000.0).abs() <= 100.0);
    r.save("Tungsten", &[PresetGroup::Color], a);

    // The same numbers as Copy settings'.
    let copied = SettingsPatch::copied_from(&picked);
    let copied_wb = copied.input_patch().and_then(|p| p.white_balance);
    let expected = AutoOr::Manual(WhiteBalanceManualPatch {
        kelvin: Some(picked.values.input.kelvin),
        tint: Some(picked.values.input.tint),
    });
    assert_eq!(copied_wb, Some(expected));
    let stored = r.core.presets.store().get("Tungsten").expect("stored");
    assert_eq!(
        stored.settings.input().white_balance,
        AutoOr::Manual(WhiteBalanceValue {
            kelvin: picked.values.input.kelvin,
            tint: picked.values.input.tint,
        })
    );
    assert!(
        stored.settings.input().white_pick.is_none(),
        "the point stays with its photo"
    );
    assert_eq!(
        stored.settings.input().light_source,
        ym_color::wb::LightSource::Auto
    );

    // Applied to another photo it is the manual white balance; the other photo's own pick, if it has one, stays.
    r.apply("Tungsten", &[b]);
    assert_eq!(
        r.settings(b).input().white_balance,
        AutoOr::Manual(WhiteBalanceValue {
            kelvin: picked.values.input.kelvin,
            tint: picked.values.input.tint,
        })
    );
    r.patch(
        b,
        &SettingsPatch::default().and(
            NodeId::Input,
            NodePatch {
                white_pick: Some(Some([0.2, 0.2])),
                ..NodePatch::default()
            },
        ),
    );
    // An Auto preset does not clear the pick: the pick stays the photo's own.
    r.save("Auto colour", &[PresetGroup::Color], r.ids[0]);
    let mut clean = DevelopSettings::with_look(r.core.looks.effective_default());
    clean.input_mut().white_balance = AutoOr::Auto;
    let auto_preset = ym_core::presets::Preset::from_settings(
        "Plain colour",
        &clean,
        &[PresetGroup::Color],
        None,
    )
    .expect("preset");
    r.core
        .presets
        .store()
        .save(&auto_preset, false)
        .expect("saved");
    let before = r.settings(b);
    r.apply("Plain colour", &[b]);
    let after = r.settings(b);
    assert!(after.input().white_balance.is_auto());
    assert_eq!(after.input().white_pick, before.input().white_pick);
    assert_eq!(after.input().white_pick, Some([0.2, 0.2]));
}

// ---------------------------------------------------------------------------------------------
// What a preset never holds
// ---------------------------------------------------------------------------------------------

#[test]
fn object_clicks_and_brush_strokes_are_not_saved_and_the_other_rows_are_kept() {
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    let mut s = DevelopSettings::with_look(r.core.looks.effective_default());
    s.main_mut().masks = every_mask();
    let mut node2 = CorrectorNode::default();
    node2.masks = every_mask();
    s.nodes.extra.insert(2, node2);
    s.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(2)),
        NodeLink::picture(NodeId::Corrector(2), NodeId::Output),
    ];
    r.set(source, s.clone());
    r.save("Masks", &[PresetGroup::Nodes], source);
    let stored = r.core.presets.store().get("Masks").expect("stored");
    let rows = |settings: &DevelopSettings| {
        vec![
            settings.main().masks.clone(),
            settings.nodes.extra[&2].masks.clone(),
        ]
    };
    let (source_masks, saved_masks) = (rows(&s), rows(&stored.settings));
    for (src, saved) in source_masks.iter().zip(&saved_masks) {
        assert_eq!(src.items.len(), saved.items.len(), "every row is there");
        assert_eq!(src.invert, saved.invert);
        for (a, b) in src.items.iter().zip(&saved.items) {
            match (&a.shape, &b.shape) {
                (MaskShape::Object(_), MaskShape::Object(o)) => {
                    assert!(
                        o.points.is_empty() && o.box_.is_none(),
                        "an Object row arrives empty"
                    );
                }
                (MaskShape::Brush(x), MaskShape::Brush(y)) => {
                    assert!(!x.strokes.is_empty());
                    assert!(y.strokes.is_empty(), "a Brush row arrives with no strokes");
                    assert_eq!(y.feather, x.feather, "it keeps its feather");
                    assert_eq!(
                        (a.mode, a.invert, a.enabled, a.opacity),
                        (b.mode, b.invert, b.enabled, b.opacity)
                    );
                }
                // Subject, People, Background, Linear, Radial and the ranges are kept as they are.
                _ => assert_eq!(a, b),
            }
        }
    }
    // Applying it puts the same empty rows on the target; Copy settings, in contrast, copies the strokes.
    r.apply("Masks", &[target]);
    assert_eq!(rows(&r.settings(target)), saved_masks);
    let copied = SettingsPatch::copied_from(&r.resolved(source));
    let strokes = copied
        .main_patch()
        .and_then(|p| p.masks.as_ref())
        .and_then(|m| m.items.as_ref())
        .map(|items| {
            items
                .iter()
                .any(|i| matches!(&i.shape, MaskShape::Brush(b) if !b.strokes.is_empty()))
        });
    assert_eq!(strokes, Some(true), "Copy settings keeps the strokes");
}

#[test]
fn the_crop_the_turn_and_the_white_pick_are_never_written_and_never_touched() {
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    r.set(source, rich_with_a_usable_look());
    let src = r.settings(source);
    assert!(src.input().crop.is_some() && src.input().white_pick.is_some());
    let entry = r.save("Everything", &PresetGroup::ALL, source);
    assert_eq!(entry.groups, PresetGroup::ALL);
    let text = std::fs::read_to_string(r.app.path().join("presets/Everything.ympreset")).unwrap();
    let doc: Value = serde_json::from_str(&text).unwrap();
    let input = &doc["settings"]["nodes"]["items"][0];
    assert_eq!(input["id"], "input");
    for key in ["crop", "orientation", "white_pick"] {
        assert!(input.get(key).is_none(), "{key} is in the file: {input}");
    }
    assert!(input.get("exposure").is_some() && input.get("pos").is_some());
    // The target keeps its own.
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    own.input_mut().orientation = ym_core::settings::Orientation::Deg270;
    own.input_mut().crop = Some(ym_core::settings::Crop {
        rect: ym_core::settings::NormRect {
            x: 0.0,
            y: 0.0,
            width: 0.5,
            height: 0.5,
        },
        angle_deg: 1.0,
        aspect: ym_core::settings::CropAspect::Free,
        auto: false,
    });
    own.input_mut().white_pick = Some([0.3, 0.4]);
    r.set(target, own.clone());
    r.apply("Everything", &[target]);
    let after = r.settings(target);
    assert_eq!(after.input().orientation, own.input().orientation);
    assert_eq!(after.input().crop, own.input().crop);
    assert_eq!(after.input().white_pick, own.input().white_pick);
    // ... also when a hand-written file says otherwise.
    let hand = r#"{"yellowmoon_preset":1,"name":"Hand","groups":["light"],"settings":{"nodes":{"items":[
        {"id":"input","orientation":90,"white_pick":[0.9,0.9],"exposure":{"kind":"manual","ev":0.1}}]}}}"#;
    std::fs::write(r.app.path().join("presets/Hand.ympreset"), hand).unwrap();
    r.apply("Hand", &[target]);
    let after = r.settings(target);
    assert_eq!(after.input().orientation, own.input().orientation);
    assert_eq!(after.input().white_pick, own.input().white_pick);
    assert_eq!(
        after.input().exposure,
        ym_core::settings::ExposureSetting::manual_ev(0.1)
    );
}

// ---------------------------------------------------------------------------------------------
// Nodes, stacking, matching
// ---------------------------------------------------------------------------------------------

#[test]
fn with_nodes_the_other_nodes_are_replaced_and_without_it_they_stay() {
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    let mut s = DevelopSettings::with_look(r.core.looks.effective_default());
    let mut sky = CorrectorNode::default();
    sky.name = "Sky".to_owned();
    sky.trim.exposure_ev = -0.5;
    s.nodes.extra.insert(2, sky);
    s.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(2)),
        NodeLink::picture(NodeId::Corrector(2), NodeId::Output),
    ];
    s.main_mut().name = "Base".to_owned();
    s.main_grade_mut().contrast = PercentSetting::manual_value(30.0);
    r.set(source, s.clone());
    r.save(
        "With nodes",
        &[PresetGroup::Nodes, PresetGroup::Light],
        source,
    );
    r.save("Light only", &[PresetGroup::Light], source);

    // The target has three correctors of its own.
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    for (id, name) in [(7, "Mine"), (8, "Also mine"), (9, "Third")] {
        let mut c = CorrectorNode::default();
        c.name = name.to_owned();
        own.nodes.extra.insert(id, c);
    }
    own.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(7)),
        NodeLink::picture(NodeId::Corrector(7), NodeId::Corrector(8)),
        NodeLink::picture(NodeId::Corrector(8), NodeId::Corrector(9)),
        NodeLink::picture(NodeId::Corrector(9), NodeId::Output),
    ];
    r.set(target, own.clone());

    // Without `nodes` a preset of Light changes Main's grade and nothing else.
    r.apply("Light only", &[target]);
    let after = r.settings(target);
    assert_eq!(
        after.main_grade().contrast,
        PercentSetting::manual_value(30.0)
    );
    assert_eq!(after.nodes.extra, own.nodes.extra);
    assert_eq!(after.nodes.links, own.nodes.links);
    assert_eq!(after.main().name, "Main", "Main's name is a Nodes field");

    // With `nodes` the preset's nodes replace them.
    r.set(target, own.clone());
    r.apply("With nodes", &[target]);
    let after = r.settings(target);
    assert_eq!(after.nodes.extra, s.nodes.extra);
    assert_eq!(after.nodes.links, s.nodes.links);
    assert_eq!(after.main().name, "Base");
    assert!(after.nodes.corrector(7).is_none());
}

#[test]
fn two_presets_stack_and_the_later_one_wins_on_a_shared_group() {
    let r = rig(3);
    let (one, two, target) = (r.ids[0], r.ids[1], r.ids[2]);
    let mut a = DevelopSettings::with_look(r.core.looks.effective_default());
    a.main_grade_mut().brightness = 10.0;
    a.main_grade_mut().saturation = PercentSetting::manual_value(40.0);
    a.output_mut().look_intensity = 55.0;
    r.set(one, a);
    let mut b = DevelopSettings::with_look(r.core.looks.effective_default());
    b.main_grade_mut().brightness = 20.0;
    b.input_mut().sharpen = AutoOr::Manual(ym_core::settings::SharpenValue {
        amount: 90.0,
        radius_px: 1.0,
    });
    r.set(two, b);
    r.save(
        "A",
        &[PresetGroup::Light, PresetGroup::Color, PresetGroup::Look],
        one,
    );
    r.save("B", &[PresetGroup::Light, PresetGroup::Detail], two);
    r.apply("A", &[target]);
    r.apply("B", &[target]);
    let s = r.settings(target);
    assert_eq!(
        s.main_grade().brightness,
        20.0,
        "the later one wins where the groups are shared"
    );
    assert_eq!(
        s.main_grade().saturation,
        PercentSetting::manual_value(40.0),
        "A's Colour is kept"
    );
    assert_eq!(s.output().look_intensity, 55.0, "A's Look is kept");
    assert_eq!(
        s.input().sharpen,
        AutoOr::Manual(ym_core::settings::SharpenValue {
            amount: 90.0,
            radius_px: 1.0
        })
    );
    // The other order gives the other winner.
    r.set(
        target,
        DevelopSettings::with_look(r.core.looks.effective_default()),
    );
    r.apply("B", &[target]);
    r.apply("A", &[target]);
    assert_eq!(r.settings(target).main_grade().brightness, 10.0);
}

#[test]
fn a_preset_matches_after_it_is_applied_and_not_after_a_field_of_its_groups_changes() {
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    r.set(source, rich_with_a_usable_look());
    r.save(
        "Rich light",
        &[PresetGroup::Light, PresetGroup::Look],
        source,
    );
    let matches = |photo: Option<PhotoId>| {
        r.core
            .presets_list(photo)
            .presets
            .iter()
            .find(|p| p.name == "Rich light")
            .expect("listed")
            .matches
    };
    assert!(!matches(Some(target)), "not yet");
    assert!(!matches(None), "false for no photo");
    r.apply("Rich light", &[target]);
    assert!(
        matches(Some(target)),
        "applying it again would change nothing"
    );
    assert!(matches(Some(source)), "and the photo it came from matches");
    // A field outside its groups does not matter.
    r.patch(target, &SettingsPatch::sharpen(120.0, 1.2));
    assert!(matches(Some(target)));
    // Any field of its groups does.
    for patch in [
        SettingsPatch::exposure_ev(3.0),
        SettingsPatch::contrast(-50.0),
        SettingsPatch::look_at(LookId::neutral(), 12.0),
    ] {
        r.apply("Rich light", &[target]);
        assert!(matches(Some(target)));
        r.patch(target, &patch);
        assert!(!matches(Some(target)), "{patch:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// A look that is not in the library
// ---------------------------------------------------------------------------------------------

fn write_cube(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(
        &path,
        ym_color::lut::Lut3d::identity(5)
            .expect("cube")
            .to_cube_string(name),
    )
    .expect("write");
    path
}

#[test]
fn a_missing_look_applies_the_rest_and_names_the_look() {
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    let src = tempfile::tempdir().unwrap();
    let imported = r
        .core
        .looks_import(
            &write_cube(src.path(), "Film warm.cube"),
            Some(LookInput::LogC4),
            &mut |_| {},
        )
        .expect("import");
    let warm = imported.added[0].id.clone();
    assert_eq!(imported.added[0].name, "Film warm");
    r.patch(
        source,
        &SettingsPatch::look_at(warm.clone(), 70.0).and(
            NodeId::MAIN,
            NodePatch {
                grade: Some(ym_core::settings::GradePatch {
                    brightness: Some(12.0),
                    ..Default::default()
                }),
                ..NodePatch::default()
            },
        ),
    );
    let entry = r.save(
        "Warm evening",
        &[PresetGroup::Look, PresetGroup::Light],
        source,
    );
    assert_eq!(
        entry.look.as_ref().map(|l| (l.id.clone(), l.name.clone())),
        Some((warm.clone(), "Film warm".to_owned()))
    );
    assert!(!entry.look_missing);
    r.apply("Warm evening", &[target]);
    assert_eq!(r.settings(target).output().look, warm);
    assert_eq!(r.settings(target).output().look_intensity, 70.0);

    // The look goes: a look of the same name from another library would be another look, by id.
    r.core.looks_remove(&warm).expect("removed");
    let own_look = LookId::neutral();
    r.patch(target, &SettingsPatch::look_at(own_look.clone(), 100.0));
    r.patch(target, &SettingsPatch::contrast(5.0));
    r.patch(
        target,
        &SettingsPatch::main_grade(ym_core::settings::GradePatch {
            brightness: Some(0.0),
            ..Default::default()
        }),
    );
    let listing = r.core.presets_list(Some(target));
    assert!(listing.presets[0].look_missing);
    let before = r.settings(target);
    let report = r.apply("Warm evening", &[target]);
    assert_eq!(report.missing_look.as_deref(), Some("Film warm"));
    assert_eq!(report.pasted.len(), 1);
    let after = r.settings(target);
    assert_eq!(
        after.output().look,
        own_look,
        "the look fields are left out of the patch"
    );
    assert_eq!(after.output().look_intensity, 100.0);
    assert_eq!(after.main_grade().brightness, 12.0, "the rest applies");
    assert_ne!(
        before.main_grade().brightness,
        after.main_grade().brightness
    );
    assert_eq!(
        missing_look_sentence("Warm evening", "Film warm"),
        "Warm evening uses the look Film warm, which isn't imported. Everything else was applied."
    );
    // A preset with a missing look matches when everything else matches.
    let m = r.core.presets_list(Some(target));
    assert!(m.presets[0].matches && m.presets[0].look_missing);
    // The tile applies it the same way.
    let tile = r
        .core
        .presets
        .thumbnail_settings("Warm evening", &before)
        .expect("tile settings");
    assert_eq!(tile.output().look, own_look);
    assert_eq!(tile.main_grade().brightness, 12.0);
}

// ---------------------------------------------------------------------------------------------
// What the Save dialog ticks
// ---------------------------------------------------------------------------------------------

#[test]
fn the_groups_a_photo_changed_are_ticked_for_the_person() {
    let r = rig(2);
    let (a, b) = (r.ids[0], r.ids[1]);
    let ticked = |id: PhotoId| {
        r.core
            .preset_inspect(&PresetSource::Photo { photo_id: id })
            .expect("inspect")
            .changed
    };
    assert_eq!(ticked(a), [], "a photo nobody touched ticks nothing");
    assert!(
        !r.core
            .preset_inspect(&PresetSource::Photo { photo_id: a })
            .unwrap()
            .older_export
    );

    let patch = |p: SettingsPatch| {
        r.set(
            b,
            DevelopSettings::with_look(r.core.looks.effective_default()),
        );
        r.patch(b, &p);
        ticked(b)
    };
    let grade = |g: ym_core::settings::GradePatch| {
        SettingsPatch::main(NodePatch {
            grade: Some(g),
            ..NodePatch::default()
        })
    };
    assert_eq!(
        patch(SettingsPatch::look_at(LookId::neutral(), 50.0)),
        [PresetGroup::Look]
    );
    assert_eq!(patch(SettingsPatch::exposure_ev(0.3)), [PresetGroup::Light]);
    assert_eq!(
        patch(grade(ym_core::settings::GradePatch {
            brightness: Some(3.0),
            ..Default::default()
        })),
        [PresetGroup::Light]
    );
    assert_eq!(
        patch(SettingsPatch::white_balance(4300.0, 3.0)),
        [PresetGroup::Color]
    );
    assert_eq!(
        patch(grade(ym_core::settings::GradePatch {
            saturation: Some(PercentSetting::manual_value(12.0)),
            ..Default::default()
        })),
        [PresetGroup::Color]
    );
    assert_eq!(
        patch(SettingsPatch::sharpen(100.0, 1.0)),
        [PresetGroup::Detail]
    );
    assert_eq!(
        patch(SettingsPatch::input(NodePatch {
            lens: Some(ym_core::settings::LensPatch {
                flares: Some(20.0),
                ..Default::default()
            }),
            ..NodePatch::default()
        })),
        [PresetGroup::Lens]
    );
    assert_eq!(
        patch(SettingsPatch::input(NodePatch {
            decode: Some(ym_core::settings::DecodePatch {
                neural_demosaic: Some(false),
                ..Default::default()
            }),
            ..NodePatch::default()
        })),
        [PresetGroup::Raw]
    );
    assert_eq!(
        patch(grade(ym_core::settings::GradePatch {
            curve: Some(ym_core::settings::CurvesPatch {
                luma: Some(vec![[0.0, 0.0], [0.5, 0.4], [1.0, 1.0]]),
                ..Default::default()
            }),
            ..Default::default()
        })),
        [PresetGroup::Curve]
    );
    // The photo's own crop and turn are not a preset's business.
    assert_eq!(
        patch(SettingsPatch::input(NodePatch {
            orientation: Some(ym_core::settings::Orientation::Deg90),
            ..NodePatch::default()
        })),
        []
    );
    // A brush stroke on Main is a Nodes field (the row is there, emptied), and so is a new node and a moved card.
    let mut s = DevelopSettings::with_look(r.core.looks.effective_default());
    s.main_mut().masks = every_mask();
    r.set(b, s);
    assert_eq!(ticked(b), [PresetGroup::Nodes]);
    let mut s = DevelopSettings::with_look(r.core.looks.effective_default());
    s.nodes.extra.insert(2, CorrectorNode::default());
    s.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(2)),
        NodeLink::picture(NodeId::Corrector(2), NodeId::Output),
    ];
    r.set(b, s);
    assert_eq!(ticked(b), [PresetGroup::Nodes]);
    // Everything at once.
    r.set(b, rich_with_a_usable_look());
    assert_eq!(ticked(b), PresetGroup::ALL);
}

#[test]
fn a_picked_white_ticks_colour_and_a_photo_graded_by_apple_auto_but_never_touched_ticks_nothing() {
    let r = pick_rig();
    let a = r.ids[0];
    r.patch(
        a,
        &SettingsPatch::default().and(
            NodeId::Input,
            NodePatch {
                white_pick: Some(Some([0.4167, 0.125])),
                ..NodePatch::default()
            },
        ),
    );
    let info = r
        .core
        .preset_inspect(&PresetSource::Photo { photo_id: a })
        .expect("inspect");
    assert_eq!(
        info.changed,
        [PresetGroup::Color],
        "a pick is saved as a manual white balance"
    );

    // Apple Auto's values on the photo: the stored settings are still Auto, so nothing is ticked.
    let dir = tempfile::tempdir().unwrap();
    let scene =
        Scene::new(384, 288, light(PROFILE, 5200.0), Layout::Patchwork, 0.3).noise(2e-4, 1e-6);
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", MODEL));
    let steps = AutoSteps {
        exposure: Some(1),
        white: Some(2),
        contrast: Some(2),
        saturation: Some(-1),
        ..AutoSteps::keep()
    };
    let answered = est.with_apple(
        Some(steps),
        AppleAuto::Answered {
            answer: AdvisorAnswer {
                steps,
                scene: String::new(),
                plan: String::new(),
                served_model: None,
            },
            info: AdvisorInfo::new("Scripted", "27.0 (27A1)"),
        },
        AutoProvider::Apple,
        tables_of(&profile),
    );
    let untouched = ResolvedSettings::resolve(
        PhotoId(1),
        &DevelopSettings::default(),
        &answered,
        &profile,
        None,
    )
    .unwrap();
    assert_ne!(
        untouched.values.main.grade.contrast, 0.0,
        "Apple Auto moved the values"
    );
    assert_ne!(untouched.values.main.grade.saturation, 0.0);
    assert_eq!(changed_groups(&untouched, &DevelopSettings::default()), []);
    // The person's own value on top of Apple Auto's ticks its group.
    let mut edited = DevelopSettings::default();
    edited.main_grade_mut().contrast = PercentSetting::manual_value(5.0);
    let edited = ResolvedSettings::resolve(PhotoId(1), &edited, &answered, &profile, None).unwrap();
    assert_eq!(
        changed_groups(&edited, &DevelopSettings::default()),
        [PresetGroup::Light]
    );
}

// ---------------------------------------------------------------------------------------------
// Applying is Paste
// ---------------------------------------------------------------------------------------------

#[test]
fn a_preset_applied_through_the_core_equals_the_same_patch_pasted_and_undo_restores_the_photo() {
    let r = rig(5);
    let (source, t1, t2, p1, p2) = (r.ids[0], r.ids[1], r.ids[2], r.ids[3], r.ids[4]);
    r.set(source, rich_with_a_usable_look());
    // Two pairs of identical targets, one with nodes of its own.
    let mut own = DevelopSettings::with_look(r.core.looks.effective_default());
    own.nodes.extra.insert(6, CorrectorNode::default());
    own.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(6)),
        NodeLink::picture(NodeId::Corrector(6), NodeId::Output),
    ];
    for id in [t2, p2] {
        r.set(id, own.clone());
    }
    for groups in [
        vec![PresetGroup::Light, PresetGroup::Color],
        vec![PresetGroup::Look, PresetGroup::Nodes, PresetGroup::Detail],
        PresetGroup::ALL.to_vec(),
    ] {
        let originals: Vec<_> = [t1, t2].iter().map(|id| r.settings(*id)).collect();
        r.save("Recipe", &groups, source);
        r.core.presets.store().get("Recipe").expect("stored");
        let preset = r.core.presets.store().get("Recipe").unwrap();
        // The patch pasted by hand, with the look check the core makes.
        let patch = preset.patch_for(&DevelopSettings::default(), true);
        assert!(patch.missing_look.is_none());
        let pasted = r
            .core
            .paste_settings(&[p1, p2], &patch.patch)
            .expect("pasted");
        let applied = r.apply("Recipe", &[t1, t2]);
        assert_eq!(applied.pasted.len(), 2);
        assert_eq!(applied.skipped, pasted.skipped);
        assert_eq!(r.settings(t1), r.settings(p1), "{groups:?}");
        assert_eq!(r.settings(t2), r.settings(p2), "{groups:?}");
        // The undo pair is Paste's: the same patches for the same change.
        assert_eq!(applied.pasted[0].before, pasted.pasted[0].before);
        assert_eq!(applied.pasted[0].after, pasted.pasted[0].after);
        assert_eq!(applied.pasted[1].before, pasted.pasted[1].before);
        // `before` puts the photo back exactly, `after` does it again.
        for (k, id) in [t1, t2].into_iter().enumerate() {
            let done = r.settings(id);
            r.patch(id, &applied.pasted[k].before);
            assert_eq!(
                r.settings(id),
                originals[k],
                "undo of {groups:?} on target {k}"
            );
            r.patch(id, &applied.pasted[k].after);
            assert_eq!(r.settings(id), done, "redo");
            // Back to the start for the next round.
            r.set(id, originals[k].clone());
        }
        r.set(p1, originals[0].clone());
        r.set(p2, originals[1].clone());
        r.core.preset_remove("Recipe").ok();
        let _ = std::fs::remove_file(r.app.path().join("presets/Recipe.ympreset"));
    }
}

#[test]
fn the_photos_that_cannot_take_a_preset_are_skipped_and_a_patch_that_fails_changes_none() {
    let r = rig(3);
    let (source, a, b) = (r.ids[0], r.ids[1], r.ids[2]);
    // A preset with a Film node 5; the second target has a corrector 5, which a patch cannot turn into a Film node.
    let mut s = DevelopSettings::with_look(r.core.looks.effective_default());
    s.nodes.films.insert(5, FilmNode::default());
    r.set(source, s);
    r.save("Film", &[PresetGroup::Nodes, PresetGroup::Light], source);
    let mut clash = DevelopSettings::with_look(r.core.looks.effective_default());
    clash.nodes.extra.insert(5, CorrectorNode::default());
    r.set(b, clash);
    let (before_a, before_b) = (r.settings(a), r.settings(b));
    let err = r
        .core
        .preset_apply("Film", &[a, b])
        .expect_err("fails on b");
    assert!(matches!(err, CoreError::Presets(_)), "{err:?}");
    assert_eq!(err.scope(), "presets");
    assert!(
        matches!(
            &err,
            CoreError::Presets(ym_core::presets::PresetError::Session(
                ym_core::session::SessionError::Settings(SettingsError::Graph(_))
            ))
        ),
        "{err:?}"
    );
    assert_eq!(
        r.settings(a),
        before_a,
        "a patch that cannot be applied to one photo changes none"
    );
    assert_eq!(r.settings(b), before_b);
    // A photo that is not in the session is skipped, and the report says so.
    let ghost = PhotoId(9999);
    let report = r.core.preset_apply("Film", &[a, ghost]).expect("applied");
    assert_eq!(report.pasted.iter().map(|p| p.id).collect::<Vec<_>>(), [a]);
    assert_eq!(report.skipped, [ghost]);
    // A preset that is gone is an error of the presets scope.
    let e = r.core.preset_apply("Nothing", &[a]).expect_err("gone");
    assert_eq!(e.scope(), "presets");
    assert_eq!(e.to_string(), "There is no preset called Nothing.");
}

// ---------------------------------------------------------------------------------------------
// The commands
// ---------------------------------------------------------------------------------------------

#[test]
fn save_update_rename_and_remove_follow_the_rules_of_the_dialog() {
    let r = rig(2);
    let (a, b) = (r.ids[0], r.ids[1]);
    r.patch(a, &SettingsPatch::exposure_ev(0.4));
    let from = |id: PhotoId| PresetSource::Photo { photo_id: id };
    let e = r
        .core
        .preset_save("  Warm evening ", &[PresetGroup::Light], &from(a), false)
        .expect("saved");
    assert_eq!(
        (e.name.as_str(), e.groups.as_slice()),
        ("Warm evening", &[PresetGroup::Light][..])
    );
    assert!(
        e.look.is_none() && !e.look_missing && e.matches,
        "it matches the photo it came from"
    );
    assert_eq!(e.revision.len(), 12);
    // Without `replace` a name that exists is refused.
    let err = r
        .core
        .preset_save("warm evening", &[PresetGroup::Light], &from(a), false)
        .expect_err("exists");
    assert_eq!(
        err.to_string(),
        "A preset called warm evening already exists."
    );
    assert_eq!(err.scope(), "presets");
    for (name, groups, text) in [
        ("", &[PresetGroup::Light][..], "Give the preset a name."),
        ("x", &[][..], "Tick at least one group."),
    ] {
        assert_eq!(
            r.core
                .preset_save(name, groups, &from(a), false)
                .expect_err("refused")
                .to_string(),
            text
        );
    }
    // Replace writes over it, with a new revision.
    r.patch(a, &SettingsPatch::exposure_ev(-0.4));
    let again = r
        .core
        .preset_save("Warm evening", &[PresetGroup::Light], &from(a), true)
        .expect("replaced");
    assert_ne!(again.revision, e.revision);
    assert_eq!(r.core.presets_list(None).presets.len(), 1);
    // Update keeps the groups and the name and takes the photo's values.
    r.patch(b, &SettingsPatch::exposure_ev(1.5));
    let updated = r.core.preset_update("Warm evening", b).expect("updated");
    assert_eq!(
        (updated.name.as_str(), updated.groups.as_slice()),
        ("Warm evening", &[PresetGroup::Light][..])
    );
    assert_ne!(updated.revision, again.revision);
    assert_eq!(
        r.core
            .presets
            .store()
            .get("Warm evening")
            .unwrap()
            .settings
            .input()
            .exposure,
        ym_core::settings::ExposureSetting::manual_ev(1.5)
    );
    // Rename.
    r.core
        .preset_save("Cool morning", &[PresetGroup::Light], &from(a), false)
        .unwrap();
    assert_eq!(
        r.core
            .preset_rename("Cool morning", "warm EVENING")
            .expect_err("exists")
            .to_string(),
        "A preset called warm EVENING already exists."
    );
    let renamed = r
        .core
        .preset_rename("Cool morning", "Golden hour")
        .expect("renamed");
    assert_eq!(renamed.name, "Golden hour");
    let names: Vec<_> = r
        .core
        .presets_list(None)
        .presets
        .into_iter()
        .map(|p| p.name)
        .collect();
    assert_eq!(names, ["Golden hour", "Warm evening"]);
    assert_eq!(
        r.core.preset_reveal("golden HOUR"),
        Some(r.app.path().join("presets/Golden hour.ympreset"))
    );
}

#[test]
fn a_preset_from_a_settings_file_and_the_damaged_files_of_the_list() {
    let r = rig(1);
    let dir = tempfile::tempdir().unwrap();
    let file = dir.path().join("settings.json");
    std::fs::write(
        &file,
        json!({ "nodes": { "items": [
            { "id": "input", "exposure": { "kind": "manual", "ev": 0.25 } },
            { "id": 1, "grade": { "contrast": { "kind": "manual", "value": 14 } } }
        ] } })
        .to_string(),
    )
    .unwrap();
    let source = PresetSource::File { path: file.clone() };
    let info = r.core.preset_inspect(&source).expect("inspect");
    assert_eq!(info.changed, [PresetGroup::Light]);
    let e = r
        .core
        .preset_save("From file", &info.changed, &source, false)
        .expect("saved");
    assert_eq!(e.groups, [PresetGroup::Light]);
    let stored = r.core.presets.store().get("From file").unwrap();
    assert_eq!(
        stored.settings.main_grade().contrast,
        PercentSetting::manual_value(14.0)
    );
    // A file that is not settings.
    let bad = dir.path().join("photo.jpg");
    std::fs::write(&bad, [0xff, 0xd8, 0xff]).unwrap();
    let err = r
        .core
        .preset_inspect(&PresetSource::File { path: bad })
        .expect_err("not settings");
    assert_eq!(err.exit_code_for_tests(), 2);
    // The list shows a damaged file with its reason and the folder; Delete removes it by its file name.
    std::fs::write(r.app.path().join("presets/junk.ympreset"), "nope").unwrap();
    let listing = r.core.presets_list(None);
    assert_eq!(
        listing.damaged,
        [DamagedPreset {
            file: "junk.ympreset".into(),
            reason: "This file isn't a Yellowmoon preset.".into()
        }]
    );
    assert_eq!(
        listing.folder,
        r.app.path().join("presets").display().to_string()
    );
}

trait ExitCode {
    fn exit_code_for_tests(&self) -> i32;
}

impl ExitCode for CoreError {
    fn exit_code_for_tests(&self) -> i32 {
        match self {
            CoreError::Presets(e) => e.exit_code(),
            _ => 1,
        }
    }
}

#[test]
fn import_and_export_go_through_the_core_and_the_events_say_when_the_folder_changed() {
    let r = rig(1);
    let source = r.ids[0];
    let changed = Arc::new(AtomicUsize::new(0));
    let c = Arc::clone(&changed);
    r.core.on_presets_changed(move || {
        c.fetch_add(1, Ordering::SeqCst);
    });
    // The first listing is the baseline and says nothing.
    assert!(r.core.presets_list(None).presets.is_empty());
    assert_eq!(changed.load(Ordering::SeqCst), 0);
    r.patch(source, &SettingsPatch::exposure_ev(0.2));
    r.save("One", &[PresetGroup::Light], source);
    assert_eq!(changed.load(Ordering::SeqCst), 1, "a save");
    r.core.preset_rename("One", "Uno").unwrap();
    assert_eq!(changed.load(Ordering::SeqCst), 2, "a rename");
    // The listing right after a change of the service's own finds nothing new.
    let _ = r.core.presets_list(None);
    assert_eq!(changed.load(Ordering::SeqCst), 2);
    // A preset made by `ym` (another process, here a file) shows up and fires the event once.
    let other = tempfile::tempdir().unwrap();
    let out = other.path().join("Made by ym.ympreset");
    r.core.preset_export("Uno", &out).expect("exported");
    std::fs::copy(&out, r.app.path().join("presets/Made by ym.ympreset")).unwrap();
    assert_eq!(
        r.core.presets_list(None).presets.len(),
        1,
        "the same name twice: one is damaged"
    );
    assert_eq!(changed.load(Ordering::SeqCst), 3);
    let _ = r.core.presets_list(None);
    assert_eq!(changed.load(Ordering::SeqCst), 3, "once");
    // Import: a name that exists comes back for the page to ask about.
    let o: ImportOutcome = r.core.presets_import(std::slice::from_ref(&out), false);
    assert_eq!(o.exists.len(), 1);
    assert_eq!(o.exists[0].name, "Uno");
    assert!(o.added.is_empty());
    let o = r
        .core
        .presets_import(&[out.clone(), other.path().join("missing.ympreset")], true);
    assert_eq!(o.replaced, ["Uno"]);
    assert_eq!(o.refused.len(), 1);
    assert_eq!(o.refused[0].file, "missing.ympreset");
    assert!(changed.load(Ordering::SeqCst) >= 4);
}

// ---------------------------------------------------------------------------------------------
// Nothing else is written
// ---------------------------------------------------------------------------------------------

/// Every file under `root`, relative, with its size.
fn tree(root: &Path) -> Vec<(String, u64)> {
    fn walk(dir: &Path, root: &Path, out: &mut Vec<(String, u64)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for e in entries.flatten() {
            let p = e.path();
            if p.is_dir() {
                walk(&p, root, out);
            } else {
                let rel = p
                    .strip_prefix(root)
                    .expect("under root")
                    .to_string_lossy()
                    .into_owned();
                out.push((rel, e.metadata().map(|m| m.len()).unwrap_or(0)));
            }
        }
    }
    let mut v = Vec::new();
    walk(root, root, &mut v);
    v.sort();
    v
}

#[test]
fn a_full_open_edit_apply_and_export_cycle_writes_no_file_in_presets_unless_a_preset_is_saved() {
    let r = rig(2);
    let (a, b) = (r.ids[0], r.ids[1]);
    assert!(
        tree(r.app.path()).is_empty(),
        "opening the core writes nothing"
    );
    r.patch(a, &SettingsPatch::exposure_ev(0.4));
    r.core.session.set_picked(a, true).unwrap();
    // Listing, inspecting and a failed apply write nothing either.
    let _ = r.core.presets_list(Some(a));
    r.core
        .preset_inspect(&PresetSource::Photo { photo_id: a })
        .unwrap();
    assert!(r.core.preset_apply("Nothing", &[a]).is_err());
    assert!(r.core.preset_remove("Nothing").is_err());
    assert!(!r.app.path().join("presets").exists());
    assert!(tree(r.app.path()).is_empty(), "{:?}", tree(r.app.path()));
    // A preset is saved: the preset and its lock are the only files, beside nothing else.
    r.save("Warm evening", &[PresetGroup::Light], a);
    assert_eq!(
        tree(r.app.path())
            .into_iter()
            .map(|(n, _)| n)
            .collect::<Vec<_>>(),
        ["presets/.lock", "presets/Warm evening.ympreset"]
    );
    // Applying it, undoing it and listing again change nothing on disk.
    let before = tree(r.app.path());
    let report = r.apply("Warm evening", &[b]);
    r.patch(b, &report.pasted[0].before);
    let _ = r.core.presets_list(Some(b));
    assert_eq!(tree(r.app.path()), before);
    // The photos on disk are untouched, and nothing is written next to them.
    assert_eq!(
        tree(&r.folder)
            .into_iter()
            .map(|(n, _)| n)
            .collect::<Vec<_>>(),
        ["DSC00001.ARW", "DSC00002.ARW"]
    );
}

#[test]
fn a_preset_tile_is_the_photo_with_the_preset_applied_through_its_look() {
    let r = rig(2);
    let (source, target) = (r.ids[0], r.ids[1]);
    r.patch(source, &SettingsPatch::exposure_ev(1.5));
    r.save("Bright", &[PresetGroup::Light], source);
    let png = |name: &str| -> Vec<u8> {
        r.core
            .spawn_preset_tile(target, name)
            .expect("queued")
            .wait()
            .expect("finished")
            .expect("rendered")
    };
    let tile = png("Bright");
    assert_eq!(&tile[..8], b"\x89PNG\r\n\x1a\n");
    let (w, h) = (
        u32::from_be_bytes(tile[16..20].try_into().unwrap()),
        u32::from_be_bytes(tile[20..24].try_into().unwrap()),
    );
    assert!(w.max(h) <= 240 && w.min(h) > 0, "{w}x{h}");
    // The target as it is, with the same look and no preset.
    let plain = r
        .core
        .renderer
        .look_thumbnail_png(
            &ym_core::pipeline::PhotoSource::from_session(&r.core.session, target).unwrap(),
            &r.settings(target).output().look,
            &ym_core::scheduler::CancelToken::new(),
        )
        .expect("plain");
    assert_ne!(tile, plain, "the preset changed the picture");
    // The photo's own settings are untouched by a tile.
    assert!(r.settings(target).input().exposure.is_auto());
    assert!(r.core.spawn_preset_tile(target, "Nothing").is_err());
}

// ---------------------------------------------------------------------------------------------
// The wire types against src/api/types.ts
// ---------------------------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("repo root")
        .to_path_buf()
}

/// The field names of `export interface <name> { ... }` in `src/api/types.ts`.
fn ts_interface_keys(source: &str, name: &str) -> BTreeSet<String> {
    let start = source
        .find(&format!("export interface {name} {{"))
        .unwrap_or_else(|| panic!("interface {name} in types.ts"));
    let body = &source[start..];
    let end = body.find("\n}").expect("end of the interface");
    body[..end]
        .lines()
        .skip(1)
        .filter_map(|line| {
            let line = line.strip_prefix("  ")?;
            if line.starts_with(' ') || line.starts_with('/') || line.starts_with('*') {
                return None;
            }
            let (key, _) = line.split_once(':')?;
            Some(key.trim_end_matches('?').to_owned())
        })
        .collect()
}

fn keys(v: &Value) -> BTreeSet<String> {
    v.as_object().expect("object").keys().cloned().collect()
}

#[test]
fn the_preset_wire_types_have_the_fields_of_the_typescript_interfaces() {
    let path = repo_root().join("src/api/types.ts");
    let Ok(source) = std::fs::read_to_string(&path) else {
        eprintln!("{} is not there; skipping", path.display());
        return;
    };
    let r = rig(1);
    let a = r.ids[0];
    r.patch(a, &SettingsPatch::exposure_ev(0.3));
    let entry = r.save("Wire", &[PresetGroup::Light], a);
    let check = |name: &str, value: Value| {
        assert_eq!(
            ts_interface_keys(&source, name),
            keys(&value),
            "interface {name} against its Rust type"
        );
    };
    check("PresetEntry", serde_json::to_value(&entry).unwrap());
    check(
        "DamagedPreset",
        serde_json::to_value(DamagedPreset {
            file: String::new(),
            reason: String::new(),
        })
        .unwrap(),
    );
    let list = r.core.presets_list(Some(a));
    check("PresetList", serde_json::to_value(&list).unwrap());
    let report = r.apply("Wire", &[r.ids[0]]);
    check("PresetApplyReport", serde_json::to_value(&report).unwrap());
    check(
        "PresetInspect",
        serde_json::to_value(
            r.core
                .preset_inspect(&PresetSource::Photo { photo_id: a })
                .unwrap(),
        )
        .unwrap(),
    );
    let import = ImportOutcome {
        added: vec![],
        replaced: vec![],
        exists: vec![ym_core::presets::ImportExists {
            name: String::new(),
            path: String::new(),
        }],
        refused: vec![ym_core::presets::ImportRefused {
            file: String::new(),
            reason: String::new(),
        }],
    };
    let v = serde_json::to_value(&import).unwrap();
    check("ImportResult", v.clone());
    assert_eq!(
        ts_interface_keys(&source, "ImportExists"),
        keys(&v["exists"][0])
    );
    assert_eq!(
        ts_interface_keys(&source, "ImportRefused"),
        keys(&v["refused"][0])
    );
    // The look object and the source union.
    check(
        "PresetLook",
        json!({ "id": "builtin:neutral", "name": "Neutral" }),
    );
    assert_eq!(
        serde_json::to_value(PresetSource::Photo { photo_id: a }).unwrap(),
        json!({ "kind": "photo", "photo_id": a.0 })
    );
    assert_eq!(
        serde_json::to_value(PresetSource::File {
            path: "x.json".into()
        })
        .unwrap(),
        json!({ "kind": "file", "path": "x.json" })
    );
    // `matches` and `revision` are the wire's own names; the groups are the ids of docs/30.
    let e = serde_json::to_value(&entry).unwrap();
    assert_eq!(e["groups"], json!(["light"]));
    assert_eq!(e["look"], Value::Null);
}

#[test]
fn the_interfaces_of_the_wire_are_there_with_the_spellings_of_the_spec() {
    let path = repo_root().join("src/api/types.ts");
    let Ok(source) = std::fs::read_to_string(&path) else {
        return;
    };
    assert!(
        source.contains("export type PresetGroup ="),
        "the group ids"
    );
    for id in [
        "look", "lens", "light", "color", "detail", "curve", "raw", "nodes",
    ] {
        assert!(source.contains(&format!("\"{id}\"")), "{id}");
    }
    let _ = (WhiteBalanceManualPatch::default(), NodeKind::Input);
}

// ---------------------------------------------------------------------------------------------
// New Preset from Exported Photo (P-06, docs/30 section 12)
// ---------------------------------------------------------------------------------------------

/// Exports every photo of the rig in `format` and returns the first file.
fn export_first(r: &Rig, format: ym_export::FormatKind, dir: &Path) -> PathBuf {
    use ym_core::export_queue::ExportJob;
    use ym_core::session::ApplyScope;
    let job = ExportJob {
        scope: ApplyScope::All,
        options: ym_export::ExportOptions {
            format,
            destination: Some(dir.to_path_buf()),
            ..ym_export::ExportOptions::default()
        },
    };
    let id = r.core.export_start(&job).expect("export starts");
    assert!(r.core.exports.wait_idle(Duration::from_secs(300)));
    let summary = r.core.export_summary(id).expect("summary");
    assert_eq!(summary.failed, 0, "{:?}", summary.errors);
    summary.files[0].clone()
}

/// The file of a preset saved from `source`, with the name taken out so that two can be compared.
fn saved_text(r: &Rig, name: &str, source: &PresetSource) -> String {
    r.core
        .preset_save(name, &PresetGroup::ALL, source, false)
        .expect("saved");
    let preset = r.core.presets.store().get(name).expect("stored");
    preset.to_json().expect("json").replace(name, "N")
}

#[test]
fn a_preset_made_from_an_export_equals_the_preset_made_from_the_photo_that_was_exported() {
    use ym_export::FormatKind;
    let r = pick_rig();
    let (a, b) = (r.ids[0], r.ids[1]);
    // a: a picked white, a manual contrast, exposure and the rest Auto. b: manual exposure and sharpening, a manual white balance.
    r.patch(
        a,
        &SettingsPatch::default().and(
            NodeId::Input,
            NodePatch {
                white_pick: Some(Some([0.4167, 0.125])),
                ..NodePatch::default()
            },
        ),
    );
    r.patch(
        a,
        &serde_json::from_value(json!({ "nodes": { "1": { "grade": { "contrast": { "kind": "manual", "value": 18.0 } } } } }))
            .expect("patch"),
    );
    r.patch(
        b,
        &serde_json::from_value(json!({ "nodes": { "input": {
            "exposure": { "kind": "manual", "ev": 0.6 },
            "sharpen": { "kind": "manual", "amount": 80.0, "radius_px": 1.1 },
            "white_balance": { "kind": "manual", "kelvin": 4300.0, "tint": 5.0 }
        } } }))
        .expect("patch"),
    );
    let out = tempfile::tempdir().expect("out");
    for (kind, ext) in [
        (FormatKind::Jpeg, "jpeg"),
        (FormatKind::Png, "png"),
        (FormatKind::Tiff, "tiff"),
    ] {
        let dir = out.path().join(ext);
        let first = export_first(&r, kind, &dir);
        let files: Vec<PathBuf> = {
            let mut f: Vec<PathBuf> = std::fs::read_dir(&dir)
                .expect("dir")
                .flatten()
                .map(|e| e.path())
                .collect();
            f.sort();
            f
        };
        assert_eq!(files.len(), 2, "{first:?}");
        for (file, photo) in files.iter().zip([a, b]) {
            let from_photo = saved_text(
                &r,
                &format!("photo-{ext}-{photo:?}"),
                &PresetSource::Photo { photo_id: photo },
            );
            let from_file = saved_text(
                &r,
                &format!("file-{ext}-{photo:?}"),
                &PresetSource::File { path: file.clone() },
            );
            assert_eq!(from_file, from_photo, "{ext} of {photo:?}");
            let inspect = r
                .core
                .preset_inspect(&PresetSource::File { path: file.clone() })
                .expect("inspected");
            assert!(!inspect.older_export);
            // The pre-ticked groups are the photo's own.
            let ticked = r
                .core
                .preset_inspect(&PresetSource::Photo { photo_id: photo })
                .expect("inspected")
                .changed;
            assert_eq!(inspect.changed, ticked, "{ext} of {photo:?}");
        }
    }
}

#[test]
fn an_export_that_holds_no_settings_is_refused_with_its_sentence() {
    let r = rig(1);
    let dir = tempfile::tempdir().expect("dir");
    let jpeg = dir.path().join("plain.jpg");
    // A JPEG that is not an export: the markers and nothing else.
    std::fs::write(&jpeg, [0xFF, 0xD8, 0xFF, 0xD9]).expect("write");
    let err = r
        .core
        .preset_inspect(&PresetSource::File { path: jpeg })
        .expect_err("refused");
    assert_eq!(
        err.to_string(),
        "This file has no Yellowmoon settings. Choose a photo that Yellowmoon exported."
    );
}

/// A file made before P-06 has no `ym:AutoFields`: it says so, keeps the numbers as written and leaves the pinned fields out of
/// the pre-ticking.
#[test]
fn an_older_export_says_so_and_ignores_the_pinned_fields_when_ticking() {
    use ym_export::FormatKind;
    let r = rig(1);
    let dir = tempfile::tempdir().expect("dir");
    let file = export_first(&r, FormatKind::Tiff, dir.path());
    // A photo of the defaults: the pinned exposure and the rest differ from the defaults' Auto, but nothing was edited.
    let current = r
        .core
        .preset_inspect(&PresetSource::File { path: file.clone() })
        .expect("inspected");
    assert!(
        !current.older_export && current.changed.is_empty(),
        "{current:?}"
    );
    // The same file with the property taken out of its packet (blanked in place, so every offset stays; a TIFF has no checksum).
    let mut bytes = std::fs::read(&file).expect("read");
    let find = |hay: &[u8], needle: &[u8], from: usize| {
        from + hay[from..]
            .windows(needle.len())
            .position(|w| w == needle)
            .expect("found")
    };
    let at = find(&bytes, b"<ym:AutoFields>", 0);
    let end = find(&bytes, b"</ym:AutoFields>", at) + b"</ym:AutoFields>".len();
    for b in &mut bytes[at..end] {
        *b = b' ';
    }
    let older = dir.path().join("older.tif");
    std::fs::write(&older, &bytes).expect("write");
    let inspect = r
        .core
        .preset_inspect(&PresetSource::File {
            path: older.clone(),
        })
        .expect("inspected");
    assert!(inspect.older_export);
    assert!(
        inspect.changed.is_empty(),
        "the pinned Light, Colour and Detail fields are not a change: {:?}",
        inspect.changed
    );
    // The numbers are saved as written when the groups are chosen anyway.
    r.core
        .preset_save(
            "Older",
            &[PresetGroup::Light],
            &PresetSource::File { path: older },
            false,
        )
        .expect("saved");
    let stored = r.core.presets.store().get("Older").expect("stored");
    assert!(
        !stored.settings.input().exposure.is_auto(),
        "fixed as written"
    );
}
