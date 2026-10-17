//! `golden truth`: the end-to-end colour of the product against the ground truth of the synthetic scenes.
//!
//! For every scene the real `ym` binary develops the file with its default settings and writes a DNG. The two defaults
//! that are not the sensor's colour are switched off, because the truth is the colour of the scene and not a look: the
//! look is Neutral (the default look is Signature since pipeline 0.3.0) and the highlight range is 0, "as recorded"
//! (`rolloff.highlights`, a gain on the highlights that is the identity at 0 below the ALEXA's clip, `docs/02` C7;
//! the default 0.5 moves the camera's clip on purpose; [`truth_settings`]). The low-light toe keeps its value. The DNG
//! holds the picture after the whole colour chain up to the look and curve
//! (C1 to C11, `docs/02`), decoded back to linear AWG4: the last place before the display transform where the colour
//! can be compared with an ALEXA-referred truth without going through a tone curve. The sidecar of the scene holds,
//! for every patch, what an ALEXA would have recorded of the same scene (the ALEXA estimate of `docs/02` C3 to C6, in
//! linear AWG4) and what Sony's own camera RGB gives through the same ARRI matrix without the sensor map (the
//! baseline).
//!
//! **Exposure.** The truth is reflectance-referred (the 18 % grey patch is `[0.18, 0.18, 0.18]`); a render has its own
//! automatic exposure. Both the render and the baseline are therefore scaled by one number, 0.18 divided by the
//! luminance of the neutral 18 % patch of that picture (`training/synth/README.md`: "compare after dividing by the grey
//! patch"). One scalar, not one per channel: the white balance error of the render stays in the number.
//!
//! **Difference.** CIEDE2000 in CIE L*a*b* of the AWG4 colours (XYZ with the D65 white, the white being the perfect
//! white of the scene, `[1, 1, 1]`), averaged over the interior of each patch in linear light. Patches whose noise-free
//! value reaches the clip point are left out of the statistics (their truth is the unclipped value).
//!
//! This is a number to report, not a gate: no threshold is defined for it yet.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::{Context, Result, ensure};
use serde::Serialize;
use ym_color::aces2::{Aces2Output, Aces2Transform};
use ym_color::consts::{Vec3, linalg};
use ym_color::deltae::{d65_white_xyz, delta_e_2000, percentile, xyz_to_lab};
use ym_color::encode::srgb_decode;
use ym_color::gamut;

use crate::dngread::{LinearRaw, read_linear_raw};
use crate::manifest::Entry;
use crate::scene::{Probe, Sidecar, Truth};
use crate::ymrun::{Output, YmBinary};

/// The value the 18 % grey patch has in the truth.
pub const GREY: f64 = 0.18;

/// Median, 95th percentile and maximum of a group of patches.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct GroupStats {
    /// Patches counted.
    pub count: usize,
    /// Median ΔE00.
    pub median: f64,
    /// 95th percentile.
    pub p95: f64,
    /// Maximum.
    pub max: f64,
}

impl GroupStats {
    /// Statistics of `values`, or `None` when there are none.
    #[must_use]
    pub fn of(values: &[f64]) -> Option<Self> {
        Some(Self {
            count: values.len(),
            median: percentile(values, 50.0)?,
            p95: percentile(values, 95.0)?,
            max: values.iter().copied().fold(0.0, f64::max),
        })
    }
}

/// One patch of one scene.
#[derive(Clone, Debug, Serialize)]
pub struct PatchResult {
    /// The probe's name (`rta-017`, `neutral-0.18`, `ladder-hue0-ev-3`, ...).
    pub name: String,
    /// The probe's group (`chart`, `neutral`, `ladder`, `shadow`, `black`, `highlight`, `gradient`, `rolloff`).
    /// `rolloff` is the scene's six-stop highlight ramps; with the highlight range at 0 (as recorded) their
    /// unclipped samples measure the colour of bright tones with C7 the identity (until pipeline 0.4.0 they went
    /// through the old roll-off shoulder at 0.5).
    pub group: String,
    /// ΔE00 between the render and the ALEXA-estimate truth.
    pub render: f64,
    /// ΔE00 between the baseline (Sony through the ARRI matrix, no sensor map) and the truth.
    pub baseline: f64,
    /// Like `render`, but after the display transform (ACES 2, sRGB) on both sides: the colour as it is seen.
    pub render_display: f64,
    /// Like `baseline`, but after the display transform.
    pub baseline_display: f64,
    /// The noise-free value reaches the clip point: left out of the statistics.
    pub clipped: bool,
    /// The render after the exposure scaling, linear AWG4.
    pub render_awg4: Vec3,
    /// The truth, linear AWG4.
    pub truth_awg4: Vec3,
}

/// The result for one scene.
#[derive(Clone, Debug, Serialize)]
pub struct SceneResult {
    /// The manifest id of the file that was rendered.
    pub id: String,
    /// The light's token (`D6500`, `bb2850`, ...).
    pub light: String,
    /// The light's class.
    pub light_class: String,
    /// The camera model of the file.
    pub body: String,
    /// The ISO.
    pub iso: u32,
    /// The ARW variant.
    pub variant: String,
    /// The CFA phase and bit depth of the mosaic (`rggb 14b`).
    pub mosaic: String,
    /// The profile `ym` used and how it matched (`exact`, `sensor_family`, `default`).
    pub profile: String,
    /// `ym`'s match of the camera.
    pub profile_match: String,
    /// Kelvin the estimate gave.
    pub estimated_kelvin: Option<f64>,
    /// The scaling applied to the render (0.18 over the luminance of its grey patch is `grey_scale / stored`).
    pub exposure_scale: f64,
    /// Statistics per probe group, render against truth.
    pub render: BTreeMap<String, GroupStats>,
    /// Statistics per probe group, baseline against truth.
    pub baseline: BTreeMap<String, GroupStats>,
    /// Statistics per probe group, render against truth after the display transform.
    pub render_display: BTreeMap<String, GroupStats>,
    /// Every patch.
    pub patches: Vec<PatchResult>,
}

fn lab_of(awg4: Vec3, white: Vec3) -> Vec3 {
    let xyz = linalg::mul_vec(&gamut::matrices().awg4_to_xyz, awg4);
    xyz_to_lab(xyz, white)
}

fn scaled(v: Vec3, s: f64) -> Vec3 {
    v.map(|c| c * s)
}

/// Lab (D65) of what the display transform (ACES 2, sRGB) makes of a scene-linear AWG4 colour.
fn displayed_lab(awg4: Vec3) -> Vec3 {
    let m = gamut::matrices();
    let ap0 = linalg::mul_vec(&m.awg4_to_ap0, awg4);
    let encoded = Aces2Transform::shared(Aces2Output::Srgb).transform_ap0(ap0);
    let linear = encoded.map(srgb_decode);
    xyz_to_lab(linalg::mul_vec(&m.rec709_to_xyz, linear), d65_white_xyz())
}

/// Compares the picture `raw` with the truth of `truth` patch by patch.
///
/// `exposure_scale` is found here from the grey reference patch of the truth.
///
/// # Errors
///
/// When the truth has no grey reference patch, a patch lies outside the picture or the grey patch is black.
pub fn compare(raw: &LinearRaw, truth: &Truth) -> Result<(f64, Vec<PatchResult>)> {
    let white = linalg::mul_vec(&gamut::matrices().awg4_to_xyz, [1.0, 1.0, 1.0]);
    let grey_probe: &Probe = truth
        .probes
        .iter()
        .find(|p| p.role == "grey_reference")
        .context("the scene has no grey reference patch")?;
    let render_grey = raw.mean_rgb(grey_probe.interior)?;
    let render_grey_y = gamut::awg4_luminance(render_grey);
    ensure!(render_grey_y > 0.0, "the grey patch of the render is black");
    let render_scale = GREY / render_grey_y;
    let baseline_grey_y = gamut::awg4_luminance(grey_probe.sony_through_arri.awg4);
    ensure!(
        baseline_grey_y > 0.0,
        "the grey patch of the baseline is black"
    );
    let baseline_scale = GREY / baseline_grey_y;

    let mut out = Vec::with_capacity(truth.probes.len());
    for p in &truth.probes {
        let render = scaled(raw.mean_rgb(p.interior)?, render_scale);
        let baseline = scaled(p.sony_through_arri.awg4, baseline_scale);
        let want = p.alexa.awg4;
        let want_lab = lab_of(want, white);
        let want_shown = displayed_lab(want);
        out.push(PatchResult {
            name: p.name.clone(),
            group: p.group.clone(),
            render: delta_e_2000(lab_of(render, white), want_lab),
            baseline: delta_e_2000(lab_of(baseline, white), want_lab),
            render_display: delta_e_2000(displayed_lab(render), want_shown),
            baseline_display: delta_e_2000(displayed_lab(baseline), want_shown),
            clipped: p.clipped_channels.iter().any(|&c| c),
            render_awg4: render,
            truth_awg4: want,
        });
    }
    Ok((render_scale, out))
}

/// Statistics per group over the patches that are not clipped.
#[must_use]
pub fn group_stats(
    patches: &[PatchResult],
    pick: impl Fn(&PatchResult) -> f64,
) -> BTreeMap<String, GroupStats> {
    let mut groups: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for p in patches.iter().filter(|p| !p.clipped) {
        groups.entry(p.group.clone()).or_default().push(pick(p));
    }
    groups
        .into_iter()
        .filter_map(|(k, v)| GroupStats::of(&v).map(|s| (k, s)))
        .collect()
}

/// The settings a scene is developed with for the truth: the entry's own, with the Neutral look and the highlight range
/// at 0 ("as recorded", `rolloff.highlights`) unless the entry sets them. The default look (Signature) and the default
/// highlight range (0.5, which places the camera's clip higher) change the picture on purpose; the truth measures the
/// colour of the scene, so a number here moves only when the sensor map or the white balance does. An entry's own
/// `rolloff.low_light` stays (a `rolloff` with only `highlights` reads the toe as its default, 0.5).
#[must_use]
pub fn truth_settings(base: &serde_json::Value) -> serde_json::Value {
    let mut settings = base.clone();
    if let Some(map) = settings.as_object_mut() {
        map.entry("look")
            .or_insert_with(|| serde_json::Value::from("builtin:neutral"));
        let rolloff = map
            .entry("rolloff")
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        // A `rolloff` that is not an object is passed on as it is (ym refuses it, loudly).
        if let Some(r) = rolloff.as_object_mut() {
            r.entry("highlights")
                .or_insert_with(|| serde_json::Value::from(0));
        }
    }
    settings
}

/// Develops the file of `entry` with `ym`, reads the DNG and compares it with the scene's truth.
///
/// # Errors
///
/// When `ym` fails, the sidecar is missing or the picture does not match the scene.
pub fn scene_result(ym: &YmBinary, testdata: &Path, entry: &Entry) -> Result<SceneResult> {
    let sidecar_rel = entry
        .sidecar
        .as_deref()
        .with_context(|| format!("{} has no sidecar (ground truth)", entry.id))?;
    let sidecar_path = testdata.join(sidecar_rel);
    let text = std::fs::read_to_string(&sidecar_path)
        .with_context(|| format!("reading {}", sidecar_path.display()))?;
    let sidecar: Sidecar = serde_json::from_str(&text)
        .with_context(|| format!("parsing {}", sidecar_path.display()))?;
    let truth = sidecar.truth()?;

    let scratch = tempfile::tempdir().context("creating a scratch folder")?;
    let rendered = ym.render(
        &testdata.join(&entry.file),
        &truth_settings(&entry.settings),
        Output::Dng,
        &scratch.path().join("out"),
        scratch.path(),
    )?;
    let bytes = std::fs::read(&rendered.output)
        .with_context(|| format!("reading {}", rendered.output.display()))?;
    let raw = read_linear_raw(&bytes)?;
    ensure!(
        (raw.width, raw.height)
            == (
                sidecar.mosaic.width as usize,
                sidecar.mosaic.height as usize
            ),
        "{}: the render is {} x {} but the scene is {} x {}",
        entry.id,
        raw.width,
        raw.height,
        sidecar.mosaic.width,
        sidecar.mosaic.height
    );
    let (exposure_scale, patches) = compare(&raw, &truth)?;
    let camera = rendered.json.get("camera");
    let text_of = |key: &str| {
        camera
            .and_then(|c| c.get(key))
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_owned()
    };
    Ok(SceneResult {
        id: entry.id.clone(),
        light: truth.light.token.clone(),
        light_class: truth.light.class.clone(),
        body: sidecar.writer.model.clone(),
        iso: sidecar.writer.iso,
        variant: entry.variant.clone().unwrap_or_default(),
        mosaic: format!(
            "{} {}b",
            entry.cfa.as_deref().unwrap_or("?").to_lowercase(),
            entry.bits.map_or("?".to_owned(), |b| b.to_string())
        ),
        profile: text_of("profile"),
        profile_match: text_of("match"),
        estimated_kelvin: rendered
            .json
            .pointer("/auto/kelvin")
            .and_then(serde_json::Value::as_f64),
        exposure_scale,
        render: group_stats(&patches, |p| p.render),
        baseline: group_stats(&patches, |p| p.baseline),
        render_display: group_stats(&patches, |p| p.render_display),
        patches,
    })
}

/// How well a variant suits the colour check: the plain mosaics first, the camera-processed one last.
fn variant_rank(variant: &str) -> u8 {
    match variant {
        "uncompressed14" => 0,
        "lossless-l" => 1,
        "packed12" => 2,
        "craw" => 3,
        "lossless-ms" => 9,
        _ => 5,
    }
}

/// One entry per scene (per sidecar): the variant that carries the colour best.
#[must_use]
pub fn one_per_scene<'a>(entries: &[&'a Entry]) -> Vec<&'a Entry> {
    let mut best: BTreeMap<&str, &'a Entry> = BTreeMap::new();
    for e in entries {
        let Some(sidecar) = e.sidecar.as_deref() else {
            continue;
        };
        let rank = |x: &Entry| variant_rank(x.variant.as_deref().unwrap_or(""));
        match best.get(sidecar) {
            Some(cur) if rank(cur) <= rank(e) => {}
            _ => {
                best.insert(sidecar, e);
            }
        }
    }
    let mut v: Vec<&Entry> = best.into_values().collect();
    v.sort_by(|a, b| a.id.cmp(&b.id));
    v
}

/// The numbers over all scenes, pooled.
#[derive(Clone, Debug, Serialize)]
pub struct Pooled {
    /// Scenes included.
    pub scenes: usize,
    /// The chart patches of every scene, render against truth.
    pub render: Option<GroupStats>,
    /// The chart patches of every scene, baseline against truth.
    pub baseline: Option<GroupStats>,
    /// The chart patches of every scene, render against truth after the display transform.
    pub render_display: Option<GroupStats>,
}

/// Pools the `chart` group of the scenes selected by `keep`.
#[must_use]
pub fn pool(scenes: &[SceneResult], keep: impl Fn(&SceneResult) -> bool) -> Pooled {
    let mut render = Vec::new();
    let mut baseline = Vec::new();
    let mut shown = Vec::new();
    let mut n = 0;
    for s in scenes.iter().filter(|s| keep(s)) {
        n += 1;
        for p in s
            .patches
            .iter()
            .filter(|p| p.group == "chart" && !p.clipped)
        {
            render.push(p.render);
            baseline.push(p.baseline);
            shown.push(p.render_display);
        }
    }
    Pooled {
        scenes: n,
        render: GroupStats::of(&render),
        baseline: GroupStats::of(&baseline),
        render_display: GroupStats::of(&shown),
    }
}

/// The table printed by `golden truth`.
#[must_use]
pub fn format_table(scenes: &[SceneResult]) -> String {
    use std::fmt::Write;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{:<7} {:<11} {:>5} {:<9} {:<14} {:>5} | {:>4} {:>4} {:>4} | {:>4} {:>4} {:>4} | {:>4} {:>4} | {:>5}",
        "light",
        "body",
        "iso",
        "mosaic",
        "variant",
        "kelv",
        "med",
        "p95",
        "max",
        "b.md",
        "b.95",
        "b.mx",
        "d.md",
        "d.95",
        "neutr"
    );
    let f = |v: Option<f64>| v.map_or("   -".to_owned(), |v| format!("{v:4.2}"));
    for r in scenes {
        let chart = r.render.get("chart");
        let base = r.baseline.get("chart");
        let shown = r.render_display.get("chart");
        let neutral = r.render.get("neutral");
        let _ = writeln!(
            s,
            "{:<7} {:<11} {:>5} {:<9} {:<14} {:>5} | {} {} {} | {} {} {} | {} {} | {:>5}",
            r.light,
            r.body,
            r.iso,
            r.mosaic,
            r.variant,
            r.estimated_kelvin
                .map_or("-".to_owned(), |k| format!("{k:.0}")),
            f(chart.map(|g| g.median)),
            f(chart.map(|g| g.p95)),
            f(chart.map(|g| g.max)),
            f(base.map(|g| g.median)),
            f(base.map(|g| g.p95)),
            f(base.map(|g| g.max)),
            f(shown.map(|g| g.median)),
            f(shown.map(|g| g.p95)),
            neutral.map_or("-".to_owned(), |g| format!("{:.2}", g.median)),
        );
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{LightTruth, Reference};
    use serde_json::json;

    fn reference(v: Vec3) -> Reference {
        Reference {
            native_balanced: v,
            awg3: v,
            awg4: v,
            logc4: v,
        }
    }

    fn probe(
        name: &str,
        group: &str,
        role: &str,
        rect: [u32; 4],
        alexa: Vec3,
        baseline: Vec3,
    ) -> Probe {
        Probe {
            name: name.into(),
            group: group.into(),
            role: role.into(),
            rect,
            interior: rect,
            irradiance: 1.0,
            camera_dn_above_black: [0.0; 3],
            clipped_channels: [false; 3],
            alexa: reference(alexa),
            sony_through_arri: reference(baseline),
        }
    }

    fn truth_of(probes: Vec<Probe>) -> Truth {
        let light: LightTruth = serde_json::from_value(json!({
            "token": "D6500", "name": "x", "class": "daylight", "class_probabilities": {"daylight": 1.0},
            "cct_k": 6500.0, "w_true_camera_rgb": [0.3, 1.0, 0.7]
        }))
        .unwrap();
        Truth {
            light,
            as_shot: serde_json::from_value(
                json!({"gains_true": [3.0, 1.0, 1.4], "log_gain_error_rb": [0.0, 0.0]}),
            )
            .unwrap(),
            noise: serde_json::from_value(
                json!({"iso": 400, "bits": 14, "a_dn_per_electron": 1.0, "b_dn2": 1.0,
                "a_normalised": 1e-4, "b_normalised": 1e-7, "gain_regime": "low gain"}),
            )
            .unwrap(),
            probes,
        }
    }

    /// A 2-patch picture: the left half is `grey`, the right half is `other` (fractions of the 16-bit range).
    fn picture(grey: Vec3, other: Vec3) -> LinearRaw {
        let mut data = Vec::new();
        for _y in 0..4 {
            for x in 0..8 {
                let v = if x < 4 { grey } else { other };
                data.extend(v.map(|c| (c * 65535.0).round() as u16));
            }
        }
        LinearRaw {
            width: 8,
            height: 4,
            data,
            baseline_exposure_ev: None,
            orientation: None,
            look_table_dims: None,
        }
    }

    fn two_patch_truth(other_alexa: Vec3) -> Truth {
        truth_of(vec![
            probe(
                "neutral-0.18",
                "neutral",
                "grey_reference",
                [0, 0, 4, 4],
                [0.18; 3],
                [0.18; 3],
            ),
            probe(
                "rta-001",
                "chart",
                "",
                [4, 0, 4, 4],
                other_alexa,
                [0.25, 0.2, 0.1],
            ),
        ])
    }

    #[test]
    fn the_truth_is_developed_without_the_default_look_and_with_the_highlights_as_recorded() {
        use serde_json::json;
        // An entry with no settings gets the Neutral look and the highlight range at 0; the toe is not named, so it
        // reads as its default.
        let plain = truth_settings(&json!({}));
        assert_eq!(plain["look"], "builtin:neutral");
        assert_eq!(plain["rolloff"], json!({"highlights": 0}));
        assert!(plain.get("white_offset_ev").is_none());
        // What the entry sets itself stays.
        let own = truth_settings(&json!({
            "look": "builtin:signature",
            "rolloff": {"highlights": 0.5, "low_light": 0.2},
            "exposure": {"ev": 1.0}
        }));
        assert_eq!(own["look"], "builtin:signature");
        assert_eq!(own["rolloff"], json!({"highlights": 0.5, "low_light": 0.2}));
        assert_eq!(own["exposure"]["ev"], 1.0);
        // An entry that sets only the toe keeps it and gets the highlights as recorded.
        let toe = truth_settings(&json!({"rolloff": {"low_light": 0.0}}));
        assert_eq!(toe["rolloff"], json!({"highlights": 0, "low_light": 0.0}));
        // Something that is not an object is passed on as it is (ym refuses it, loudly), and so is a `rolloff` that
        // is not one.
        assert_eq!(truth_settings(&json!(3)), json!(3));
        assert_eq!(truth_settings(&json!({"rolloff": 1}))["rolloff"], json!(1));
    }

    #[test]
    fn a_render_that_equals_the_truth_up_to_exposure_has_no_difference() {
        let truth = two_patch_truth([0.4, 0.2, 0.1]);
        // the render is 0.18 -> 0.045 (two stops down) and the patch scaled alike
        let raw = picture([0.045; 3], [0.1, 0.05, 0.025]);
        let (scale, patches) = compare(&raw, &truth).unwrap();
        assert!((scale - 4.0).abs() < 1e-3, "{scale}");
        assert!(patches.iter().all(|p| p.render < 0.01), "{patches:?}");
        // the baseline [0.25, 0.2, 0.1] is a different colour from [0.4, 0.2, 0.1]
        assert!(patches[1].baseline > 3.0, "{}", patches[1].baseline);
        let stats = group_stats(&patches, |p| p.render);
        assert_eq!(stats["chart"].count, 1);
        assert!(stats["neutral"].max < 0.01);
    }

    #[test]
    fn a_color_shift_shows_and_a_clipped_patch_is_left_out() {
        let mut truth = two_patch_truth([0.4, 0.2, 0.1]);
        let raw = picture([0.18; 3], [0.4, 0.2, 0.16]);
        let (_, patches) = compare(&raw, &truth).unwrap();
        assert!(patches[1].render > 3.0, "{}", patches[1].render);
        truth.probes[1].clipped_channels = [true, false, false];
        let (_, patches) = compare(&raw, &truth).unwrap();
        assert!(patches[1].clipped);
        assert!(!group_stats(&patches, |p| p.render).contains_key("chart"));
    }

    #[test]
    fn a_scene_without_grey_or_a_black_grey_is_an_error() {
        let truth = truth_of(vec![probe(
            "rta-001",
            "chart",
            "",
            [0, 0, 4, 4],
            [0.1; 3],
            [0.1; 3],
        )]);
        assert!(compare(&picture([0.2; 3], [0.2; 3]), &truth).is_err());
        let truth = two_patch_truth([0.1; 3]);
        assert!(compare(&picture([0.0; 3], [0.2; 3]), &truth).is_err());
        let wide = truth_of(vec![probe(
            "neutral-0.18",
            "neutral",
            "grey_reference",
            [0, 0, 40, 4],
            [0.18; 3],
            [0.18; 3],
        )]);
        assert!(compare(&picture([0.2; 3], [0.2; 3]), &wide).is_err());
    }

    #[test]
    fn one_entry_per_scene_prefers_the_plain_mosaic() {
        let make = |id: &str, variant: &str| Entry {
            id: id.into(),
            tier: "synthetic".into(),
            set: "default".into(),
            file: format!("{id}.ARW"),
            sidecar: Some("scenes/a.json".into()),
            settings: json!({}),
            variant: Some(variant.into()),
            light: None,
            body: None,
            cfa: None,
            bits: None,
        };
        let (a, b, c) = (
            make("a_ms", "lossless-ms"),
            make("a_craw", "craw"),
            make("a_l", "lossless-l"),
        );
        let picked = one_per_scene(&[&a, &b, &c]);
        assert_eq!(picked.len(), 1);
        assert_eq!(picked[0].id, "a_l");
    }

    #[test]
    fn pooling_and_the_table_work_on_a_result() {
        let truth = two_patch_truth([0.4, 0.2, 0.1]);
        let raw = picture([0.18; 3], [0.4, 0.2, 0.1]);
        let (scale, patches) = compare(&raw, &truth).unwrap();
        let scene = SceneResult {
            id: "x".into(),
            light: "D6500".into(),
            light_class: "daylight".into(),
            body: "ILCE-7RM4".into(),
            iso: 400,
            variant: "lossless-l".into(),
            mosaic: "rggb 14b".into(),
            profile: "sony-ilce-7rm4".into(),
            profile_match: "exact".into(),
            estimated_kelvin: Some(6500.0),
            exposure_scale: scale,
            render: group_stats(&patches, |p| p.render),
            baseline: group_stats(&patches, |p| p.baseline),
            render_display: group_stats(&patches, |p| p.render_display),
            patches,
        };
        let pooled = pool(std::slice::from_ref(&scene), |_| true);
        assert_eq!(pooled.scenes, 1);
        assert_eq!(pooled.render.unwrap().count, 1);
        assert!(format_table(std::slice::from_ref(&scene)).contains("D6500"));
        assert_eq!(pool(&[scene], |_| false).render, None);
    }
}
