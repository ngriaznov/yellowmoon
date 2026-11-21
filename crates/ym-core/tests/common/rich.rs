//! A settings graph with a value in every field (the presets' and albums' round trips), and the resolved settings of
//! a synthetic photo.

#![allow(clippy::field_reassign_with_default)]

use ym_color::warp::WarpPin;
use ym_color::wb::LightSource;
use ym_core::session::PhotoId;
use ym_core::settings::{
    AutoOr, BrushMask, BrushStroke, ColorBalance, ColorWarp, ColorWarpPin, CorrectorNode, Crop,
    CropAspect, Curves, DevelopSettings, LensKind, LensSettings, MaskItem, MaskSettings, MaskShape,
    NodeId, NodeLink, NodeTrim, NoiseReductionValue, NormRect, ObjectMask, Orientation,
    ResolvedSettings, Rolloff, SharpenValue,
};
use ym_looks::library::LookId;

use super::{Layout, Scene, estimates_of_file, light};

/// A node's masks with one row of each kind that the tests need (Object with clicks and a box, Brush with a stroke, and the
/// rest).
pub fn every_mask() -> MaskSettings {
    let row = MaskItem::new;
    let mut items = vec![
        row(MaskShape::Object(ObjectMask {
            points: vec![(0.41, 0.33, true), (0.52, 0.31, false)],
            box_: Some([0.3, 0.2, 0.6, 0.5]),
            feather: 0.1,
            grow: -0.2,
            refine: 0.7,
        })),
        row(MaskShape::Subject(Default::default())),
        row(MaskShape::People(Default::default())),
        row(MaskShape::Background(Default::default())),
        row(MaskShape::Linear(Default::default())),
        row(MaskShape::Radial(Default::default())),
        row(MaskShape::Brush(BrushMask {
            strokes: vec![BrushStroke {
                size: 0.05,
                hardness: 0.8,
                erase: true,
                points: vec![[0.4, 0.5], [0.45, 0.52]],
            }],
            feather: 0.25,
            ..Default::default()
        })),
        row(MaskShape::ColorRange(Default::default())),
        row(MaskShape::BrightnessRange(Default::default())),
    ];
    items[1].mode = ym_core::settings::MaskMode::Subtract;
    items[2].invert = true;
    items[3].enabled = false;
    items[4].opacity = 0.5;
    MaskSettings {
        invert: true,
        items,
        ..MaskSettings::default()
    }
}

/// A graph with a manual value in every field of the Input, Main and the Output, two more correctors with masks, and links.
pub fn rich() -> DevelopSettings {
    let mut s = DevelopSettings::with_look(LookId::parse("user:film-warm").unwrap());
    s.output_mut().look_intensity = 62.5;
    s.input_mut().exposure = AutoOr::Manual(ym_core::settings::ExposureValue { ev: -1.25 });
    s.main_grade_mut().contrast = ym_core::settings::PercentSetting::manual_value(12.0);
    s.main_grade_mut().brightness = -7.5;
    s.main_grade_mut().highlights = ym_core::settings::PercentSetting::manual_value(-30.0);
    s.main_grade_mut().shadows = ym_core::settings::PercentSetting::manual_value(25.0);
    s.main_grade_mut().white = 4.0;
    s.main_grade_mut().black = ym_core::settings::PercentSetting::manual_value(-9.0);
    s.input_mut().light_source = LightSource::Tungsten;
    s.input_mut().white_balance = AutoOr::Manual(ym_core::settings::WhiteBalanceValue {
        kelvin: 3250.0,
        tint: -4.5,
    });
    s.main_grade_mut().color_balance = ColorBalance {
        shadows: ym_core::settings::BalanceSetting::manual_point([0.2, -0.3]),
        midtones: [0.0, 0.0],
        highlights: ym_core::settings::BalanceSetting::manual_point([-0.5, 0.5]),
    };
    s.main_grade_mut().saturation = ym_core::settings::PercentSetting::manual_value(15.0);
    s.main_grade_mut().color_warp = ColorWarp {
        pins: vec![
            ColorWarpPin {
                source: [0.12, 0.03],
                target: [0.08, 0.1],
                locked: false,
                range: 0.06,
                shadows: 0.5,
                highlights: 1.0,
                pivot_ev: -1.5,
                exposure_ev: 0.25,
                saturation: -0.4,
            },
            ColorWarpPin::from(WarpPin::locked_at([-0.05, -0.1])),
        ],
    };
    s.input_mut().lens = LensSettings {
        kind: LensKind::Vintage,
        flares: 60.0,
        edges: 45.0,
        anamorphic: 30.0,
    };
    s.input_mut().sharpen = AutoOr::Manual(SharpenValue {
        amount: 150.0,
        radius_px: 1.1,
    });
    s.output_mut().output_sharpening = 140.0;
    s.input_mut().noise_reduction = AutoOr::Manual(NoiseReductionValue {
        luminance: 35.0,
        color: 60.0,
        detail: 20.0,
    });
    s.main_grade_mut().curve = Curves {
        luma: vec![[0.0, 0.0], [0.25, 0.2], [0.75, 0.82], [1.0, 1.0]],
        red: vec![[0.0, 0.02], [1.0, 1.0]],
        green: vec![[0.0, 0.0], [1.0, 1.0]],
        blue: vec![[0.0, 0.0], [0.5, 0.45], [1.0, 0.98]],
    };
    s.input_mut().decode.neural_demosaic = false;
    s.input_mut().decode.chromatic_aberration = false;
    s.input_mut().rolloff = Rolloff {
        highlights: 0.8,
        low_light: 0.1,
    };
    s.input_mut().crop = Some(Crop {
        rect: NormRect {
            x: 0.1,
            y: 0.2,
            width: 0.7,
            height: 0.6,
        },
        angle_deg: -2.5,
        aspect: CropAspect::FourByFive,
        auto: true,
    });
    s.input_mut().orientation = Orientation::Deg90;
    s.input_mut().white_pick = Some([0.25, 0.75]);
    s.main_mut().trim = NodeTrim {
        exposure_ev: 0.25,
        temperature: -10.0,
        tint: 5.0,
    };
    s.main_mut().masks = every_mask();
    let mut mountain = CorrectorNode::default();
    mountain.name = "Mountain".to_owned();
    mountain.pos = [1.0, 0.0];
    mountain.trim.exposure_ev = 0.35;
    mountain.grade.highlights = ym_core::settings::PercentSetting::manual_value(-12.0);
    mountain.masks = every_mask();
    s.nodes.extra.insert(2, mountain);
    let mut sky = CorrectorNode::default();
    sky.name = "Sky".to_owned();
    sky.bypass = true;
    sky.pos = [2.0, 0.0];
    sky.trim.temperature = -18.0;
    s.nodes.extra.insert(3, sky);
    s.output_mut().pos = [3.0, 0.0];
    s.nodes.links = vec![
        NodeLink::picture(NodeId::Input, NodeId::MAIN),
        NodeLink::picture(NodeId::MAIN, NodeId::Corrector(2)),
        NodeLink::picture(NodeId::Corrector(2), NodeId::Corrector(3)),
        NodeLink::picture(NodeId::Corrector(3), NodeId::Output),
    ];
    s
}

/// The resolved settings of `settings` against the estimates of a small synthetic scene (a photo that has no session).
pub fn resolved_of(settings: &DevelopSettings) -> ResolvedSettings {
    let dir = tempfile::tempdir().unwrap();
    let scene = Scene::new(
        256,
        192,
        light("sony-ilce-7rm4", 5200.0),
        Layout::Patchwork,
        0.5,
    );
    let (_, est, profile) = estimates_of_file(&scene.write(dir.path(), "a.ARW", "ILCE-7RM4"));
    ResolvedSettings::resolve(PhotoId(3), settings, &est, &profile, None).unwrap()
}
