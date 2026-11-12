//! The Face Refinement node's retouch on the device against its CPU reference (`docs/27-skin-retouch.md` sections 6.2 and 7,
//! S-13): `ym_retouch::retouch_skin_region_toned` on the linear AWG4 pixels of a region, with the LogC4 decode before it and
//! the encode after it that `ym-core`'s `film_pass` puts around it. The device follows the CPU's own plan
//! (`ym_retouch::plan_skin_region`) and runs the per-pixel and per-plane maths in f32 where the CPU uses f64.
//!
//! The picture compared is the node's picture in LogC4 / AWG4 with the pixels no tool changed taken from the input (the chain
//! passes those through), so the number is what the colour pass would blend. The error is in LogC4 code units (a stop is
//! about 0.0716 of it), the unit the other node tests use.
//!
//! | Case | Asserted | Gate (LogC4, absolute) |
//! |---|---|---|
//! | blemish removal alone, scales 1 and 0.5 | device against CPU | `GATE_BLEMISH` |
//! | dark circles and even skin, half-resolution and full-resolution layers | device against CPU | `GATE_TOOLS` |
//! | skin grading: contrast, shadow desaturation, shine removal | device against CPU | `GATE_TOOLS` |
//! | teeth whitening (the radix select of `L_max`) | device against CPU | `GATE_TOOLS` |
//! | every tool, several faces, a protection plane | device against CPU; a protected pixel is untouched | `GATE_TOOLS` |
//! | the tone relation: the CPU's curves, and the device's own estimate | device against CPU | `GATE_TONE` |
//! | a region at another place in the output, a repeat | the same bits | bit for bit |
//!
//! The device is the lavapipe software Vulkan adapter in CI (`WGPU_BACKEND=vulkan`, `YM_REQUIRE_GPU=1`).

mod common;

use std::sync::Arc;

use common::{TestGpu, gpus};
use ym_color::logc::{logc4_decode_rgb, logc4_encode_rgb};
use ym_gpu::kernels::face::{
    BlemishStep, FacePlan, FaceRegion, FaceStep, LowStep, Protect, TeethStep, ToneCurve,
    ToneSource, face_region_host,
};
use ym_gpu::queue::{JobSpec, Priority};
use ym_retouch::mock::{self, MockProposer};
use ym_retouch::{
    BlemishMap, DevicePlan, Face, LinearAwg4Image, Mask, Models, Placement, Rect, RetouchCache,
    SkinSettings, ToneRelation, plan_skin_region, prepare_skin, retouch_regions,
    retouch_skin_region_toned, tone_quantiles,
};

/// Gates, in LogC4 code values. Measured on lavapipe (f32 against f64, 1 ulp of a code value is 6e-8 to 1.2e-7): blemish removal
/// 1.2e-7, the tools 1.8e-7, the tone relation 5.7e-7 (the device's own estimate of it); the gates are ten times that or more, for
/// the compilers of other devices (fused multiply-adds, re-associated sums).
const GATE_BLEMISH: f64 = 2e-6;
const GATE_TOOLS: f64 = 2e-6;
const GATE_TONE: f64 = 5e-6;

/// Every device: the face kernel's planes are f32 in every storage mode.
fn f32_devices() -> impl Iterator<Item = &'static TestGpu> {
    gpus().iter()
}

fn hash(x: u32, y: u32) -> f32 {
    let mut h = x.wrapping_mul(0x9e37_79b1) ^ y.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2c1b_3c6d);
    h ^= h >> 12;
    (h & 0xffff) as f32 / 65535.0 - 0.5
}

/// A smooth skin-coloured picture with a gradient, fine noise, spots and dark patches around a face.
fn synthetic(w: u32, h: u32, face: &Face) -> LinearAwg4Image {
    let r = face.rect;
    let (cx, cy) = (
        f64::from(r.x_px) + f64::from(r.width_px) / 2.0,
        f64::from(r.y_px) + f64::from(r.height_px) / 2.0,
    );
    let unit = f64::from(r.width_px) / 500.0;
    let mut data = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (f64::from(x), f64::from(y));
            let mut luma = 0.28 + 0.1 * (fx / f64::from(w)) + 0.05 * (fy / f64::from(h));
            luma += 0.01 * f64::from(hash(x, y));
            for (sx, sy) in [(-80.0, 60.0), (60.0, -20.0), (20.0, 120.0)] {
                let d = ((fx - cx - sx * unit).powi(2) + (fy - cy - sy * unit).powi(2)).sqrt();
                luma *= 1.0 - 0.25 * (-(d / (9.0 * unit)).powi(2)).exp();
            }
            let d =
                ((fx - cx + 80.0 * unit).powi(2) + (fy - cy - 60.0 * unit - 40.0).powi(2)).sqrt();
            luma *= 1.0 - 0.2 * (-(d / (40.0 * unit)).powi(2)).exp();
            let l = luma as f32;
            data.push([l, l * 0.78, l * 0.62]);
        }
    }
    LinearAwg4Image::new(w, h, data).unwrap()
}

fn downscale2(img: &LinearAwg4Image) -> LinearAwg4Image {
    let (w, h) = (img.width_px as usize / 2, img.height_px as usize / 2);
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let mut acc = [0.0f64; 3];
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let p = img.data[(2 * y + dy) * img.width_px as usize + 2 * x + dx];
                for c in 0..3 {
                    acc[c] += f64::from(p[c]) / 4.0;
                }
            }
            data.push([acc[0] as f32, acc[1] as f32, acc[2] as f32]);
        }
    }
    LinearAwg4Image::new(w as u32, h as u32, data).unwrap()
}

fn settings(blemish: f32, dark: f32, even: f32) -> SkinSettings {
    SkinSettings {
        blemish,
        dark_circles: dark,
        even_skin: even,
        texture: 0.0,
        ..SkinSettings::default()
    }
}

/// Every tool at once.
fn all() -> SkinSettings {
    SkinSettings {
        skin_contrast: 50.0,
        shadow_desaturation: 60.0,
        shine_removal: 60.0,
        teeth_whitening: 70.0,
        ..settings(60.0, 60.0, 60.0)
    }
}

struct Scene {
    full: LinearAwg4Image,
    size: (u32, u32),
    faces: Vec<(Face, SkinSettings)>,
    cache: RetouchCache,
}

/// A picture with faces (two apart, two that overlap, as the part-equals-whole test of `ym-retouch`) and its prepared cache.
fn scene(mix: &[SkinSettings]) -> Scene {
    let faces_at = [
        Face {
            rect: Rect::new(60, 40, 220, 260),
            scale: 0.26,
        },
        Face {
            rect: Rect::new(700, 60, 200, 240),
            scale: 0.24,
        },
        Face {
            rect: Rect::new(380, 220, 180, 210),
            scale: 0.21,
        },
        Face {
            rect: Rect::new(470, 240, 170, 200),
            scale: 0.2,
        },
    ];
    let (w, h) = (1000u32, 520u32);
    let mut full = synthetic(w, h, &faces_at[0]);
    for f in &faces_at[1..] {
        let extra = synthetic(w, h, f);
        for (a, b) in full.data.iter_mut().zip(&extra.data) {
            *a = [
                (a[0] + b[0]) * 0.5,
                (a[1] + b[1]) * 0.5,
                (a[2] + b[2]) * 0.5,
            ];
        }
    }
    // A bright mouth under each face, for the teeth mask.
    for f in &faces_at {
        mock::paint_mouth(&mut full, &f.rect, 0.9);
    }
    let models = Models {
        proposer: Box::new(MockProposer::scale(192, 0.9)),
        ..mock::models(vec![Rect::new(560, 300, 24, 24)])
    };
    let faces: Vec<(Face, SkinSettings)> = faces_at
        .iter()
        .copied()
        .zip(mix.iter().copied().cycle())
        .collect();
    let mut cache = RetouchCache::new(11, 11);
    prepare_skin(&full, &faces, &models, &mut cache).unwrap();
    Scene {
        full,
        size: (w, h),
        faces,
        cache,
    }
}

/// One big face (D = 600 px at scale 1): the layers are at half resolution with radii up to 45 px, the rows are several boxes
/// of the sliding sum long, and the region is not a multiple of anything.
fn big_scene(mix: SkinSettings) -> Scene {
    let face = Face {
        rect: Rect::new(350, 60, 500, 600),
        scale: 600.0 / 1200.0,
    };
    let (w, h) = (1203u32, 901u32);
    let mut full = synthetic(w, h, &face);
    mock::paint_mouth(&mut full, &face.rect, 0.9);
    let models = Models {
        proposer: Box::new(MockProposer::scale(192, 0.9)),
        ..mock::models(vec![Rect::new(560, 300, 24, 24)])
    };
    let faces = vec![(face, mix)];
    let mut cache = RetouchCache::new(12, 12);
    prepare_skin(&full, &faces, &models, &mut cache).unwrap();
    Scene {
        full,
        size: (w, h),
        faces,
        cache,
    }
}

struct MapOf(Arc<BlemishMap>);

impl AsRef<[[f32; 4]]> for MapOf {
    fn as_ref(&self) -> &[[f32; 4]] {
        &self.0.data
    }
}

/// The CPU's plan as the device takes it.
fn plain(plan: &DevicePlan) -> FacePlan {
    FacePlan {
        origin_px: plan.origin_px,
        size_px: plan.size_px,
        render_scale: plan.render_scale,
        faces: plan
            .faces
            .iter()
            .map(|f| FaceStep {
                blemish: f.blemish.as_ref().map(|b| BlemishStep {
                    rect: (
                        b.map.rect.x_px,
                        b.map.rect.y_px,
                        b.map.rect.width_px,
                        b.map.rect.height_px,
                    ),
                    size: (b.map.width_px, b.map.height_px),
                    data: Arc::new(MapOf(Arc::clone(&b.map))),
                    amount: b.a,
                }),
                low: f
                    .tones
                    .as_ref()
                    .and_then(|t| t.low.as_ref())
                    .map(|l| LowStep {
                        rect: (l.rect.x_px, l.rect.y_px, l.rect.width_px, l.rect.height_px),
                        ds: l.params.ds,
                        radii: [l.params.r_tex, l.params.r_dc, l.params.r_se, l.params.r_g],
                        strengths: [l.params.k_dc, l.params.sp_dc, l.params.k_se, l.params.sp_se],
                        grading: [l.gc, l.gd, l.gs],
                        dc: l.dc.clone(),
                        se: l.se.clone(),
                        grade: l.grade.clone(),
                    }),
                teeth: f
                    .tones
                    .as_ref()
                    .and_then(|t| t.teeth.as_ref())
                    .map(|t| TeethStep {
                        rect: (t.rect.x_px, t.rect.y_px, t.rect.width_px, t.rect.height_px),
                        bytes: t.bytes.clone(),
                        amount: t.tw,
                    }),
            })
            .collect(),
    }
}

fn curves_of(rel: &ToneRelation) -> Option<[ToneCurve; 3]> {
    rel.curve_data().map(|c| {
        c.map(|c| ToneCurve {
            ln_analysis: c.ln_analysis,
            ln_input: c.ln_input,
            slope_lo: c.slope_lo,
            slope_hi: c.slope_hi,
        })
    })
}

/// One region of a working picture.
struct Case<'a> {
    scene: &'a Scene,
    working: &'a LinearAwg4Image,
    scale: f32,
    rect: Rect,
    /// The node's input over the region as LogC4, planar.
    logc4: Vec<f32>,
    /// The same pixels decoded as `film_pass` decodes them.
    part: LinearAwg4Image,
}

fn crop(img: &LinearAwg4Image, r: Rect) -> Vec<[f32; 3]> {
    let (w, h) = (r.width_px as usize, r.height_px as usize);
    let mut data = Vec::with_capacity(w * h);
    for y in 0..h {
        let s = (r.y_px as usize + y) * img.width_px as usize + r.x_px as usize;
        data.extend_from_slice(&img.data[s..s + w]);
    }
    data
}

impl<'a> Case<'a> {
    fn new(scene: &'a Scene, working: &'a LinearAwg4Image, scale: f32, rect: Rect) -> Self {
        let (w, h) = (rect.width_px as usize, rect.height_px as usize);
        let lin = crop(working, rect);
        let mut logc4 = vec![0.0f32; 3 * w * h];
        let mut dec = Vec::with_capacity(w * h);
        for (i, p) in lin.iter().enumerate() {
            let e = logc4_encode_rgb([f64::from(p[0]), f64::from(p[1]), f64::from(p[2])]);
            for c in 0..3 {
                logc4[c * w * h + i] = e[c] as f32;
            }
            let l = logc4_decode_rgb([
                f64::from(logc4[i]),
                f64::from(logc4[w * h + i]),
                f64::from(logc4[2 * w * h + i]),
            ]);
            dec.push([l[0] as f32, l[1] as f32, l[2] as f32]);
        }
        let part = LinearAwg4Image::new(rect.width_px, rect.height_px, dec).unwrap();
        Self {
            scene,
            working,
            scale,
            rect,
            logc4,
            part,
        }
    }

    fn place(&self) -> Placement {
        Placement {
            origin_px: (self.rect.x_px, self.rect.y_px),
            working_px: (self.working.width_px, self.working.height_px),
        }
    }

    fn plan(&self, faces: &[(Face, SkinSettings)]) -> DevicePlan {
        plan_skin_region(
            (self.rect.width_px, self.rect.height_px),
            self.place(),
            self.scene.size,
            faces,
            self.scale,
            &self.scene.cache,
        )
        .unwrap()
    }

    /// The CPU's picture: LogC4 where a tool changed the pixel, the input's value elsewhere.
    fn cpu(
        &self,
        faces: &[(Face, SkinSettings)],
        protect: Option<&Mask>,
        tone: Option<&ToneRelation>,
    ) -> Vec<f32> {
        let out = retouch_skin_region_toned(
            &self.part,
            self.place(),
            self.scene.size,
            faces,
            self.scale,
            &self.scene.cache,
            protect,
            tone,
        )
        .unwrap();
        let n = self.part.data.len();
        let mut pic = self.logc4.clone();
        for (i, (new, old)) in out.data.iter().zip(&self.part.data).enumerate() {
            if new != old {
                let e = logc4_encode_rgb([f64::from(new[0]), f64::from(new[1]), f64::from(new[2])]);
                for c in 0..3 {
                    pic[c * n + i] = e[c] as f32;
                }
            }
        }
        pic
    }

    /// The device's picture, with the input's value where it made none, and the number of pixels it changed.
    fn device(
        &self,
        dev: &TestGpu,
        plan: &FacePlan,
        tone: &ToneSource,
        protect: Option<&[u8]>,
    ) -> (Vec<f32>, usize) {
        let (plan, tone, logc4) = (plan.clone(), tone.clone(), self.logc4.clone());
        let protect = protect.map(<[u8]>::to_vec);
        let raw = dev
            .gpu
            .run(JobSpec::new(Priority::Viewport), move |ctx, _| {
                let region = FaceRegion {
                    plan: &plan,
                    tone: &tone,
                    protect: protect.as_deref().map_or(Protect::None, Protect::Bytes),
                };
                face_region_host(ctx, &region, &logc4)
            })
            .expect("face region");
        merge(&raw, &self.logc4)
    }
}

/// Zeros mean "no picture": the input's value.
fn merge(raw: &[f32], input: &[f32]) -> (Vec<f32>, usize) {
    let n = raw.len() / 3;
    let mut changed = 0;
    let out = (0..raw.len())
        .map(|k| {
            let i = k % n;
            let none = (0..3).all(|c| raw[c * n + i] == 0.0);
            if !none && k < n {
                changed += 1;
            }
            if none { input[k] } else { raw[k] }
        })
        .collect();
    (out, changed)
}

fn worst(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| f64::from((x - y).abs()))
        .fold(0.0, f64::max)
}

fn regions(scene: &Scene, working: &LinearAwg4Image, scale: f32) -> Vec<Rect> {
    retouch_regions(
        (working.width_px, working.height_px),
        scene.size,
        &scene.faces,
        scale,
        &scene.cache,
    )
    .unwrap()
}

/// The device against the CPU over every region of the scene at `scale`, for `faces`.
fn compare(
    name: &str,
    scene: &Scene,
    working: &LinearAwg4Image,
    scale: f32,
    gate: f64,
    protect: bool,
) -> f64 {
    let mut worst_all: f64 = 0.0;
    let rects = regions(scene, working, scale);
    assert!(!rects.is_empty(), "{name}: the faces have regions");
    let mut seen = false;
    for dev in f32_devices() {
        seen = true;
        for r in &rects {
            let case = Case::new(scene, working, scale, *r);
            let (w, h) = (r.width_px as usize, r.height_px as usize);
            let mask = protect.then(|| {
                Mask::new(
                    w as u32,
                    h as u32,
                    (0..w * h)
                        .map(|i| {
                            if (i % w / 40 + i / w / 40) % 3 == 0 {
                                255
                            } else if i % 7 == 0 {
                                128
                            } else {
                                0
                            }
                        })
                        .collect(),
                )
                .unwrap()
            });
            let cpu = case.cpu(&scene.faces, mask.as_ref(), None);
            let plan = plain(&case.plan(&scene.faces));
            let (got, changed) = case.device(
                dev,
                &plan,
                &ToneSource::Plain,
                mask.as_ref().map(|m| m.data.as_slice()),
            );
            let e = worst(&got, &cpu);
            let cpu_changed = cpu.iter().zip(&case.logc4).filter(|(a, b)| a != b).count() / 3;
            eprintln!(
                "[gpu_face] {name} scale {scale} region {w} x {h}: worst {e:.2e} (gate {gate:.0e}); \
                 device changed {changed}, CPU {cpu_changed}"
            );
            assert!(changed > 0, "{name}: the device changed some pixels");
            assert!(e <= gate, "{name}: worst {e:e} over {gate:e}");
            worst_all = worst_all.max(e);
        }
    }
    if !seen {
        eprintln!("[gpu_face] no f32 device here");
    }
    worst_all
}

#[test]
fn blemish_removal_matches_the_cpu() {
    let s = scene(&[settings(70.0, 0.0, 0.0)]);
    let half = downscale2(&s.full);
    compare("blemish", &s, &s.full, 1.0, GATE_BLEMISH, false);
    compare("blemish", &s, &half, 0.5, GATE_BLEMISH, false);
}

#[test]
fn dark_circles_and_even_skin_match_the_cpu() {
    // D is about 260 px at scale 1, so the layers are at half resolution; at scale 0.5 they are not.
    let s = scene(&[
        settings(0.0, 80.0, 0.0),
        settings(0.0, 0.0, 70.0),
        settings(0.0, 50.0, 90.0),
    ]);
    let half = downscale2(&s.full);
    compare(
        "dark circles, even skin",
        &s,
        &s.full,
        1.0,
        GATE_TOOLS,
        false,
    );
    compare("dark circles, even skin", &s, &half, 0.5, GATE_TOOLS, false);
}

#[test]
fn skin_grading_matches_the_cpu() {
    let contrast = SkinSettings {
        skin_contrast: 60.0,
        ..SkinSettings::default()
    };
    let shadow = SkinSettings {
        shadow_desaturation: 80.0,
        skin_contrast: -40.0,
        ..SkinSettings::default()
    };
    let shine = SkinSettings {
        shine_removal: 90.0,
        ..SkinSettings::default()
    };
    let s = scene(&[contrast, shadow, shine]);
    let half = downscale2(&s.full);
    compare("grading", &s, &s.full, 1.0, GATE_TOOLS, false);
    compare("grading", &s, &half, 0.5, GATE_TOOLS, false);
}

#[test]
fn teeth_whitening_matches_the_cpu() {
    let teeth = SkinSettings {
        teeth_whitening: 85.0,
        ..SkinSettings::default()
    };
    let s = scene(&[teeth]);
    let half = downscale2(&s.full);
    compare("teeth", &s, &s.full, 1.0, GATE_TOOLS, false);
    compare("teeth", &s, &half, 0.5, GATE_TOOLS, false);
}

#[test]
fn every_tool_on_several_faces_with_a_protection_plane_matches_the_cpu() {
    let s = scene(&[
        all(),
        settings(70.0, 0.0, 0.0),
        all(),
        settings(0.0, 80.0, 40.0),
    ]);
    let half = downscale2(&s.full);
    for protect in [false, true] {
        compare("all tools", &s, &s.full, 1.0, GATE_TOOLS, protect);
        compare("all tools", &s, &half, 0.5, GATE_TOOLS, protect);
    }
}

/// A protected pixel is untouched, and a repeat and another place in the output give the same bits.
#[test]
fn protection_holds_and_a_region_is_the_same_wherever_it_is_made() {
    let s = scene(&[all()]);
    let rects = regions(&s, &s.full, 1.0);
    let r = rects[0];
    let case = Case::new(&s, &s.full, 1.0, r);
    let (w, h) = (r.width_px as usize, r.height_px as usize);
    let bytes: Vec<u8> = (0..w * h)
        .map(|i| {
            if (i % w / 16 + i / w / 16) % 2 == 0 {
                255
            } else {
                0
            }
        })
        .collect();
    let plan = plain(&case.plan(&s.faces));
    for dev in f32_devices() {
        let (a, changed) = case.device(dev, &plan, &ToneSource::Plain, Some(&bytes));
        let (b, _) = case.device(dev, &plan, &ToneSource::Plain, Some(&bytes));
        assert_eq!(a, b, "a repeat gives the same bits");
        assert!(changed > 0);
        for (i, &p) in bytes.iter().enumerate() {
            if p == 255 {
                for c in 0..3 {
                    assert_eq!(a[c * w * h + i], case.logc4[c * w * h + i], "{i}");
                }
            }
        }
    }
}

/// The CPU's relation applied as numbers, and the device's own estimate of it from its quantiles of the node's input.
#[test]
fn the_tone_relation_matches_the_cpu_made_and_estimated() {
    let s = scene(&[settings(80.0, 0.0, 0.0)]);
    // The node's input is the analysis picture through a power curve, as an upstream tone curve makes it.
    let powered = |p: f64| {
        let data = s
            .full
            .data
            .iter()
            .map(|v| v.map(|c| f64::from(c).powf(p) as f32))
            .collect();
        LinearAwg4Image::new(s.full.width_px, s.full.height_px, data).unwrap()
    };
    let working = powered(0.7);
    for dev in f32_devices() {
        for r in regions(&s, &working, 1.0) {
            let case = Case::new(&s, &working, 1.0, r);
            // The analysis picture over the region (at scale 1 it is the full picture).
            let analysis = crop(&s.full, r);
            let rel = ToneRelation::estimate(&analysis, &case.part.data);
            assert!(!rel.is_plain(), "a power curve is a relation");
            let q = tone_quantiles(&analysis).expect("enough values");
            let cpu = case.cpu(&s.faces, None, Some(&rel));
            let plain_cpu = case.cpu(&s.faces, None, None);
            let plan = plain(&case.plan(&s.faces));
            let (curves, _) = case.device(
                dev,
                &plan,
                &ToneSource::Curves(Box::new(curves_of(&rel).expect("curves"))),
                None,
            );
            let (estimated, _) = case.device(dev, &plan, &ToneSource::Estimate(Box::new(q)), None);
            let (e_curves, e_est, e_plain) = (
                worst(&curves, &cpu),
                worst(&estimated, &cpu),
                worst(&plain_cpu, &cpu),
            );
            eprintln!(
                "[gpu_face] tone region {} x {}: CPU curves {e_curves:.2e}, device estimate {e_est:.2e} (the plain gain is {e_plain:.2e} from the relation)",
                r.width_px, r.height_px
            );
            assert!(e_plain > 10.0 * GATE_TONE, "the relation matters here");
            assert!(e_curves <= GATE_TONE, "curves {e_curves:e}");
            assert!(e_est <= GATE_TONE, "estimate {e_est:e}");
        }
    }
}

/// A face of 600 px at scale 1 and 300 px at scale 0.5: the larger radii and the longer rows of the sliding box filter, with every
/// tool on and a protection plane.
#[test]
fn a_big_face_matches_the_cpu() {
    let s = big_scene(all());
    let half = downscale2(&s.full);
    for protect in [false, true] {
        compare("big face", &s, &s.full, 1.0, GATE_TOOLS, protect);
        compare("big face", &s, &half, 0.5, GATE_TOOLS, protect);
    }
}
