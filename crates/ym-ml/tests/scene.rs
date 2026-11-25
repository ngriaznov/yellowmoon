//! The scene of a picture through the real networks, to the soft masks (`docs/34` sections 4b.2, 4b.3 and 4a.3, RG-04): the three
//! committed pictures of the segmenter's fixture (`tests/data/eomt/`: a landscape, a street with a person, a shore) go through
//! `ym_ml::scene::make_scene` (the segmenter, MediaPipe's Selfie Multiclass for skin, the subject rule) and the planes of
//! `ym_color::mask_element` (the guided filter, guide the L\*/100 of the picture).
//!
//! What is asserted is what must hold of any scene, whatever the networks say about a drawn picture: the same bits on a second run
//! and on 1 and 4 threads; the class bits agree with the segmenter's own label map through the class table; the subject follows its
//! rule; skin lies inside `person`; the soft planes are in [0, 1], equal 1 well inside a big element and 0 well outside every
//! element, and their mean is near the element's share. Without the weights each test prints a plain `SKIPPED` line and passes;
//! `YM_REQUIRE_EOMT=1` makes a missing file a failure (as `tests/eomt.rs` does), and `YM_REQUIRE_SELFIE=1` the skin network's.
//!
//! The network is large: run with `cargo test -p ym-ml --release --test scene`.

use std::path::{Path, PathBuf};

use ym_color::mask_element::{ElementScene, soft_masks_of_picture};
use ym_ml::eomt::{Element, Eomt, element_of};
use ym_ml::mediapipe::selfie::{SelfieNet, SelfieSegmenter};
use ym_ml::scene::{
    BUILT, ELEMENT_COUNT, GROUND, SKIN, SKY, SUBJECT, SceneBits, VEGETATION, WATER, class_bits,
    make_scene, person_mask,
};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn flag(name: &str) -> bool {
    std::env::var_os(name).is_some_and(|v| v != "0" && !v.is_empty())
}

/// An 8-bit RGB PNG: (width, height, bytes).
fn read_png(path: &Path) -> (usize, usize, Vec<u8>) {
    let file = std::fs::File::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let mut reader = png::Decoder::new(std::io::BufReader::new(file))
        .read_info()
        .expect("png info");
    let mut buf = vec![0; reader.output_buffer_size().expect("png size")];
    let info = reader.next_frame(&mut buf).expect("png frame");
    assert_eq!(info.bit_depth, png::BitDepth::Eight);
    let channels = info.color_type.samples();
    let (w, h) = (info.width as usize, info.height as usize);
    let rgb = match channels {
        3 => buf[..w * h * 3].to_vec(),
        4 => buf[..w * h * 4]
            .chunks_exact(4)
            .flat_map(|p| [p[0], p[1], p[2]])
            .collect(),
        n => panic!("{n} channels"),
    };
    (w, h, rgb)
}

/// The networks, or `None` (with a plain line) when the weights are not there.
fn networks() -> Option<(Eomt, Option<SelfieSegmenter>)> {
    let eomt_dir = repo().join("data/models/eomt");
    let eomt = match Eomt::load(&eomt_dir) {
        Ok(m) => m,
        Err(e) => {
            assert!(
                !flag("YM_REQUIRE_EOMT"),
                "YM_REQUIRE_EOMT is set and the segmenter did not load: {e}"
            );
            println!("SKIPPED: the segmenter's weights are not in data/models/eomt ({e})");
            return None;
        }
    };
    let selfie = match SelfieNet::load(&repo().join("data/models/mediapipe")) {
        Ok(n) => Some(SelfieSegmenter::new(n)),
        Err(e) => {
            assert!(
                !flag("YM_REQUIRE_SELFIE"),
                "YM_REQUIRE_SELFIE is set and the skin network did not load: {e}"
            );
            println!("NOTE: no skin network ({e}); the scenes have no skin");
            None
        }
    };
    Some((eomt, selfie))
}

/// The guide the render makes (`ColorChain::scene_guide_pixel`) for a picture that already is 8-bit sRGB: its L\*/100.
fn lightness(rgb: &[u8]) -> Vec<f32> {
    let decode = |c: u8| {
        let e = f64::from(c) / 255.0;
        if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }
    };
    rgb.chunks_exact(3)
        .map(|p| {
            let y = 0.2126 * decode(p[0]) + 0.7152 * decode(p[1]) + 0.0722 * decode(p[2]);
            let f = if y > 216.0 / 24389.0 {
                y.cbrt()
            } else {
                (24389.0 / 27.0 * y + 16.0) / 116.0
            };
            ((116.0 * f - 16.0) / 100.0).clamp(0.0, 1.0) as f32
        })
        .collect()
}

fn pictures() -> Vec<(&'static str, usize, usize, Vec<u8>)> {
    ["meadow", "street", "shore"]
        .into_iter()
        .map(|name| {
            let (w, h, rgb) = read_png(
                &repo()
                    .join("crates/ym-ml/tests/data/eomt")
                    .join(format!("{name}.png")),
            );
            (name, w, h, rgb)
        })
        .collect()
}

#[test]
fn a_picture_goes_through_the_segmenter_to_the_scene_and_to_the_soft_planes() {
    let Some((eomt, selfie)) = networks() else {
        return;
    };
    for (name, w, h, rgb) in pictures() {
        let scene = make_scene(&eomt, selfie.as_ref(), &rgb, w, h, &|| false).expect("a scene");
        assert_eq!((scene.width, scene.height), (w, h), "{name}");
        assert_eq!(scene.skin_available, selfie.is_some(), "{name}");
        // The class bits are the segmenter's own label map through the table of section 4b.2, at the picture's grid.
        let labels = eomt
            .segment(&rgb, w, h)
            .expect("labels")
            .resized_nearest(w, h);
        let class = class_bits(&labels);
        let mask = SKY | VEGETATION | WATER | GROUND | BUILT;
        for (i, (&b, &c)) in scene.bits.iter().zip(&class).enumerate() {
            assert_eq!(b & mask, c, "{name}: pixel {i}");
        }
        // Skin lies inside person; the subject is the person at 3 % or more of the picture.
        let person = person_mask(&labels);
        let share = person.iter().filter(|&&p| p).count() as f64 / person.len() as f64;
        for (i, &b) in scene.bits.iter().enumerate() {
            assert!(
                b & SKIN == 0 || person[i],
                "{name}: skin outside person at {i}"
            );
            if share >= 0.03 {
                assert_eq!(
                    b & SUBJECT != 0,
                    person[i],
                    "{name}: the subject is the person"
                );
            }
        }
        let shares: Vec<f64> = (0..ELEMENT_COUNT).map(|e| scene.share(1 << e)).collect();
        println!(
            "{name} {w}x{h}: sky {:.3} vegetation {:.3} water {:.3} ground {:.3} built {:.3} skin {:.3} subject {:.3} (person {share:.3})",
            shares[0], shares[1], shares[2], shares[3], shares[4], shares[5], shares[6]
        );
        assert!(
            shares[..5].iter().any(|&s| s > 0.05),
            "{name}: the segmenter found no big element: {shares:?}"
        );
        // The soft planes of the elements that cover at least 2 % (what a fit makes).
        let covering = scene.covering(0.02);
        let element_scene = ElementScene {
            key: 1,
            width: w,
            height: h,
            bits: scene.bits.clone(),
        };
        let guide = lightness(&rgb);
        let planes = soft_masks_of_picture(&element_scene, &covering, &guide, w, h);
        for e in 0..ELEMENT_COUNT {
            let Some(plane) = &planes[e] else {
                assert!(
                    !covering[e] || shares[e] == 0.0,
                    "{name}: element {e} has no plane"
                );
                continue;
            };
            assert_eq!(plane.len(), w * h);
            assert!(plane.iter().all(|v| (0.0..=1.0).contains(v)), "{name}: {e}");
            let mean = plane.iter().map(|&v| f64::from(v)).sum::<f64>() / plane.len() as f64;
            assert!(
                (mean - shares[e]).abs() < 0.05,
                "{name}: element {e}: the plane's mean {mean:.3} is far from its share {:.3}",
                shares[e]
            );
            // Far from every pixel of the element the mask is near 0, far inside it near 1 (the guide may pull single pixels, since an
            // edge in the picture inside a region moves the mask there: the means are what must hold).
            let r = (0.02 * w.min(h) as f64).round().max(1.0) as usize;
            let far = 3 * r;
            let (mut inside, mut outside) = ((0.0f64, 0usize), (0.0f64, 0usize));
            let bit = 1u8 << e;
            for y in far..h - far {
                for x in far..w - far {
                    let around = |f: &dyn Fn(u8) -> bool| {
                        (y - far..=y + far).step_by(far).all(|yy| {
                            (x - far..=x + far)
                                .step_by(far)
                                .all(|xx| f(scene.bits[yy * w + xx]))
                        })
                    };
                    let v = f64::from(plane[y * w + x]);
                    if around(&|b| b & bit != 0) {
                        inside = (inside.0 + v, inside.1 + 1);
                    } else if around(&|b| b & bit == 0) {
                        outside = (outside.0 + v, outside.1 + 1);
                    }
                }
            }
            println!(
                "  element {e}: inside mean {:.3} over {} pixels, outside mean {:.3} over {}",
                inside.0 / inside.1.max(1) as f64,
                inside.1,
                outside.0 / outside.1.max(1) as f64,
                outside.1
            );
            if inside.1 > 100 {
                assert!(
                    inside.0 / inside.1 as f64 > 0.8,
                    "{name}: element {e} inside"
                );
            }
            if outside.1 > 100 {
                assert!(
                    outside.0 / (outside.1 as f64) < 0.2,
                    "{name}: element {e} outside"
                );
            }
        }
        // The same planes again: the same bits.
        let again = soft_masks_of_picture(&element_scene, &covering, &guide, w, h);
        assert!(again == planes, "{name}: a second run is other bits");
    }
}

#[test]
fn the_same_scene_on_one_and_four_threads_and_on_a_second_run() {
    let Some((eomt, selfie)) = networks() else {
        return;
    };
    let (name, w, h, rgb) = pictures().remove(1);
    let run = |threads: usize| -> SceneBits {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| make_scene(&eomt, selfie.as_ref(), &rgb, w, h, &|| false).expect("scene"))
    };
    let one = run(1);
    assert!(run(4) == one, "{name}: 4 threads give another scene");
    assert!(run(1) == one, "{name}: a second run gives another scene");
}

#[test]
fn a_cancelled_scene_stops() {
    let Some((eomt, selfie)) = networks() else {
        return;
    };
    let (_, w, h, rgb) = pictures().remove(0);
    let r = make_scene(&eomt, selfie.as_ref(), &rgb, w, h, &|| true);
    assert!(
        matches!(r, Err(ym_ml::model::MlError::Cancelled)),
        "{:?}",
        r.map(|s| s.width)
    );
}

#[test]
fn the_class_elements_are_what_the_table_says_of_the_element_enum() {
    // Pure: no weights. Every class id maps to the bit of its element, and an `Element` and its bit agree.
    for c in 0..=255u8 {
        let e = element_of(c);
        let bit = ym_ml::scene::class_bit(c);
        let expect = match e {
            Element::Sky => SKY,
            Element::Vegetation => VEGETATION,
            Element::Water => WATER,
            Element::Ground => GROUND,
            Element::Architecture | Element::Interior => BUILT,
            _ => 0,
        };
        assert_eq!(bit, expect, "class {c}");
    }
}

/// Real photographs of people (`tools/fetch/mediapipe.sh --photos` writes the folder; `YM_SCENE_PHOTOS=/path` names another): the person
/// is found, the subject is the person, and **skin from MediaPipe lies inside the person** and is not empty. A local check: it runs
/// where the photos and both networks are, and prints a plain `SKIPPED` line otherwise.
#[test]
fn real_photographs_of_people_have_skin_inside_the_person() {
    let dir = std::env::var_os("YM_SCENE_PHOTOS")
        .map_or_else(|| repo().join("testdata/mediapipe/photos"), PathBuf::from);
    let Ok(listing) = std::fs::read_dir(&dir) else {
        println!("SKIPPED: no photographs in {}", dir.display());
        return;
    };
    let Some((eomt, Some(selfie))) = networks() else {
        println!("SKIPPED: this check needs both networks");
        return;
    };
    let mut names: Vec<PathBuf> = listing
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "jpg"))
        .collect();
    names.sort();
    for path in names {
        let mut d =
            zune_jpeg::JpegDecoder::new(std::io::Cursor::new(std::fs::read(&path).expect("photo")));
        let rgb = d.decode().expect("decode");
        let info = d.info().expect("header");
        let (w, h) = (usize::from(info.width), usize::from(info.height));
        let scene = make_scene(&eomt, Some(&selfie), &rgb, w, h, &|| false).expect("a scene");
        let labels = eomt
            .segment(&rgb, w, h)
            .expect("labels")
            .resized_nearest(w, h);
        let person = person_mask(&labels);
        let person_share = person.iter().filter(|&&p| p).count() as f64 / person.len() as f64;
        let (skin, subject) = (scene.share(SKIN), scene.share(SUBJECT));
        println!(
            "{} {w}x{h}: person {person_share:.3}, skin {skin:.3}, subject {subject:.3}",
            path.file_name().and_then(|n| n.to_str()).unwrap_or("?")
        );
        for (i, &b) in scene.bits.iter().enumerate() {
            assert!(b & SKIN == 0 || person[i], "skin outside the person at {i}");
        }
        if person_share >= 0.03 {
            assert!(skin > 0.0, "a person of {person_share:.3} and no skin");
            assert!(
                (subject - person_share).abs() < 1e-12,
                "the subject is the person"
            );
        }
    }
}
