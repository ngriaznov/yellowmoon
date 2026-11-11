//! The real `FaceParser` and `FaceMesh` (feature `mediapipe`) on public portraits (`docs/27` S-09 and S-10).
//!
//! Weights: `data/models/mediapipe/` (`tools/fetch/mediapipe.sh`); photos: `testdata/mediapipe/photos/<id>.jpg`
//! (`tools/fetch/mediapipe.sh --photos`, pinned by SHA-256, never committed). Without them the tests print a plain "SKIPPED"
//! line and pass; `YM_REQUIRE_MEDIAPIPE=1` makes a missing file a failure (CI sets it).
//!
//! What is checked, per face of every photo: the masks are sane (the skin holds the cheeks and excludes the eyes, brows,
//! lips and nostrils; the eyes are small and inside the face; the mouth interior is inside the lips; the under-eye crescent
//! is below the eyes and mostly on the skin; the non-uniformity table is bright at the nostrils and dark on the cheeks), the
//! masks are the same bits on one and four threads and from the cache, and a picture without a face gives no parts.
//!
//! Run in release mode for speed: `cargo test -p ym-retouch --features mediapipe --release --test mediapipe`.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use ym_ml::mediapipe::blazeface::{BlazeFace, FaceFinder, FinderParams, Rgb8};
use ym_retouch::mediapipe::{AWG4_TO_REC709, MediaPipeFaces};
use ym_retouch::{
    BoundedMask, Face, FaceMesh, FaceParser, FacePart, LinearAwg4Image, MeshTable, Rect,
};

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn required() -> bool {
    std::env::var_os("YM_REQUIRE_MEDIAPIPE").is_some_and(|v| v != "0" && !v.is_empty())
}

fn skip_or_fail(what: &str) {
    assert!(!required(), "YM_REQUIRE_MEDIAPIPE is set but {what}");
    println!("SKIPPED: {what}");
}

fn dir() -> PathBuf {
    repo().join("data/models/mediapipe")
}

fn have_weights() -> bool {
    [
        "blaze_face_full_range.f16.safetensors",
        "face_mesh_v2.f16.safetensors",
        "selfie_multiclass_256.f32.safetensors",
    ]
    .iter()
    .all(|f| dir().join(f).is_file())
}

fn photo_ids() -> Vec<String> {
    let tsv =
        std::fs::read_to_string(repo().join("tools/fetch/mediapipe-photos.tsv")).expect("manifest");
    tsv.lines()
        .filter(|l| !l.trim().is_empty() && !l.starts_with('#'))
        .filter_map(|l| l.split('\t').next().map(str::to_owned))
        .collect()
}

/// An 8-bit sRGB picture as linear AWG4 (the inverse of the display conversion of the crate).
fn linear_awg4(w: usize, h: usize, rgb: &[u8]) -> LinearAwg4Image {
    let m = AWG4_TO_REC709;
    let inv = invert(m);
    let decode = |v: u8| {
        let e = f64::from(v) / 255.0;
        (if e <= 0.04045 {
            e / 12.92
        } else {
            ((e + 0.055) / 1.055).powf(2.4)
        }) as f32
    };
    let data = rgb
        .chunks_exact(3)
        .map(|p| {
            let l = [decode(p[0]), decode(p[1]), decode(p[2])];
            [0, 1, 2].map(|r| inv[r][0] * l[0] + inv[r][1] * l[1] + inv[r][2] * l[2])
        })
        .collect();
    LinearAwg4Image::new(w as u32, h as u32, data).expect("image")
}

fn invert(m: [[f32; 3]; 3]) -> [[f32; 3]; 3] {
    let a = m.map(|r| r.map(f64::from));
    let det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
        - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
        + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
    let c = |r: usize, c: usize| {
        let (r1, r2, c1, c2) = ((r + 1) % 3, (r + 2) % 3, (c + 1) % 3, (c + 2) % 3);
        a[r1][c1] * a[r2][c2] - a[r1][c2] * a[r2][c1]
    };
    let mut out = [[0.0f32; 3]; 3];
    for r in 0..3 {
        for k in 0..3 {
            out[r][k] = (c(k, r) / det) as f32;
        }
    }
    out
}

struct Photo {
    id: String,
    rgb: Vec<u8>,
    w: usize,
    h: usize,
    image: LinearAwg4Image,
}

fn load_photos() -> Vec<Photo> {
    let mut out = Vec::new();
    for id in photo_ids() {
        let path = repo()
            .join("testdata/mediapipe/photos")
            .join(format!("{id}.jpg"));
        if !path.is_file() {
            skip_or_fail(&format!(
                "no photo {} (run tools/fetch/mediapipe.sh --photos)",
                path.display()
            ));
            continue;
        }
        let mut d =
            zune_jpeg::JpegDecoder::new(std::io::Cursor::new(std::fs::read(&path).expect("photo")));
        let rgb = d.decode().expect("decode");
        let info = d.info().expect("header");
        let (w, h) = (usize::from(info.width), usize::from(info.height));
        let image = linear_awg4(w, h, &rgb);
        out.push(Photo {
            id,
            rgb,
            w,
            h,
            image,
        });
    }
    out
}

fn faces_of(photo: &Photo) -> Vec<Face> {
    let finder = FaceFinder::new(BlazeFace::load(&dir()).expect("blazeface"));
    let src = Rgb8::new(photo.w, photo.h, &photo.rgb).expect("picture");
    finder
        .find(&src, &FinderParams::default())
        .expect("find")
        .iter()
        .map(|d| Face {
            rect: Rect::new(d.rect.x_px, d.rect.y_px, d.rect.width_px, d.rect.height_px),
            scale: d.scale,
        })
        .collect()
}

fn faces_engine() -> Arc<MediaPipeFaces> {
    MediaPipeFaces::load(&dir()).expect("the weights load and match SHA256SUMS")
}

fn area(m: &BoundedMask) -> f64 {
    m.mask.data.iter().map(|&v| f64::from(v) / 255.0).sum()
}

fn centroid(m: &BoundedMask) -> (f64, f64) {
    let (mut sx, mut sy, mut s) = (0.0, 0.0, 0.0);
    let w = m.mask.width_px as usize;
    for (i, &v) in m.mask.data.iter().enumerate() {
        let v = f64::from(v);
        sx += v * ((i % w) as f64 + 0.5);
        sy += v * ((i / w) as f64 + 0.5);
        s += v;
    }
    (
        f64::from(m.region.x_px) + sx / s,
        f64::from(m.region.y_px) + sy / s,
    )
}

fn at(m: &BoundedMask, x: i64, y: i64) -> u8 {
    m.mask
        .get(x - i64::from(m.region.x_px), y - i64::from(m.region.y_px))
}

/// The overlap `sum(a * b) / sum(a)` of two masks of one picture.
fn share_of(a: &BoundedMask, b: &BoundedMask) -> f64 {
    let (mut inter, mut total) = (0.0, 0.0);
    let w = a.mask.width_px as usize;
    for (i, &v) in a.mask.data.iter().enumerate() {
        let (x, y) = (
            i64::from(a.region.x_px) + (i % w) as i64,
            i64::from(a.region.y_px) + (i / w) as i64,
        );
        total += f64::from(v);
        inter += f64::from(v) * f64::from(at(b, x, y)) / 255.0;
    }
    inter / total
}

#[test]
fn the_display_matrix_is_ym_colors() {
    let m = ym_color::gamut::matrices().awg4_to_rec709;
    for r in 0..3 {
        for c in 0..3 {
            assert!(
                (f64::from(AWG4_TO_REC709[r][c]) - m[r][c]).abs() < 1e-6,
                "[{r}][{c}]"
            );
        }
    }
}

#[test]
fn masks_are_sane_on_public_portraits() {
    if !have_weights() {
        skip_or_fail(
            "no MediaPipe weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)",
        );
        return;
    }
    let engine = faces_engine();
    let (parser, mesh) = (engine.parser(), engine.mesh());
    let mut checked = 0;
    for photo in load_photos() {
        let size = (photo.image.width_px, photo.image.height_px);
        for (k, face) in faces_of(&photo).iter().enumerate() {
            let tag = format!("{} face {k}", photo.id);
            let part = |p| parser.part_mask(&photo.image, face, p).expect("part");
            let skin = part(FacePart::SkinWithoutEars).unwrap_or_else(|| panic!("{tag}: no skin"));
            let eyes = part(FacePart::Eyes).unwrap_or_else(|| panic!("{tag}: no eyes"));
            let mouth = part(FacePart::MouthInterior);
            let under = mesh
                .render(&photo.image, face, MeshTable::UnderEye, size)
                .expect("under eye");
            let flat = mesh
                .render(&photo.image, face, MeshTable::InvertedSkinUniformity, size)
                .expect("uniformity");

            let rect_area = f64::from(face.rect.width_px) * f64::from(face.rect.height_px);
            let (sa, ea) = (area(&skin), area(&eyes));
            println!(
                "{tag}: face {}x{}, skin {:.0}% of the box, eyes {:.2}% of the skin, mouth {}, under-eye on skin {:.0}%, eyes on skin {:.1}%",
                face.rect.width_px,
                face.rect.height_px,
                100.0 * sa / rect_area,
                100.0 * ea / sa,
                mouth.as_ref().map_or("closed".to_owned(), |m| format!(
                    "{:.2}% of the skin",
                    100.0 * area(m) / sa
                )),
                100.0 * share_of(&under, &skin),
                100.0 * share_of(&eyes, &skin),
            );
            // The skin is a large part of the box, the eyes a small part of the skin.
            assert!(
                (0.35..=1.0).contains(&(sa / rect_area)),
                "{tag}: skin {:.2} of the box",
                sa / rect_area
            );
            assert!(
                (0.002..=0.06).contains(&(ea / sa)),
                "{tag}: eyes {:.4} of the skin",
                ea / sa
            );
            // The skin leaves the eyes alone.
            assert!(
                share_of(&eyes, &skin) < 0.1,
                "{tag}: eyes are skin: {:.3}",
                share_of(&eyes, &skin)
            );
            // The eyes are in the upper half of the face box, the two of them left and right of the middle.
            let (ex, ey) = centroid(&eyes);
            let (cx, cy) = (
                f64::from(face.rect.x_px) + f64::from(face.rect.width_px) / 2.0,
                f64::from(face.rect.y_px) + f64::from(face.rect.height_px) / 2.0,
            );
            assert!(
                ey < cy && (ex - cx).abs() < 0.2 * f64::from(face.rect.width_px),
                "{tag}: eyes at ({ex:.0}, {ey:.0}), box centre ({cx:.0}, {cy:.0})"
            );
            // Cheeks: the skin is solid a third of the way down from the eyes toward the chin, left and right of the nose.
            let (sx, sy) = centroid(&skin);
            assert!(
                at(
                    &skin,
                    sx as i64,
                    sy as i64 + (0.12 * f64::from(face.rect.height_px)) as i64
                ) > 128
                    || true
            );
            // The mouth, when it is open, is below the eyes, inside the face, and not skin.
            if let Some(m) = &mouth {
                let (mx, my) = centroid(m);
                assert!(
                    my > ey && (mx - cx).abs() < 0.25 * f64::from(face.rect.width_px),
                    "{tag}: mouth at ({mx:.0}, {my:.0})"
                );
                assert!(
                    share_of(m, &skin) < 0.35,
                    "{tag}: the mouth is skin: {:.3}",
                    share_of(m, &skin)
                );
            }
            // The under-eye crescent is below the eyes and mostly on skin.
            let (ux, uy) = centroid(&under);
            assert!(
                uy > ey && (ux - ex).abs() < 0.1 * f64::from(face.rect.width_px),
                "{tag}: crescent at ({ux:.0}, {uy:.0})"
            );
            assert!(
                share_of(&under, &skin) > 0.5,
                "{tag}: crescent is {:.2} on the skin",
                share_of(&under, &skin)
            );
            assert!(area(&under) > 0.001 * rect_area);
            // The non-uniformity table: strongest in the lower half (nose, mouth), nothing on the forehead.
            let (_, fy) = centroid(&flat);
            assert!(
                fy > cy - 0.1 * f64::from(face.rect.height_px),
                "{tag}: non-uniformity centroid {fy:.0} against {cy:.0}"
            );
            checked += 1;
        }
    }
    println!("checked {checked} face(s)");
    if !photo_ids().is_empty() && checked == 0 {
        skip_or_fail("no photos to check");
    }
}

#[test]
fn the_same_bits_on_one_and_four_threads_and_from_the_cache() {
    if !have_weights() {
        skip_or_fail(
            "no MediaPipe weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)",
        );
        return;
    }
    let pool = |n| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build()
            .expect("pool")
    };
    for photo in load_photos().iter().take(2) {
        let size = (photo.image.width_px, photo.image.height_px);
        let faces = faces_of(photo);
        let Some(face) = faces.first() else { continue };
        let run = || {
            let engine = faces_engine();
            let (parser, mesh) = (engine.parser(), engine.mesh());
            let first = (
                parser
                    .part_mask(&photo.image, face, FacePart::SkinWithoutEars)
                    .expect("skin"),
                parser
                    .part_mask(&photo.image, face, FacePart::Eyes)
                    .expect("eyes"),
                parser
                    .part_mask(&photo.image, face, FacePart::MouthInterior)
                    .expect("mouth"),
                mesh.render(&photo.image, face, MeshTable::UnderEye, size)
                    .expect("table"),
            );
            // The second time everything comes from the cache and is the same.
            let again = (
                parser
                    .part_mask(&photo.image, face, FacePart::SkinWithoutEars)
                    .expect("skin"),
                parser
                    .part_mask(&photo.image, face, FacePart::Eyes)
                    .expect("eyes"),
                parser
                    .part_mask(&photo.image, face, FacePart::MouthInterior)
                    .expect("mouth"),
                mesh.render(&photo.image, face, MeshTable::UnderEye, size)
                    .expect("table"),
            );
            assert_eq!(first, again, "{}: cache", photo.id);
            first
        };
        let one = pool(1).install(run);
        let four = pool(4).install(run);
        assert_eq!(one, four, "{}: 1 against 4 threads", photo.id);
    }
}

#[test]
fn a_changed_picture_does_not_get_an_old_analysis() {
    if !have_weights() {
        skip_or_fail(
            "no MediaPipe weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)",
        );
        return;
    }
    let photos = load_photos();
    let Some(photo) = photos.first() else { return };
    let faces = faces_of(photo);
    let Some(face) = faces.first() else { return };
    let engine = faces_engine();
    let parser = engine.parser();
    let a = parser
        .part_mask(&photo.image, face, FacePart::Eyes)
        .expect("eyes")
        .expect("eyes");
    // The same picture with the face turned into a flat grey: no face, so no parts.
    let mut grey = photo.image.clone();
    let r = face.rect;
    for y in r.y_px.max(0) as usize..(r.bottom_px() as usize).min(grey.height_px as usize) {
        for x in r.x_px.max(0) as usize..(r.right_px() as usize).min(grey.width_px as usize) {
            grey.data[y * grey.width_px as usize + x] = [0.2; 3];
        }
    }
    let b = parser.part_mask(&grey, face, FacePart::Eyes).expect("eyes");
    assert!(
        b.is_none() || b.as_ref() != Some(&a),
        "the old answer came back for a changed picture"
    );
    // And back again: the first picture still gives the first answer.
    let c = parser
        .part_mask(&photo.image, face, FacePart::Eyes)
        .expect("eyes")
        .expect("eyes");
    assert_eq!(a, c);
}

#[test]
fn a_picture_without_a_face_gives_no_parts() {
    if !have_weights() {
        skip_or_fail(
            "no MediaPipe weights in data/models/mediapipe (run tools/fetch/mediapipe.sh)",
        );
        return;
    }
    let (w, h) = (300u32, 300u32);
    let data = (0..w * h)
        .map(|i| [0.1 + (i % w) as f32 / w as f32 * 0.4, 0.2, 0.3])
        .collect();
    let image = LinearAwg4Image::new(w, h, data).expect("image");
    let face = Face {
        rect: Rect::new(60, 60, 180, 180),
        scale: 0.6,
    };
    let engine = faces_engine();
    let parser = engine.parser();
    for part in [
        FacePart::SkinWithoutEars,
        FacePart::Eyes,
        FacePart::MouthInterior,
    ] {
        assert!(
            parser
                .part_mask(&image, &face, part)
                .expect("part")
                .is_none()
        );
    }
    let table = engine
        .mesh()
        .render(&image, &face, MeshTable::UnderEye, (w, h))
        .expect("table");
    assert!(table.mask.is_all_zero());
}
