//! The scene of a picture for Reference Grade: which pixels are sky, vegetation, water, ground, built surfaces, skin and the
//! subject (`docs/34-reference-grade.md` sections 4a.3, 4b.2 and 4b.3, RG-04).
//!
//! The pieces are the segmenter's label map ([`crate::eomt`], RG-03), MediaPipe's Selfie Multiclass ([`crate::mediapipe::selfie`]) and
//! two rules written here:
//!
//! * **The class table** of section 4b.2 as bits ([`element_bit`]): `sky`, `vegetation`, `water`, `ground` and `built` (architecture
//!   plus interior) are the pixels whose class belongs to them. `person`, `other_thing` and `other_stuff` have no grade of their
//!   own.
//! * **Skin** ([`skin_mask`]): the classes `BodySkin` and `FaceSkin` of MediaPipe, intersected with `person`, run on each 4-connected
//!   component of `person` that covers at least 0.5 % of the picture, on the square crop of the component's box times 1.15; each pixel
//!   takes the class of the highest score. It runs only when `person` covers at least 3 % of the picture.
//! * **The subject** ([`subject_mask`]): `person` when it covers at least 3 %; else the `other_thing` pixels in the central 60 % of the
//!   frame when they cover at least 1 %; else none. (Apple Vision is never used: `docs/34` section 4b.3.)
//!
//! A scene is [`SceneBits`]: one byte per pixel of the picture's grid, a bit for each of the seven elements. The grid is the one of
//! the picture the segmenter read (the Input's picture, 1024 px on the long edge); the label map of the model's own grid (at most
//! 640 px) is carried to it by nearest neighbour, centre-mapped, like every other size. Everything is integer or fixed-order
//! arithmetic over the same labels, so the same picture gives the same scene on every run, thread count and platform.

use crate::eomt::{self, Element, Eomt, LabelMap};
use crate::mediapipe::blazeface::{PixelSource, Resample, Rgb8};
use crate::mediapipe::roi::Roi;
use crate::mediapipe::selfie::{Class, ClassMap, SelfieSegmenter};
use crate::model::{MlError, Result};

/// The version of the rules of this module: the class table, the skin and subject rules and their numbers. It is part of the key of
/// a photo's scene (`docs/34` section 7.1), so a change of a rule makes the scenes again.
pub const RULES_VERSION: u32 = 1;

/// The number of elements a scene holds: six elements and the subject.
pub const ELEMENT_COUNT: usize = 7;
/// Bit of `sky`.
pub const SKY: u8 = 1 << 0;
/// Bit of `vegetation`.
pub const VEGETATION: u8 = 1 << 1;
/// Bit of `water`.
pub const WATER: u8 = 1 << 2;
/// Bit of `ground`.
pub const GROUND: u8 = 1 << 3;
/// Bit of `built` (architecture and interior).
pub const BUILT: u8 = 1 << 4;
/// Bit of `skin`.
pub const SKIN: u8 = 1 << 5;
/// Bit of `subject`.
pub const SUBJECT: u8 = 1 << 6;

/// The names of the seven elements, in bit order (the order of `ElementGrade`s in `ym_color::hue`).
pub const ELEMENT_NAMES: [&str; ELEMENT_COUNT] = [
    "sky",
    "vegetation",
    "water",
    "ground",
    "built",
    "skin",
    "subject",
];

/// `person` must cover this share of the picture to be the subject and for skin to be looked for.
pub const PERSON_MIN_SHARE: f64 = 0.03;
/// A component of `person` must cover this share of the picture to get a crop of its own.
pub const SKIN_COMPONENT_MIN_SHARE: f64 = 0.005;
/// The side of a component's crop, times the longer side of its box.
pub const SKIN_CROP_MARGIN: f64 = 1.15;
/// The `other_thing` pixels of the central window must cover this share of the picture to be the subject.
pub const SUBJECT_THING_MIN_SHARE: f64 = 0.01;
/// The central window of the subject rule, as shares of the width and the height: from this to [`SUBJECT_WINDOW_TO`].
pub const SUBJECT_WINDOW_FROM: f64 = 0.2;
/// The end of the central window of the subject rule.
pub const SUBJECT_WINDOW_TO: f64 = 0.8;

/// A scene: for every pixel of a `width x height` grid a byte whose bits say which elements the pixel belongs to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneBits {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// `width * height` bytes, rows top to bottom, with the bits of [`SKY`] to [`SUBJECT`].
    pub bits: Vec<u8>,
    /// `width * height` bytes: the segmenter's element number of every pixel ([`Element`] as its number: unlabelled 0, sky 1,
    /// vegetation 2, water 3, ground 4, architecture 5, interior 6, person 7, other thing 8, other stuff 9). The bits fold these
    /// into the grade's elements; the fit's measures (`docs/34` section 4b.4) need the finer table.
    pub classes: Vec<u8>,
    /// Whether skin could be looked for: false when the MediaPipe weights were not there (the scene has no skin then).
    pub skin_available: bool,
}

impl SceneBits {
    /// The share of the pixels that have `bit`.
    #[must_use]
    pub fn share(&self, bit: u8) -> f64 {
        if self.bits.is_empty() {
            return 0.0;
        }
        self.bits.iter().filter(|&&b| b & bit != 0).count() as f64 / self.bits.len() as f64
    }

    /// Which elements cover at least `min_share` of the picture (`docs/34` section 4a.3, step 5: a fit makes the masks of every
    /// element of at least 2 %).
    #[must_use]
    pub fn covering(&self, min_share: f64) -> [bool; ELEMENT_COUNT] {
        let mut out = [false; ELEMENT_COUNT];
        for (e, o) in out.iter_mut().enumerate() {
            *o = self.share(1 << e) >= min_share && self.share(1 << e) > 0.0;
        }
        out
    }
}

/// The bit of the grade's element a segmenter element belongs to (`docs/34` section 4b.2): `built` is architecture and interior;
/// `person`, `other_thing`, `other_stuff` and unlabelled pixels have none (skin and the subject are made by their own rules).
#[must_use]
pub const fn element_bit(element: Element) -> u8 {
    match element {
        Element::Sky => SKY,
        Element::Vegetation => VEGETATION,
        Element::Water => WATER,
        Element::Ground => GROUND,
        Element::Architecture | Element::Interior => BUILT,
        Element::Unlabelled | Element::Person | Element::OtherThing | Element::OtherStuff => 0,
    }
}

/// The class table as bits: the bit of the element of every class id (`eomt::UNLABELLED` and above have none).
#[must_use]
pub fn class_bit(class: u8) -> u8 {
    element_bit(eomt::element_of(class))
}

/// The bits of the five class elements for every pixel of a label map (no skin, no subject).
#[must_use]
pub fn class_bits(labels: &LabelMap) -> Vec<u8> {
    let mut table = [0u8; 256];
    for (c, t) in table.iter_mut().enumerate() {
        *t = class_bit(c.min(255) as u8);
    }
    labels
        .classes
        .iter()
        .map(|&c| table[usize::from(c)])
        .collect()
}

/// Which pixels of a label map are `person`.
#[must_use]
pub fn person_mask(labels: &LabelMap) -> Vec<bool> {
    labels
        .classes
        .iter()
        .map(|&c| eomt::element_of(c) == Element::Person)
        .collect()
}

/// The subject of a picture (`docs/34` section 4b.3): `person` when it covers at least [`PERSON_MIN_SHARE`]; else the `other_thing`
/// pixels in the central window (the pixel's left and top edge, in shares of the width and the height, from
/// [`SUBJECT_WINDOW_FROM`] up to but not including [`SUBJECT_WINDOW_TO`]) when they cover at least [`SUBJECT_THING_MIN_SHARE`];
/// else nothing. `elements` holds the segmenter's element number of every pixel of a `width x height` grid.
#[must_use]
pub fn subject_mask(elements: &[u8], width: usize, height: usize) -> Vec<bool> {
    let n = (width * height) as f64;
    let person: Vec<bool> = elements
        .iter()
        .map(|&e| e == Element::Person as u8)
        .collect();
    if n > 0.0 && person.iter().filter(|&&p| p).count() as f64 / n >= PERSON_MIN_SHARE {
        return person;
    }
    let central = |i: usize| {
        let (x, y) = (
            (i % width) as f64 / width as f64,
            (i / width) as f64 / height as f64,
        );
        (SUBJECT_WINDOW_FROM..SUBJECT_WINDOW_TO).contains(&x)
            && (SUBJECT_WINDOW_FROM..SUBJECT_WINDOW_TO).contains(&y)
    };
    let thing: Vec<bool> = (0..width * height)
        .map(|i| elements[i] == Element::OtherThing as u8 && central(i))
        .collect();
    if n > 0.0 && thing.iter().filter(|&&p| p).count() as f64 / n >= SUBJECT_THING_MIN_SHARE {
        thing
    } else {
        vec![false; width * height]
    }
}

/// The 4-connected components of a mask: for each, its pixel count and its box `[x0, y0, x1, y1]` (inclusive), in the order of their
/// first pixel (row by row). Iterative, so a large component cannot overflow the stack.
#[must_use]
pub fn components(mask: &[bool], width: usize, height: usize) -> Vec<(usize, [usize; 4])> {
    let mut seen = vec![false; width * height];
    let mut out = Vec::new();
    for start in 0..width * height {
        if !mask[start] || seen[start] {
            continue;
        }
        let (mut count, mut bb) = (0usize, [width, height, 0usize, 0usize]);
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(i) = stack.pop() {
            let (x, y) = (i % width, i / width);
            count += 1;
            bb = [bb[0].min(x), bb[1].min(y), bb[2].max(x), bb[3].max(y)];
            for (nx, ny) in [
                (x.wrapping_sub(1), y),
                (x + 1, y),
                (x, y.wrapping_sub(1)),
                (x, y + 1),
            ] {
                if nx < width && ny < height && mask[ny * width + nx] && !seen[ny * width + nx] {
                    seen[ny * width + nx] = true;
                    stack.push(ny * width + nx);
                }
            }
        }
        out.push((count, bb));
    }
    out
}

/// The skin of a picture (`docs/34` section 4b.3): `segment` gives the class probabilities of a square crop of the picture (the
/// network of MediaPipe's Selfie Multiclass on it); the result is the pixels of `person` that the crop of their component calls
/// `BodySkin` or `FaceSkin`, where the class of a pixel is the one of the highest probability (the last of equals, as the POC took
/// it). Nothing is looked for when `person` covers less than [`PERSON_MIN_SHARE`] of the picture, and a component that covers less
/// than [`SKIN_COMPONENT_MIN_SHARE`] has no crop and no skin.
///
/// # Errors
///
/// The error of `segment`, the first time it fails.
pub fn skin_mask(
    person: &[bool],
    width: usize,
    height: usize,
    segment: &dyn Fn(&Roi) -> Result<ClassMap>,
) -> Result<Vec<bool>> {
    let mut skin = vec![false; width * height];
    let n = (width * height) as f64;
    if n == 0.0 || (person.iter().filter(|&&p| p).count() as f64) / n < PERSON_MIN_SHARE {
        return Ok(skin);
    }
    const CLASSES: [Class; 6] = [
        Class::Background,
        Class::Hair,
        Class::BodySkin,
        Class::FaceSkin,
        Class::Clothes,
        Class::Others,
    ];
    for (count, bb) in components(person, width, height) {
        if (count as f64) < SKIN_COMPONENT_MIN_SHARE * n {
            continue;
        }
        let (bw, bh) = ((bb[2] - bb[0] + 1) as f64, (bb[3] - bb[1] + 1) as f64);
        let roi = Roi {
            center_px: [bb[0] as f64 + bw / 2.0, bb[1] as f64 + bh / 2.0],
            side_px: bw.max(bh) * SKIN_CROP_MARGIN,
            rotation_rad: 0.0,
        };
        let map = segment(&roi)?;
        for y in bb[1]..=bb[3] {
            for x in bb[0]..=bb[2] {
                let i = y * width + x;
                if !person[i] {
                    continue;
                }
                let [u, v] = roi.from_picture([x as f64 + 0.5, y as f64 + 0.5]);
                if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                    continue;
                }
                let best = CLASSES
                    .into_iter()
                    .max_by(|a, b| map.prob(*a, u, v).total_cmp(&map.prob(*b, u, v)));
                skin[i] = matches!(best, Some(Class::BodySkin | Class::FaceSkin));
            }
        }
    }
    Ok(skin)
}

/// The scene of a label map on a `width x height` grid, from the picture `rgb` of that grid (8 bits, rows top to bottom, three bytes
/// a pixel): the five class elements, the subject, and skin when `selfie` is given (`skin_available` says whether it was). The
/// label map is carried to the grid by nearest neighbour.
///
/// # Errors
///
/// [`MlError::Shape`] when `rgb` is not `width * height * 3` bytes or a side is 0, and the errors of MediaPipe's network.
pub fn scene_from_labels(
    labels: &LabelMap,
    rgb: &[u8],
    width: usize,
    height: usize,
    selfie: Option<&SelfieSegmenter>,
) -> Result<SceneBits> {
    let src = Rgb8::new(width, height, rgb)?;
    let grid = labels.resized_nearest(width, height);
    let mut bits = class_bits(&grid);
    let elements = grid.elements();
    let subject = subject_mask(&elements, width, height);
    let skin = match selfie {
        Some(seg) => skin_mask(&person_mask(&grid), width, height, &|roi| {
            seg.segment(&src as &dyn PixelSource, roi, Resample::Box)
        })?,
        None => vec![false; width * height],
    };
    for (i, b) in bits.iter_mut().enumerate() {
        if subject[i] {
            *b |= SUBJECT;
        }
        if skin[i] {
            *b |= SKIN;
        }
    }
    Ok(SceneBits {
        width,
        height,
        bits,
        classes: elements,
        skin_available: selfie.is_some(),
    })
}

/// The scene of a picture: the segmenter on it ([`Eomt::segment_with`]), then [`scene_from_labels`] on the same grid. `cancel` is
/// asked between the network's blocks.
///
/// # Errors
///
/// [`MlError::Cancelled`] when `cancel` says so, and the errors of the networks and of [`scene_from_labels`].
pub fn make_scene(
    eomt: &Eomt,
    selfie: Option<&SelfieSegmenter>,
    rgb: &[u8],
    width: usize,
    height: usize,
    cancel: &dyn Fn() -> bool,
) -> Result<SceneBits> {
    let labels = eomt.segment_with(rgb, width, height, cancel)?;
    if cancel() {
        return Err(MlError::Cancelled);
    }
    scene_from_labels(&labels, rgb, width, height, selfie)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eomt::{CLASS_NAMES, CLASSES, UNLABELLED};
    use crate::mediapipe::selfie::{CLASSES as SELFIE_CLASSES, INPUT_PX};

    /// The class table of section 4b.2, written out by name once more: each listed name has its bit, every other class has none.
    #[test]
    fn the_class_table_gives_each_class_its_element_and_nothing_else() {
        let named = |names: &[&str]| -> Vec<usize> {
            names
                .iter()
                .map(|n| {
                    CLASS_NAMES
                        .iter()
                        .position(|c| c == n)
                        .unwrap_or_else(|| panic!("no class called {n}"))
                })
                .collect()
        };
        let groups: [(u8, Vec<usize>); 5] = [
            (SKY, named(&["sky-other-merged"])),
            (
                VEGETATION,
                named(&["tree-merged", "grass-merged", "flower", "potted plant"]),
            ),
            (WATER, named(&["sea", "river", "water-other"])),
            (
                GROUND,
                named(&[
                    "sand",
                    "dirt-merged",
                    "gravel",
                    "road",
                    "pavement-merged",
                    "mountain-merged",
                    "rock-merged",
                    "snow",
                    "playingfield",
                    "railroad",
                ]),
            ),
            (
                BUILT,
                named(&[
                    // architecture
                    "building-other-merged",
                    "house",
                    "roof",
                    "bridge",
                    "fence-merged",
                    "wall-brick",
                    "wall-stone",
                    "wall-tile",
                    "wall-wood",
                    "wall-other-merged",
                    "window-other",
                    "platform",
                    "stairs",
                    "tent",
                    // interior
                    "ceiling-merged",
                    "floor-wood",
                    "floor-other-merged",
                    "curtain",
                    "rug-merged",
                    "cabinet-merged",
                    "counter",
                    "shelf",
                    "window-blind",
                    "door-stuff",
                    "mirror-stuff",
                    "pillow",
                    "blanket",
                    "towel",
                    "table-merged",
                    "light",
                ]),
            ),
        ];
        let mut listed = 0;
        for (bit, classes) in &groups {
            for &c in classes {
                assert_eq!(class_bit(c as u8), *bit, "{}", CLASS_NAMES[c]);
                listed += 1;
            }
        }
        // 1 + 4 + 3 + 10 + 30 classes; the other 85 (person, the other things, the other stuff) and "unlabelled" have none.
        assert_eq!(listed, 48);
        for (c, name) in CLASS_NAMES.iter().enumerate().take(CLASSES) {
            if !groups.iter().any(|(_, g)| g.contains(&c)) {
                assert_eq!(class_bit(c as u8), 0, "{name}");
            }
        }
        assert_eq!(class_bit(UNLABELLED), 0);
        assert_eq!(class_bit(255), 0);
        // Person is a class of its own, never a bit: skin and the subject come from their rules.
        assert_eq!(class_bit(0), 0);
        assert_eq!(CLASS_NAMES[0], "person");
    }

    fn labels(w: usize, h: usize, f: impl Fn(usize, usize) -> u8) -> LabelMap {
        let mut classes = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                classes.push(f(x, y));
            }
        }
        LabelMap {
            width: w,
            height: h,
            classes,
        }
    }

    const PERSON: u8 = 0;
    const OTHER_THING: u8 = 1; // bicycle
    const SKY_CLASS: u8 = 119; // checked below by name

    #[test]
    fn the_test_class_ids_are_what_they_are_called() {
        assert_eq!(CLASS_NAMES[usize::from(PERSON)], "person");
        assert_eq!(CLASS_NAMES[usize::from(OTHER_THING)], "bicycle");
        assert_eq!(CLASS_NAMES[usize::from(SKY_CLASS)], "sky-other-merged");
    }

    /// The subject: person at 3 % or more; below it the other things of the central window at 1 % or more; else none.
    #[test]
    fn the_subject_is_the_person_then_the_central_thing_then_nothing() {
        let (w, h) = (100, 100);
        let elements = |l: &LabelMap| l.elements();
        // Person 3 % (300 of 10,000 pixels): the subject is the person, wherever it is.
        let l = labels(
            w,
            h,
            |x, y| if x < 30 && y < 10 { PERSON } else { SKY_CLASS },
        );
        let s = subject_mask(&elements(&l), w, h);
        assert_eq!(s.iter().filter(|&&b| b).count(), 300);
        assert!(s[0] && !s[w * 50]);
        // Person 299 pixels: under 3 %. No thing in the window: no subject.
        let l = labels(
            w,
            h,
            |x, y| {
                if (y * w + x) < 299 { PERSON } else { SKY_CLASS }
            },
        );
        assert!(subject_mask(&elements(&l), w, h).iter().all(|&b| !b));
        // Under 3 % of person, and things: those of the central window (x, y from 0.2 up to 0.8 by the pixel's left edge) at 1 %.
        let l = labels(w, h, |x, y| {
            let thing = (10..30).contains(&x) && (10..40).contains(&y);
            if (y * w + x) < 100 {
                PERSON
            } else if thing {
                OTHER_THING
            } else {
                SKY_CLASS
            }
        });
        let s = subject_mask(&elements(&l), w, h);
        // The things of x 20..30, y 20..40 are in the window: 10 * 20 = 200 pixels; the ones outside it are not the subject.
        assert_eq!(s.iter().filter(|&&b| b).count(), 200);
        assert!(s[25 * w + 25] && !s[15 * w + 15] && !s[0]);
        // Things in the window at 99 of 10,000 pixels (just under 1 %): none. At 100: the subject.
        let thing_at = |n: usize| {
            labels(w, h, move |x, y| {
                if (50..60).contains(&y) && (20..80).contains(&x) && (y - 50) * 60 + (x - 20) < n {
                    OTHER_THING
                } else {
                    SKY_CLASS
                }
            })
        };
        assert!(
            subject_mask(&elements(&thing_at(99)), w, h)
                .iter()
                .all(|&b| !b)
        );
        assert_eq!(
            subject_mask(&elements(&thing_at(100)), w, h)
                .iter()
                .filter(|&&b| b)
                .count(),
            100
        );
        // The edge of the window: x = 80 is 0.8, out; x = 79 is in; x = 20 is 0.2, in; x = 19 out.
        let l = labels(w, h, |x, y| {
            if (20..80).contains(&y) && (x == 19 || x == 20 || x == 79 || x == 80) {
                OTHER_THING
            } else {
                SKY_CLASS
            }
        });
        let s = subject_mask(&elements(&l), w, h);
        assert!(s[30 * w + 20] && s[30 * w + 79] && !s[30 * w + 19] && !s[30 * w + 80]);
    }

    #[test]
    fn components_are_four_connected_with_their_boxes() {
        let (w, h) = (6, 4);
        let m = [
            "##..#.", //
            "#...##", //
            "..#...", //
            "..##..",
        ];
        let mask: Vec<bool> = m.iter().flat_map(|r| r.chars().map(|c| c == '#')).collect();
        let c = components(&mask, w, h);
        assert_eq!(
            c,
            vec![(3, [0, 0, 1, 1]), (3, [4, 0, 5, 1]), (3, [2, 2, 3, 3])]
        );
        // A diagonal pair is two components.
        let mask = [true, false, false, true];
        assert_eq!(components(&mask, 2, 2).len(), 2);
    }

    /// A class map whose crop is `face` (face skin) in the left half and `hair` in the right, in every row.
    fn halves(roi: &Roi) -> ClassMap {
        let mut probs = vec![0.0f32; INPUT_PX * INPUT_PX * SELFIE_CLASSES];
        for y in 0..INPUT_PX {
            for x in 0..INPUT_PX {
                let class = if x < INPUT_PX / 2 {
                    Class::FaceSkin
                } else {
                    Class::Hair
                };
                probs[(y * INPUT_PX + x) * SELFIE_CLASSES + class as usize] = 0.9;
                probs[(y * INPUT_PX + x) * SELFIE_CLASSES + Class::Background as usize] = 0.1;
            }
        }
        ClassMap { roi: *roi, probs }
    }

    #[test]
    fn skin_is_the_skin_classes_inside_person_on_a_crop_of_each_big_component() {
        let (w, h) = (200, 200);
        // One person block 60 x 100 pixels at x 40..100, y 50..150: 15 % of the picture. Its crop is centred on the box with a side of
        // 100 * 1.15, so the left half of the crop is skin.
        let person: Vec<bool> = (0..w * h)
            .map(|i| (40..100).contains(&(i % w)) && (50..150).contains(&(i / w)))
            .collect();
        let seen = std::cell::RefCell::new(Vec::new());
        let skin = skin_mask(&person, w, h, &|roi| {
            seen.borrow_mut().push(*roi);
            Ok(halves(roi))
        })
        .expect("skin");
        let seen = seen.into_inner();
        assert_eq!(seen.len(), 1, "one crop for one component");
        assert_eq!(seen[0].center_px, [70.0, 100.0]);
        assert!((seen[0].side_px - 115.0).abs() < 1e-9);
        assert_eq!(seen[0].rotation_rad, 0.0);
        // The crop's left half is x < 70 (the centre): skin there, inside person only.
        assert!(skin[100 * w + 50] && skin[100 * w + 69]);
        assert!(!skin[100 * w + 80], "hair side");
        assert!(
            !skin[100 * w + 30],
            "outside person, though the crop covers it"
        );
        assert!(!skin[10 * w + 50]);
        // Skin is a subset of person.
        assert!(skin.iter().zip(&person).all(|(&s, &p)| !s || p));
    }

    #[test]
    fn skin_is_not_looked_for_under_three_percent_of_person_and_small_components_have_no_crop() {
        let (w, h) = (100, 100);
        let called = std::cell::Cell::new(0);
        let seg = |roi: &Roi| {
            called.set(called.get() + 1);
            Ok(halves(roi))
        };
        // 299 person pixels (2.99 %): the network is not run.
        let person: Vec<bool> = (0..w * h).map(|i| i < 299).collect();
        let skin = skin_mask(&person, w, h, &seg).expect("skin");
        assert_eq!(called.get(), 0);
        assert!(skin.iter().all(|&s| !s));
        // 3 % of person (300 pixels) made of one big component (251 pixels) and a small one of 49 pixels (0.49 %, under 0.5 %): one crop.
        let mut person = vec![false; w * h];
        for y in 0..10 {
            for x in 0..25 {
                person[y * w + x] = true; // 250 pixels
            }
        }
        person[10 * w] = true; // 251 pixels
        for y in 50..57 {
            for x in 50..57 {
                person[y * w + x] = true; // 49 pixels
            }
        }
        let skin = skin_mask(&person, w, h, &seg).expect("skin");
        assert_eq!(called.get(), 1);
        assert!(!skin[50 * w + 50], "the small component has no skin");
        assert!(skin.iter().any(|&s| s));
        // At exactly 0.5 % (50 pixels) a second component gets a crop.
        person[57 * w + 50] = true;
        let skin = skin_mask(&person, w, h, &seg).expect("skin");
        assert_eq!(called.get(), 3, "two crops this time (one already counted)");
        assert!(skin.iter().any(|&s| s));
    }

    #[test]
    fn a_failing_network_fails_the_scene() {
        let person = vec![true; 100];
        let r = skin_mask(&person, 10, 10, &|_| Err(MlError::Input("no".into())));
        assert!(matches!(r, Err(MlError::Input(_))));
    }

    /// The whole assembly on a label map: the five class bits, the subject, and skin when asked; `skin_available` says whether the
    /// network was there.
    #[test]
    fn a_scene_has_the_class_bits_the_subject_and_a_flag_for_skin() {
        let (w, h) = (64, 48);
        // Sky in the top third, a person block in the middle, the ground below.
        let l = labels(w, h, |x, y| {
            if y < 16 {
                SKY_CLASS
            } else if (20..44).contains(&x) && (16..40).contains(&y) {
                PERSON
            } else {
                // dirt-merged
                CLASS_NAMES
                    .iter()
                    .position(|c| *c == "dirt-merged")
                    .unwrap() as u8
            }
        });
        let rgb = vec![128u8; w * h * 3];
        let scene = scene_from_labels(&l, &rgb, w, h, None).expect("scene");
        assert!(!scene.skin_available);
        assert_eq!(scene.bits[0], SKY);
        assert_eq!(
            scene.bits[30 * w + 30],
            SUBJECT,
            "a person pixel is the subject, no class bit"
        );
        assert_eq!(scene.bits[45 * w], GROUND);
        assert!(scene.bits.iter().all(|&b| b & SKIN == 0));
        // The label map of another grid is carried by nearest neighbour onto the scene's.
        let small = labels(16, 12, |x, y| {
            if y < 4 {
                SKY_CLASS
            } else if x < 8 {
                PERSON
            } else {
                255
            }
        });
        let scene = scene_from_labels(&small, &rgb, w, h, None).expect("scene");
        assert_eq!(scene.bits[0], SKY);
        assert_eq!(scene.bits[30 * w + 10], SUBJECT);
        assert_eq!(scene.bits[30 * w + 60], 0);
        // Shares and the covering test.
        let c = scene.covering(0.02);
        assert!(c[0] && c[6] && !c[1] && !c[5]);
        assert!((scene.share(SKY) - 4.0 / 12.0).abs() < 1e-9);
    }
}
