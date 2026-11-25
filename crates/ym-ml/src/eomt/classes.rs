//! The 133 COCO panoptic classes of the checkpoint and the elements Reference Grade grades (`docs/34` section 4b.2).
//!
//! The names are the checkpoint's `id2label` (`config.json` of `tue-mps/coco_panoptic_eomt_small_640_2x`; ids 0 to 79 are the
//! 80 "thing" classes, 80 to 132 the 53 "stuff" classes of COCO's merged panoptic set). A test compares this table with the
//! `id2label` that `training/eomt/export.py` writes into the weights' header and that `tools/eomt/parity_fixture.py` writes into
//! the fixture, so a table that drifts from the checkpoint fails.

/// The number of classes of the checkpoint (the head has one more output, "none").
pub const CLASSES: usize = 133;
/// The first stuff class: ids from here to [`CLASSES`] are the stuff classes (`stuff_classes=list(range(80, 133))`).
pub const FIRST_STUFF: usize = 80;
/// The value of a pixel that no segment claims, in a label map: one past the last class.
pub const UNLABELLED: u8 = CLASSES as u8;

/// What the grade tells apart (`docs/34` section 4b.2). The numbers are the POC's element ids (`classmap.py`).
/// `built` is [`Element::Architecture`] plus [`Element::Interior`]; skin is not a class (it comes from MediaPipe).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Element {
    /// No segment claims the pixel.
    Unlabelled = 0,
    /// `sky-other-merged`.
    Sky = 1,
    /// Trees, grass, flowers and potted plants.
    Vegetation = 2,
    /// Sea, river and other water.
    Water = 3,
    /// Sand, dirt, gravel, roads, pavement, mountains, rock, snow, playing fields and railroads.
    Ground = 4,
    /// Buildings, houses, roofs, bridges, fences, walls, windows, platforms, stairs and tents.
    Architecture = 5,
    /// Ceilings, floors, curtains, rugs, furniture, blinds, doors, mirrors, pillows, blankets, towels, tables and lights.
    Interior = 6,
    /// `person`.
    Person = 7,
    /// Every other thing class (ids 0 to 79).
    OtherThing = 8,
    /// Every other stuff class (ids 80 to 132).
    OtherStuff = 9,
}

/// Every element, in the order of their numbers.
pub const ELEMENTS: [Element; 10] = [
    Element::Unlabelled,
    Element::Sky,
    Element::Vegetation,
    Element::Water,
    Element::Ground,
    Element::Architecture,
    Element::Interior,
    Element::Person,
    Element::OtherThing,
    Element::OtherStuff,
];

impl Element {
    /// The element's name as the spec and the POC write it.
    pub const fn name(self) -> &'static str {
        match self {
            Element::Unlabelled => "unlabelled",
            Element::Sky => "sky",
            Element::Vegetation => "vegetation",
            Element::Water => "water",
            Element::Ground => "ground",
            Element::Architecture => "architecture",
            Element::Interior => "interior",
            Element::Person => "person",
            Element::OtherThing => "other_thing",
            Element::OtherStuff => "other_stuff",
        }
    }

    /// `architecture` or `interior`: the element `built` of the grade.
    pub const fn is_built(self) -> bool {
        matches!(self, Element::Architecture | Element::Interior)
    }
}

use Element as E;

/// The class names, by id: the checkpoint's `id2label`.
pub const CLASS_NAMES: [&str; CLASSES] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
    "banner",
    "blanket",
    "bridge",
    "cardboard",
    "counter",
    "curtain",
    "door-stuff",
    "floor-wood",
    "flower",
    "fruit",
    "gravel",
    "house",
    "light",
    "mirror-stuff",
    "net",
    "pillow",
    "platform",
    "playingfield",
    "railroad",
    "river",
    "road",
    "roof",
    "sand",
    "sea",
    "shelf",
    "snow",
    "stairs",
    "tent",
    "towel",
    "wall-brick",
    "wall-stone",
    "wall-tile",
    "wall-wood",
    "water-other",
    "window-blind",
    "window-other",
    "tree-merged",
    "fence-merged",
    "ceiling-merged",
    "sky-other-merged",
    "cabinet-merged",
    "table-merged",
    "floor-other-merged",
    "pavement-merged",
    "mountain-merged",
    "grass-merged",
    "dirt-merged",
    "paper-merged",
    "food-other-merged",
    "building-other-merged",
    "rock-merged",
    "wall-other-merged",
    "rug-merged",
];

/// The element of each class, by id (`docs/34` section 4b.2).
const CLASS_ELEMENTS: [Element; CLASSES] = [
    E::Person,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::Vegetation,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherThing,
    E::OtherStuff,
    E::Interior,
    E::Architecture,
    E::OtherStuff,
    E::Interior,
    E::Interior,
    E::Interior,
    E::Interior,
    E::Vegetation,
    E::OtherStuff,
    E::Ground,
    E::Architecture,
    E::Interior,
    E::Interior,
    E::OtherStuff,
    E::Interior,
    E::Architecture,
    E::Ground,
    E::Ground,
    E::Water,
    E::Ground,
    E::Architecture,
    E::Ground,
    E::Water,
    E::Interior,
    E::Ground,
    E::Architecture,
    E::Architecture,
    E::Interior,
    E::Architecture,
    E::Architecture,
    E::Architecture,
    E::Architecture,
    E::Water,
    E::Interior,
    E::Architecture,
    E::Vegetation,
    E::Architecture,
    E::Interior,
    E::Sky,
    E::Interior,
    E::Interior,
    E::Interior,
    E::Ground,
    E::Ground,
    E::Vegetation,
    E::Ground,
    E::OtherStuff,
    E::OtherStuff,
    E::Architecture,
    E::Ground,
    E::Architecture,
    E::Interior,
];

/// The name of class `id`, or `None` past the last class.
pub fn class_name(id: u8) -> Option<&'static str> {
    CLASS_NAMES.get(usize::from(id)).copied()
}

/// The element of class `id`; [`UNLABELLED`] (and any value past the last class) is [`Element::Unlabelled`].
pub fn element_of(id: u8) -> Element {
    CLASS_ELEMENTS
        .get(usize::from(id))
        .copied()
        .unwrap_or(Element::Unlabelled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_table_has_the_shape_of_the_spec() {
        assert_eq!(CLASS_NAMES.len(), 133);
        assert_eq!(CLASS_NAMES[0], "person");
        assert_eq!(CLASS_NAMES[79], "toothbrush");
        assert_eq!(CLASS_NAMES[80], "banner");
        assert_eq!(CLASS_NAMES[132], "rug-merged");
        let count = |e: Element| (0..CLASSES).filter(|&i| CLASS_ELEMENTS[i] == e).count();
        assert_eq!(count(Element::Sky), 1);
        assert_eq!(count(Element::Vegetation), 4);
        assert_eq!(count(Element::Water), 3);
        assert_eq!(count(Element::Ground), 10);
        assert_eq!(count(Element::Architecture), 14);
        assert_eq!(count(Element::Interior), 16);
        assert_eq!(count(Element::Person), 1);
        assert_eq!(count(Element::OtherStuff), 6);
        assert_eq!(count(Element::OtherThing), 78);
    }

    #[test]
    fn every_name_the_spec_lists_is_in_the_table_and_maps_to_its_element() {
        let listed: [(Element, &[&str]); 7] = [
            (Element::Sky, &["sky-other-merged"]),
            (
                Element::Vegetation,
                &["tree-merged", "grass-merged", "flower", "potted plant"],
            ),
            (Element::Water, &["sea", "river", "water-other"]),
            (
                Element::Ground,
                &[
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
                ],
            ),
            (
                Element::Architecture,
                &[
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
                ],
            ),
            (
                Element::Interior,
                &[
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
                ],
            ),
            (Element::Person, &["person"]),
        ];
        for (element, names) in listed {
            for name in names {
                let id = CLASS_NAMES
                    .iter()
                    .position(|n| n == name)
                    .unwrap_or_else(|| panic!("{name} is not a class"));
                assert_eq!(element_of(id as u8), element, "{name}");
            }
        }
        for name in [
            "banner",
            "cardboard",
            "fruit",
            "net",
            "paper-merged",
            "food-other-merged",
        ] {
            let id = CLASS_NAMES
                .iter()
                .position(|n| *n == name)
                .expect("a class");
            assert_eq!(element_of(id as u8), Element::OtherStuff, "{name}");
        }
    }

    #[test]
    fn an_unlisted_class_is_other_thing_or_other_stuff_by_its_id() {
        for (id, name) in CLASS_NAMES.iter().enumerate() {
            let element = element_of(id as u8);
            if element == Element::OtherThing {
                assert!(id < FIRST_STUFF, "{id} {name}");
            }
            if element == Element::OtherStuff {
                assert!(id >= FIRST_STUFF, "{id} {name}");
            }
            if id < FIRST_STUFF && *name != "person" && *name != "potted plant" {
                assert_eq!(element, Element::OtherThing, "{name}");
            }
        }
        assert_eq!(element_of(UNLABELLED), Element::Unlabelled);
        assert_eq!(element_of(255), Element::Unlabelled);
    }

    #[test]
    fn the_names_are_unique() {
        let mut sorted = CLASS_NAMES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), CLASSES);
    }
}
