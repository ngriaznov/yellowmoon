//! RG-12b (`docs/34-reference-grade.md` section 9.6): the plan that ends at a Reference Grade node's mask shows one element's soft
//! mask when the chain is given an element view (`ChainParams::element_view`). This is the CPU reference of the view: the code
//! values of the mask view are the element's plane value `m` in all three channels, read from the render's planes at the pixel,
//! with no tolerance (the view reads a u16 plane code and divides it by its one, so the same code gives the same value).
//!
//! The chain's own mask rows, the other elements, a pixel with no plane and a node that is not the viewed one do not move it.

use std::collections::BTreeMap;
use std::sync::Arc;

use ym_color::consts::linalg::Mat3;
use ym_color::display::{ChainParams, ColorChain, MainNode, NodeParams, NodeSet, ReferenceStep};
use ym_color::hue::{ELEMENT_COUNT, ElementGrade, ElementStep, HueStep};
use ym_color::mask::{PhotoFrame, PhotoSpot};
use ym_color::mask_element::{ElementPlanes, ElementRow, ElementScene};
use ym_color::plan::{MAIN_NODE, PictureSource, PlanGraph, PlanNode, Stop, compile};
use ym_color::tone::ToneParams;

/// The plane's one: a u16 code of 65535 is a mask of 1.
const ONE: f64 = 65535.0;

fn matrix() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn node(id: u32, from: u32) -> PlanNode {
    PlanNode {
        id,
        pos: [f64::from(id), 0.0],
        picture: Some(PictureSource::Node(from)),
        key: None,
    }
}

fn scene() -> Arc<ElementScene> {
    Arc::new(ElementScene {
        key: 1,
        width: 1,
        height: 1,
        bits: vec![0],
        classes: Vec::new(),
    })
}

/// Main, then Reference Grade node 2 (a live sky grade) and a second one, 3, in a line; the plan ends at the mask of `stop`.
fn chain(stop: Stop, view: Option<(u32, usize)>, live: [bool; ELEMENT_COUNT]) -> ColorChain {
    let plan = compile(
        &PlanGraph {
            nodes: vec![
                PlanNode {
                    id: MAIN_NODE,
                    pos: [1.0, 0.0],
                    picture: Some(PictureSource::Input),
                    key: None,
                },
                node(2, MAIN_NODE),
                node(3, 2),
            ],
            output: PictureSource::Node(3),
        },
        stop,
    )
    .expect("plan");
    let mut grades = [ElementGrade::IDENTITY; ELEMENT_COUNT];
    grades[0] = ElementGrade {
        exposure_ev: 0.5,
        hue_deg: 0.0,
        chroma: 1.0,
    };
    let reference = NodeParams {
        reference: Some(ReferenceStep {
            hue: HueStep::default(),
            elements: ElementStep::new(grades),
        }),
        ..NodeParams::default()
    };
    let mut others = BTreeMap::new();
    others.insert(2, reference.clone());
    others.insert(3, reference);
    let mut p = ChainParams::new(matrix(), ToneParams::IDENTITY);
    p.nodes = NodeSet {
        plan,
        main: MainNode::default(),
        others,
    };
    p.elements = vec![ElementRow {
        node: 2,
        live,
        scene: scene(),
    }];
    p.element_view = view;
    ColorChain::new(&p).expect("chain")
}

/// Planes of node 2: element `e` at the code `code`, the others at another code, over one pixel.
fn planes(e: usize, code: u16) -> ElementPlanes {
    let mut codes: [Vec<u16>; ELEMENT_COUNT] = std::array::from_fn(|_| vec![9000]);
    codes[e] = vec![code];
    ElementPlanes {
        width: 1,
        height: 1,
        nodes: BTreeMap::from([(2, codes)]),
    }
}

fn value(chain: &ColorChain, planes: Option<&ElementPlanes>) -> [f64; 3] {
    let frame = PhotoFrame::new(1.0, 1.0, 1.0);
    let spot = planes.map(|p| PhotoSpot::new([0.5, 0.5], frame).with_raster(p, 0, 0));
    chain.apply_pixel_at([0.18; 3], spot)
}

#[test]
fn the_view_is_the_element_plane_in_all_three_channels() {
    let mut live = [false; ELEMENT_COUNT];
    for e in 0..ELEMENT_COUNT {
        live[e] = true;
        let c = chain(Stop::Mask(2), Some((2, e)), live);
        live[e] = false;
        for code in [0u16, 1, 20000, 40000, 65535] {
            let got = value(&c, Some(&planes(e, code)));
            let want = f64::from(code) / ONE;
            assert_eq!(got, [want; 3], "element {e} at code {code}");
        }
    }
}

#[test]
fn a_pixel_with_no_plane_is_zero_and_the_other_elements_do_not_show() {
    let mut live = [false; ELEMENT_COUNT];
    live[2] = true;
    let c = chain(Stop::Mask(2), Some((2, 2)), live);
    assert_eq!(value(&c, None), [0.0; 3], "no planes, no mask");
    let other = ElementPlanes {
        width: 1,
        height: 1,
        nodes: BTreeMap::from([(
            2,
            std::array::from_fn(|e| if e == 2 { Vec::new() } else { vec![65535] }),
        )]),
    };
    assert_eq!(
        value(&c, Some(&other)),
        [0.0; 3],
        "an empty plane of the element is 0 whatever the others hold"
    );
}

#[test]
fn another_node_or_no_view_shows_the_nodes_own_mask() {
    let mut live = [false; ELEMENT_COUNT];
    live[0] = true;
    // No view: a node with no mask rows has m = 1.
    let none = chain(Stop::Mask(2), None, live);
    assert_eq!(value(&none, Some(&planes(0, 20000))), [1.0; 3]);
    // The view names node 2, the plan ends at node 3's mask: node 3's own m.
    let elsewhere = chain(Stop::Mask(3), Some((2, 0)), live);
    assert_eq!(value(&elsewhere, Some(&planes(0, 20000))), [1.0; 3]);
}
