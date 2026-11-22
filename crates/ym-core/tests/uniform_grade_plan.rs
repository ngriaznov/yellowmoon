//! A Colour Uniform node with a grade is two steps of the plan (`docs/32` section 12a.4, variant W2): the pull with the node's masks,
//! then a corrector's steps that read its picture and its mask as their key. The cards stay one per node, with their numbers.

use serde_json::json;
use ym_color::plan::Stop;
use ym_core::pipeline::color::{GRADE_TAIL, is_grade_tail, node_plan};
use ym_core::settings::DevelopSettings;
use ym_looks::library::LookId;

fn settings(grade: bool) -> DevelopSettings {
    let mut uniform = json!({ "id": 2, "kind": "color_uniform", "pos": [1.0, 0.0],
        "target": { "hue_deg": 26, "saturation": 9.7, "lightness_ev": 0 },
        "compress": { "hue": 60, "saturation": 60, "lightness": 0 } });
    if grade {
        uniform["trim"] = json!({ "exposure_ev": 0.5 });
    }
    DevelopSettings::from_value(
        json!({ "nodes": { "items": [ uniform, { "id": 3, "kind": "corrector", "pos": [2.0, 0.0] } ],
            "links": [
              { "from": { "node": "input", "port": "picture" }, "to": { "node": 1, "port": "picture" } },
              { "from": { "node": 1, "port": "picture" }, "to": { "node": 2, "port": "picture" } },
              { "from": { "node": 2, "port": "picture" }, "to": { "node": 3, "port": "picture" } },
              { "from": { "node": 3, "port": "picture" }, "to": { "node": "output", "port": "picture" } } ] } }),
        &LookId::neutral(),
    )
    .expect("settings")
    .0
}

#[test]
fn a_graded_node_is_two_steps_and_one_card() {
    let plain = node_plan(&settings(false).nodes, Stop::Output).expect("plan");
    let graded = node_plan(&settings(true).nodes, Stop::Output).expect("plan");
    let ids = |p: &ym_color::plan::Plan| p.node_ids();
    assert!(!ids(&plain).iter().any(|&i| is_grade_tail(i)));
    assert_eq!(ids(&graded).len(), ids(&plain).len() + 1);
    assert!(ids(&graded).contains(&(2 | GRADE_TAIL)));
    // The cards and their numbers are the same: the grade step is no card.
    assert_eq!(graded.cards, plain.cards);
    // The next node reads the grade step's picture, and the grade step reads the pull's picture and mask.
    let steps: Vec<_> = graded.passes.iter().flat_map(|p| p.steps.iter()).collect();
    let tail = steps
        .iter()
        .find(|s| s.node == (2 | GRADE_TAIL))
        .expect("the grade step");
    assert!(tail.key.is_some(), "the grade step reads the pull's mask");
    let pull = steps.iter().find(|s| s.node == 2).expect("the pull");
    assert!(pull.key_out.is_some(), "the pull gives its mask");
}

#[test]
fn a_thumbnail_stops_after_the_grade_step_and_the_mask_view_at_the_pull() {
    let s = settings(true);
    let picture = node_plan(&s.nodes, Stop::Picture(2)).expect("plan");
    assert!(picture.node_ids().contains(&(2 | GRADE_TAIL)));
    let mask = node_plan(&s.nodes, Stop::Mask(2)).expect("plan");
    assert!(!mask.node_ids().iter().any(|&i| is_grade_tail(i)));
}
