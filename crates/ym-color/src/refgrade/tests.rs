//! Tests of the Reference Grade fit: the network (the proof of concept's recorded check vectors, every refusal), the measures, the
//! objective's cases and the search's properties on a synthetic renderer.

use std::collections::BTreeMap;

use sha2::{Digest, Sha256};

use super::measure::{
    LabConv, PixelScene, Px, class, measure, network_inputs, oklab_hue, sector_exclusions, wrap_deg,
};
use super::network::{NETWORK_TEXT, Network, NetworkErrorKind, NetworkValues, input_names};
use super::objective::{
    Coverage, Fixed, Terms, allowed_turn_deg, candidate_measure, clip_allowance, clip_over,
    clip_term, content_turn_share, h4_inner_limit, is_white, licence_allowed,
    skin_chroma_down_limit, skin_chroma_up_limit, skin_hue_limit_deg, terms, turn_weight,
};
use super::params::{N_PARAMS, Params, to_numbers};
use super::search::{CandidateRenderer, FitError, FitSetup, Refused, Start, fit};
use super::{SECTORS, sample_pixels};

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn shipped() -> Network {
    Network::shipped().expect("the shipped network file parses")
}

// ---------------------------------------------------------------------------------------------
// The network
// ---------------------------------------------------------------------------------------------

#[test]
fn the_shipped_file_is_round_15s_and_expands_to_432_neurons() {
    // The file's hash is pinned: a weight changes only by a spec change with the owner's agreement (docs/34 section 4b.5).
    assert_eq!(
        sha256_hex(NETWORK_TEXT.as_bytes()),
        "6f52dd30aeafdbf1f9e2fc8e9a84bee5c8afc8c5b2fb10e00cc3f457c0395a5a"
    );
    let net = shipped();
    assert_eq!(net.len(), 432);
    let outputs = net.outputs();
    assert_eq!(outputs.iter().filter(|n| n.starts_with("T.")).count(), 62);
    assert_eq!(outputs.iter().filter(|n| n.starts_with("G.")).count(), 21);
    // The expanded text is pinned too: every weight with its number.
    assert_eq!(
        sha256_hex(net.dump().as_bytes()),
        "faa3da68a20c1d3de8a9fb2df99c1787c94dfe99c8905e3e1959e587bcc7e331",
        "the expanded network changed"
    );
}

fn expect_refusal(text: &str) -> (usize, NetworkErrorKind) {
    let e = Network::parse(text).expect_err("the file must be refused");
    (e.line, e.kind)
}

#[test]
fn every_refusal_of_the_grammar_names_the_line_and_the_name() {
    // A node not defined earlier.
    let (line, kind) = expect_refusal("a = lin( 0 + b )\nb = lin( 1 )");
    assert_eq!(line, 1);
    assert_eq!(kind, NetworkErrorKind::UnknownNode("b".into()));
    // A reference to a name that is neither an input nor a neuron.
    let (_, kind) = expect_refusal("a = lin( 0 + P.nothing )");
    assert_eq!(kind, NetworkErrorKind::UnknownNode("P.nothing".into()));
    // A duplicate name, against an earlier neuron and against an input.
    let (line, kind) = expect_refusal("a = lin( 1 )\n\na = lin( 2 )");
    assert_eq!((line, kind), (3, NetworkErrorKind::DuplicateName));
    let (_, kind) = expect_refusal("P.p2 = lin( 1 )");
    assert_eq!(kind, NetworkErrorKind::DuplicateName);
    // An unknown activation.
    let (_, kind) = expect_refusal("a = tanh( 1 )");
    assert_eq!(kind, NetworkErrorKind::UnknownActivation("tanh".into()));
    // A template without `end`.
    let (line, kind) = expect_refusal("# c\ntemplate t(x)\n{x}.a = lin( 1 )\n");
    assert_eq!((line, kind), (2, NetworkErrorKind::TemplateWithoutEnd));
    // A template value with a missing constant.
    let (line, kind) =
        expect_refusal("const one k=1\ntemplate t(x)\n{x}.a = lin( {k} )\nend\napply t one two");
    assert_eq!(line, 5);
    assert!(
        matches!(kind, NetworkErrorKind::MissingConstant(_)),
        "{kind:?}"
    );
    // An apply of a template that is not there.
    let (_, kind) = expect_refusal("apply nothing a b");
    assert_eq!(kind, NetworkErrorKind::UnknownTemplate("nothing".into()));
    // A term that is not a number times a node.
    let (_, kind) = expect_refusal("a = lin( 0 + x*P.p2 )");
    assert!(matches!(kind, NetworkErrorKind::BadTerm(_)), "{kind:?}");
    let (_, kind) = expect_refusal("a = lin( 0 + 2*P.p2*P.p50 )");
    assert!(matches!(kind, NetworkErrorKind::BadTerm(_)), "{kind:?}");
    let (_, kind) = expect_refusal("a = lin( 0 + 2* )");
    assert!(matches!(kind, NetworkErrorKind::BadTerm(_)), "{kind:?}");
    // A weight that is not finite is not a number either.
    let (_, kind) = expect_refusal("a = lin( 0 + inf*P.p2 )");
    assert!(matches!(kind, NetworkErrorKind::BadTerm(_)), "{kind:?}");
    // A bias that is missing, a sign with nothing after it, two terms with no sign between them, a line with no equals sign.
    let (_, kind) = expect_refusal("a = lin( + P.p2 )");
    assert!(matches!(kind, NetworkErrorKind::BadBias(_)), "{kind:?}");
    let (_, kind) = expect_refusal("a = lin( 1 + )");
    assert_eq!(kind, NetworkErrorKind::SignWithoutTerm);
    let (_, kind) = expect_refusal("a = lin( 1 P.p2 )");
    assert!(
        matches!(kind, NetworkErrorKind::ExpectedSign(_)),
        "{kind:?}"
    );
    let (_, kind) = expect_refusal("a lin( 1 )");
    assert_eq!(kind, NetworkErrorKind::NoEquals);
    // The text of an error says the line and the name.
    let e = Network::parse("# one\nz = lin( 1 + q )").unwrap_err();
    assert_eq!(
        e.to_string(),
        "network line 2, z: refers to q, which is not an input and not defined earlier"
    );
}

#[test]
fn a_small_file_with_constants_templates_and_apply_expands_and_evaluates_in_f64() {
    let text = "
# a comment
const a k=2 m=0.5
const b k=3 m=-1.5
template t(x)
{x}.sum  = relu( {m} + {k}*P.p2 - R.p2 )
{x}.back = lin( -{k} + {x}.sum )
end
apply t a b
out = relu( -1000 + a.back )
";
    let net = Network::parse(text).unwrap();
    assert_eq!(net.len(), 5);
    let mut inputs = vec![0.0; net.inputs().len()];
    let at = |name: &str| net.inputs().iter().position(|n| n == name).unwrap();
    inputs[at("P.p2")] = 10.0;
    inputs[at("R.p2")] = 4.0;
    let v = net.eval(&inputs);
    // a.sum = relu(0.5 + 2*10 - 4) = 16.5 ; a.back = -2 + 16.5 ; b.sum = relu(-1.5 + 30 - 4) = 24.5 ; b.back = -3 + 24.5
    assert_eq!(v.get("a.sum"), 16.5);
    assert_eq!(v.get("a.back"), 14.5);
    assert_eq!(v.get("b.sum"), 24.5);
    assert_eq!(v.get("b.back"), 21.5);
    assert_eq!(v.get("out"), 0.0);
}

/// The proof of concept's input vectors for its network check: a mid-key landscape with sky, vegetation and ground, in the photo and
/// in the reference alike.
fn same_inputs() -> BTreeMap<String, f64> {
    let mut p: BTreeMap<String, f64> = input_names().into_iter().map(|n| (n, 0.0)).collect();
    for x in ["P", "R"] {
        for (k, v) in [
            ("p2", 6.0),
            ("p50", 52.0),
            ("p98", 94.0),
            ("iqr_rel", 0.42),
            ("logsat", 15f64.ln()),
            ("subj", 1.0),
            ("sep", 6.0),
            ("tint_s_a", 1.0),
            ("tint_s_b", 2.0),
            ("tint_m_a", 0.5),
            ("tint_m_b", 1.0),
            ("tint_h_a", -0.5),
            ("tint_h_b", 1.5),
        ] {
            p.insert(format!("{x}.{k}"), v);
        }
        for (e, share, l, h, c) in [
            ("sky", 0.30, 74.0, 255.0, 22.0),
            ("vegetation", 0.20, 42.0, 120.0, 26.0),
            ("ground", 0.10, 45.0, 70.0, 14.0),
        ] {
            p.insert(format!("{x}.{e}.share"), share);
            p.insert(format!("{x}.{e}.L"), l);
            p.insert(format!("{x}.{e}.h"), h);
            p.insert(format!("{x}.{e}.C"), c);
            p.insert(format!("{x}.{e}.logC"), f64::max(c, 1.0).ln());
        }
        for (j, share, logc) in [(0, 0.08, 2.4), (3, 0.07, 2.6)] {
            p.insert(format!("{x}.s{j}.share"), share);
            p.insert(format!("{x}.s{j}.logC"), logc);
        }
    }
    p
}

fn run(net: &Network, m: &BTreeMap<String, f64>) -> NetworkValues {
    let inputs: Vec<f64> = net.inputs().iter().map(|n| m[n]).collect();
    net.eval(&inputs)
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() < 1e-9, "{what}: got {got}, want {want}");
}

/// A one-colour amber reference against a neutral photo: the vector of checks 4, 20 and 22.
fn amber() -> BTreeMap<String, f64> {
    let mut p = same_inputs();
    for (k, v) in [
        ("P.cast_a", 0.0),
        ("P.cast_b", 1.0),
        ("P.conc", 0.1),
        ("R.cast_a", 6.0),
        ("R.cast_b", 28.0),
        ("R.conc", 0.9),
    ] {
        p.insert(k.into(), v);
    }
    p
}

/// The network check of `docs/34` section 10: the proof of concept's 22 recorded checks, on its input vectors, to 1e-9.
#[test]
fn the_network_check_of_the_proof_of_concept_passes_on_its_recorded_vectors() {
    let net = shipped();
    let same = same_inputs();
    let out1 = run(&net, &same);
    // Check 1: R identical to P gives P's own values.
    for k in [
        "p2", "p98", "iqr_rel", "p50", "logsat", "tint_s_a", "tint_s_b", "tint_m_a", "tint_m_b",
        "tint_h_a", "tint_h_b", "sep",
    ] {
        close(
            out1.get(&format!("T.{k}")),
            same[&format!("P.{k}")],
            &format!("check 1, T.{k}"),
        );
    }
    for e in ["sky", "vegetation", "ground"] {
        close(
            out1.get(&format!("T.{e}.h")),
            same[&format!("P.{e}.h")],
            &format!("check 1, T.{e}.h"),
        );
        close(
            out1.get(&format!("T.{e}.logC")),
            same[&format!("P.{e}.logC")],
            &format!("check 1, T.{e}.logC"),
        );
        close(
            out1.get(&format!("T.{e}.off")),
            same[&format!("P.{e}.L")] - same["P.p50"],
            &format!("check 1, T.{e}.off"),
        );
    }
    for j in [0, 3] {
        close(out1.get(&format!("T.s{j}.dh")), 0.0, "check 1, T.s.dh");
        close(
            out1.get(&format!("T.s{j}.dlogC")),
            0.0,
            "check 1, T.s.dlogC",
        );
    }
    // Check 2: a teal sky (255 -> 205), a darker floor (6 -> 0), 1.35x saturation, a brighter sky (74 -> 84).
    let mut r = same.clone();
    r.insert("R.p2".into(), 0.0);
    r.insert("R.logsat".into(), 15f64.ln() + 1.35f64.ln());
    r.insert("R.sky.h".into(), 205.0);
    r.insert("R.sky.L".into(), 84.0);
    r.insert(
        "D.sky.dh".into(),
        ((205.0f64 - 255.0 + 180.0).rem_euclid(360.0)) - 180.0,
    );
    let out2 = run(&net, &r);
    close(out2.get("T.p2"), 6.0 - 4.8, "check 2, T.p2");
    close(out2.get("T.sky.h"), 255.0 - 0.85 * 40.0, "check 2, T.sky.h");
    close(
        out2.get("T.logsat"),
        15f64.ln() + 0.85 * 1.35f64.ln(),
        "check 2, T.logsat",
    );
    close(out2.get("T.sky.off"), 27.0, "check 2, T.sky.off");
    close(out2.get("T.vegetation.h"), 120.0, "check 2, T.vegetation.h");
    close(out2.get("T.p98"), 94.0, "check 2, T.p98");
    // Check 3: a low-key reference (median 20) on the same high-key photo: key mismatch; ramps; the exposure and subject gates shut.
    let mut r3 = same.clone();
    r3.insert("R.p50".into(), 20.0);
    r3.insert("R.p2".into(), 0.0);
    r3.insert("R.sep".into(), 14.0);
    let out3 = run(&net, &r3);
    close(out3.get("G.key_mismatch"), 1.0, "check 3, G.key_mismatch");
    close(out3.get("T.p2"), 3.6, "check 3, T.p2");
    close(out3.get("T.sky.off"), 22.0, "check 3, T.sky.off");
    close(out3.get("T.sep"), 6.0, "check 3, T.sep");
    // Check 4: unify opens, the frame cast moves 0.8 of the (clipped) difference, the wall opens to 45.
    let out4 = run(&net, &amber());
    close(out4.get("G.unify"), 1.0, "check 4, G.unify");
    close(out4.get("T.cast_b"), 13.0, "check 4, T.cast_b");
    close(out4.get("T.cast_a"), 4.8, "check 4, T.cast_a");
    close(out4.get("T.wall_deg"), 45.0, "check 4, T.wall_deg");
    // Check 5: both pictures one-colour amber: nothing to unify.
    let mut p5 = same.clone();
    for x in ["P", "R"] {
        p5.insert(format!("{x}.cast_a"), 6.0);
        p5.insert(format!("{x}.cast_b"), 28.0);
        p5.insert(format!("{x}.conc"), 0.9);
    }
    let out5 = run(&net, &p5);
    close(out5.get("G.unify"), 0.0, "check 5, G.unify");
    close(out5.get("T.wall_deg"), 25.0, "check 5, T.wall_deg");
    // Check 6: the reference's "sky" has median L* 10 (a black backdrop): the dark-sky rule shuts the sky's gate.
    let mut p6 = same.clone();
    p6.insert("R.sky.L".into(), 10.0);
    close(
        run(&net, &p6).get("G.sky.matched"),
        0.0,
        "check 6, G.sky.matched",
    );
    // Checks 7 and 8: the whites' chroma cap is 5 + 0.8 (|a*| + |b*|) of the reference's highlight tint, at most 14.
    let mut p7 = same.clone();
    p7.insert("R.tint_h_a".into(), 2.0);
    p7.insert("R.tint_h_b".into(), 9.0);
    close(run(&net, &p7).get("T.hl_cmax"), 13.8, "check 7, T.hl_cmax");
    let mut p8 = same.clone();
    p8.insert("R.tint_h_a".into(), -20.0);
    p8.insert("R.tint_h_b".into(), -20.0);
    close(
        run(&net, &p8).get("T.hl_cmax"),
        14.0,
        "check 8, T.hl_cmax (capped)",
    );
    close(
        out1.get("T.hl_cmax"),
        5.0 + 0.8 * (0.5 + 1.5),
        "check 8, the reference's own highlight tint",
    );
    // Checks 8 and 9 (v8): purple is allowed where the reference holds a tenth of its frame in magenta, and not where it holds none.
    let mut p9 = same.clone();
    p9.insert("R.magenta_share".into(), 0.10);
    close(
        run(&net, &p9).get("T.purple_ok"),
        1.0,
        "check 8 (v8), T.purple_ok",
    );
    close(out1.get("T.purple_ok"), 0.0, "check 9 (v8), T.purple_ok");
    // Check 10 (v8): a reference with highlight tint (a*, b*) = (-8, 4).
    let mut p10 = same.clone();
    p10.insert("R.tint_h_a".into(), -8.0);
    p10.insert("R.tint_h_b".into(), 4.0);
    let out10 = run(&net, &p10);
    close(out10.get("T.white_a"), -6.4, "check 10, T.white_a");
    close(out10.get("T.white_b"), 3.2, "check 10, T.white_b");
    close(out10.get("T.white_tol"), 6.0, "check 10, T.white_tol");
    close(
        out10.get("T.skin_drift"),
        6.0,
        "check 10, T.skin_drift (no matched skin)",
    );
    // Checks 11 and 12 (v9): the memory colours' limits depend on whether the reference matches them.
    let mut pu = same.clone();
    pu.insert("R.sky.share".into(), 0.0);
    pu.insert("R.vegetation.share".into(), 0.0);
    let out_u = run(&net, &pu);
    close(out_u.get("T.mem.sky_deg"), 10.0, "check 11, T.mem.sky_deg");
    close(
        out_u.get("T.mem.vegetation_deg"),
        12.0,
        "check 11, T.mem.vegetation_deg",
    );
    close(out1.get("T.mem.sky_deg"), 45.0, "check 12, T.mem.sky_deg");
    close(
        out1.get("T.mem.vegetation_deg"),
        40.0,
        "check 12, T.mem.vegetation_deg",
    );
    // Checks 13 and 14 (v10): one pixel in a hundred may turn purple, a quarter where the reference holds magenta.
    close(
        out1.get("T.purple_max_share"),
        0.01,
        "check 13, T.purple_max_share",
    );
    close(
        run(&net, &p9).get("T.purple_max_share"),
        0.25,
        "check 14, T.purple_max_share",
    );
    // Checks 15 and 16 (v11): a turn of 10 degrees into the purple band, 45 where the reference holds purple.
    close(
        out1.get("T.purple_rot_deg"),
        10.0,
        "check 15, T.purple_rot_deg",
    );
    close(
        run(&net, &p9).get("T.purple_rot_deg"),
        45.0,
        "check 16, T.purple_rot_deg",
    );
    // Checks 17 and 18 (v11): skin chroma may rise 0.12 (log), 0.30 where the reference's skin is matched.
    let mut p17 = same.clone();
    p17.insert("P.skin.share".into(), 0.30);
    p17.insert("R.skin.share".into(), 0.30);
    close(
        run(&net, &p17).get("T.skin_dlogC_max"),
        0.30,
        "check 17, T.skin_dlogC_max",
    );
    let mut p18 = same.clone();
    p18.insert("P.skin.share".into(), 0.30);
    p18.insert("R.skin.share".into(), 0.0);
    close(
        run(&net, &p18).get("T.skin_dlogC_max"),
        0.12,
        "check 18, T.skin_dlogC_max",
    );
    // Checks 19 and 20 (v12): a strong colour may turn 20 degrees, 30 when unify is open; 3 % of the frame may turn further; a
    // reference may license 2 points of clipping.
    close(out1.get("T.turn_deg"), 20.0, "check 19, T.turn_deg");
    close(out1.get("T.turn_share"), 0.03, "check 19, T.turn_share");
    close(out1.get("T.clip_extra"), 0.02, "check 19, T.clip_extra");
    let open = run(&net, &amber());
    close(open.get("G.unify"), 1.0, "check 20, G.unify");
    close(open.get("T.turn_deg"), 30.0, "check 20, T.turn_deg");
    // Checks 21 and 22 (v13, v14): the weak-colour turn budget and the turn that counts in full, unify shut and open.
    for (out, unify) in [(&out1, 0.0), (&open, 1.0)] {
        close(out.get("G.unify"), unify, "check 21, G.unify");
        close(out.get("T.turn_cref"), 25.0, "check 21, T.turn_cref");
        close(out.get("T.turn_cap"), 45.0, "check 21, T.turn_cap");
        close(out.get("T.turn_full"), 60.0, "check 22, T.turn_full");
    }
}

// ---------------------------------------------------------------------------------------------
// The objective's cases
// ---------------------------------------------------------------------------------------------

fn px_of(lab: [f64; 3], ok: [f64; 2]) -> Px {
    Px::new(lab, ok, false, lab[0] < 2.5)
}

/// A pixel at CIELAB chroma `c` and an Oklab hue `h_deg` (the Oklab vector has unit length; only its angle matters).
fn colour(c: f64, h_deg: f64) -> Px {
    let (s, co) = h_deg.to_radians().sin_cos();
    px_of([50.0, c * co, c * s], [co, s])
}

#[test]
fn the_allowed_turn_and_the_turn_weight_follow_section_5_4() {
    // allowed(C) by chroma, unify shut (turn 20) and open (turn 30): 45 at C* 8 to 12, then 35, 25 and 20 (shut) or 45, 39 and 30
    // (open) at 15, 20 and 25 and above; C* 0 gives the cap.
    let shut = |c| allowed_turn_deg(c, 20.0, 25.0, 45.0);
    let open = |c| allowed_turn_deg(c, 30.0, 25.0, 45.0);
    for c in [8.0, 10.0, 12.0] {
        assert!(
            (shut(c) - 45.0).abs() < 1e-9 || shut(c) > 40.0,
            "C* {c}: {}",
            shut(c)
        );
    }
    assert!((shut(8.0) - 45.0).abs() < 1e-9);
    assert!(
        (shut(12.0) - 45.0).abs() < 0.5,
        "C* 12 shut: {}",
        shut(12.0)
    );
    assert!(
        (shut(15.0) - 35.0).abs() < 0.5,
        "C* 15 shut: {}",
        shut(15.0)
    );
    assert!(
        (shut(20.0) - 25.0).abs() < 0.5,
        "C* 20 shut: {}",
        shut(20.0)
    );
    assert!((shut(25.0) - 20.0).abs() < 1e-9 && (shut(60.0) - 20.0).abs() < 1e-9);
    assert!(
        (open(15.0) - 45.0).abs() < 1e-9,
        "C* 15 open: {}",
        open(15.0)
    );
    assert!(
        (open(20.0) - 39.0).abs() < 0.5,
        "C* 20 open: {}",
        open(20.0)
    );
    assert!((open(25.0) - 30.0).abs() < 1e-9 && (open(60.0) - 30.0).abs() < 1e-9);
    assert_eq!(shut(0.0), 45.0);
    // The weight ramps from the allowed turn to a whole pixel at T.turn_full (60): 30 degrees where 20 are allowed is a quarter.
    assert_eq!(turn_weight(20.0, 20.0, 60.0), 0.0);
    assert!((turn_weight(30.0, 20.0, 60.0) - 0.25).abs() < 1e-12);
    assert!((turn_weight(40.0, 20.0, 60.0) - 0.5).abs() < 1e-12);
    assert_eq!(turn_weight(60.0, 20.0, 60.0), 1.0);
    assert_eq!(turn_weight(170.0, 20.0, 60.0), 1.0);
    // A step when the full turn is not past the allowed one.
    assert_eq!(turn_weight(46.0, 45.0, 40.0), 1.0);
    assert_eq!(turn_weight(44.0, 45.0, 40.0), 0.0);
}

/// The share of a sample of one colour turned by `turn` degrees, over `total` pixels of which `n` are that colour.
fn turned_share(c_star: f64, turn: f64, n: usize, total: usize) -> f64 {
    let p: Vec<Px> = (0..total)
        .map(|i| {
            if i < n {
                colour(c_star, 60.0 + turn)
            } else {
                colour(1.0, 0.0)
            }
        })
        .collect();
    let p_cstar: Vec<f64> = (0..total)
        .map(|i| if i < n { c_star } else { 1.0 })
        .collect();
    let p_okh: Vec<f64> = (0..total).map(|i| if i < n { 60.0 } else { 0.0 }).collect();
    // Hue 60 in the photo: the candidate's Oklab hue is 60 + turn.
    content_turn_share(
        &p,
        &p_cstar,
        &p_okh,
        &vec![true; total],
        (20.0, 25.0, 45.0, 60.0),
    )
}

#[test]
fn a_pale_pixel_turned_35_does_not_count_and_a_strong_one_turned_25_does() {
    // A pale wall pixel (C* 12, allowed 45) turned 35: no weight at all.
    assert_eq!(turned_share(12.0, 35.0, 100, 100), 0.0);
    // A strong colour (C* 40, allowed 20) turned 25: counts a little: (25 - 20) / (60 - 20) of a pixel.
    let s = turned_share(40.0, 25.0, 100, 100);
    assert!((s - 5.0 / 40.0).abs() < 1e-9, "{s}");
    // 30 degrees is a quarter, 60 a whole pixel; the share is over every pixel of the sample.
    assert!((turned_share(40.0, 30.0, 10, 100) - 0.025).abs() < 1e-9);
    assert!((turned_share(40.0, 60.0, 4, 100) - 0.04).abs() < 1e-9);
    // Under C* 8 a pixel is never counted, and a skin pixel never is.
    assert_eq!(turned_share(7.0, 120.0, 100, 100), 0.0);
}

#[test]
fn a_turn_budget_costs_nothing_for_10_percent_turned_30_and_much_for_4_percent_reversed() {
    let cost = |s: f64, limit: f64| {
        if s > limit {
            6.0 * ((s - limit) / 0.01f64).powi(2)
        } else {
            0.0
        }
    };
    // 10 % of strong colours turned 30 degrees: s = 10 * 0.25 / 100 = 0.025, under the 3 % budget.
    assert_eq!(cost(turned_share(40.0, 30.0, 10, 100), 0.03), 0.0);
    // 4 % that reverse count whole: s = 0.04, 6 ((0.04 - 0.03) / 0.01)^2 = 6.
    assert!((cost(turned_share(40.0, 100.0, 4, 100), 0.03) - 6.0).abs() < 1e-9);
}

#[test]
fn the_clip_allowance_and_term_follow_the_formula() {
    // allow(p, r) = max(p, min(r, p + extra)): the reference licenses at most two points, never more than its own share.
    assert_eq!(clip_allowance(0.01, 0.50, 0.02), 0.03);
    assert_eq!(clip_allowance(0.01, 0.02, 0.02), 0.02);
    assert_eq!(clip_allowance(0.10, 0.02, 0.02), 0.10);
    // over(o, a) = ((o - a - 0.005)+ / 0.01)^2.
    assert_eq!(clip_over(0.03, 0.03), 0.0);
    assert_eq!(clip_over(0.034, 0.03), 0.0);
    assert!((clip_over(0.045, 0.03) - 1.0).abs() < 1e-9);
    assert!((clip_over(0.065, 0.03) - 9.0).abs() < 1e-9);
}

/// A photo and a reference of a few hundred pixels each, with a scene, for the terms.
struct Pair {
    p_px: Vec<Px>,
    r_px: Vec<Px>,
    scene: PixelScene,
    r_scene: PixelScene,
}

fn conv(srgb: [f64; 3]) -> Px {
    LabConv::new().px_of_codes(srgb)
}

/// A frame of sky above, vegetation below, with skin and whites among them.
fn pair() -> Pair {
    let n = 1024usize;
    let mut p_px = Vec::new();
    let mut classes = Vec::new();
    let mut skin = Vec::new();
    for i in 0..n {
        let u = i as f64 / n as f64;
        let (px, c, s) = if u < 0.30 {
            (
                conv([0.35 + 0.1 * u, 0.55, 0.85 - 0.1 * u]),
                class::SKY,
                false,
            )
        } else if u < 0.55 {
            (conv([0.25, 0.45 + 0.1 * u, 0.20]), class::VEGETATION, false)
        } else if u < 0.65 {
            (conv([0.80, 0.55, 0.45 + 0.05 * u]), class::PERSON, true)
        } else if u < 0.80 {
            (
                conv([0.92, 0.93, 0.92 - 0.05 * u]),
                class::ARCHITECTURE,
                false,
            )
        } else {
            (conv([0.45 + 0.2 * u, 0.40, 0.30]), class::GROUND, false)
        };
        p_px.push(px);
        classes.push(c);
        skin.push(s);
    }
    let scene = PixelScene {
        class: classes.clone(),
        skin,
        subject: vec![false; n],
    };
    // The reference: the same layout, warmer and darker.
    let r_px: Vec<Px> = p_px
        .iter()
        .map(|q| {
            let lab = [(q.lab[0] * 0.9).max(0.0), q.lab[1] + 4.0, q.lab[2] + 6.0];
            px_of(lab, [q.ok[0] + 0.01, q.ok[1] + 0.02])
        })
        .collect();
    let r_scene = scene.clone();
    Pair {
        p_px,
        r_px,
        scene,
        r_scene,
    }
}

fn fixed_for(pr: &Pair, skin_protection: bool) -> Fixed {
    Fixed::new(
        &shipped(),
        &pr.p_px,
        &pr.scene,
        &pr.r_px,
        &pr.r_scene,
        skin_protection,
    )
    .expect("the fixed part")
}

fn with_regularisers(mut t: Terms, f: &Fixed, c: &super::objective::CandidateMeasure) -> Terms {
    t.clip = clip_term(f, &c.m);
    t
}

#[test]
fn the_photo_against_itself_has_no_look_term_and_no_wall() {
    let pr = pair();
    let mut same = pr.r_px.clone();
    same.clone_from(&pr.p_px);
    let f = Fixed::new(&shipped(), &pr.p_px, &pr.scene, &same, &pr.scene, true).unwrap();
    let c = candidate_measure(&pr.p_px, &pr.scene, &f);
    let t = with_regularisers(terms(&c, &f), &f, &c);
    assert_eq!(t.total(), 0.0, "{:?}", t.named());
    assert_eq!(t.look_only_total(), 0.0);
}

#[test]
fn the_look_only_total_is_the_full_total_minus_the_walls_and_the_clip_term() {
    let pr = pair();
    let f = fixed_for(&pr, true);
    // Candidates: the reference itself, a hard push toward purple and a clipped picture, so the walls and the clip term all fire.
    let mut pushed = pr.p_px.clone();
    for q in &mut pushed {
        let lab = [q.lab[0], q.lab[1] + 20.0, q.lab[2] - 40.0];
        *q = px_of(lab, [-0.3, -0.5]);
    }
    let mut clipped = pr.p_px.clone();
    for q in clipped.iter_mut().take(300) {
        q.hi = true;
        q.lab[0] = 2.0;
        q.lo25 = true;
    }
    let mut saw_wall = false;
    for cand in [&pr.r_px, &pushed, &clipped] {
        let c = candidate_measure(cand, &pr.scene, &f);
        let mut t = terms(&c, &f);
        t.clip = clip_term(&f, &c.m);
        t.reg_global = 0.1;
        t.reg_local = 0.05;
        let walls = t.walls_and_clip();
        saw_wall |= walls > 0.0;
        let want = t.total() - walls;
        assert!(
            (t.look_only_total() - want).abs() < 1e-9 * (1.0 + want.abs()),
            "look-only {} against {}",
            t.look_only_total(),
            want
        );
        assert!(t.look_only_total() <= t.total() + 1e-12);
    }
    assert!(
        saw_wall,
        "a wall or the clip term fired in at least one case"
    );
    // A refused candidate keeps its full total, so it stays worse than anything.
    let refused = Terms {
        clip: super::objective::REFUSED_CLIP,
        ..Terms::default()
    };
    assert_eq!(refused.look_only_total(), refused.total());
}

#[test]
fn skin_protection_switches_the_two_skin_walls_and_nothing_else() {
    let pr = pair();
    let on = fixed_for(&pr, true);
    let off = fixed_for(&pr, false);
    // The skin turned toward green and made much more chromatic.
    let mut cand = pr.p_px.clone();
    for (i, q) in cand.iter_mut().enumerate() {
        if pr.scene.skin[i] {
            *q = px_of([q.lab[0], -10.0, 60.0], [-0.1, 0.3]);
        }
    }
    let c_on = candidate_measure(&cand, &pr.scene, &on);
    let c_off = candidate_measure(&cand, &pr.scene, &off);
    let (t_on, t_off) = (terms(&c_on, &on), terms(&c_off, &off));
    assert!(
        t_on.guard_skin_drift > 0.0 && t_on.guard_skin_chroma > 0.0,
        "{:?}",
        t_on.named()
    );
    assert_eq!(
        (t_off.guard_skin_drift, t_off.guard_skin_chroma),
        (0.0, 0.0)
    );
    // Every other term is the same.
    let rest = |t: &Terms| t.total() - t.guard_skin_drift - t.guard_skin_chroma;
    assert!((rest(&t_on) - rest(&t_off)).abs() < 1e-12);
}

#[test]
fn opposite_tinted_whites_do_not_cancel_in_the_chroma_cap() {
    let pr = pair();
    let f = fixed_for(&pr, true);
    // Whites: the architecture pixels (L* of about 90). Half of them tinted one way, half the other: their mean (a*, b*) is the
    // neutral, but the cap is on the mean per-pixel chroma.
    let mut cand = pr.p_px.clone();
    let mut flip = 1.0;
    for (i, q) in cand.iter_mut().enumerate() {
        if pr.scene.class[i] == class::ARCHITECTURE {
            flip = -flip;
            *q = px_of([q.lab[0], 12.0 * flip, 12.0 * flip], q.ok);
        }
    }
    let c = candidate_measure(&cand, &pr.scene, &f);
    let t = terms(&c, &f);
    assert!(t.guard_whites > 0.0, "{:?}", t.named());
}

#[test]
fn a_purple_turn_costs_and_staying_put_does_not() {
    let pr = pair();
    let f = fixed_for(&pr, true);
    // The vegetation turned into the purple band (Oklab hue 300 to 355) by far more than the 10 degrees the reference allows.
    let mut cand = pr.p_px.clone();
    for (i, q) in cand.iter_mut().enumerate() {
        if pr.scene.class[i] == class::VEGETATION {
            let (s, co) = 330f64.to_radians().sin_cos();
            *q = px_of([q.lab[0], 40.0 * co, 40.0 * s], [co, s]);
        }
    }
    let c = candidate_measure(&cand, &pr.scene, &f);
    let t = terms(&c, &f);
    assert!(
        t.guard_purple > 0.0 && t.guard_purple_pixels > 0.0,
        "{:?}",
        t.named()
    );
    assert!(t.guard_hue > 0.0);
    let calm = terms(&candidate_measure(&pr.p_px, &pr.scene, &f), &f);
    assert_eq!(
        (calm.guard_purple, calm.guard_purple_pixels, calm.guard_hue),
        (0.0, 0.0, 0.0)
    );
}

#[test]
fn measures_follow_their_definitions_on_a_picture_made_by_hand() {
    // 100 pixels, L* 0 to 99 in order: p2 at rank 0.02 * 99 = 1.98, p50 at 49.5, p98 at 97.02.
    let px: Vec<Px> = (0..100)
        .map(|i| px_of([i as f64, 0.0, 0.0], [0.0, 0.0]))
        .collect();
    let scene = PixelScene::blank(100);
    let m = measure(&px, &scene, &[false; 6]);
    assert!(
        (m.p2 - 1.98).abs() < 1e-9 && (m.p50 - 49.5).abs() < 1e-9 && (m.p98 - 97.02).abs() < 1e-9
    );
    let (p75, p25) = (74.25, 24.75);
    assert!((m.iqr_rel - (p75 - p25) / (97.02 - 1.98)).abs() < 1e-9);
    // No colour at all: the overall chroma is ln(max(0, 1)) = 0, the cast 0, conc 0.
    assert_eq!((m.logsat, m.cast_a, m.cast_b, m.conc), (0.0, 0.0, 0.0, 0.0));
    // No subject, no element.
    assert!(!m.subj && m.sep == 0.0 && m.el.iter().all(|e| e.share == 0.0));
}

#[test]
fn the_sector_exclusion_needs_both_pictures_at_two_percent() {
    let pr = pair();
    let p = measure(&pr.p_px, &pr.scene, &[false; 6]);
    let r = measure(&pr.r_px, &pr.r_scene, &[false; 6]);
    let excl = sector_exclusions(&p, &r);
    assert_eq!(
        excl,
        [true, true, false, true, true, false],
        "sky, vegetation, ground and built are in both; skin is never excluded"
    );
    let none = sector_exclusions(
        &p,
        &measure(&pr.r_px, &PixelScene::blank(pr.r_px.len()), &[false; 6]),
    );
    assert_eq!(none, [false; 6]);
    let inputs = network_inputs(&p, &r);
    assert_eq!(inputs.len(), input_names().len());
    assert!(wrap_deg(190.0) == -170.0 && oklab_hue([0.0, -1.0]) == 270.0 && SECTORS == 6);
}

// ---------------------------------------------------------------------------------------------
// The sample and the parameters
// ---------------------------------------------------------------------------------------------

#[test]
fn the_sample_is_8192_pixels_one_per_cell_seeded_and_the_same_every_time() {
    let s = sample_pixels(512, 341);
    assert_eq!(s.len(), 8192);
    assert_eq!(s, sample_pixels(512, 341));
    // One pixel in each cell of the 128 x 64 grid, inside the picture.
    for (n, &(x, y)) in s.iter().enumerate() {
        let (gx, gy) = (n % 128, n / 128);
        assert!(x < 512 && y < 341);
        // The pixel lies in its cell, to the pixel (a cell is not a whole number of pixels).
        let (x0, x1) = (gx as f64 / 128.0 * 512.0, (gx + 1) as f64 / 128.0 * 512.0);
        let (y0, y1) = (gy as f64 / 64.0 * 341.0, (gy + 1) as f64 / 64.0 * 341.0);
        assert!(
            x as f64 >= x0.floor() && x as f64 <= x1.floor(),
            "cell {n}: x {x}"
        );
        assert!(
            y as f64 >= y0.floor() && y as f64 <= y1.floor(),
            "cell {n}: y {y}"
        );
    }
    // The first pixel is the one SplitMix64 seeded with 0x5EED_2026_1124 draws.
    let mut rng = super::SplitMix64(super::SAMPLE_SEED);
    let (u, v) = (rng.unit(), rng.unit());
    assert_eq!(
        s[0],
        (
            ((u / 128.0) * 512.0) as usize,
            ((v / 64.0) * 341.0) as usize
        )
    );
    // Another picture size gives another sample.
    assert_ne!(sample_pixels(512, 300), s);
}

#[test]
fn the_identity_numbers_are_the_identity_and_a_curve_is_increasing_by_construction() {
    let zero = [0.0; N_PARAMS];
    let n = to_numbers(&zero);
    assert!(n.is_identity());
    assert_eq!(n.hue_vs_sat, [1.0; 6]);
    // The extremes of every number still give monotone knots with the end points added.
    for corner in [-1.0, 1.0] {
        let x: Params = [corner; N_PARAMS];
        let n = to_numbers(&x);
        let curves = std::iter::once(&n.luma).chain(n.rgb.iter());
        for c in curves {
            let c = c.as_ref().expect("a moved curve has knots");
            assert_eq!(c.first(), Some(&[0.0, 0.0]));
            assert_eq!(c.last(), Some(&[1.0, 1.0]));
            for w in c.windows(2) {
                assert!(w[1][0] > w[0][0] && w[1][1] >= w[0][1], "{c:?}");
            }
            assert!(
                c.iter()
                    .all(|p| (0.0..=1.0).contains(&p[0]) && (0.0..=1.0).contains(&p[1]))
            );
        }
        assert!(n.elements.iter().all(|e| (0.5..=1.6).contains(&e.chroma)));
        assert_eq!(n.luma.as_ref().map(Vec::len), Some(10));
    }
    // A hue number of 1 turns by 30 degrees; an element's hue by its cap.
    let mut x = [0.0; N_PARAMS];
    x[24] = 1.0;
    x[31] = 1.0;
    let n = to_numbers(&x);
    assert_eq!(n.hue_vs_hue_deg[0], 30.0);
    assert_eq!(n.elements[0].hue_deg, 40.0);
}

// ---------------------------------------------------------------------------------------------
// The search
// ---------------------------------------------------------------------------------------------

/// A renderer with no chain behind it: the sample's pixels are a fixed picture; the candidate lifts the lightness by the first
/// number, scales the chroma by the saturation number, and turns the hue by the first `hue_vs_hue` number.
struct Toy {
    base: Vec<Px>,
    calls: usize,
    refuse_all: bool,
}

impl CandidateRenderer for Toy {
    fn sample_len(&self) -> usize {
        self.base.len()
    }
    fn render(&mut self, x: &Params, out: &mut [Px]) -> Result<(), Refused> {
        self.calls += 1;
        if self.refuse_all {
            return Err(Refused);
        }
        let lift = 6.0 * x[0] + 5.0 * x[1];
        let chroma = 1.0 + 0.7 * x[17];
        let turn = (6.0 * x[24]).to_radians();
        let (s, c) = turn.sin_cos();
        for (o, b) in out.iter_mut().zip(&self.base) {
            let (a, bb) = (b.lab[1], b.lab[2]);
            let lab = [
                (b.lab[0] + lift).clamp(0.0, 100.0),
                chroma * (a * c - bb * s),
                chroma * (a * s + bb * c),
            ];
            *o = px_of(lab, b.ok);
        }
        Ok(())
    }
}

fn setup_pair() -> (Toy, Fixed, PixelScene) {
    let pr = pair();
    // The photo is the base picture; the reference is brighter and more colourful, so there is something to find.
    let r_px: Vec<Px> = pr
        .p_px
        .iter()
        .map(|q| {
            px_of(
                [(q.lab[0] + 9.0).min(100.0), q.lab[1] * 1.3, q.lab[2] * 1.3],
                q.ok,
            )
        })
        .collect();
    let f = Fixed::new(&shipped(), &pr.p_px, &pr.scene, &r_px, &pr.scene, true).unwrap();
    (
        Toy {
            base: pr.p_px.clone(),
            calls: 0,
            refuse_all: false,
        },
        f,
        pr.scene,
    )
}

fn run_fit(
    toy: &mut Toy,
    f: &Fixed,
    scene: &PixelScene,
) -> Result<super::search::FitResult, FitError> {
    let setup = FitSetup {
        fixed: f,
        scene,
        mask_ok: [true; 7],
    };
    fit(&setup, toy, &|| false, &mut |_| {})
}

#[test]
fn the_search_is_no_worse_than_the_identity_or_the_neutral_search_and_keeps_its_numbers_in_range() {
    let (mut toy, f, scene) = setup_pair();
    let r = run_fit(&mut toy, &f, &scene).expect("a fit");
    assert!(
        r.total <= r.identity_total,
        "{} against the identity {}",
        r.total,
        r.identity_total
    );
    assert!(r.total <= r.neutral_total && r.total <= r.look_total.max(r.neutral_total));
    assert_eq!(r.total, r.neutral_total.min(r.look_total));
    assert!(r.x.iter().all(|v| (-1.0..=1.0).contains(v)), "{:?}", r.x);
    assert!(
        !r.identity,
        "the reference is brighter and more colourful: the fit moves"
    );
    assert!(r.total < r.identity_total);
    assert_eq!(r.searches.len(), 3);
    // Every golden pass of a search only lowers its objective.
    for s in &r.searches {
        for l in &s.log {
            assert!(l.after <= l.before + 1e-12, "{l:?}");
        }
    }
    // A tie keeps the neutral search.
    if r.look_total >= r.neutral_total {
        assert_eq!(r.start, Start::Neutral);
    }
    // The neutral search moved lightness up and chroma up (the toy's numbers 0, 1 and 17).
    let neutral = &r.searches[0];
    assert!(neutral.x[0] > 0.0 || neutral.x[1] > 0.0);
    assert!(neutral.x[17] > 0.0);
}

#[test]
fn an_element_that_fails_the_need_test_keeps_its_numbers_at_exactly_zero() {
    let (mut toy, f, scene) = setup_pair();
    let r = run_fit(&mut toy, &f, &scene).unwrap();
    let neutral = &r.searches[0];
    // The toy chain cannot move an element on its own, so an element that failed the test has all three numbers at 0.
    for k in 0..6 {
        if !neutral.need.local[k] {
            assert_eq!(
                neutral.x[30 + 3 * k..33 + 3 * k],
                [0.0, 0.0, 0.0],
                "element {k}"
            );
        }
    }
    if !neutral.need.local[6] {
        assert_eq!(neutral.x[48], 0.0);
    }
    // The toy chain ignores the local numbers, so no candidate improves on 0 by moving them: a tie keeps the start.
    assert!(neutral.x[30..].iter().all(|&v| v == 0.0));
}

#[test]
fn the_same_inputs_give_the_same_bits_twice() {
    let (mut a, f, scene) = setup_pair();
    let (mut b, _, _) = setup_pair();
    let ra = run_fit(&mut a, &f, &scene).unwrap();
    let rb = run_fit(&mut b, &f, &scene).unwrap();
    assert_eq!(ra.x.map(f64::to_bits), rb.x.map(f64::to_bits));
    assert_eq!(ra.total.to_bits(), rb.total.to_bits());
    assert_eq!(ra.evals, rb.evals);
    assert_eq!(a.calls, b.calls);
}

#[test]
fn a_photo_already_like_the_reference_is_the_identity() {
    let pr = pair();
    let f = Fixed::new(&shipped(), &pr.p_px, &pr.scene, &pr.p_px, &pr.scene, true).unwrap();
    let mut toy = Toy {
        base: pr.p_px.clone(),
        calls: 0,
        refuse_all: false,
    };
    let r = run_fit(&mut toy, &f, &pr.scene).unwrap();
    assert!(r.identity, "{:?}", r.x);
    assert_eq!(r.start, Start::Neutral);
    assert_eq!(r.total, r.identity_total);
}

#[test]
fn cancel_stops_the_fit_between_candidates_and_a_refusing_chain_is_a_failure() {
    let (mut toy, f, scene) = setup_pair();
    let setup = FitSetup {
        fixed: &f,
        scene: &scene,
        mask_ok: [true; 7],
    };
    let calls = std::cell::Cell::new(0usize);
    let cancel = || {
        calls.set(calls.get() + 1);
        calls.get() > 500
    };
    let err = fit(&setup, &mut toy, &cancel, &mut |_| {}).unwrap_err();
    assert_eq!(err, FitError::Cancelled);
    // It stopped within a candidate of the signal.
    assert!(
        toy.calls <= 502,
        "{} renders after a cancel at 500",
        toy.calls
    );
    let mut refusing = Toy {
        base: toy.base.clone(),
        calls: 0,
        refuse_all: true,
    };
    assert_eq!(
        fit(&setup, &mut refusing, &|| false, &mut |_| {}).unwrap_err(),
        FitError::ChainRefused
    );
}

#[test]
fn progress_names_the_search_and_the_stage_and_never_goes_back() {
    let (mut toy, f, scene) = setup_pair();
    let setup = FitSetup {
        fixed: &f,
        scene: &scene,
        mask_ok: [true; 7],
    };
    let mut seen: Vec<super::search::FitProgress> = Vec::new();
    fit(&setup, &mut toy, &|| false, &mut |p| seen.push(p)).unwrap();
    assert_eq!(
        seen.first().map(|p| (p.search, p.stage)),
        Some((1, super::search::Stage::Tone))
    );
    assert_eq!(
        seen.last().map(|p| (p.search, p.stage)),
        Some((3, super::search::Stage::Polish))
    );
    assert!(
        seen.windows(2)
            .all(|w| w[0].fraction() <= w[1].fraction() && w[0].evals <= w[1].evals)
    );
    assert!(
        seen.iter()
            .all(|p| p.searches == 3 && (0.0..=1.0).contains(&p.fraction()))
    );
}

#[test]
fn two_searches_side_by_side_give_the_bits_of_one_after_the_other() {
    use super::search::fit_parallel;
    let (mut one, f, scene) = setup_pair();
    let (mut a, _, _) = setup_pair();
    let (mut b, _, _) = setup_pair();
    let setup = FitSetup {
        fixed: &f,
        scene: &scene,
        mask_ok: [true; 7],
    };
    let seq = run_fit(&mut one, &f, &scene).unwrap();
    let mut seen: Vec<super::search::FitProgress> = Vec::new();
    for threads in [1usize, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        seen.clear();
        let par = pool
            .install(|| fit_parallel(&setup, &mut a, &mut b, &|| false, &mut |p| seen.push(p)))
            .unwrap();
        assert_eq!(
            par.x.map(f64::to_bits),
            seq.x.map(f64::to_bits),
            "{threads} threads"
        );
        assert_eq!(par.total.to_bits(), seq.total.to_bits());
        assert_eq!(par.neutral_total.to_bits(), seq.neutral_total.to_bits());
        assert_eq!(par.look_total.to_bits(), seq.look_total.to_bits());
        assert_eq!(par.identity_total.to_bits(), seq.identity_total.to_bits());
        assert_eq!(par.evals, seq.evals);
        assert_eq!(par.start, seq.start);
        assert_eq!(par.need, seq.need);
        for (p, s) in par.searches.iter().zip(&seq.searches) {
            assert_eq!(p.x.map(f64::to_bits), s.x.map(f64::to_bits));
            assert_eq!(p.after.total().to_bits(), s.after.total().to_bits());
            assert_eq!(p.log, s.log);
        }
        // One line for the whole fit: it starts at "1 of 3", ends at "3 of 3" and never goes back.
        assert_eq!(seen.len(), 18);
        assert_eq!(seen.first().map(|p| p.search), Some(1));
        assert_eq!(
            seen.last().map(|p| (p.search, p.stage)),
            Some((3, super::search::Stage::Polish))
        );
        assert!(
            seen.windows(2)
                .all(|w| w[0].fraction() <= w[1].fraction() && w[0].evals <= w[1].evals)
        );
    }
}

#[test]
fn a_cancel_stops_both_searches_and_a_refusing_chain_is_a_failure() {
    use super::search::fit_parallel;
    let (mut a, f, scene) = setup_pair();
    let (mut b, _, _) = setup_pair();
    let setup = FitSetup {
        fixed: &f,
        scene: &scene,
        mask_ok: [true; 7],
    };
    let calls = std::sync::atomic::AtomicUsize::new(0);
    let cancel = || calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed) > 500;
    let err = fit_parallel(&setup, &mut a, &mut b, &cancel, &mut |_| {}).unwrap_err();
    assert_eq!(err, FitError::Cancelled);
    assert!(
        a.calls + b.calls <= 520,
        "{} renders after a cancel at 500",
        a.calls + b.calls
    );
    let mut refusing = Toy {
        base: a.base.clone(),
        calls: 0,
        refuse_all: true,
    };
    let mut other = Toy {
        base: a.base.clone(),
        calls: 0,
        refuse_all: true,
    };
    assert_eq!(
        fit_parallel(&setup, &mut refusing, &mut other, &|| false, &mut |_| {}).unwrap_err(),
        FitError::ChainRefused
    );
}

// ---------------------------------------------------------------------------------------------
// RG-13: the skin walls measure what H2 measures, the whites walls what H4 measures
// ---------------------------------------------------------------------------------------------

/// The skin of `cand` moved: its (a\*, b\*) and Oklab (a, b) turned by `turn_deg` and scaled by `scale`.
fn skin_moved(pr: &Pair, turn_deg: f64, scale: f64) -> Vec<Px> {
    let (sn, cs) = turn_deg.to_radians().sin_cos();
    pr.p_px
        .iter()
        .enumerate()
        .map(|(i, q)| {
            if !pr.scene.skin[i] {
                return *q;
            }
            let (a, b) = (q.lab[1], q.lab[2]);
            let lab = [
                q.lab[0],
                scale * (a * cs - b * sn),
                scale * (a * sn + b * cs),
            ];
            px_of(
                lab,
                [q.ok[0] * cs - q.ok[1] * sn, q.ok[0] * sn + q.ok[1] * cs],
            )
        })
        .collect()
}

fn skin_walls(pr: &Pair, f: &Fixed, cand: &[Px]) -> (f64, f64) {
    let t = terms(&candidate_measure(cand, &pr.scene, f), f);
    (t.guard_skin_drift, t.guard_skin_chroma)
}

#[test]
fn the_inner_limits_are_eight_tenths_of_h2s_lines() {
    assert!((skin_chroma_down_limit() - (1.0f64 / 0.55).ln() * 0.8).abs() < 1e-12);
    // Ratios 0.620 and 1.195.
    assert!(((-skin_chroma_down_limit()).exp() - 0.620).abs() < 5e-4);
    let pr = pair();
    let f = fixed_for(&pr, true);
    assert!(
        (skin_chroma_up_limit(&f.t).exp() - 1.195).abs() < 5e-3
            || skin_chroma_up_limit(&f.t) <= 0.30
    );
    assert!(skin_hue_limit_deg(&f.t) <= 8.0 && skin_hue_limit_deg(&f.t) >= 6.0);
    assert!(skin_chroma_up_limit(&f.t) <= 0.8 * 1.25f64.ln() + 1e-12);
}

#[test]
fn the_skin_walls_are_two_sided_in_chroma_and_inside_h2s_lines() {
    let pr = pair();
    let f = fixed_for(&pr, true);
    // Inside the inner limits: nothing. Chroma 0.7 and 1.05 of the photo's, hue turned 4 degrees.
    assert_eq!(skin_walls(&pr, &f, &skin_moved(&pr, 0.0, 0.7)), (0.0, 0.0));
    assert_eq!(skin_walls(&pr, &f, &skin_moved(&pr, 0.0, 1.05)), (0.0, 0.0));
    assert_eq!(skin_walls(&pr, &f, &skin_moved(&pr, 4.0, 1.0)), (0.0, 0.0));
    // Outside, on either side: the chroma wall fires below 0.620 and above the up limit, and the hue wall past the hue limit. All of them
    // are moves H2 would flag at the next step (0.55, 1.25, 10 degrees), so the wall is in front of the line.
    assert!(
        skin_walls(&pr, &f, &skin_moved(&pr, 0.0, 0.5)).1 > 0.0,
        "falls"
    );
    assert!(
        skin_walls(&pr, &f, &skin_moved(&pr, 0.0, 1.5)).1 > 0.0,
        "rises"
    );
    assert!(
        skin_walls(&pr, &f, &skin_moved(&pr, 12.0, 1.0)).0 > 0.0,
        "turns"
    );
    assert!(
        skin_walls(&pr, &f, &skin_moved(&pr, -12.0, 1.0)).0 > 0.0,
        "turns the other way"
    );
    // A fall of the chroma to nothing costs more than a fall to 0.5.
    assert!(
        skin_walls(&pr, &f, &skin_moved(&pr, 0.0, 0.05)).1
            > skin_walls(&pr, &f, &skin_moved(&pr, 0.0, 0.5)).1
    );
}

#[test]
fn the_skin_walls_start_at_one_percent_of_the_frame_on_the_analysis_picture() {
    let pr = pair();
    let moved = skin_moved(&pr, 12.0, 0.4);
    let at = |skin: f64| {
        let f = Fixed::with_coverage(
            &shipped(),
            &pr.p_px,
            &pr.scene,
            &pr.r_px,
            &pr.r_scene,
            true,
            Coverage { skin, whites: 0.1 },
        )
        .unwrap();
        skin_walls(&pr, &f, &moved)
    };
    // The gate is the coverage given (the analysis picture's), not the sample's own share of 10 %.
    assert_eq!(at(0.009), (0.0, 0.0));
    let (hue, chroma) = at(0.011);
    assert!(hue > 0.0 && chroma > 0.0);
    // With Fixed::new the coverage is the sample's own.
    let f = fixed_for(&pr, true);
    assert!((f.coverage.skin - 0.10).abs() < 0.02);
    assert!(skin_walls(&pr, &f, &moved).0 > 0.0);
}

#[test]
fn there_is_no_absolute_band_for_the_skin() {
    // A photo whose skin is green and nearly grey: outside the old band (CIELAB hue 30 to 70, C* 12 to 38) in hue and in chroma.
    let mut pr = pair();
    for (i, q) in pr.p_px.iter_mut().enumerate() {
        if pr.scene.skin[i] {
            *q = px_of([q.lab[0], -14.0, 8.0], [-0.04, 0.02]);
        }
    }
    pr.r_px.clone_from(&pr.p_px);
    let f = fixed_for(&pr, true);
    let c = candidate_measure(&pr.p_px, &pr.scene, &f);
    let t = with_regularisers(terms(&c, &f), &f, &c);
    assert_eq!(
        t.total(),
        0.0,
        "the photo as shot costs nothing: {:?}",
        t.named()
    );
    assert!(t.named().iter().all(|(k, _)| *k != "guard_skin"));
    // And dragging it into the old band costs: the wall is on the move, not on the place.
    let into_band: Vec<Px> = pr
        .p_px
        .iter()
        .enumerate()
        .map(|(i, q)| {
            if pr.scene.skin[i] {
                px_of([q.lab[0], 14.0, 16.0], [0.04, 0.06])
            } else {
                *q
            }
        })
        .collect();
    let (hue, chroma) = skin_walls(&pr, &f, &into_band);
    assert!(hue > 0.0 || chroma > 0.0);
}

#[test]
fn the_licence_is_the_references_whites_at_most_three_above_the_photos_plus_one() {
    assert_eq!(licence_allowed(2.0, 6.0), 6.0); // max(2, min(6, 5)) + 1
    assert_eq!(licence_allowed(2.0, 3.0), 4.0); // max(2, min(3, 5)) + 1
    assert_eq!(licence_allowed(2.0, 2.0), 3.0); // r = p: nothing but the margin
    assert_eq!(licence_allowed(5.0, 1.0), 6.0); // a reference with less never asks the photo to lose chroma
    // The inner limit of H4's line: 0.8 of max(8, the reference's own whites + 4).
    assert!((h4_inner_limit(0.0) - 6.4).abs() < 1e-12);
    assert!((h4_inner_limit(3.0) - 6.4).abs() < 1e-12);
    assert!((h4_inner_limit(10.0) - 11.2).abs() < 1e-12);
}

/// A photo whose whites (the architecture pixels) have (a\*, b\*) = (`a`, `b`) at L\* 92, and a reference made of it with its whites
/// given by `ref_whites` ((a\*, b\*) and the share of the whites kept white), the rest of the reference strongly coloured.
fn whites_pair(a: f64, b: f64, ref_whites: Option<[f64; 2]>) -> Pair {
    let mut pr = pair();
    for (i, q) in pr.p_px.iter_mut().enumerate() {
        if pr.scene.class[i] == class::ARCHITECTURE {
            *q = px_of([92.0, a, b], [a / 100.0, b / 100.0]);
        }
    }
    pr.r_px = pr
        .p_px
        .iter()
        .enumerate()
        .map(
            |(i, q)| match (pr.scene.class[i] == class::ARCHITECTURE, ref_whites) {
                (true, Some(w)) => px_of([92.0, w[0], w[1]], [w[0] / 100.0, w[1] / 100.0]),
                (true, None) => px_of([92.0, 14.0, 20.0], [0.14, 0.2]),
                _ => *q,
            },
        )
        .collect();
    pr
}

fn whites_wall(pr: &Pair, f: &Fixed, a: f64, b: f64) -> f64 {
    let cand: Vec<Px> = pr
        .p_px
        .iter()
        .enumerate()
        .map(|(i, q)| {
            if pr.scene.class[i] == class::ARCHITECTURE {
                px_of([92.0, a, b], [a / 100.0, b / 100.0])
            } else {
                *q
            }
        })
        .collect();
    terms(&candidate_measure(&cand, &pr.scene, f), f).guard_whites
}

#[test]
fn a_reference_with_no_whites_licenses_the_photo_only_the_margin() {
    // The photo's whites have C* 3; the reference has none (its pixels there are coloured).
    let pr = whites_pair(1.2, 2.75, None);
    let f = fixed_for(&pr, true);
    let l = f.licence;
    assert!(l.on);
    assert!(is_white(&pr.p_px[800]));
    assert!((l.photo_c - 1.2f64.hypot(2.75)).abs() < 0.05, "{l:?}");
    assert_eq!(l.reference_c, l.photo_c, "no whites of its own: r = p");
    assert!((l.allowed - (l.photo_c + 1.0)).abs() < 1e-9);
    assert!((l.cap - l.allowed).abs() < 1e-9);
    // The targets follow it: the cap is the licence, whatever the network asked for.
    assert_eq!(f.t.hl_cmax, l.cap);
    assert!(whites_wall(&pr, &f, 1.2, 2.75) == 0.0);
    // C* 5 is above the licence (3 + 1): the cap wall fires; C* 3.8 is not.
    assert!(whites_wall(&pr, &f, 2.0, 4.6) > 0.0);
    assert_eq!(whites_wall(&pr, &f, 1.4, 3.5), 0.0);
}

#[test]
fn a_reference_with_whites_of_its_own_licenses_them_up_to_three_above_the_photos() {
    // The reference's whites (at a quarter of the frame) have C* 4.5; the photo's, 2.
    let pr = whites_pair(0.5, 1.94, Some([1.5, 4.24]));
    let f = fixed_for(&pr, true);
    let l = f.licence;
    assert!((l.reference_c - 1.5f64.hypot(4.24)).abs() < 0.05);
    let want = licence_allowed(l.photo_c, l.reference_c);
    assert!((l.allowed - want).abs() < 1e-9);
    assert!(l.allowed > l.photo_c + 1.0, "more than the margin: {l:?}");
    assert!((l.cap - l.allowed).abs() < 1e-9);
    assert_eq!(
        whites_wall(&pr, &f, 1.5, 3.9),
        0.0,
        "at the reference's own whites"
    );
    assert!(whites_wall(&pr, &f, 3.0, 6.5) > 0.0, "past the licence");
}

#[test]
fn the_cap_stays_under_the_inner_limit_of_h4s_line_and_the_targets_never_ask_for_more() {
    // Whites of C* 5.8, a reference with none: the licence would be 6.8, but H4's inner limit is 6.4 (0.8 of 8).
    let pr = whites_pair(2.0, 5.44, None);
    let f = fixed_for(&pr, true);
    let l = f.licence;
    assert!((l.allowed - (l.photo_c + 1.0)).abs() < 1e-9 && l.allowed > l.inner);
    assert!((l.cap - 6.4).abs() < 1e-9 && l.cap < 8.0, "{l:?}");
    assert_eq!(f.t.hl_cmax, 6.4);
    // The white target and the highlights band of the tint term are within the licence's extra chroma of the photo's own.
    let n = f.p.tint[2];
    let extra = l.cap - l.photo_c;
    let tint = f.t.tint[2];
    assert!((tint[0] - n[0]).hypot(tint[1] - n[1]) <= extra + 1e-9);
    // The photo's own whites cost nothing in the direction wall: the tolerance covers them.
    let c = candidate_measure(&pr.p_px, &pr.scene, &f);
    assert_eq!(terms(&c, &f).guard_white_direction, 0.0);
    // A photo whose own whites are above the line is never asked to lose chroma: the cap is at least the photo's own.
    let pr = whites_pair(4.0, 9.0, None);
    let f = fixed_for(&pr, true);
    assert!(f.licence.cap >= f.licence.photo_c - 1e-9);
}

#[test]
fn the_whites_guard_starts_at_one_percent_of_the_frame_on_the_analysis_picture() {
    let pr = whites_pair(1.2, 2.75, None);
    let at = |whites: f64| {
        Fixed::with_coverage(
            &shipped(),
            &pr.p_px,
            &pr.scene,
            &pr.r_px,
            &pr.r_scene,
            true,
            Coverage { skin: 0.1, whites },
        )
        .unwrap()
    };
    let (off, on) = (at(0.009), at(0.011));
    assert!(!off.licence.on && on.licence.on);
    assert_eq!(whites_wall(&pr, &off, 3.0, 7.0), 0.0, "shut under 1 %");
    assert!(whites_wall(&pr, &on, 3.0, 7.0) > 0.0);
    // The targets are the network's own where the guard does not exist.
    assert_eq!(off.t.hl_cmax, off.net_values.get("T.hl_cmax"));
    assert_eq!(on.t.hl_cmax, on.licence.cap);
}

// ---------------------------------------------------------------------------------------------
// Match brightness (docs/34 section 5.8, RG-16)
// ---------------------------------------------------------------------------------------------

use super::brightness::{
    BISECTION_STEPS, GAP_HOLD_EV, Gap, SHIFT_RANGE_EV, asked_ev, luminance_of_lstar, solve_shift,
};
use super::objective::clip_shares;

#[test]
fn luminance_is_cielabs_inverse_with_lstar_held_at_one() {
    assert!((luminance_of_lstar(100.0) - 1.0).abs() < 1e-12);
    // L* 50 is Y 0.1842...
    assert!((luminance_of_lstar(50.0) - 0.184_187).abs() < 1e-5);
    // The linear toe: below L* 8 it is L* over kappa.
    assert!((luminance_of_lstar(4.0) - 4.0 * 27.0 / 24389.0).abs() < 1e-12);
    // Held at 1 and at 100: black has a luminance, so a ratio is a number.
    assert_eq!(luminance_of_lstar(0.0), luminance_of_lstar(1.0));
    assert_eq!(luminance_of_lstar(-5.0), luminance_of_lstar(1.0));
    assert_eq!(luminance_of_lstar(120.0), 1.0);
    // The inverse of the forward function the fit measures with.
    for l in [10.0, 25.0, 40.0, 61.0, 83.0] {
        let y = luminance_of_lstar(l);
        let back = 116.0 * y.cbrt() - 16.0;
        assert!((back - l).abs() < 1e-9, "{l}");
    }
}

#[test]
fn the_gap_is_held_at_one_and_a_half_stops_and_the_move_is_a_share_of_it() {
    // One stop brighter in the reference: a gap of 1 stop, not held.
    let photo_l = 116.0 * 0.1f64.cbrt() - 16.0;
    let ref_l = 116.0 * 0.2f64.cbrt() - 16.0;
    let g = Gap::of_lstar(photo_l, ref_l);
    assert!((g.raw_ev - 1.0).abs() < 1e-9 && !g.limited);
    assert_eq!(g.held_ev, g.raw_ev);
    // Three stops brighter: held at +1.5; and darker: -1.5.
    let ref_l = 116.0 * 0.8f64.cbrt() - 16.0;
    let g = Gap::of_lstar(photo_l, ref_l);
    assert!((g.raw_ev - 3.0).abs() < 1e-9);
    assert!(g.limited && g.held_ev == GAP_HOLD_EV);
    let g = Gap::of_lstar(ref_l, photo_l);
    assert!(g.limited && g.held_ev == -GAP_HOLD_EV);
    // The move asked for is b * g: 0 at 0, half at 50, all at 100 and monotone in between.
    assert_eq!(asked_ev(0.0, &g), 0.0);
    assert_eq!(asked_ev(100.0, &g), -GAP_HOLD_EV);
    assert!((asked_ev(50.0, &g) + 0.75).abs() < 1e-12);
    let up = Gap::of_lstar(photo_l, ref_l);
    let mut last = -1.0;
    for pct in 0..=100 {
        let a = asked_ev(f64::from(pct), &up);
        assert!(a >= last, "monotone at {pct}");
        last = a;
    }
    // A value outside 0 to 100 is held to it.
    assert_eq!(asked_ev(250.0, &up), up.held_ev);
    assert_eq!(asked_ev(-5.0, &up), 0.0);
}

/// A toy display: the photo's linear luminances, scaled by `2^d` and rolled off like a tone curve (so the *displayed* move is less
/// than the exposure shift: 0.7 stop for each stop).
fn displayed_median_y(ys: &[f64], d: f64) -> f64 {
    let mut v: Vec<f64> = ys
        .iter()
        .map(|&y| {
            let t = y * d.exp2();
            t.powf(0.7)
        })
        .collect();
    v.sort_unstable_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
fn the_bisection_lands_within_a_hundredth_of_a_stop_of_its_target() {
    let ys: Vec<f64> = (0..2001)
        .map(|i| 0.002 + 0.7 * f64::from(i) / 2000.0)
        .collect();
    let y0 = displayed_median_y(&ys, 0.0);
    let mut calls = 0usize;
    for ev in [-1.25, -1.0, -0.4, -0.1, 0.2, 0.75, 1.0, 1.25] {
        let target = y0 * f64::exp2(ev);
        let d = solve_shift::<()>(target, |d| {
            calls += 1;
            Ok(displayed_median_y(&ys, d))
        })
        .unwrap();
        assert!((SHIFT_RANGE_EV.0..=SHIFT_RANGE_EV.1).contains(&d));
        let got = (displayed_median_y(&ys, d) / y0).log2();
        assert!(
            (got - ev).abs() < 0.01,
            "asked {ev}, moved {got} with shift {d}"
        );
    }
    assert_eq!(calls, 8 * BISECTION_STEPS, "16 renders for each");
    // The same closure gives the same shift, bit for bit.
    let run = || {
        solve_shift::<()>(y0 * 0.6, |d| Ok(displayed_median_y(&ys, d)))
            .unwrap()
            .to_bits()
    };
    assert_eq!(run(), run());
    // A target no shift reaches takes the nearer end.
    let top = solve_shift::<()>(10.0, |d| Ok(displayed_median_y(&ys, d))).unwrap();
    assert_eq!(top, SHIFT_RANGE_EV.1);
    let bottom = solve_shift::<()>(0.0, |d| Ok(displayed_median_y(&ys, d))).unwrap();
    assert_eq!(bottom, SHIFT_RANGE_EV.0);
    // An error of the closure ends the search.
    let mut n = 0;
    let e = solve_shift(1.0, |_| {
        n += 1;
        if n == 3 { Err("cancelled") } else { Ok(0.0) }
    })
    .unwrap_err();
    assert_eq!((e, n), ("cancelled", 3));
}

/// The photo lifted by `by` L* points (a pure brightness move that clips nothing).
fn lifted(px: &[Px], by: f64) -> Vec<Px> {
    px.iter()
        .map(|q| {
            Px::new(
                [(q.lab[0] + by).min(100.0), q.lab[1], q.lab[2]],
                q.ok,
                false,
                false,
            )
        })
        .collect()
}

#[test]
fn moving_the_photo_costs_the_walls_nothing_and_a_moved_photo_that_is_the_photo_is_todays_fixed_part()
 {
    let pr = pair();
    let coverage = Coverage::of_sample(&pr.p_px, &pr.scene);
    // P' = P: the same fixed part as today's, field for field.
    let today = Fixed::with_coverage(
        &shipped(),
        &pr.p_px,
        &pr.scene,
        &pr.r_px,
        &pr.r_scene,
        true,
        coverage,
    )
    .unwrap();
    let same = Fixed::with_moved_photo(
        &shipped(),
        &pr.p_px,
        &pr.p_px,
        &pr.scene,
        &pr.r_px,
        &pr.r_scene,
        true,
        coverage,
    )
    .unwrap();
    assert_eq!(format!("{today:?}"), format!("{same:?}"));
    assert_eq!(today.clip_base, [today.p.hi, today.p.lo25]);
    // P' = a pure exposure move (every pixel lifted, nothing clipped). The walls read P', so the moved photo itself costs them nothing.
    let moved = lifted(&pr.p_px, 6.0);
    let f = Fixed::with_moved_photo(
        &shipped(),
        &pr.p_px,
        &moved,
        &pr.scene,
        &pr.r_px,
        &pr.r_scene,
        true,
        coverage,
    )
    .unwrap();
    assert!(
        (f.p.p50 - today.p.p50 - 6.0).abs() < 1.0,
        "the network reads P''s median"
    );
    let c = candidate_measure(&moved, &pr.scene, &f);
    let t = terms(&c, &f);
    let walls = [
        t.guard_skin_drift,
        t.guard_skin_chroma,
        t.guard_white_direction,
    ];
    assert!(walls.iter().all(|&w| w == 0.0), "{:?}", t.named());
    assert_eq!(clip_term(&f, &c.m), 0.0);
    // The same candidate against the fixed part made from P, which would count the move as harm somewhere, is no better: the walls
    // of the moved fixed part are never above them.
    let c0 = candidate_measure(&moved, &pr.scene, &today);
    let t0 = terms(&c0, &today);
    assert!(t.walls_and_clip() <= t0.walls_and_clip() + 1e-12);
}

#[test]
fn the_clip_term_keeps_the_photo_as_its_baseline() {
    let pr = pair();
    let coverage = Coverage::of_sample(&pr.p_px, &pr.scene);
    // P' clips a fifth of the frame at the top; P clips nothing.
    let mut moved = lifted(&pr.p_px, 4.0);
    for q in moved.iter_mut().take(200) {
        q.hi = true;
    }
    assert_eq!(clip_shares(&pr.p_px), [0.0, 0.0]);
    let [moved_hi, _] = clip_shares(&moved);
    assert!(moved_hi > 0.19);
    // With the baseline made from P the clip term counts the clipping of a candidate that is P'.
    let f = Fixed::with_moved_photo(
        &shipped(),
        &pr.p_px,
        &moved,
        &pr.scene,
        &pr.r_px,
        &pr.r_scene,
        true,
        coverage,
    )
    .unwrap();
    assert_eq!(f.clip_base, [0.0, 0.0]);
    let c = candidate_measure(&moved, &pr.scene, &f);
    let counted = clip_term(&f, &c.m);
    assert!(counted > 1.0, "the clip term reads P: {counted}");
    // Were the baseline P''s own (as the walls' are), the same candidate would clip nothing more than it.
    let own = Fixed::with_coverage(
        &shipped(),
        &moved,
        &pr.scene,
        &pr.r_px,
        &pr.r_scene,
        true,
        coverage,
    )
    .unwrap();
    let c2 = candidate_measure(&moved, &pr.scene, &own);
    assert_eq!(clip_term(&own, &c2.m), 0.0);
}
