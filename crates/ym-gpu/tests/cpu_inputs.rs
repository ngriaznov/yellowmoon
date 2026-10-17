//! Shared plumbing of the raw-algorithms package (`docs/algorithms/INTEGRATION.md` sections 2 to
//! 4): the noise model on the 16-bit working scale, `demosaic_with` and the CA curve type.
//!
//! * `noise_dn_matches_the_vectors`: [`NoiseDn::from_model`] against the numbers the C++ oracle
//!   wrote into the vectors' `derived` fields (adaptive ratio `g_dn`/`r_dn`, log-guided
//!   `read_noise_dn`, diffraction `sigma0_dn`/`sigma1_dn` through section 2.3 of its spec).
//! * `demosaic_with_ignores_its_inputs_for_the_existing_methods`: the existing methods give the
//!   same pixels through `demosaic`, `demosaic_with` with unit inputs and `demosaic_with` with
//!   gains, noise and CA curves (they do not read them).
//! * `ca_curves_from_the_vector_*`: the `ca_weighting` curves load into [`CaCurves`] and give the
//!   misregistration the spec states for that tile (0.68 to 0.72 px).
//!
//! Added in review:
//!
//! * `noise_dn_matches_the_oracle_on_models_beyond_the_vectors`: 40 more models, bit-exact
//!   against the C++ references' own derivation (zero model, extremes, log-uniform draws).
//! * `radial_curves_evaluate_like_the_oracle`: 4 curves (2 to 9 knots, one with negative radii)
//!   at 9 to 21 radii each, bit-exact against the reference's `curve_eval`.
//! * `the_noise_field_of_the_chain_changes_nothing_yet`: `Chain::noise` as `None`, zero and rich
//!   gives the same bits, whole frame and tile (the "no behaviour change" criterion).
//! * `demosaic_with_survives_every_small_size_and_keeps_the_placement`: 1x1 to 5x3 in all four
//!   phases, and the placement of a sub-buffer through the banded runner.

use std::path::{Path, PathBuf};

use serde_json::Value;
use ym_gpu::cpu::chain::{Chain, RawFrame};
use ym_gpu::cpu::demosaic::{self, CaCurves, DemosaicInputs, Method, RadialCurve};
use ym_gpu::cpu::denoise::BilateralParams;
use ym_gpu::cpu::detail::NoiseReduction;
use ym_gpu::cpu::normalize::Levels;
use ym_gpu::cpu::sharpen::SharpenParams;
use ym_gpu::cpu::synth;
use ym_gpu::cpu::tiling::Rect;
use ym_gpu::cpu::vignette::VignetteProfile;
use ym_gpu::cpu::{Cfa, Mosaic, NoiseDn};
use ym_noise::model::NoiseModel;

fn vector_dir(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/cpu_data")
        .join(name)
}

/// Every vector of one algorithm, sorted by file name.
fn vectors(dir: &str) -> Vec<(String, Value)> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(vector_dir(dir))
        .expect("vector directory")
        .map(|e| e.expect("entry").path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    files
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).expect("vector file");
            let v: Value = serde_json::from_str(&text).expect("vector json");
            (
                p.file_stem().expect("stem").to_string_lossy().into_owned(),
                v,
            )
        })
        .collect()
}

fn num(v: &Value, key: &str) -> f64 {
    v[key]
        .as_f64()
        .unwrap_or_else(|| panic!("number {key} in {v}"))
}

/// The `noise {a, b, sigma_row, q}` object of a vector's params as a model (no offsets).
fn noise_model(params: &Value) -> NoiseModel {
    let n = &params["noise"];
    NoiseModel {
        a: num(n, "a"),
        b: num(n, "b"),
        delta: [0.0; 4],
        sigma_row: num(n, "sigma_row"),
        q: num(n, "q"),
    }
}

fn assert_rel(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!(
        (got - want).abs() <= tol,
        "{what}: got {got}, vector says {want} (relative tolerance {rel})"
    );
}

#[test]
fn noise_dn_matches_the_vectors() {
    // Adaptive ratio: g_dn and r_dn are printed with 14 to 17 digits.
    let adaptive = vectors("adaptive_ratio_demosaic");
    assert!(adaptive.len() >= 15, "found {} vectors", adaptive.len());
    let mut worst: f64 = 0.0;
    for (name, v) in &adaptive {
        let p = &v["params"];
        let n = NoiseDn::from_model(&noise_model(p));
        let d = &p["derived"];
        assert_rel(n.g, num(d, "g_dn"), 1e-12, &format!("{name} g_dn"));
        assert_rel(n.r, num(d, "r_dn"), 1e-12, &format!("{name} r_dn"));
        worst = worst.max((n.g / num(d, "g_dn") - 1.0).abs());
        worst = worst.max((n.r / num(d, "r_dn") - 1.0).abs());
        assert!(n.is_valid(), "{name}");
    }
    eprintln!("adaptive ratio: worst relative difference {worst:e}");

    // Log-guided: only the read noise is used, printed with 9 digits.
    let log_guided = vectors("log_guided_demosaic");
    assert!(log_guided.len() >= 14, "found {} vectors", log_guided.len());
    let mut worst: f64 = 0.0;
    for (name, v) in &log_guided {
        let p = &v["params"];
        let n = NoiseDn::from_model(&noise_model(p));
        let want = num(&p["derived"], "read_noise_dn");
        assert_rel(n.r, want, 1e-7, &format!("{name} read_noise_dn"));
        worst = worst.max((n.r / want - 1.0).abs());
    }
    eprintln!("log-guided: worst relative difference {worst:e}");

    // Diffraction: spec section 2.3. With d = sum(luma_c * g0_c) and s = min(2 * scale, 1),
    //   sigma0 = max(d * r * sqrt(s), 1 DN),  sigma1 = d * g * s.
    // The planner vectors carry both (printed with 9 digits).
    let diffraction = vectors("diffraction_deconvolution");
    let mut checked = 0;
    let mut worst: f64 = 0.0;
    for (name, v) in &diffraction {
        let e = &v["expected"];
        // Only the planner vectors of active plans carry the noise constants (an inactive plan
        // reports zeros).
        if e.get("sigma0_dn").is_none() || num(e, "active") != 1.0 {
            continue;
        }
        let p = &v["params"];
        let n = NoiseDn::from_model(&noise_model(p));
        let sum = |key: &str| -> Vec<f64> {
            p[key]
                .as_array()
                .expect("array")
                .iter()
                .map(|x| x.as_f64().expect("number"))
                .collect()
        };
        let (luma, g0) = (sum("luma"), sum("g0"));
        let d: f64 = luma.iter().zip(&g0).map(|(l, g)| l * g).sum();
        let s = (2.0 * num(p, "scale")).min(1.0);
        let sigma0 = (d * n.r * s.sqrt()).max(1.0);
        let sigma1 = d * n.g * s;
        assert_rel(sigma0, num(e, "sigma0_dn"), 1e-6, &format!("{name} sigma0"));
        assert_rel(sigma1, num(e, "sigma1_dn"), 1e-6, &format!("{name} sigma1"));
        worst = worst
            .max((sigma0 / num(e, "sigma0_dn") - 1.0).abs())
            .max((sigma1 / num(e, "sigma1_dn") - 1.0).abs());
        checked += 1;
    }
    eprintln!("diffraction: {checked} planner vectors, worst relative difference {worst:e}");
    assert!(checked >= 10, "only {checked} planner vectors checked");
}

fn curves_from_vector() -> (CaCurves, Value) {
    let (_, v) = vectors("adaptive_ratio_demosaic")
        .into_iter()
        .find(|(n, _)| n == "ca_weighting")
        .expect("ca_weighting vector");
    let ca = &v["params"]["ca"];
    let column = |curve: &str, key: &str| -> Vec<f64> {
        ca[curve][key]
            .as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().expect("number"))
            .collect()
    };
    let build = |curve: &str| {
        RadialCurve::new(column(curve, "radius_px"), column(curve, "shift_px"))
            .unwrap_or_else(|e| panic!("{curve} curve of the vector: {e}"))
    };
    let curves = CaCurves::new(
        num(ca, "centre_x"),
        num(ca, "centre_y"),
        build("red"),
        build("blue"),
    )
    .expect("curves");
    (curves, v)
}

#[test]
fn ca_curves_from_the_vector_load_and_give_the_stated_misregistration() {
    let (ca, v) = curves_from_vector();
    assert_eq!((ca.centre_x, ca.centre_y), (3024.0, 2012.0));
    assert_eq!(ca.red.len(), 458);
    assert_eq!(ca.blue.len(), 458);

    // The vector tile (32 x 32 at frame (4800, 3330)): the spec says `ca` is about 0.68 to 0.72 px
    // across it (docs/algorithms/01 section 5, vector table).
    let origin = (
        num(&v["params"]["ca"], "origin_x"),
        num(&v["params"]["ca"], "origin_y"),
    );
    assert_eq!(origin, (4800.0, 3330.0));
    let (mut lo, mut hi) = (f64::MAX, f64::MIN);
    for y in 0..32 {
        for x in 0..32 {
            let dx = origin.0 + f64::from(x) + 0.5 - ca.centre_x;
            let dy = origin.1 + f64::from(y) + 0.5 - ca.centre_y;
            let m = ca.misregistration_px(dx.hypot(dy));
            lo = lo.min(m);
            hi = hi.max(m);
        }
    }
    eprintln!("misregistration over the tile: {lo:.4} to {hi:.4} px");
    assert!(
        (0.67..=0.73).contains(&lo) && (0.67..=0.73).contains(&hi),
        "{lo} to {hi}"
    );

    // The curves are exact at their knots.
    for curve in [&ca.red, &ca.blue] {
        for (r, s) in curve.radius_px().iter().zip(curve.shift_px()) {
            assert_eq!(curve.shift_at(*r), *s);
        }
    }
}

fn mosaic(cfa: Cfa, w: usize, h: usize) -> Mosaic {
    synth::mosaic_from_rgb(&synth::natural(w, h, 11), cfa)
}

/// The methods that work on a balanced mosaic and ignore the gains, the noise and the CA curves.
/// The package's methods read them: `LogGuided` the read noise (`tests/cpu_log_guided.rs`),
/// `AdaptiveRatio` all three.
const IGNORING_INPUTS: [Method; 3] = [Method::Bilinear, Method::Menon, Method::Lmmse];

#[test]
fn demosaic_with_ignores_its_inputs_for_the_existing_methods() {
    let (ca, _) = curves_from_vector();
    let rich = DemosaicInputs {
        gains: [2.0, 1.0, 1.6],
        noise: NoiseDn::new(1.3107, 6.56),
        ca: Some(&ca),
    };
    // 150 rows: two bands of the banded runner, so the closure path is exercised.
    for cfa in Cfa::ALL {
        let m = mosaic(cfa, 61, 150);
        for method in IGNORING_INPUTS {
            let plain = demosaic::demosaic(&m, method);
            let unit = demosaic::demosaic_with(&m, method, &DemosaicInputs::default());
            let with = demosaic::demosaic_with(&m, method, &rich);
            assert_eq!(plain, unit, "{method:?} {} unit inputs", cfa.name());
            assert_eq!(plain, with, "{method:?} {} rich inputs", cfa.name());
            let whole = demosaic::demosaic_whole(&m, method);
            assert_eq!(
                whole,
                demosaic::demosaic_whole_with(&m, method, &rich),
                "{method:?} {} whole frame",
                cfa.name()
            );
        }
    }
}

#[test]
fn tiny_frames_keep_falling_back_to_bilinear() {
    let data: Vec<f32> = (0..8).map(|i| 0.1 * (i as f32 + 1.0)).collect();
    let m = Mosaic::from_data(8, 1, Cfa::RGGB, data).expect("mosaic");
    let want = demosaic::bilinear(&m);
    assert_eq!(demosaic::demosaic(&m, Method::Menon), want);
    assert_eq!(
        demosaic::demosaic_with(&m, Method::Menon, &DemosaicInputs::default()),
        want
    );
}

/// Models and the values the development oracle (the C++ references, built outside the
/// repository) computes from them. Each row is `[a, b, sigma_row, q, g_dn, r_dn, read_noise_dn,
/// sigma0_dn]`:
///
/// * `g_dn`, `r_dn`: `derive()` of the adaptive ratio reference (`g_dn`, `r_dn`);
/// * `read_noise_dn`: `derive()` of the log-guided reference (the same formula, with the
///   spec 02 rule that an invalid variance counts as 0);
/// * `sigma0_dn`: `make_plan()` of the diffraction reference for a 3.76 um pitch, f/8, `scale`
///   0.4 (so `s = 0.8`), `g0 = (2, 1, 1.5)` and the default luma weights 0.299/0.587/0.114.
///
/// The rows were printed with 17 significant digits by a scratch driver that links the three
/// references; they are not a vector of the package and nothing in the repository regenerates
/// them. They cover models the vectors do not: the zero model, a unit model, the extremes of each
/// field (1e-14 to 1) and 30 log-uniform draws with some fields zero.
#[rustfmt::skip]
const ORACLE_ROWS: &[[f64; 8]] = &[
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    [2e-05, 1e-08, 3e-05, 1.5e-05, 1.3107000000000002, 6.847937136957286, 6.847937136957286, 8.305474491912186],
    [1.0, 1.0, 1.0, 1.0, 65535.0, 94591.62472835532, 94591.62472835532, 114724.81575363291],
    [1e-09, 1e-14, 1e-09, 1e-09, 6.5535e-05, 0.006553854971636445, 0.006553854971636445, 1.0],
    [0.005, 4e-06, 0.002, 6.1e-05, 327.675, 185.364563919679, 185.364563919679, 224.81816444117902],
    [2.5e-05, 4e-08, 5e-05, 0.0, 1.6383750000000001, 13.51038635874267, 13.51038635874267, 16.38598121364669],
    [0.1, 0.0, 0.0, 0.0, 6553.5, 0.0, 0.0, 1.0],
    [0.0, 0.0, 0.001, 0.0, 0.0, 65.535, 65.535, 79.4836839097082],
    [0.0, 0.0, 0.0, 0.001, 0.0, 18.91832494567106, 18.91832494567106, 22.944963150726576],
    [0.0, 0.0, 0.0, 6.103888176768602e-05, 0.0, 1.154753399601481, 1.154753399601481, 1.400534892921112],
    [0.003809238600570521, 1.7335610302181084e-07, 4.35086671137577e-06, 2.866720582694637e-07, 249.6384516883891, 27.287678573343605, 27.287678573343605, 33.09567739915099],
    [4.1108822260137623e-05, 1.066913006462136e-09, 2.3795358165017019e-07, 9.450631457847219e-06, 2.6940666668181192, 2.1481214390210646, 2.1481214390210646, 2.605334637351306],
    [0.00010119877265989034, 1.0941277701465373e-08, 4.483059616137945e-06, 0.0, 6.6320615662659135, 6.861290890243869, 6.861290890243869, 8.321670501757902],
    [0.00010727535505023626, 8.052898600684123e-07, 0.0, 0.0005144971194980674, 7.030290393217233, 59.60979449202478, 59.60979449202478, 72.29733826698349],
    [0.00019446581291108133, 0.0, 2.6498226491611276e-05, 2.4444152517446455e-07, 12.744317049127716, 1.7365674304864374, 1.7365674304864374, 2.1061841265381576],
    [6.926620776907175e-05, 3.451015308977416e-10, 2.4238654054140984e-05, 2.328116152931734e-06, 4.5393609261461165, 2.001840093559784, 2.001840093559784, 2.4279182915127273],
    [1.3230432488415413e-05, 3.771549441649227e-11, 0.00029855153825712256, 1.3338970163000752e-06, 0.8670563931283041, 19.569730353221196, 19.569730353221196, 23.735015817405294],
    [3.3966994103213545e-07, 4.5154260393022323e-07, 0.0007846920289744483, 0.0, 0.022260269585540997, 67.70384610165294, 67.70384610165294, 82.1141543147221],
    [0.0009493477396825388, 3.243541593223462e-08, 4.890545499542358e-06, 5.09041355774629e-07, 62.215504120095176, 11.807100132001368, 11.807100132001368, 14.320161971195096],
    [1.6222452799729378e-07, 3.352548488568643e-11, 0.0003240598985169276, 2.4849273604340874e-07, 0.010631384442302648, 21.240655647396924, 21.240655647396924, 25.76158632047998],
    [6.198050377097844e-07, 1.1441035150738248e-10, 0.0, 3.2164473608879176e-06, 0.04061892314631072, 0.7036166711242612, 0.7036166711242612, 1.0],
    [0.00163085288945616, 3.831858584877878e-11, 0.00011702162842138179, 1.4173429606704738e-05, 106.87794411050943, 7.684414178791823, 7.684414178791823, 9.319990045294412],
    [0.0001715033614811447, 1.786700305440839e-09, 6.164370684021935e-07, 0.0, 11.239472794666819, 2.77041823735225, 2.77041823735225, 3.3600831231463677],
    [1.2577881075324803e-07, 2.673386316314851e-07, 3.3243819807996664e-07, 3.187124305229877e-06, 0.00824291436271411, 33.88480115938077, 33.88480115938077, 41.09695315015694],
    [8.481302204999068e-06, 5.063953674627532e-10, 7.863925192929442e-07, 0.00028292968127833666, 0.5558221400046139, 5.552242169439248, 5.552242169439248, 6.733999566428042],
    [0.0022968412664731304, 0.0, 2.0642004926917337e-05, 0.00017638908447652454, 150.5234923983166, 3.600760004824929, 3.600760004824929, 4.367157550289526],
    [0.00036721443232461614, 8.936361711958959e-10, 0.0003600344945824226, 3.5689769533879584e-07, 24.06539782239372, 23.676053586646145, 23.676053586646145, 28.715342328683843],
    [1.1460689746065885e-08, 1.7621340227715234e-10, 0.0, 0.0, 0.0007510763025084277, 0.8699469543773409, 0.8699469543773409, 1.055109311664636],
    [6.059659137172322e-08, 5.628588879324631e-07, 2.4116270969996144e-05, 0.0003956807655372403, 0.003971197615545882, 49.75860790870109, 49.75860790870109, 60.349392886279155],
    [0.00028515394986365356, 8.720212355778312e-10, 8.102086151394003e-05, 0.0004782093314369366, 18.687564104314536, 10.666999497941905, 10.666999497941905, 12.93739858639552],
    [3.5019193334063742e-06, 4.974399261195801e-08, 2.5864723452567095e-07, 3.1168047088627956e-05, 0.22949828351478674, 14.628406360040831, 14.628406360040831, 17.74196425153354],
    [2.8946576466055876e-08, 7.377348363384032e-10, 6.090243384736729e-06, 7.167699686974492e-05, 0.0018970138887029718, 2.2729958182932295, 2.2729958182932295, 2.756787688247617],
    [3.0810987045738876e-05, 2.415628179919403e-07, 0.002194853207300948, 0.0, 2.0191980360424973, 147.40194523372364, 147.40194523372364, 178.77545773454506],
    [0.0034213364126595334, 9.18054782615944e-06, 0.0011135321282304165, 2.7380953569690258e-06, 224.21728180364252, 211.55224071737487, 211.55224071737487, 256.5797120862184],
    [5.5520291949611666e-05, 1.5283148283770837e-06, 0.0, 0.0002857073184481849, 3.6385223329178005, 81.19776497033429, 81.19776497033429, 98.48016304382128],
    [1.574971658297513e-07, 7.010208664303921e-12, 4.18375274902302e-06, 0.00014558435164560343, 0.01032157676265275, 2.773259405703573, 2.773259405703573, 3.3635290150693216],
    [9.387618978288575e-06, 0.0, 0.0023936987412857707, 4.213914288235764e-05, 0.6152176097421418, 156.873072645022, 156.873072645022, 190.26245090503735],
    [0.0009612480491043301, 1.4920893007696772e-07, 0.00017950097378586195, 0.0, 62.99539089805228, 27.91433531894889, 27.91433531894889, 33.85571381766899],
    [1.250599745758723e-05, 4.958953605277576e-08, 0.001099548220779589, 0.0009216213065979303, 0.8195805433829791, 75.56097338721922, 75.56097338721922, 91.64361828976281],
    [5.657006510969247e-08, 2.2190342766978182e-12, 0.001012029309967418, 0.00013852030714208905, 0.003707319216963696, 66.37516453939826, 66.37516453939826, 80.50267181970555],
];

#[test]
fn noise_dn_matches_the_oracle_on_models_beyond_the_vectors() {
    let luma = [0.299_f32, 0.587, 0.114].map(f64::from);
    let g0 = [2.0_f32, 1.0, 1.5].map(f64::from);
    let d: f64 = luma.iter().zip(&g0).map(|(l, g)| l * g).sum();
    let s = (2.0_f64 * 0.4).min(1.0);
    assert_eq!(ORACLE_ROWS.len(), 40);
    for (i, &[a, b, sigma_row, q, g_dn, r_dn, read_noise_dn, sigma0_dn]) in
        ORACLE_ROWS.iter().enumerate()
    {
        let n = NoiseDn::from_model(&NoiseModel {
            a,
            b,
            delta: [0.0; 4],
            sigma_row,
            q,
        });
        // Same operation order as the references, so the doubles are the same.
        assert_eq!(n.g, g_dn, "row {i} g");
        assert_eq!(n.r, r_dn, "row {i} r");
        assert_eq!(
            n.r, read_noise_dn,
            "row {i} read noise of the log-guided reference"
        );
        assert!(n.is_valid(), "row {i}");
        // Through spec 03 section 2.3 (the floor is 1 DN).
        let sigma0 = (d * n.r * s.sqrt()).max(1.0);
        assert_rel(sigma0, sigma0_dn, 1e-12, &format!("row {i} sigma0"));
    }
}

/// A curve and the values of the reference's `curve_eval` (the adaptive ratio oracle) at some
/// radii: below, at and above the ends, inside, and at every knot. Printed with 17 significant
/// digits by a scratch driver, like [`ORACLE_ROWS`].
struct OracleCurve {
    knots: &'static [(f64, f64)],
    points: &'static [(f64, f64)],
}

#[rustfmt::skip]
const ORACLE_CURVES: &[OracleCurve] = &[
    OracleCurve {
        knots: &[(149.78016289558295, 0.1817695313784342), (482.8181908563091, -0.7183414164193358)],
        points: &[
            (49.78016289558295, 0.4520422546795452),
            (149.77916289558294, 0.18177223410566723),
            (149.78016289558295, 0.1817695313784342),
            (316.29917687594605, -0.2682859425204509),
            (482.8181908563091, -0.7183414164193358),
            (482.8191908563091, -0.7183441191465688),
            (732.8181908563091, -1.3940232246721131),
            (-3.0, 0.5946926383001998),
            (0.0, 0.5865844566011664),
            (365.0880676010652, -0.4001490061516335),
            (149.78016289558295, 0.1817695313784342),
            (482.8181908563091, -0.7183414164193358),
        ],
    },
    OracleCurve {
        knots: &[(30.288179348598216, 0.1715518227001862), (240.37298478529195, 1.5251612664443606), (552.0903793021308, 0.1925806375650323)],
        points: &[
            (-69.71182065140178, -0.4727639053842356),
            (30.287179348598215, 0.17154537954290536),
            (30.288179348598216, 0.1715518227001862),
            (291.1892793253645, 1.3079234384415699),
            (552.0903793021308, 0.1925806375650323),
            (552.0913793021308, 0.19257636260106703),
            (802.0903793021308, -0.8761603537300031),
            (-3.0, -0.042929152435782575),
            (0.0, -0.023599680593249905),
            (149.3939559474039, 0.9389690743833861),
            (465.80653301223344, 0.5614409712314861),
            (30.288179348598216, 0.1715518227001862),
            (240.37298478529195, 1.5251612664443606),
            (552.0903793021308, 0.1925806375650323),
        ],
    },
    OracleCurve {
        knots: &[(247.45544438723886, -0.5210132489818529), (317.5522722916645, -0.6806586051156868), (435.5815201263679, -0.2547504298839878), (589.0938823357202, 1.6165354474524807), (874.8065965193879, 1.274446777878329)],
        points: &[
            (147.45544438723886, -0.2932634906739531),
            (247.45444438723885, -0.5210109714842698),
            (247.45544438723886, -0.5210132489818529),
            (561.1310204533133, 1.275673592567478),
            (874.8065965193879, 1.274446777878329),
            (874.8075965193879, 1.2744455805614),
            (1124.8065965193878, 0.9751175455832596),
            (-3.0, 0.04939842027906016),
            (0.0, 0.04256592752982313),
            (258.5499928280917, -0.5462810562412481),
            (369.60584839914515, -0.49282343696084124),
            (538.8600406360883, 1.00419467447498),
            (686.6976873746917, 1.4996727593269088),
            (247.45544438723886, -0.5210132489818529),
            (317.5522722916645, -0.6806586051156868),
            (435.5815201263679, -0.2547504298839878),
            (589.0938823357202, 1.6165354474524807),
            (874.8065965193879, 1.274446777878329),
        ],
    },
    OracleCurve {
        knots: &[(72.09693198311186, 0.835344073685508), (398.4580266407121, 1.8323041305253174), (433.86894278290714, 0.6867801610837503), (501.2939402374227, -0.7512235923353898), (681.494587404842, -0.5497015696037248), (928.098334370954, -0.12703271167429908), (1131.5451248403206, 1.0454759942272887), (1377.2328808661646, 0.7421930801150176), (1742.6869922462727, -0.7417721875507124)],
        points: &[
            (-27.90306801688814, 0.5298664683092409),
            (72.09593198311185, 0.8353410189094542),
            (72.09693198311186, 0.835344073685508),
            (907.3919621146922, -0.16252259705150063),
            (1742.6869922462727, -0.7417721875507124),
            (1742.6879922462726, -0.7417762481570138),
            (1992.6869922462727, -1.7569237629849406),
            (-3.0, 0.6059397641524539),
            (0.0, 0.6151040923137419),
            (331.0152714171916, 1.62628161686873),
            (416.68390587728953, 1.2427068028027302),
            (487.5100602144238, -0.45724847986878814),
            (521.628772124353, -0.7284827391909764),
            (790.4377438385806, -0.36297740147742813),
            (969.4558018032441, 0.11131949091649582),
            (1167.0052824107802, 1.0017031172357382),
            (1686.7428902542022, -0.5146052144566986),
            (72.09693198311186, 0.835344073685508),
            (398.4580266407121, 1.8323041305253174),
            (433.86894278290714, 0.6867801610837503),
            (501.2939402374227, -0.7512235923353898),
            (681.494587404842, -0.5497015696037248),
            (928.098334370954, -0.12703271167429908),
            (1131.5451248403206, 1.0454759942272887),
            (1377.2328808661646, 0.7421930801150176),
            (1742.6869922462727, -0.7417721875507124),
        ],
    },
];

#[test]
fn radial_curves_evaluate_like_the_oracle() {
    for (i, c) in ORACLE_CURVES.iter().enumerate() {
        let curve = RadialCurve::from_knots(c.knots).expect("the oracle accepts it");
        for &(radius, want) in c.points {
            // The same expression as `curve_eval`, so the doubles agree.
            assert_eq!(curve.shift_at(radius), want, "curve {i} at radius {radius}");
        }
    }
}

/// The chain with every raw stage on except R9, `noise` as given.
fn full_chain(noise: Option<NoiseDn>) -> Chain {
    Chain {
        levels: Levels {
            black: [512.0, 514.0, 513.0, 511.0],
            white: [16000.0; 4],
        },
        vignette: Some(
            VignetteProfile::from_tag(&[0, -400, -1500, -3500, -6000]).expect("vignette"),
        ),
        gains: [2.0, 1.0, 1.6],
        highlight_recovery: true,
        method: Method::Menon,
        denoise: Some(BilateralParams::default()),
        noise_reduction: NoiseReduction {
            luminance: 40.0,
            color: 60.0,
            detail: 70.0,
        },
        iso: 800.0,
        sigma: 0.01,
        sigma_map: None,
        noise,
        sharpen: Some(
            SharpenParams {
                amount: 90.0,
                radius_px: 0.8,
            }
            .into(),
        ),
        lens: None,
    }
}

#[test]
fn the_noise_field_of_the_chain_changes_nothing_yet() {
    // The field alone is plumbing: no stage of this chain reads `Chain::noise`, so whole-frame and
    // tiled output are the same bits with no model, the zero model and a rich one. (The adaptive
    // ratio demosaic and the diffraction deconvolution read the field; this chain uses neither.)
    let (w, h) = (97, 71);
    let img = synth::natural(w, h, 5);
    let m = synth::mosaic_from_rgb(&img, Cfa::RGGB);
    let mut rng = synth::Rng::new(12);
    let data: Vec<u16> = m
        .data
        .iter()
        .map(|&v| {
            let x = 512.0 + (v * 1.3 + 0.004 * rng.gaussian()) * 15488.0;
            x.clamp(0.0, 16000.0) as u16
        })
        .collect();
    let raw = RawFrame {
        width: w,
        height: h,
        cfa: Cfa::RGGB,
        data: &data,
        qstep: None,
    };
    let rich = NoiseDn::from_model(&NoiseModel {
        a: 2.0e-4,
        b: 4.0e-8,
        delta: [0.0; 4],
        sigma_row: 1.0e-4,
        q: 2.0e-4,
    });
    let none = full_chain(None);
    let want = none.run(&raw).expect("run without a model");
    for noise in [NoiseDn::ZERO, rich, NoiseDn::new(1.0e6, 1.0e6)] {
        let chain = full_chain(Some(noise));
        assert_eq!(chain.stages().len(), none.stages().len(), "{noise:?}");
        let got = chain.run(&raw).expect("run");
        assert_eq!(got, want, "{noise:?}: whole frame");
        // And through the tile path with halos.
        let prepared = chain.prepare(&raw).expect("prepare");
        let tile = Rect::new(20, 11, 40, 33);
        assert_eq!(
            chain.run_tile(&raw, &prepared, tile).expect("tile"),
            none.run_tile(&raw, &none.prepare(&raw).expect("prepare"), tile)
                .expect("tile"),
            "{noise:?}: tile"
        );
    }
}

#[test]
fn demosaic_with_survives_every_small_size_and_keeps_the_placement() {
    let (ca, _) = curves_from_vector();
    let rich = DemosaicInputs {
        gains: [2.0, 1.0, 1.6],
        noise: NoiseDn::new(1.3107, 6.56),
        ca: Some(&ca),
    };
    let mut rng = synth::Rng::new(77);
    for cfa in Cfa::ALL {
        for (w, h) in [
            (1, 1),
            (1, 7),
            (7, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (3, 3),
            (5, 3),
            (3, 5),
        ] {
            let data: Vec<f32> = (0..w * h).map(|_| rng.range(0.0, 1.0)).collect();
            let m = Mosaic::from_data(w, h, cfa, data).expect("mosaic");
            for method in Method::ALL {
                let plain = demosaic::demosaic(&m, method);
                let with = demosaic::demosaic_with(&m, method, &rich);
                assert_eq!((with.width, with.height), (w, h));
                if IGNORING_INPUTS.contains(&method) {
                    assert_eq!(plain, with, "{method:?} {} {w}x{h}", cfa.name());
                }
                assert!(
                    with.data.iter().all(|v| v.is_finite()),
                    "{method:?} {} {w}x{h}",
                    cfa.name()
                );
            }
        }
    }
    // A sub-buffer keeps its place in the frame, also through the banded runner.
    let whole = mosaic(Cfa::GRBG, 80, 200);
    let part = whole.extract(Rect::new(6, 10, 61, 150));
    for method in Method::ALL {
        let out = demosaic::demosaic_with(&part, method, &rich);
        assert_eq!(out.place, part.place, "{method:?}");
        assert_eq!((out.place.x0, out.place.y0), (6, 10));
    }
}
