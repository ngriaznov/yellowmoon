//! M1-11 (C12, C13): the whole display path against OpenColorIO, end to end.
//!
//! `aces2_vectors.rs` checks the ACES 2 port on AP0 inputs; `display_transform.rs` and `chain_cpu.rs`
//! compose the finished stage modules. Neither would notice a wrong colour space between them (the
//! AWG4 to AP0 step, the LogC4 decode, which output uses which limiting gamut, which transfer C13
//! applies). This file does: LogC4 code values go through `ColorChain` (identity matrix, no
//! roll-off, zero grade, Neutral look, identity curves, ACES 2) and the result is compared with
//! OpenColorIO 2.6.0's built-in `ARRI_LOGC4_to_ACES2065-1`, followed by the built-in ACES 2.0
//! SDR 100 nit output transform and the built-in display transform of the same output
//! (float32 CPU renderer). The vectors below were generated with this script (numpy and
//! PyOpenColorIO; `pip install opencolorio==2.6.0`):
//!
//! ```python
//! def bt(style):
//!     t = ocio.BuiltinTransform(); t.setStyle(style); return t
//! g = ocio.GroupTransform()
//! g.appendTransform(bt("ARRI_LOGC4_to_ACES2065-1"))
//! g.appendTransform(bt("ACES-OUTPUT - ACES2065-1_to_CIE-XYZ-D65 - SDR-100nit-P3-D65_2.0"))  # or ...-REC709_2.0 for sRGB
//! g.appendTransform(bt("DISPLAY - CIE-XYZ-D65_to_DisplayP3"))   # or _to_sRGB, _to_REC.1886-REC.2020
//! proc = ocio.Config.CreateRaw().getProcessor(g).getDefaultCPUProcessor()
//! proc.applyRGB(np.array(INPUTS, dtype=np.float32))              # in place
//! ```
//!
//! Inputs: seven greys from -6 to +5 stops, eleven chromatic and near-black/bright patches, and
//! 22 seeded random triples in LogC4 `[0.1, 0.7]`.
//!
//! **Tolerance.** The gate for the display transform is 1 code value at 10 bits (docs/02, C12).
//! The measured difference is at most 0.016 code values; most of it is C13 (the chain encodes with IEC
//! 61966-2-1 as docs/02 specifies, OpenColorIO's display transform uses the continuous "moncurve",
//! which differs by up to 9.8e-6). The test allows 0.05 code values so that a real change in any
//! stage (a different CAT, a wrong limiting gamut, a different transfer) fails it, while float32
//! noise does not.

use ym_color::consts::Vec3;
use ym_color::display::{ChainParams, ColorChain, DISPLAY_PLACEMENT_EV, OutputTarget};
use ym_color::logc::logc4_decode_rgb;
use ym_color::tone::ToneParams;

/// Largest allowed difference in 10-bit code values (the gate is 1.0).
const TOLERANCE_CODES_10BIT: f64 = 0.05;
/// The gate of docs/02, C12.
const GATE_CODES_10BIT: f64 = 1.0;

const INPUTS: [[f64; 3]; 40] = [
    [0.101610272, 0.101610272, 0.101610272],
    [0.147008611, 0.147008611, 0.147008611],
    [0.225637065, 0.225637065, 0.225637065],
    [0.278395837, 0.278395837, 0.278395837],
    [0.336536969, 0.336536969, 0.336536969],
    [0.460806553, 0.460806553, 0.460806553],
    [0.589018364, 0.589018364, 0.589018364],
    [0.45, 0.3, 0.2],
    [0.3, 0.45, 0.25],
    [0.25, 0.3, 0.5],
    [0.55, 0.5, 0.15],
    [0.2, 0.4, 0.45],
    [0.5, 0.25, 0.45],
    [0.6, 0.15, 0.12],
    [0.12, 0.6, 0.15],
    [0.12, 0.15, 0.6],
    [0.095, 0.1, 0.11],
    [0.75, 0.7, 0.65],
    [0.207361, 0.483948, 0.380361],
    [0.3223, 0.31295, 0.574311],
    [0.643086, 0.206412, 0.491671],
    [0.278982, 0.680177, 0.65191],
    [0.481522, 0.551639, 0.409092],
    [0.595537, 0.369028, 0.303287],
    [0.26674, 0.2358, 0.41549],
    [0.358547, 0.497908, 0.107704],
    [0.368621, 0.319108, 0.217239],
    [0.45692, 0.361188, 0.279995],
    [0.22565, 0.624774, 0.578477],
    [0.464026, 0.30706, 0.668092],
    [0.438026, 0.359658, 0.64027],
    [0.291605, 0.517597, 0.288292],
    [0.256932, 0.520505, 0.236735],
    [0.395866, 0.448017, 0.213344],
    [0.538744, 0.42909, 0.472902],
    [0.323287, 0.352116, 0.396898],
    [0.381983, 0.505384, 0.446307],
    [0.349762, 0.101081, 0.576419],
    [0.411631, 0.295927, 0.399973],
    [0.156068, 0.642822, 0.693842],
];

/// viewport: SDR-100nit-P3-D65_2.0 then DISPLAY - CIE-XYZ-D65_to_DisplayP3
const EXPECTED_VIEWPORT: [[f64; 3]; 40] = [
    [0.00056012, 0.00056012, 0.00056012],
    [0.04796797, 0.04796804, 0.04796797],
    [0.22143641, 0.2214365, 0.22143644],
    [0.34918833, 0.34918863, 0.34918845],
    [0.49467438, 0.49467504, 0.4946748],
    [0.77100194, 0.77100217, 0.77100217],
    [0.92841178, 0.92841232, 0.92841208],
    [0.97298056, 0.33815753, 0.24520659],
    [0.18602294, 0.85026294, 0.32632661],
    [0.0, 0.28730845, 0.92331654],
    [1.00000668, 0.83417636, 0.41596642],
    [0.0, 0.64514709, 0.76476938],
    [1.00000668, 0.24143058, 0.73912573],
    [1.00000668, 0.47858697, 0.42281789],
    [0.54567093, 1.00000668, 0.54506493],
    [0.38209474, 0.51613814, 1.00000668],
    [2.18e-06, 0.00026499, 0.00062054],
    [0.9952727, 0.98038751, 0.97313863],
    [0.0, 0.89413446, 0.5823127],
    [0.31969574, 0.45997316, 1.00000668],
    [1.00000668, 0.56444621, 0.6903196],
    [0.81527084, 1.00000668, 0.95535678],
    [0.75750214, 0.96167773, 0.68996334],
    [1.00000668, 0.54169017, 0.52757484],
    [0.25052238, 0.0, 0.71444941],
    [0.15242878, 0.98683494, 0.0],
    [0.65197486, 0.45389605, 0.23451307],
    [0.92259508, 0.53590798, 0.40894318],
    [0.65643352, 1.00000668, 0.89364344],
    [0.62290591, 0.63957983, 1.00000668],
    [0.57455629, 0.58861804, 1.00000668],
    [0.05797843, 0.98913854, 0.40997145],
    [0.0, 0.99110335, 0.36139357],
    [0.56704438, 0.83985215, 0.19643155],
    [1.00000668, 0.67489427, 0.78888607],
    [0.40929511, 0.52122295, 0.65115243],
    [0.40237623, 0.93313408, 0.71122003],
    [0.38843334, 0.45978743, 1.00000668],
    [0.8037318, 0.24048114, 0.68299598],
    [0.81180364, 0.95804948, 1.00000668],
];

/// srgb: SDR-100nit-REC709_2.0 then DISPLAY - CIE-XYZ-D65_to_sRGB
const EXPECTED_SRGB: [[f64; 3]; 40] = [
    [0.00056012, 0.00056012, 0.00056012],
    [0.04796804, 0.04796801, 0.04796801],
    [0.22143644, 0.22143644, 0.22143644],
    [0.34918845, 0.34918863, 0.34918845],
    [0.49467427, 0.49467498, 0.49467486],
    [0.77100217, 0.77100205, 0.77100229],
    [0.92841202, 0.9284119, 0.9284119],
    [1.00000668, 0.31273082, 0.22737861],
    [0.07366603, 0.82336694, 0.35710457],
    [0.0, 0.32015631, 0.91871858],
    [1.00000668, 0.82532281, 0.4216533],
    [0.0, 0.64648879, 0.74192148],
    [1.00000668, 0.26519099, 0.71757388],
    [1.00000668, 0.46121615, 0.41961047],
    [0.45602512, 1.00000668, 0.54412365],
    [0.36691344, 0.52807516, 1.00000668],
    [0.0, 0.00027714, 0.00057885],
    [0.99837583, 0.97979468, 0.97226435],
    [0.0, 0.86528254, 0.59874046],
    [0.306537, 0.47502732, 1.00000668],
    [1.00000668, 0.55352724, 0.68153518],
    [0.79178429, 1.00000668, 0.95387048],
    [0.70046693, 0.96905082, 0.6606732],
    [1.00000668, 0.52597624, 0.52408081],
    [0.279322, 0.0, 0.7393949],
    [0.0, 0.94832903, 0.00299544],
    [0.68691552, 0.44302544, 0.18326762],
    [0.97759718, 0.51285893, 0.38094446],
    [0.60087502, 1.00000668, 0.89158171],
    [0.62079126, 0.64159679, 1.00000668],
    [0.57359439, 0.59165019, 1.00000668],
    [0.0, 0.95032865, 0.45186621],
    [0.0, 0.94984525, 0.41417098],
    [0.51329207, 0.83260971, 0.11976411],
    [1.00000668, 0.66961932, 0.78481907],
    [0.3782998, 0.5252865, 0.66359985],
    [0.24189806, 0.92875713, 0.7050916],
    [0.38453346, 0.47123694, 1.00000668],
    [0.86467397, 0.19046387, 0.69540018],
    [0.79200047, 0.96162778, 1.00000668],
];

/// display_p3: SDR-100nit-P3-D65_2.0 then DISPLAY - CIE-XYZ-D65_to_DisplayP3
const EXPECTED_DISPLAY_P3: [[f64; 3]; 40] = [
    [0.00056012, 0.00056012, 0.00056012],
    [0.04796797, 0.04796804, 0.04796797],
    [0.22143641, 0.2214365, 0.22143644],
    [0.34918833, 0.34918863, 0.34918845],
    [0.49467438, 0.49467504, 0.4946748],
    [0.77100194, 0.77100217, 0.77100217],
    [0.92841178, 0.92841232, 0.92841208],
    [0.97298056, 0.33815753, 0.24520659],
    [0.18602294, 0.85026294, 0.32632661],
    [0.0, 0.28730845, 0.92331654],
    [1.00000668, 0.83417636, 0.41596642],
    [0.0, 0.64514709, 0.76476938],
    [1.00000668, 0.24143058, 0.73912573],
    [1.00000668, 0.47858697, 0.42281789],
    [0.54567093, 1.00000668, 0.54506493],
    [0.38209474, 0.51613814, 1.00000668],
    [2.18e-06, 0.00026499, 0.00062054],
    [0.9952727, 0.98038751, 0.97313863],
    [0.0, 0.89413446, 0.5823127],
    [0.31969574, 0.45997316, 1.00000668],
    [1.00000668, 0.56444621, 0.6903196],
    [0.81527084, 1.00000668, 0.95535678],
    [0.75750214, 0.96167773, 0.68996334],
    [1.00000668, 0.54169017, 0.52757484],
    [0.25052238, 0.0, 0.71444941],
    [0.15242878, 0.98683494, 0.0],
    [0.65197486, 0.45389605, 0.23451307],
    [0.92259508, 0.53590798, 0.40894318],
    [0.65643352, 1.00000668, 0.89364344],
    [0.62290591, 0.63957983, 1.00000668],
    [0.57455629, 0.58861804, 1.00000668],
    [0.05797843, 0.98913854, 0.40997145],
    [0.0, 0.99110335, 0.36139357],
    [0.56704438, 0.83985215, 0.19643155],
    [1.00000668, 0.67489427, 0.78888607],
    [0.40929511, 0.52122295, 0.65115243],
    [0.40237623, 0.93313408, 0.71122003],
    [0.38843334, 0.45978743, 1.00000668],
    [0.8037318, 0.24048114, 0.68299598],
    [0.81180364, 0.95804948, 1.00000668],
];

/// rec2020: SDR-100nit-P3-D65_2.0 then DISPLAY - CIE-XYZ-D65_to_REC.1886-REC.2020
const EXPECTED_REC2020: [[f64; 3]; 40] = [
    [0.01520704, 0.01520704, 0.01520704],
    [0.09759998, 0.09760004, 0.09759998],
    [0.26202509, 0.26202515, 0.26202509],
    [0.38311699, 0.3831172, 0.38311705],
    [0.52101851, 0.52101898, 0.5210188],
    [0.78294033, 0.78294051, 0.78294051],
    [0.93214405, 0.9321444, 0.93214422],
    [0.87672246, 0.42972204, 0.28384525],
    [0.47018862, 0.83817601, 0.37940127],
    [0.29410923, 0.33721435, 0.92150176],
    [0.95369864, 0.84752494, 0.45668834],
    [0.38344097, 0.6524542, 0.77562642],
    [0.90517342, 0.37796301, 0.74732625],
    [0.91103601, 0.54307038, 0.45261532],
    [0.68685472, 0.98178005, 0.58059675],
    [0.49401158, 0.54590917, 0.99476945],
    [0.00689048, 0.01099897, 0.01581137],
    [0.99175036, 0.98197627, 0.97463477],
    [0.47588965, 0.87930381, 0.61142385],
    [0.44847411, 0.49474883, 0.9944154],
    [0.92522341, 0.61678511, 0.70413715],
    [0.87026477, 0.992392, 0.95859671],
    [0.81115329, 0.95332283, 0.71175623],
    [0.9186976, 0.59582907, 0.55156296],
    [0.31140834, 0.1350448, 0.72429168],
    [0.51973164, 0.96355581, 0.18334484],
    [0.62613517, 0.49142829, 0.27910262],
    [0.85356778, 0.58320916, 0.44118705],
    [0.76651704, 0.98705202, 0.90132701],
    [0.66906697, 0.66354167, 0.9956702],
    [0.62732637, 0.61662853, 0.99524599],
    [0.51555753, 0.96613777, 0.45967132],
    [0.51109505, 0.96768039, 0.41747063],
    [0.64372814, 0.83443332, 0.26951873],
    [0.94077921, 0.71167946, 0.79787683],
    [0.47769237, 0.54375148, 0.66765428],
    [0.59978545, 0.91912329, 0.73099303],
    [0.48626614, 0.49670058, 0.99439335],
    [0.74255812, 0.34471709, 0.69478476],
    [0.86094892, 0.95501566, 0.99950957],
];

fn expected(target: OutputTarget) -> &'static [[f64; 3]; 40] {
    match target {
        OutputTarget::Viewport => &EXPECTED_VIEWPORT,
        OutputTarget::Srgb => &EXPECTED_SRGB,
        OutputTarget::DisplayP3 => &EXPECTED_DISPLAY_P3,
        OutputTarget::Rec2020 => &EXPECTED_REC2020,
    }
}

/// A chain that does nothing before C12: roll-off out of reach, zero grade. The ACES 2 path applies
/// the display placement `2^DISPLAY_PLACEMENT_EV` before the transform (`docs/02` C12 step 1a) and
/// OpenColorIO's vectors do not include it, so the fused matrix is `2^-k` times the identity: the
/// placement then cancels and the chain is compared with the transform itself.
fn bare(target: OutputTarget) -> ColorChain {
    let g = (-DISPLAY_PLACEMENT_EV).exp2();
    let scale = [[g, 0.0, 0.0], [0.0, g, 0.0], [0.0, 0.0, g]];
    let mut p = ChainParams::new(scale, ToneParams::new(0.5, 0.0, 1.0e6, 0.0).unwrap());
    p.output = target;
    ColorChain::new(&p).unwrap()
}

fn codes_10bit(a: Vec3, b: Vec3) -> f64 {
    (0..3).map(|k| (a[k] - b[k]).abs()).fold(0.0, f64::max) * 1023.0
}

#[test]
fn logc4_to_display_code_values_match_opencolorio_for_every_output() {
    for target in OutputTarget::ALL {
        let chain = bare(target);
        let mut worst = 0.0_f64;
        for (logc4, want) in INPUTS.iter().zip(expected(target)) {
            // The camera-side input is the linear AWG4 value of the LogC4 code.
            let got = chain.apply_pixel(logc4_decode_rgb(*logc4));
            // OpenColorIO leaves the Rec.2020 cyan overshoot (up to 1.0005) unclamped; C13 clamps.
            let want = want.map(|v| v.clamp(0.0, 1.0));
            let d = codes_10bit(got, want);
            assert!(
                d <= TOLERANCE_CODES_10BIT,
                "{target:?} LogC4 {logc4:?}: chain {got:?} vs OpenColorIO {want:?} ({d:.4} codes at 10 bits)"
            );
            worst = worst.max(d);
        }
        println!("{target:?}: worst difference to OpenColorIO {worst:.4} codes at 10 bits");
        assert!(worst < GATE_CODES_10BIT);
    }
}

#[test]
fn the_vectors_tell_the_outputs_apart() {
    // Guards against a vacuous comparison: the four outputs really are different renderings, and
    // the chain reproduces each one's own table, not another's.
    let differ = |a: &[[f64; 3]; 40], b: &[[f64; 3]; 40]| {
        a.iter()
            .zip(b)
            .filter(|(x, y)| codes_10bit(**x, **y) > 4.0)
            .count()
    };
    assert!(differ(&EXPECTED_VIEWPORT, &EXPECTED_SRGB) >= 10);
    assert!(differ(&EXPECTED_VIEWPORT, &EXPECTED_REC2020) >= 10);
    assert!(differ(&EXPECTED_SRGB, &EXPECTED_REC2020) >= 10);
    // Viewport and the Display P3 export are the same rendering.
    assert_eq!(EXPECTED_VIEWPORT, EXPECTED_DISPLAY_P3);
    for (target, other) in [
        (OutputTarget::Viewport, OutputTarget::Srgb),
        (OutputTarget::Srgb, OutputTarget::Rec2020),
        (OutputTarget::Rec2020, OutputTarget::Viewport),
    ] {
        let chain = bare(target);
        let off_table = INPUTS
            .iter()
            .zip(expected(other))
            .filter(|(logc4, want)| {
                codes_10bit(
                    chain.apply_pixel(logc4_decode_rgb(**logc4)),
                    want.map(|v| v.clamp(0.0, 1.0)),
                ) > 4.0
            })
            .count();
        assert!(off_table >= 10, "{target:?} agrees with {other:?}'s table");
    }
}
