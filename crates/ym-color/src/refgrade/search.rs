//! The staged two-start search of `docs/34` section 5.5.
//!
//! Golden section, one parameter at a time, in stages with fixed counts; the need test; the local stage; the joint polish; and
//! round 15's two starts (from neutral, and from the look-only optimum), keeping the lower full objective with ties to neutral.
//! Nothing here is random. The chain's candidate renderer is passed in ([`CandidateRenderer`]) and does the per-pixel work; every
//! statistic is reduced in pixel order, so the same bits come out on one thread and on many. The two independent searches (from
//! neutral, and look-only then from the look) run side by side with [`fit_parallel`]: each is a pure function of its inputs, so the
//! result is the same as that of [`fit`], which runs them one after the other (`docs/34` section 5.7).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, PoisonError};

use super::ELEMENT_NAMES;
use super::measure::{PixelScene, Px, wrap_deg};
use super::objective::{
    CandidateMeasure, Fixed, GATE_OPEN, REFUSED_CLIP, Terms, candidate_measure, clip_term, terms,
};
use super::params::{
    HUE_VS_HUE_START, N_GLOBAL, N_LOCAL, N_PARAMS, Numbers, Params, SATURATION_INDEX,
    SUBJECT_INDEX, to_numbers,
};

/// A chain that could not make the candidate's picture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Refused;

/// The chain's candidate renderer: the photo's pixels from the node's input, through the node at the candidate's numbers, every
/// per-pixel step after it, the Output and the display encode (`docs/34` section 2.2). It runs the chain on the 8192 sample pixels
/// and returns each as a measured pixel ([`Px`]); the order of the pixels is the sample's, whatever the threads.
pub trait CandidateRenderer {
    /// The number of sample pixels.
    fn sample_len(&self) -> usize;

    /// Renders the sample with the node at `x` into `out` (`out.len()` is [`CandidateRenderer::sample_len`]).
    ///
    /// # Errors
    ///
    /// [`Refused`] when the chain cannot make the picture (a non-finite result).
    fn render(&mut self, x: &Params, out: &mut [Px]) -> Result<(), Refused>;
}

/// Why a fit wrote nothing.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FitError {
    /// The person pressed Cancel.
    #[error("The fit was cancelled.")]
    Cancelled,
    /// The chain could not make the photo's picture with the node at its identity.
    #[error("The photo could not be rendered for the fit.")]
    ChainRefused,
    /// The objective was not a number where it must be.
    #[error("The fit did not give a number.")]
    NotFinite,
}

/// The stages of one search, as the progress says them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Tone: the luma curve.
    Tone,
    /// Colour per band: the red, green and blue curves.
    Colour,
    /// Saturation and `hue_vs_sat`.
    Saturation,
    /// `hue_vs_hue`.
    Hue,
    /// The elements' local grades.
    Local,
    /// The joint polish.
    Polish,
}

impl Stage {
    /// Every stage, in order.
    pub const ALL: [Stage; 6] = [
        Stage::Tone,
        Stage::Colour,
        Stage::Saturation,
        Stage::Hue,
        Stage::Local,
        Stage::Polish,
    ];

    /// The stage's name for the log and the report.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Stage::Tone => "tone",
            Stage::Colour => "colour per band",
            Stage::Saturation => "saturation and hue_vs_sat",
            Stage::Hue => "hue_vs_hue",
            Stage::Local => "local grades",
            Stage::Polish => "joint polish",
        }
    }
}

/// Where the fit is (`docs/34` section 5.7): which third of the fit and which stage of it. With the searches run one after the other
/// ([`fit`]) these are the search and its stage; with two side by side ([`fit_parallel`]) the stages of both are counted together,
/// so the line never goes back.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FitProgress {
    /// The search, 1 to 3: from neutral, the look-only search, and the full objective from the look (the third of the fit).
    pub search: u8,
    /// The number of searches (3).
    pub searches: u8,
    /// The stage of the search.
    pub stage: Stage,
    /// The candidates evaluated so far.
    pub evals: usize,
}

impl FitProgress {
    /// How far the fit is, 0 to 1, by stage: an estimate (the stages differ in cost; the fraction does not go back).
    #[must_use]
    pub fn fraction(&self) -> f64 {
        let stage = Stage::ALL
            .iter()
            .position(|&s| s == self.stage)
            .unwrap_or(0);
        let done = f64::from(self.search - 1) * 6.0 + stage as f64;
        (done / (f64::from(self.searches) * 6.0)).clamp(0.0, 1.0)
    }
}

/// Which start a search began from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Start {
    /// From `x = 0`, on the full objective.
    Neutral,
    /// From the look-only optimum, on the full objective.
    Look,
}

/// What the need test said (`docs/34` section 5.5, step 2), element by element: sky, vegetation, water, ground, built, skin, subject.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Need {
    /// Which elements got a local grade searched.
    pub local: [bool; 7],
}

/// One stage's line of the log.
#[derive(Debug, Clone, PartialEq)]
pub struct StageLog {
    /// The stage.
    pub stage: Stage,
    /// The half range of the pass, in numbers' units.
    pub half_range: f64,
    /// The golden-section steps of the pass after the first two points.
    pub steps: usize,
    /// The objective before the pass.
    pub before: f64,
    /// The objective after it.
    pub after: f64,
}

/// One search's result.
#[derive(Debug, Clone)]
pub struct SearchReport {
    /// The numbers it ended at.
    pub x: Params,
    /// The need test's answer after its global stages.
    pub need: Need,
    /// The terms at the start (the objective under which the search ran, or the full one for the look-only search).
    pub before: Terms,
    /// The terms at the end under the **full** objective.
    pub after: Terms,
    /// Its stages.
    pub log: Vec<StageLog>,
}

/// The fit's result.
#[derive(Debug, Clone)]
pub struct FitResult {
    /// The numbers (all 0: the identity).
    pub x: Params,
    /// The numbers in the node's units.
    pub numbers: Numbers,
    /// Whether the result is the identity: no better than leaving the photo alone.
    pub identity: bool,
    /// Which start won.
    pub start: Start,
    /// The full objective at the identity.
    pub identity_total: f64,
    /// The full objective of the search from neutral.
    pub neutral_total: f64,
    /// The full objective of the search from the look.
    pub look_total: f64,
    /// The full objective of the result.
    pub total: f64,
    /// The result's terms.
    pub terms: Terms,
    /// The need test of the chosen search.
    pub need: Need,
    /// The three searches, in order (neutral, look-only, from the look).
    pub searches: Vec<SearchReport>,
    /// The candidates evaluated.
    pub evals: usize,
}

/// A pass of golden section: (half range, steps).
type Pass = (f64, usize);

/// The global stages: each stage's parameters and its two passes (`docs/34` section 5.5, step 1).
const GLOBAL_STAGES: [(Stage, std::ops::Range<usize>, [Pass; 2]); 4] = [
    (Stage::Tone, 0..8, [(1.0, 8), (0.4, 6)]),
    (Stage::Colour, 8..17, [(1.0, 7), (0.4, 6)]),
    (
        Stage::Saturation,
        SATURATION_INDEX..HUE_VS_HUE_START,
        [(1.0, 7), (0.4, 6)],
    ),
    (Stage::Hue, HUE_VS_HUE_START..N_GLOBAL, [(1.0, 7), (0.4, 6)]),
];
/// The local stage's passes.
const LOCAL_PASSES: [Pass; 2] = [(1.0, 7), (0.4, 6)];
/// The joint polish's pass.
const POLISH: Pass = (0.25, 6);
/// The need test's thresholds: hue (degrees), log chroma and exposure (L\*).
const NEED_HUE_DEG: f64 = 6.0;
const NEED_LOGC: f64 = 0.14;
const NEED_EXPOSURE_L: f64 = 4.0;

/// What the search reads about the photo besides the sample: the photo's scene at the sample and which elements have a mask of at
/// least 2 % of the picture.
pub struct FitSetup<'a> {
    /// The fixed part of the objective.
    pub fixed: &'a Fixed,
    /// The photo's scene at the sample pixels.
    pub scene: &'a PixelScene,
    /// Which of the seven element grades have a mask of at least 2 % of the picture (sky ... skin, subject).
    pub mask_ok: [bool; 7],
}

/// Where a search tells the fit it started a stage and evaluated a candidate.
trait Announce {
    /// A stage of a search began.
    fn stage_started(&self);
    /// One candidate was evaluated (for the progress's count).
    fn evaluated(&self);
}

/// The progress as the person sees it: **one line for the whole fit**, whichever searches run side by side. The `n`-th stage
/// started, counting both searches' stages together, is stage `n % 6` of "search" `n / 6 + 1` (three searches of six stages: the
/// fit's three thirds), so the line never goes back and ends at "3 of 3" (`docs/34` section 5.7).
struct Units<F> {
    /// The callback and how many stages have started.
    state: Mutex<(F, usize)>,
    /// The candidates evaluated so far, by every search.
    evals: AtomicUsize,
}

impl<F: FnMut(FitProgress)> Units<F> {
    fn new(sink: F) -> Self {
        Self {
            state: Mutex::new((sink, 0)),
            evals: AtomicUsize::new(0),
        }
    }
}

impl<F: FnMut(FitProgress)> Announce for Units<F> {
    fn stage_started(&self) {
        let mut g = self.state.lock().unwrap_or_else(PoisonError::into_inner);
        let unit = g.1.min(UNITS - 1);
        g.1 += 1;
        let p = FitProgress {
            search: (unit / Stage::ALL.len() + 1) as u8,
            searches: 3,
            stage: Stage::ALL[unit % Stage::ALL.len()],
            evals: self.evals.load(Ordering::Relaxed),
        };
        (g.0)(p);
    }

    fn evaluated(&self) {
        self.evals.fetch_add(1, Ordering::Relaxed);
    }
}

/// The stages of the whole fit: three searches of six stages.
const UNITS: usize = 18;

struct Evaluator<'a, 'c> {
    renderer: &'a mut dyn CandidateRenderer,
    setup: &'a FitSetup<'a>,
    look_only: bool,
    evals: usize,
    buf: Vec<Px>,
    cancel: &'c dyn Fn() -> bool,
    announce: &'c dyn Announce,
}

impl Evaluator<'_, '_> {
    fn announce(&mut self) {
        self.announce.stage_started();
    }

    /// The candidate's terms and measures; a refused candidate has the refused clip term and no measures.
    fn terms_of(&mut self, x: &Params) -> Result<(Terms, Option<CandidateMeasure>), FitError> {
        if (self.cancel)() {
            return Err(FitError::Cancelled);
        }
        let reg_global = x[..N_GLOBAL].iter().map(|v| v * v).sum::<f64>() / N_GLOBAL as f64;
        let reg_local = x[N_GLOBAL..].iter().map(|v| v * v).sum::<f64>() / N_LOCAL as f64;
        let mut t = Terms {
            reg_global,
            reg_local,
            ..Terms::default()
        };
        if self.renderer.render(x, &mut self.buf).is_err() {
            t.clip = REFUSED_CLIP;
            return Ok((t, None));
        }
        let cm = candidate_measure(&self.buf, self.setup.scene, self.setup.fixed);
        let mut full = terms(&cm, self.setup.fixed);
        full.reg_global = reg_global;
        full.reg_local = reg_local;
        full.clip = clip_term(self.setup.fixed, &cm.m);
        Ok((full, Some(cm)))
    }

    /// The objective the search minimises, and the candidate's terms.
    fn objective(&mut self, x: &Params) -> Result<(f64, Terms), FitError> {
        let (t, _) = self.terms_of(x)?;
        self.evals += 1;
        self.announce.evaluated();
        let total = if self.look_only {
            t.look_only_total()
        } else {
            t.total()
        };
        Ok((total, t))
    }
}

/// Golden section of one coordinate over `[x_i - half, x_i + half]` clipped to -1 to 1: two interior points, then `iters` more
/// steps; keeps the best of every point seen, the start included, so a step never raises the objective and a tie keeps the start.
fn golden(
    ev: &mut Evaluator<'_, '_>,
    x: &mut Params,
    i: usize,
    (half, iters): Pass,
    current: f64,
) -> Result<f64, FitError> {
    const INV_PHI: f64 = 0.618_033_988_749_894_9;
    let x0 = x[i];
    let (mut a, mut b) = ((x0 - half).max(-1.0), (x0 + half).min(1.0));
    let eval_at = |ev: &mut Evaluator<'_, '_>, x: &Params, v: f64| -> Result<f64, FitError> {
        let mut t = *x;
        t[i] = v;
        Ok(ev.objective(&t)?.0)
    };
    let (mut c, mut d) = (b - INV_PHI * (b - a), a + INV_PHI * (b - a));
    let (mut fc, mut fd) = (eval_at(ev, x, c)?, eval_at(ev, x, d)?);
    let mut best = (x0, current);
    for (v, f) in [(c, fc), (d, fd)] {
        if f < best.1 {
            best = (v, f);
        }
    }
    for _ in 0..iters {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INV_PHI * (b - a);
            fc = eval_at(ev, x, c)?;
            if fc < best.1 {
                best = (c, fc);
            }
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INV_PHI * (b - a);
            fd = eval_at(ev, x, d)?;
            if fd < best.1 {
                best = (d, fd);
            }
        }
    }
    x[i] = best.0;
    Ok(best.1)
}

/// The need test (`docs/34` section 5.5, step 2): which elements get a local grade after the global stages.
fn need_test(ev: &mut Evaluator<'_, '_>, x: &Params) -> Result<Need, FitError> {
    let (_, cm) = ev.terms_of(x)?;
    let mut need = Need::default();
    let Some(cm) = cm else {
        return Ok(need);
    };
    let o = &ev.setup.fixed.t;
    let mask_ok = ev.setup.mask_ok;
    for k in 0..6 {
        let matched = o.matched[k] > GATE_OPEN;
        let hue_on = o.hue_gate[k] > GATE_OPEN;
        let exposure_on = o.exposure_gate[k] > GATE_OPEN;
        let em = &cm.m.el[k];
        let hue_err = if hue_on && em.has_hue {
            wrap_deg(em.h - o.el_h[k]).abs()
        } else {
            0.0
        };
        let logc_err = if matched {
            (em.logc - o.el_logc[k]).abs()
        } else {
            0.0
        };
        let exposure_err = if exposure_on {
            ((em.l - cm.m.p50) - o.el_off[k]).abs()
        } else {
            0.0
        };
        let want = matched
            && (hue_err > NEED_HUE_DEG || logc_err > NEED_LOGC || exposure_err > NEED_EXPOSURE_L);
        need.local[k] = want && mask_ok[k];
    }
    let sep_err = if o.subj > GATE_OPEN {
        (cm.m.sep - o.sep).abs()
    } else {
        0.0
    };
    need.local[6] = sep_err > NEED_EXPOSURE_L && mask_ok[6];
    Ok(need)
}

/// The search from `x0`: the global stages, the need test, the local stage, the joint polish.
fn search_from(ev: &mut Evaluator<'_, '_>, x0: Params) -> Result<SearchReport, FitError> {
    let mut x = x0;
    let (_, before) = ev.objective(&x)?;
    let mut log = Vec::new();
    let mut f;
    for (stage, range, passes) in &GLOBAL_STAGES {
        ev.announce();
        for &(half, steps) in passes {
            // Every stage sees the whole objective, so each pass can only lower it.
            f = ev.objective(&x)?.0;
            let start = f;
            for i in range.clone() {
                f = golden(ev, &mut x, i, (half, steps), f)?;
            }
            log.push(StageLog {
                stage: *stage,
                half_range: half,
                steps,
                before: start,
                after: f,
            });
        }
    }
    let f_global = ev.objective(&x)?.0;
    let need = need_test(ev, &x)?;
    let mut local_idx: Vec<usize> = Vec::new();
    for (k, &n) in need.local.iter().enumerate() {
        if n {
            if k < 6 {
                local_idx.extend([N_GLOBAL + 3 * k, N_GLOBAL + 3 * k + 1, N_GLOBAL + 3 * k + 2]);
            } else {
                local_idx.push(SUBJECT_INDEX);
            }
        }
    }
    ev.announce();
    f = f_global;
    for &(half, steps) in &LOCAL_PASSES {
        let start = f;
        for &i in &local_idx {
            f = golden(ev, &mut x, i, (half, steps), f)?;
        }
        log.push(StageLog {
            stage: Stage::Local,
            half_range: half,
            steps,
            before: start,
            after: f,
        });
    }
    ev.announce();
    let start = f;
    let polish: Vec<usize> = (0..N_GLOBAL).chain(local_idx.iter().copied()).collect();
    for i in polish {
        f = golden(ev, &mut x, i, POLISH, f)?;
    }
    log.push(StageLog {
        stage: Stage::Polish,
        half_range: POLISH.0,
        steps: POLISH.1,
        before: start,
        after: f,
    });
    // The result's terms under the full objective, whichever objective the search ran on.
    let look = ev.look_only;
    ev.look_only = false;
    let after = ev.objective(&x);
    ev.look_only = look;
    let (_, after) = after?;
    Ok(SearchReport {
        x,
        need,
        before,
        after,
        log,
    })
}

/// The three searches' reports and the candidates they evaluated, as the two ways of running them make them.
struct Searches {
    neutral: SearchReport,
    look_only: SearchReport,
    from_look: SearchReport,
    evals: usize,
}

/// Makes the fit's result from the three searches: the lower full objective wins, a tie keeps the neutral search.
fn result_of(s: Searches) -> Result<FitResult, FitError> {
    let Searches {
        neutral,
        look_only,
        from_look,
        evals,
    } = s;
    let (neutral_total, look_total) = (neutral.after.total(), from_look.after.total());
    let identity_total = neutral.before.total();
    let use_look = look_total < neutral_total;
    let (chosen, start) = if use_look {
        (&from_look, Start::Look)
    } else {
        (&neutral, Start::Neutral)
    };
    let total = chosen.after.total();
    if !total.is_finite() || !identity_total.is_finite() {
        return Err(FitError::NotFinite);
    }
    let x = chosen.x;
    let numbers = to_numbers(&x);
    let result = FitResult {
        x,
        numbers,
        identity: x.iter().all(|&v| v == 0.0),
        start,
        identity_total,
        neutral_total,
        look_total,
        total,
        terms: chosen.after,
        need: chosen.need,
        evals,
        searches: vec![neutral.clone(), look_only.clone(), from_look.clone()],
    };
    Ok(result)
}

/// An evaluator on `renderer` for one of the searches.
fn evaluator<'a, 'c>(
    renderer: &'a mut dyn CandidateRenderer,
    setup: &'a FitSetup<'a>,
    cancel: &'c dyn Fn() -> bool,
    announce: &'c dyn Announce,
) -> Evaluator<'a, 'c> {
    let n = renderer.sample_len();
    Evaluator {
        renderer,
        setup,
        look_only: false,
        evals: 0,
        buf: vec![Px::default(); n],
        cancel,
        announce,
    }
}

/// The photo at the identity must be renderable: a chain that refuses it cannot be fitted.
fn check_identity(ev: &mut Evaluator<'_, '_>) -> Result<(), FitError> {
    let zero = [0.0; N_PARAMS];
    if ev.renderer.render(&zero, &mut ev.buf).is_err() {
        return Err(FitError::ChainRefused);
    }
    Ok(())
}

/// Runs the fit: searches twice (from neutral, and from the look-only optimum) and keeps the result with the lower full objective;
/// a tie keeps the neutral search. `cancel` is asked before every candidate; `progress` is told at the start of every stage. The
/// searches run one after the other on `renderer`; [`fit_parallel`] gives the same result on two renderers at once.
///
/// # Errors
///
/// [`FitError::Cancelled`] when `cancel` says so (nothing is changed: the caller writes nothing), [`FitError::ChainRefused`] when the
/// photo cannot be rendered with the node at its identity, [`FitError::NotFinite`] when the result's objective is not a number.
pub fn fit(
    setup: &FitSetup<'_>,
    renderer: &mut dyn CandidateRenderer,
    cancel: &dyn Fn() -> bool,
    progress: &mut dyn FnMut(FitProgress),
) -> Result<FitResult, FitError> {
    let units = Units::new(progress);
    let mut ev = evaluator(renderer, setup, cancel, &units);
    check_identity(&mut ev)?;
    let zero = [0.0; N_PARAMS];
    let neutral = search_from(&mut ev, zero)?;
    ev.look_only = true;
    let look_only = search_from(&mut ev, zero)?;
    ev.look_only = false;
    let from_look = search_from(&mut ev, look_only.x)?;
    let evals = ev.evals;
    result_of(Searches {
        neutral,
        look_only,
        from_look,
        evals,
    })
}

/// [`fit`] with the two independent searches side by side (`docs/34` section 5.7): the search from neutral on `neutral_renderer`,
/// the look-only search and then the search from the look on `look_renderer`. Every search is a pure function of the fixed inputs and
/// of its own start, and each renderer gives the same pixels for the same numbers, so **the result is the same, bit for bit, as
/// [`fit`]'s** (and the same on any number of threads; the candidates' own pixels are split in fixed chunks by the renderers). The
/// candidates evaluated are the sum of both sides'. `progress` is one line for the whole fit (see [`FitProgress`]); it is called
/// from either side, one call at a time. A cancel, or the failure of one side, stops the other.
///
/// # Errors
///
/// As [`fit`].
pub fn fit_parallel(
    setup: &FitSetup<'_>,
    neutral_renderer: &mut (dyn CandidateRenderer + Send),
    look_renderer: &mut (dyn CandidateRenderer + Send),
    cancel: &(dyn Fn() -> bool + Sync),
    progress: &mut (dyn FnMut(FitProgress) + Send),
) -> Result<FitResult, FitError> {
    let units = Units::new(progress);
    let stop = AtomicBool::new(false);
    let cancelled = || stop.load(Ordering::Relaxed) || cancel();
    // The photo at the identity must be renderable, before anything runs.
    {
        let mut ev = evaluator(&mut *neutral_renderer, setup, &cancelled, &units);
        check_identity(&mut ev)?;
    }
    let zero = [0.0; N_PARAMS];
    let (neutral, looked) = rayon::join(
        || {
            let mut ev = evaluator(neutral_renderer, setup, &cancelled, &units);
            let r = search_from(&mut ev, zero);
            if r.is_err() {
                stop.store(true, Ordering::Relaxed);
            }
            r.map(|r| (r, ev.evals))
        },
        || {
            let mut ev = evaluator(look_renderer, setup, &cancelled, &units);
            ev.look_only = true;
            let run = (|| {
                let look_only = search_from(&mut ev, zero)?;
                ev.look_only = false;
                let from_look = search_from(&mut ev, look_only.x)?;
                Ok((look_only, from_look, ev.evals))
            })();
            if run.is_err() {
                stop.store(true, Ordering::Relaxed);
            }
            run
        },
    );
    let (neutral, neutral_evals) = neutral?;
    let (look_only, from_look, look_evals) = looked?;
    result_of(Searches {
        neutral,
        look_only,
        from_look,
        evals: neutral_evals + look_evals,
    })
}

/// The full objective of the numbers `x` and its terms, as the search scores a candidate (for the evaluation tool, which scores a solution
/// that was made elsewhere under this objective). A candidate the chain refuses has the refused clip term.
///
/// # Errors
///
/// [`FitError::ChainRefused`] when the photo cannot be rendered at the identity.
pub fn score(
    setup: &FitSetup<'_>,
    renderer: &mut dyn CandidateRenderer,
    x: &Params,
) -> Result<(f64, Terms), FitError> {
    let units = Units::new(|_: FitProgress| {});
    let mut ev = evaluator(renderer, setup, &|| false, &units);
    check_identity(&mut ev)?;
    ev.objective(x)
}

/// The element names in the order of the local numbers and the need test.
pub const LOCAL_NAMES: [&str; 7] = [
    ELEMENT_NAMES[0],
    ELEMENT_NAMES[1],
    ELEMENT_NAMES[2],
    ELEMENT_NAMES[3],
    ELEMENT_NAMES[4],
    ELEMENT_NAMES[5],
    "subject",
];
