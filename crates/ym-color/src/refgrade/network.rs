//! The hand-written network of the Reference Grade fit (`docs/34-reference-grade.md` section 4b.5).
//!
//! A text file, `data/refgrade/network.ywn`, holds one neuron per line: `name = relu( bias + w*node - node ... )` or
//! `lin( ... )`. This module reads it (with its constants, templates and `apply` lines), refuses a file that breaks a rule with the
//! line and the name, and evaluates it in `f64` in file order. Every weight in it was written by hand; nothing is trained.
//!
//! The file is round 15's, which is the proof of concept's v14 file unchanged. Its SHA-256 and its expanded text are pinned by a
//! test, and a weight changes only by a spec change (`docs/34` section 4b.5).

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use super::{ELEMENT_NAMES, SECTORS};

/// The network's file, compiled into the library (the app and `ym` both ship it).
pub const NETWORK_TEXT: &str = include_str!("../../../../data/refgrade/network.ywn");

/// Why a network file was refused, with where.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkError {
    /// The line of the file (1 is the first). For a neuron made by `apply`, the line of the `apply`.
    pub line: usize,
    /// The neuron's name, when the line got as far as one.
    pub name: Option<String>,
    /// What is wrong.
    pub kind: NetworkErrorKind,
}

/// What is wrong with a line of a network file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetworkErrorKind {
    /// The line has no `=`.
    NoEquals,
    /// The right side is not `act( ... )`.
    NoParentheses,
    /// An activation that is not `relu` or `lin`.
    UnknownActivation(String),
    /// The first number (the bias) is missing or is not a finite number.
    BadBias(String),
    /// A sign (`+` or `-`) was expected and something else was found.
    ExpectedSign(String),
    /// A sign with no term after it.
    SignWithoutTerm,
    /// A term that is not `number*node` or `node` (the weight is not a finite number, or the node is empty).
    BadTerm(String),
    /// A node that is neither an input nor defined on an earlier line.
    UnknownNode(String),
    /// The name is already an input or a neuron.
    DuplicateName,
    /// A `const` with no name or a part that is not `key=value`.
    BadConst(String),
    /// A `template` with no `(var)`.
    BadTemplate(String),
    /// An `apply` of a template that was not defined.
    UnknownTemplate(String),
    /// A template value with no constant for a `{key}` of the body.
    MissingConstant(String),
    /// A `template` that has no `end`.
    TemplateWithoutEnd,
    /// An `apply` with no template name.
    ApplyWithoutName,
    /// A `T.*` or `G.*` output the fit reads is not in the file.
    MissingOutput(String),
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use NetworkErrorKind as K;
        write!(f, "network line {}", self.line)?;
        if let Some(n) = &self.name {
            write!(f, ", {n}")?;
        }
        write!(f, ": ")?;
        match &self.kind {
            K::NoEquals => write!(f, "no '='"),
            K::NoParentheses => write!(f, "no activation with parentheses"),
            K::UnknownActivation(a) => write!(f, "unknown activation {a}"),
            K::BadBias(t) => write!(f, "the bias is not a number: {t}"),
            K::ExpectedSign(t) => write!(f, "expected + or -, got {t}"),
            K::SignWithoutTerm => write!(f, "a sign with no term"),
            K::BadTerm(t) => write!(f, "{t} is not a number times a node"),
            K::UnknownNode(n) => write!(
                f,
                "refers to {n}, which is not an input and not defined earlier"
            ),
            K::DuplicateName => write!(f, "the name is defined twice"),
            K::BadConst(t) => write!(f, "bad constant row: {t}"),
            K::BadTemplate(t) => write!(f, "bad template line: {t}"),
            K::UnknownTemplate(t) => write!(f, "no template called {t}"),
            K::MissingConstant(t) => write!(f, "constant {t} is missing"),
            K::TemplateWithoutEnd => write!(f, "a template has no `end`"),
            K::ApplyWithoutName => write!(f, "`apply` without a template name"),
            K::MissingOutput(n) => write!(
                f,
                "the fit needs the output {n}, which the file does not define"
            ),
        }
    }
}

impl std::error::Error for NetworkError {}

/// Every input name the network may read, in the order [`Network::eval`] takes the values (`docs/34` section 4b.5).
#[must_use]
pub fn input_names() -> Vec<String> {
    let mut v = Vec::new();
    for x in ["P", "R"] {
        for n in [
            "p2",
            "p50",
            "p98",
            "iqr_rel",
            "logsat",
            "subj",
            "sep",
            "cast_a",
            "cast_b",
            "conc",
            "magenta_share",
        ] {
            v.push(format!("{x}.{n}"));
        }
        for b in ["s", "m", "h"] {
            for c in ["a", "b"] {
                v.push(format!("{x}.tint_{b}_{c}"));
            }
        }
        for e in ELEMENT_NAMES {
            for n in ["share", "L", "h", "C", "logC"] {
                v.push(format!("{x}.{e}.{n}"));
            }
        }
        for j in 0..SECTORS {
            v.push(format!("{x}.s{j}.share"));
            v.push(format!("{x}.s{j}.logC"));
        }
    }
    for e in ELEMENT_NAMES {
        v.push(format!("D.{e}.dh"));
    }
    for j in 0..SECTORS {
        v.push(format!("D.s{j}.dh"));
    }
    v
}

/// Where a term reads from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    /// The input with this number.
    Input(usize),
    /// The neuron with this number (defined earlier).
    Neuron(usize),
}

/// One neuron: `name = act(bias + sum w * node)`.
#[derive(Debug, Clone)]
struct Neuron {
    name: String,
    relu: bool,
    bias: f64,
    terms: Vec<(f64, Source)>,
}

/// The hand-written network, expanded and resolved.
#[derive(Debug, Clone)]
pub struct Network {
    neurons: Vec<Neuron>,
    index: BTreeMap<String, usize>,
    inputs: Vec<String>,
}

/// One line of the expanded file, with the line of the source it came from.
struct Line {
    text: String,
    source: usize,
}

fn err(line: usize, name: Option<&str>, kind: NetworkErrorKind) -> NetworkError {
    NetworkError {
        line,
        name: name.map(str::to_owned),
        kind,
    }
}

/// A finite decimal number.
fn number(t: &str) -> Option<f64> {
    t.parse::<f64>().ok().filter(|v| v.is_finite())
}

/// One neuron line, against the inputs and the neurons defined so far.
fn parse_neuron(
    line: &Line,
    inputs: &BTreeMap<String, usize>,
    defined: &BTreeMap<String, usize>,
) -> Result<Neuron, NetworkError> {
    let at = line.source;
    let (name, rest) = line
        .text
        .split_once('=')
        .ok_or_else(|| err(at, None, NetworkErrorKind::NoEquals))?;
    let name = name.trim().to_owned();
    let rest = rest.trim();
    let named = |kind| err(at, Some(&name), kind);
    let (open, close) = match (rest.find('('), rest.rfind(')')) {
        (Some(o), Some(c)) if o < c => (o, c),
        _ => return Err(named(NetworkErrorKind::NoParentheses)),
    };
    let relu = match rest[..open].trim() {
        "relu" => true,
        "lin" => false,
        other => return Err(named(NetworkErrorKind::UnknownActivation(other.to_owned()))),
    };
    let toks: Vec<&str> = rest[open + 1..close].split_whitespace().collect();
    let first = toks.first().copied().unwrap_or("");
    let bias = number(first).ok_or_else(|| named(NetworkErrorKind::BadBias(first.to_owned())))?;
    let mut terms = Vec::new();
    let mut i = 1;
    while i < toks.len() {
        let sign = match toks[i] {
            "+" => 1.0,
            "-" => -1.0,
            other => return Err(named(NetworkErrorKind::ExpectedSign(other.to_owned()))),
        };
        let t = *toks
            .get(i + 1)
            .ok_or_else(|| named(NetworkErrorKind::SignWithoutTerm))?;
        let (w, node) = match t.split_once('*') {
            Some((w, n)) => (
                number(w).ok_or_else(|| named(NetworkErrorKind::BadTerm(t.to_owned())))?,
                n,
            ),
            None => (1.0, t),
        };
        if node.is_empty() || node.contains('*') {
            return Err(named(NetworkErrorKind::BadTerm(t.to_owned())));
        }
        let source = match (defined.get(node), inputs.get(node)) {
            (Some(&n), _) => Source::Neuron(n),
            (None, Some(&n)) => Source::Input(n),
            (None, None) => return Err(named(NetworkErrorKind::UnknownNode(node.to_owned()))),
        };
        terms.push((sign * w, source));
        i += 2;
    }
    Ok(Neuron {
        name,
        relu,
        bias,
        terms,
    })
}

impl Network {
    /// Reads and expands a network file (constants, templates, `apply`). A file that breaks a rule is refused with its line and the
    /// name, never patched: a node not defined earlier, a template value with a missing constant, a duplicate name (an input's too),
    /// an unknown activation, a template without `end`, a term that is not a number times a node.
    ///
    /// # Errors
    ///
    /// [`NetworkError`] for the first line that breaks a rule.
    pub fn parse(text: &str) -> Result<Self, NetworkError> {
        let inputs = input_names();
        let input_index: BTreeMap<String, usize> = inputs
            .iter()
            .enumerate()
            .map(|(i, n)| (n.clone(), i))
            .collect();
        let mut consts: BTreeMap<String, Vec<(String, String)>> = BTreeMap::new();
        let mut templates: BTreeMap<String, (String, Vec<String>)> = BTreeMap::new();
        let mut current: Option<(String, String, Vec<String>, usize)> = None;
        let mut lines: Vec<Line> = Vec::new();
        for (n, raw) in text.lines().enumerate() {
            let at = n + 1;
            let l = raw.trim();
            if l.is_empty() || l.starts_with('#') {
                continue;
            }
            if let Some((name, var, body, _)) = current.as_mut() {
                if l == "end" {
                    templates.insert(name.clone(), (var.clone(), std::mem::take(body)));
                    current = None;
                } else {
                    body.push(l.to_owned());
                }
                continue;
            }
            if let Some(rest) = l.strip_prefix("const ") {
                let mut it = rest.split_whitespace();
                let name = it
                    .next()
                    .ok_or_else(|| err(at, None, NetworkErrorKind::BadConst(l.to_owned())))?
                    .to_owned();
                let kv = it
                    .map(|p| {
                        p.split_once('=')
                            .map(|(k, v)| (k.to_owned(), v.to_owned()))
                            .ok_or_else(|| {
                                err(at, Some(&name), NetworkErrorKind::BadConst(p.to_owned()))
                            })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                consts.insert(name, kv);
            } else if let Some(rest) = l.strip_prefix("template ") {
                let rest = rest.trim();
                let bad = || err(at, None, NetworkErrorKind::BadTemplate(l.to_owned()));
                let open = rest.find('(').ok_or_else(bad)?;
                let close = rest.find(')').ok_or_else(bad)?;
                if open > close {
                    return Err(bad());
                }
                current = Some((
                    rest[..open].trim().to_owned(),
                    rest[open + 1..close].trim().to_owned(),
                    Vec::new(),
                    at,
                ));
            } else if let Some(rest) = l.strip_prefix("apply ") {
                let mut it = rest.split_whitespace();
                let tname = it
                    .next()
                    .ok_or_else(|| err(at, None, NetworkErrorKind::ApplyWithoutName))?;
                let (var, body) = templates.get(tname).ok_or_else(|| {
                    err(
                        at,
                        None,
                        NetworkErrorKind::UnknownTemplate(tname.to_owned()),
                    )
                })?;
                for value in it {
                    for b in body {
                        let mut s = b.replace(&format!("{{{var}}}"), value);
                        if let Some(kv) = consts.get(value) {
                            for (k, v) in kv {
                                s = s.replace(&format!("{{{k}}}"), v);
                            }
                        }
                        if let Some(p) = s.find('{') {
                            let end = s[p..].find('}').map_or(s.len(), |e| p + e + 1);
                            let name = s.split_once('=').map(|(n, _)| n.trim().to_owned());
                            return Err(err(
                                at,
                                name.as_deref(),
                                NetworkErrorKind::MissingConstant(format!(
                                    "{} for {value} in template {tname}",
                                    &s[p..end]
                                )),
                            ));
                        }
                        lines.push(Line {
                            text: s,
                            source: at,
                        });
                    }
                }
            } else {
                lines.push(Line {
                    text: l.to_owned(),
                    source: at,
                });
            }
        }
        if let Some((name, _, _, at)) = current {
            return Err(err(at, Some(&name), NetworkErrorKind::TemplateWithoutEnd));
        }
        let mut neurons: Vec<Neuron> = Vec::new();
        let mut index: BTreeMap<String, usize> = BTreeMap::new();
        for l in &lines {
            let n = parse_neuron(l, &input_index, &index)?;
            if index.contains_key(&n.name) || input_index.contains_key(&n.name) {
                return Err(err(
                    l.source,
                    Some(&n.name),
                    NetworkErrorKind::DuplicateName,
                ));
            }
            index.insert(n.name.clone(), neurons.len());
            neurons.push(n);
        }
        Ok(Self {
            neurons,
            index,
            inputs,
        })
    }

    /// The network of `data/refgrade/network.ywn`, the file the library ships.
    ///
    /// # Errors
    ///
    /// [`NetworkError`] if the compiled-in file is broken (a test pins that it is not).
    pub fn shipped() -> Result<Self, NetworkError> {
        Self::parse(NETWORK_TEXT)
    }

    /// The number of neurons after expansion.
    #[must_use]
    pub fn len(&self) -> usize {
        self.neurons.len()
    }

    /// Whether the network has no neuron.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.neurons.is_empty()
    }

    /// The input names, in the order [`Network::eval`] takes their values.
    #[must_use]
    pub fn inputs(&self) -> &[String] {
        &self.inputs
    }

    /// The neuron's position, when the network has it.
    #[must_use]
    pub fn position(&self, name: &str) -> Option<usize> {
        self.index.get(name).copied()
    }

    /// The names of the outputs: every neuron called `T.*` or `G.*`, in file order.
    #[must_use]
    pub fn outputs(&self) -> Vec<&str> {
        self.neurons
            .iter()
            .filter(|n| n.name.starts_with("T.") || n.name.starts_with("G."))
            .map(|n| n.name.as_str())
            .collect()
    }

    /// Evaluates every neuron in file order, in `f64`. `inputs` holds the value of each input of [`Network::inputs`], in that order.
    ///
    /// # Panics
    ///
    /// When `inputs` is not as long as [`Network::inputs`]: a mistake of the caller.
    #[must_use]
    pub fn eval(&self, inputs: &[f64]) -> NetworkValues {
        assert_eq!(inputs.len(), self.inputs.len(), "one value per input");
        let mut vals: Vec<f64> = Vec::with_capacity(self.neurons.len());
        for n in &self.neurons {
            let mut s = n.bias;
            for &(w, node) in &n.terms {
                let v = match node {
                    Source::Input(i) => inputs[i],
                    Source::Neuron(i) => vals[i],
                };
                s += w * v;
            }
            vals.push(if n.relu { s.max(0.0) } else { s });
        }
        NetworkValues {
            index: self.index.clone(),
            vals,
        }
    }

    /// The expanded network, every weight with its number (what a test pins).
    #[must_use]
    pub fn dump(&self) -> String {
        let name_of = |s: Source| match s {
            Source::Input(i) => self.inputs[i].as_str(),
            Source::Neuron(i) => self.neurons[i].name.as_str(),
        };
        let mut out = String::new();
        for n in &self.neurons {
            out.push_str(&format!(
                "{} = {}( {:+}",
                n.name,
                if n.relu { "relu" } else { "lin" },
                n.bias
            ));
            for &(w, node) in &n.terms {
                out.push_str(&format!(" {:+}*{}", w, name_of(node)));
            }
            out.push_str(" )\n");
        }
        out
    }

    /// The set of names that are targets or gates, for a check that the outputs the fit reads are there.
    pub(crate) fn names(&self) -> BTreeSet<&str> {
        self.index.keys().map(String::as_str).collect()
    }
}

/// The values of every neuron for one pair, by name.
#[derive(Debug, Clone)]
pub struct NetworkValues {
    index: BTreeMap<String, usize>,
    vals: Vec<f64>,
}

impl NetworkValues {
    /// The value of the neuron called `name`: 0 when the network has none (the fit checks the names it reads once, up front).
    #[must_use]
    pub fn get(&self, name: &str) -> f64 {
        self.index.get(name).map_or(0.0, |&i| self.vals[i])
    }

    /// Every neuron's value, in file order.
    #[must_use]
    pub fn all(&self) -> &[f64] {
        &self.vals
    }
}
