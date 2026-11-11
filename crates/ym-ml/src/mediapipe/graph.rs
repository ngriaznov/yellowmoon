//! A small runner for the TFLite graphs of the MediaPipe face networks, read from the safetensors files that
//! `tools/mediapipe/convert.py` writes (`docs/27` S-08, `docs/07`).
//!
//! The file holds the constants of the graph (`t<index>`, f16 or f32) and, in its header metadata under `graph`, the
//! operators in execution order with their tensor indices and builtin options. This module implements the operators the
//! networks need, in NHWC like TFLite, and refuses a graph with any other (`MlError::Unsupported`), so a network whose
//! operators are not written yet fails at load time and never at run time.
//!
//! Implemented: `CONV_2D`, `DEPTHWISE_CONV_2D`, `ADD`, `RELU`, `PRELU`, `LOGISTIC`, `MAX_POOL_2D`, `PAD`,
//! `RESIZE_BILINEAR`, `RESHAPE` and `CONCATENATION`: those of BlazeFace and Face Mesh V2. Selfie Multiclass also uses
//! `MUL`, `SUM`, `SOFTMAX`, `TRANSPOSE`, `TRANSPOSE_CONV` and `RESIZE_NEAREST_NEIGHBOR` (S-10).
//!
//! # Determinism
//!
//! Every convolution output is one chain of `f32` operations in a fixed order: the bias, then the taps in (row, column,
//! input channel) order, a multiply and a separate add, never a fused one. The products are the packed matrix product of
//! `sam2/kernels.rs`; the depthwise convolution adds the taps in the same order, with the channels side by side. The
//! work is cut into rows or row blocks, which changes no bit, so the outputs are the same on any number of threads and
//! on any instruction set.

use std::collections::HashMap;

use rayon::prelude::*;
use serde::Deserialize;

use crate::cpu_runtime::read_float_tensors;
use crate::model::{MlError, Result};
use crate::sam2::kernels::{Act, Packed, gemm, gemm_par, relu, sigmoid};

/// A dense `f32` tensor, row-major (NHWC for images).
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// The size of each dimension.
    pub shape: Vec<usize>,
    /// The values.
    pub data: Vec<f32>,
}

impl Tensor {
    /// A tensor of `shape` holding `data`.
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when the number of values is not the product of the shape.
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Result<Self> {
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(MlError::Shape(format!(
                "{} values for shape {shape:?}",
                data.len()
            )));
        }
        Ok(Self { shape, data })
    }
}

/// An input or output of the graph.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct Io {
    /// The TFLite tensor index.
    pub index: usize,
    /// The tensor's name in the TFLite file.
    pub name: String,
    /// Its shape.
    pub shape: Vec<usize>,
}

#[derive(Deserialize)]
struct IntConst {
    shape: Vec<usize>,
    data: Vec<i64>,
}

#[derive(Deserialize)]
struct OpDoc {
    op: String,
    #[serde(rename = "in")]
    ins: Vec<i64>,
    out: Vec<usize>,
    #[serde(default)]
    opt: HashMap<String, serde_json::Value>,
}

#[derive(Deserialize)]
struct Doc {
    version: u32,
    inputs: Vec<Io>,
    outputs: Vec<Io>,
    #[serde(default)]
    ints: HashMap<String, IntConst>,
    ops: Vec<OpDoc>,
}

impl OpDoc {
    fn opt_i(&self, key: &str, default: i64) -> i64 {
        self.opt
            .get(key)
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(default)
    }

    fn opt_b(&self, key: &str) -> bool {
        self.opt.get(key).is_some_and(|v| {
            v.as_bool()
                .unwrap_or_else(|| v.as_i64().is_some_and(|n| n != 0))
        })
    }

    fn opt_list(&self, key: &str) -> Vec<i64> {
        self.opt
            .get(key)
            .and_then(|v| v.as_array())
            .map(|a| a.iter().filter_map(serde_json::Value::as_i64).collect())
            .unwrap_or_default()
    }
}

/// `SAME` (0) or `VALID` (1) padding of the TFLite options.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Padding {
    Same,
    Valid,
}

#[derive(Debug, Clone, Copy)]
struct Window {
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    padding: Padding,
}

/// The geometry of a windowed operator on an `h` x `w` map: output size and the padding before (top, left).
struct Geometry {
    oh: usize,
    ow: usize,
    top: usize,
    left: usize,
}

impl Window {
    fn geometry(&self, h: usize, w: usize) -> Result<Geometry> {
        let (oh, ow, ph, pw) = match self.padding {
            Padding::Same => {
                let oh = h.div_ceil(self.sh);
                let ow = w.div_ceil(self.sw);
                let ph = ((oh - 1) * self.sh + self.kh).saturating_sub(h);
                let pw = ((ow - 1) * self.sw + self.kw).saturating_sub(w);
                (oh, ow, ph, pw)
            }
            Padding::Valid => {
                if h < self.kh || w < self.kw {
                    return Err(MlError::Shape(format!(
                        "a {}x{} window does not fit a {h}x{w} map",
                        self.kh, self.kw
                    )));
                }
                (
                    (h - self.kh) / self.sh + 1,
                    (w - self.kw) / self.sw + 1,
                    0,
                    0,
                )
            }
        };
        Ok(Geometry {
            oh,
            ow,
            top: ph / 2,
            left: pw / 2,
        })
    }
}

#[derive(Debug)]
enum Kind {
    Conv {
        w: Packed,
        win: Window,
        cin: usize,
        act: Act,
    },
    Depthwise {
        w: Vec<f32>,
        bias: Vec<f32>,
        win: Window,
        c: usize,
        act: Act,
    },
    Add {
        act: Act,
    },
    Relu,
    Prelu {
        alpha: Vec<f32>,
    },
    Logistic,
    MaxPool {
        win: Window,
    },
    Pad {
        pads: [(usize, usize); 4],
    },
    ResizeBilinear {
        h: usize,
        w: usize,
        align_corners: bool,
        half_pixel: bool,
    },
    Reshape {
        shape: Vec<i64>,
    },
    Concat {
        axis: usize,
    },
}

#[derive(Debug)]
struct Node {
    kind: Kind,
    /// Tensor indices of the activations the operator reads (constants are inside `kind`).
    ins: Vec<usize>,
    out: usize,
}

/// A loaded graph: the operators with their constants prepared, ready to [`Graph::run`].
#[derive(Debug)]
pub struct Graph {
    inputs: Vec<Io>,
    outputs: Vec<Io>,
    nodes: Vec<Node>,
    /// For each tensor index, the node after which it is not needed any more (`usize::MAX` for the graph outputs).
    last_use: Vec<usize>,
    tensors: usize,
    parameters: usize,
}

/// Looks at the activation or constant a node reads.
fn get(values: &[Option<Tensor>], index: usize) -> Result<&Tensor> {
    values
        .get(index)
        .and_then(Option::as_ref)
        .ok_or_else(|| MlError::Weights(format!("tensor {index} is not available")))
}

fn fused(code: i64, op: &str) -> Result<Act> {
    match code {
        0 => Ok(Act::None),
        1 => Ok(Act::Relu),
        other => Err(MlError::Unsupported(format!(
            "{op} with fused activation {other}"
        ))),
    }
}

fn padding(code: i64) -> Result<Padding> {
    match code {
        0 => Ok(Padding::Same),
        1 => Ok(Padding::Valid),
        other => Err(MlError::Weights(format!("padding code {other}"))),
    }
}

fn dim(v: i64, what: &str) -> Result<usize> {
    usize::try_from(v)
        .ok()
        .filter(|&n| n >= 1)
        .ok_or_else(|| MlError::Weights(format!("{what} is {v}")))
}

impl Graph {
    /// Builds the graph from the bytes of a file written by `tools/mediapipe/convert.py`.
    ///
    /// # Errors
    ///
    /// [`MlError::Weights`] when the file is not such a file, lacks a constant or holds one that is not finite,
    /// [`MlError::Unsupported`] when the graph uses an operator or an option this module does not implement,
    /// [`MlError::TensorShape`] when a constant has a shape its operator cannot use.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let (_, meta) = safetensors::SafeTensors::read_metadata(bytes)
            .map_err(|e| MlError::Weights(e.to_string()))?;
        let header = meta.metadata().clone().unwrap_or_default();
        if header.get("format").map(String::as_str) != Some("yellowmoon-tflite-graph-1") {
            return Err(MlError::Weights(
                "not a yellowmoon-tflite-graph-1 file".into(),
            ));
        }
        let doc: Doc = serde_json::from_str(
            header
                .get("graph")
                .ok_or_else(|| MlError::Weights("header field `graph` missing".into()))?,
        )
        .map_err(|e| MlError::Weights(format!("graph: {e}")))?;
        if doc.version != 1 {
            return Err(MlError::Unsupported(format!(
                "graph version {}",
                doc.version
            )));
        }
        let mut consts: HashMap<usize, Tensor> = HashMap::new();
        let mut parameters = 0;
        for (name, (shape, data)) in read_float_tensors(bytes)? {
            let index: usize = name
                .strip_prefix('t')
                .and_then(|n| n.parse().ok())
                .ok_or_else(|| MlError::Weights(format!("unexpected tensor `{name}`")))?;
            if data.iter().any(|v| !v.is_finite()) {
                return Err(MlError::Weights(format!(
                    "tensor `{name}` holds a value that is not finite"
                )));
            }
            parameters += data.len();
            consts.insert(index, Tensor { shape, data });
        }
        let ints: HashMap<usize, &IntConst> = doc
            .ints
            .iter()
            .filter_map(|(k, v)| k.parse().ok().map(|k| (k, v)))
            .collect();

        let mut nodes = Vec::with_capacity(doc.ops.len());
        for op in &doc.ops {
            nodes.push(Self::node(op, &consts, &ints)?);
        }
        let tensors = 1 + nodes
            .iter()
            .flat_map(|n| n.ins.iter().copied().chain([n.out]))
            .chain(doc.inputs.iter().map(|i| i.index))
            .chain(doc.outputs.iter().map(|i| i.index))
            .max()
            .unwrap_or(0);
        let mut last_use = vec![usize::MAX; tensors];
        let mut seen = vec![false; tensors];
        for i in &doc.inputs {
            seen[i.index] = true;
        }
        for (n, node) in nodes.iter().enumerate() {
            for &i in &node.ins {
                if !seen[i] {
                    return Err(MlError::Weights(format!(
                        "operator {n} reads tensor {i} before it is computed"
                    )));
                }
                last_use[i] = n;
            }
            seen[node.out] = true;
        }
        for o in &doc.outputs {
            if !seen[o.index] {
                return Err(MlError::Weights(format!(
                    "output tensor {} is never computed",
                    o.index
                )));
            }
            last_use[o.index] = usize::MAX;
        }
        Ok(Self {
            inputs: doc.inputs,
            outputs: doc.outputs,
            nodes,
            last_use,
            tensors,
            parameters,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn node(
        op: &OpDoc,
        consts: &HashMap<usize, Tensor>,
        ints: &HashMap<usize, &IntConst>,
    ) -> Result<Node> {
        // The constant (float) or integer tensor at input position `pos`.
        let konst = |pos: usize| -> Result<&Tensor> {
            let idx = op
                .ins
                .get(pos)
                .and_then(|&i| usize::try_from(i).ok())
                .ok_or_else(|| MlError::Weights(format!("{} lacks input {pos}", op.op)))?;
            consts.get(&idx).ok_or_else(|| {
                MlError::Weights(format!("{}: tensor {idx} is not a constant", op.op))
            })
        };
        let int_const = |pos: usize| -> Result<&IntConst> {
            let idx = op
                .ins
                .get(pos)
                .and_then(|&i| usize::try_from(i).ok())
                .ok_or_else(|| MlError::Weights(format!("{} lacks input {pos}", op.op)))?;
            ints.get(&idx).copied().ok_or_else(|| {
                MlError::Weights(format!(
                    "{}: tensor {idx} is not an integer constant",
                    op.op
                ))
            })
        };
        let act_in = |pos: usize| -> Result<usize> {
            op.ins
                .get(pos)
                .and_then(|&i| usize::try_from(i).ok())
                .ok_or_else(|| MlError::Weights(format!("{} lacks input {pos}", op.op)))
        };
        let out = *op
            .out
            .first()
            .ok_or_else(|| MlError::Weights(format!("{} has no output", op.op)))?;
        if op.out.len() != 1 {
            return Err(MlError::Unsupported(format!(
                "{} with several outputs",
                op.op
            )));
        }
        let window = |kh, kw| -> Result<Window> {
            Ok(Window {
                kh,
                kw,
                sh: dim(op.opt_i("StrideH", 1), "stride")?,
                sw: dim(op.opt_i("StrideW", 1), "stride")?,
                padding: padding(op.opt_i("Padding", 0))?,
            })
        };
        let (kind, ins) = match op.op.as_str() {
            "CONV_2D" => {
                if op.opt_i("DilationHFactor", 1) != 1 || op.opt_i("DilationWFactor", 1) != 1 {
                    return Err(MlError::Unsupported("CONV_2D with dilation".into()));
                }
                let w = konst(1)?;
                let [cout, kh, kw, cin] = w.shape[..] else {
                    return Err(MlError::Weights("CONV_2D weight is not 4-D".into()));
                };
                let bias = op.ins.get(2).copied().filter(|&i| i >= 0);
                let bias = match bias {
                    Some(_) => Some(&konst(2)?.data[..]),
                    None => None,
                };
                if bias.is_some_and(|b| b.len() != cout) {
                    return Err(MlError::Weights("CONV_2D bias size".into()));
                }
                let packed = Packed::from_nk(&w.data, cout, kh * kw * cin, bias);
                (
                    Kind::Conv {
                        w: packed,
                        win: window(kh, kw)?,
                        cin,
                        act: fused(op.opt_i("FusedActivationFunction", 0), "CONV_2D")?,
                    },
                    vec![act_in(0)?],
                )
            }
            "DEPTHWISE_CONV_2D" => {
                if op.opt_i("DepthMultiplier", 1) != 1
                    || op.opt_i("DilationHFactor", 1) != 1
                    || op.opt_i("DilationWFactor", 1) != 1
                {
                    return Err(MlError::Unsupported(
                        "DEPTHWISE_CONV_2D with a depth multiplier or dilation".into(),
                    ));
                }
                let w = konst(1)?;
                let [1, kh, kw, c] = w.shape[..] else {
                    return Err(MlError::Weights("DEPTHWISE_CONV_2D weight shape".into()));
                };
                let bias = if op.ins.get(2).is_some_and(|&i| i >= 0) {
                    konst(2)?.data.clone()
                } else {
                    vec![0.0; c]
                };
                if bias.len() != c {
                    return Err(MlError::Weights("DEPTHWISE_CONV_2D bias size".into()));
                }
                (
                    Kind::Depthwise {
                        w: w.data.clone(),
                        bias,
                        win: window(kh, kw)?,
                        c,
                        act: fused(op.opt_i("FusedActivationFunction", 0), "DEPTHWISE_CONV_2D")?,
                    },
                    vec![act_in(0)?],
                )
            }
            "ADD" => (
                Kind::Add {
                    act: fused(op.opt_i("FusedActivationFunction", 0), "ADD")?,
                },
                vec![act_in(0)?, act_in(1)?],
            ),
            "RELU" => (Kind::Relu, vec![act_in(0)?]),
            "LOGISTIC" => (Kind::Logistic, vec![act_in(0)?]),
            "PRELU" => {
                let alpha = konst(1)?;
                (
                    Kind::Prelu {
                        alpha: alpha.data.clone(),
                    },
                    vec![act_in(0)?],
                )
            }
            "MAX_POOL_2D" => (
                Kind::MaxPool {
                    win: window(
                        dim(op.opt_i("FilterHeight", 1), "filter")?,
                        dim(op.opt_i("FilterWidth", 1), "filter")?,
                    )?,
                },
                vec![act_in(0)?],
            ),
            "PAD" => {
                let p = int_const(1)?;
                if p.shape != [4, 2] {
                    return Err(MlError::Unsupported(
                        "PAD of other than 4 dimensions".into(),
                    ));
                }
                let mut pads = [(0usize, 0usize); 4];
                for (d, pad) in pads.iter_mut().enumerate() {
                    let before = usize::try_from(p.data[2 * d]);
                    let after = usize::try_from(p.data[2 * d + 1]);
                    let (Ok(before), Ok(after)) = (before, after) else {
                        return Err(MlError::Weights("negative PAD".into()));
                    };
                    *pad = (before, after);
                }
                (Kind::Pad { pads }, vec![act_in(0)?])
            }
            "RESIZE_BILINEAR" => {
                let size = int_const(1)?;
                if size.data.len() != 2 {
                    return Err(MlError::Weights("RESIZE_BILINEAR size".into()));
                }
                (
                    Kind::ResizeBilinear {
                        h: dim(size.data[0], "resize height")?,
                        w: dim(size.data[1], "resize width")?,
                        align_corners: op.opt_b("AlignCorners"),
                        half_pixel: op.opt_b("HalfPixelCenters"),
                    },
                    vec![act_in(0)?],
                )
            }
            "RESHAPE" => {
                let shape = if op.ins.get(1).is_some_and(|&i| i >= 0) {
                    int_const(1)?.data.clone()
                } else {
                    op.opt_list("NewShape")
                };
                if shape.is_empty() {
                    return Err(MlError::Weights("RESHAPE without a shape".into()));
                }
                (Kind::Reshape { shape }, vec![act_in(0)?])
            }
            "CONCATENATION" => {
                if op.opt_i("FusedActivationFunction", 0) != 0 {
                    return Err(MlError::Unsupported(
                        "CONCATENATION with an activation".into(),
                    ));
                }
                let axis = usize::try_from(op.opt_i("Axis", 0))
                    .map_err(|_| MlError::Unsupported("CONCATENATION on a negative axis".into()))?;
                let ins = (0..op.ins.len()).map(act_in).collect::<Result<Vec<_>>>()?;
                (Kind::Concat { axis }, ins)
            }
            other => {
                return Err(MlError::Unsupported(format!(
                    "TFLite operator {other} (not implemented in ym-ml yet)"
                )));
            }
        };
        Ok(Node { kind, ins, out })
    }

    /// The graph's inputs.
    pub fn inputs(&self) -> &[Io] {
        &self.inputs
    }

    /// The graph's outputs, in the order [`Graph::run`] returns them.
    pub fn outputs(&self) -> &[Io] {
        &self.outputs
    }

    /// The number of constants (weights) in the graph.
    pub fn parameters(&self) -> usize {
        self.parameters
    }

    /// Runs the graph on its inputs (one tensor per input, in order) and returns its outputs.
    ///
    /// # Errors
    ///
    /// [`MlError::Shape`] when an input does not have the shape the graph was converted with, [`MlError::Input`] when an
    /// input holds a value that is not finite.
    pub fn run(&self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        if inputs.len() != self.inputs.len() {
            return Err(MlError::Shape(format!(
                "{} inputs given, {} expected",
                inputs.len(),
                self.inputs.len()
            )));
        }
        let mut values: Vec<Option<Tensor>> = vec![None; self.tensors];
        for (spec, input) in self.inputs.iter().zip(inputs) {
            if input.shape != spec.shape {
                return Err(MlError::Shape(format!(
                    "input `{}` has shape {:?}, {:?} expected",
                    spec.name, input.shape, spec.shape
                )));
            }
            if input.data.iter().any(|v| !v.is_finite()) {
                return Err(MlError::Input(format!(
                    "input `{}` holds a value that is not finite",
                    spec.name
                )));
            }
            values[spec.index] = Some(input.clone());
        }
        for (n, node) in self.nodes.iter().enumerate() {
            let result = Self::eval(node, &values)?;
            values[node.out] = Some(result);
            for &i in &node.ins {
                if self.last_use[i] == n {
                    values[i] = None;
                }
            }
        }
        self.outputs
            .iter()
            .map(|o| {
                values[o.index]
                    .take()
                    .ok_or_else(|| MlError::Weights(format!("output {} missing", o.index)))
            })
            .collect()
    }

    fn eval(node: &Node, values: &[Option<Tensor>]) -> Result<Tensor> {
        let x = get(values, node.ins[0])?;
        match &node.kind {
            Kind::Conv { w, win, cin, act } => conv(x, w, *win, *cin, *act),
            Kind::Depthwise {
                w,
                bias,
                win,
                c,
                act,
            } => depthwise(x, w, bias, *win, *c, *act),
            Kind::Add { act } => {
                let y = get(values, node.ins[1])?;
                if x.shape != y.shape {
                    return Err(MlError::Shape(format!(
                        "ADD of shapes {:?} and {:?}",
                        x.shape, y.shape
                    )));
                }
                let data = x
                    .data
                    .iter()
                    .zip(&y.data)
                    .map(|(&a, &b)| match act {
                        Act::Relu => relu(a + b),
                        _ => a + b,
                    })
                    .collect();
                Ok(Tensor {
                    shape: x.shape.clone(),
                    data,
                })
            }
            Kind::Relu => Ok(Tensor {
                shape: x.shape.clone(),
                data: x.data.iter().map(|&v| relu(v)).collect(),
            }),
            Kind::Logistic => Ok(Tensor {
                shape: x.shape.clone(),
                data: x.data.iter().map(|&v| sigmoid(v)).collect(),
            }),
            Kind::Prelu { alpha } => {
                let c = *x.shape.last().unwrap_or(&1);
                if alpha.len() != c && alpha.len() != 1 {
                    return Err(MlError::Shape(format!(
                        "PRELU slope of {} values on {c} channels",
                        alpha.len()
                    )));
                }
                let mut data = x.data.clone();
                for px in data.chunks_exact_mut(c) {
                    for (ch, v) in px.iter_mut().enumerate() {
                        if *v < 0.0 {
                            *v *= alpha[if alpha.len() == 1 { 0 } else { ch }];
                        }
                    }
                }
                Ok(Tensor {
                    shape: x.shape.clone(),
                    data,
                })
            }
            Kind::MaxPool { win } => max_pool(x, *win),
            Kind::Pad { pads } => pad(x, pads),
            Kind::ResizeBilinear {
                h,
                w,
                align_corners,
                half_pixel,
            } => resize_bilinear(x, *h, *w, *align_corners, *half_pixel),
            Kind::Reshape { shape } => reshape(x, shape),
            Kind::Concat { axis } => {
                let parts = node
                    .ins
                    .iter()
                    .map(|&i| get(values, i))
                    .collect::<Result<Vec<_>>>()?;
                concat(&parts, *axis)
            }
        }
    }
}

fn nhwc(x: &Tensor) -> Result<(usize, usize, usize)> {
    match x.shape[..] {
        [1, h, w, c] => Ok((h, w, c)),
        _ => Err(MlError::Shape(format!(
            "expected a 1 x H x W x C tensor, found {:?}",
            x.shape
        ))),
    }
}

fn conv(x: &Tensor, w: &Packed, win: Window, cin: usize, act: Act) -> Result<Tensor> {
    let (h, wd, c) = nhwc(x)?;
    if c != cin {
        return Err(MlError::Shape(format!(
            "CONV_2D takes {cin} channels, found {c}"
        )));
    }
    let g = win.geometry(h, wd)?;
    let cout = w.n;
    let mut out = vec![0.0f32; g.oh * g.ow * cout];
    if win.kh == 1 && win.kw == 1 && win.sh == 1 && win.sw == 1 {
        gemm_par(&x.data, h * wd, w, act, None, &mut out);
    } else {
        let k = win.kh * win.kw * cin;
        out.par_chunks_mut(g.ow * cout)
            .enumerate()
            .for_each(|(oy, row)| {
                let mut cols = vec![0.0f32; g.ow * k];
                for ox in 0..g.ow {
                    for ky in 0..win.kh {
                        let iy = (oy * win.sh + ky).wrapping_sub(g.top);
                        if iy >= h {
                            continue;
                        }
                        for kx in 0..win.kw {
                            let ix = (ox * win.sw + kx).wrapping_sub(g.left);
                            if ix >= wd {
                                continue;
                            }
                            let src = &x.data[(iy * wd + ix) * cin..][..cin];
                            cols[ox * k + (ky * win.kw + kx) * cin..][..cin].copy_from_slice(src);
                        }
                    }
                }
                gemm(&cols, k, g.ow, w, act, None, row, cout);
            });
    }
    Ok(Tensor {
        shape: vec![1, g.oh, g.ow, cout],
        data: out,
    })
}

fn depthwise(
    x: &Tensor,
    w: &[f32],
    bias: &[f32],
    win: Window,
    c: usize,
    act: Act,
) -> Result<Tensor> {
    let (h, wd, cin) = nhwc(x)?;
    if cin != c {
        return Err(MlError::Shape(format!(
            "DEPTHWISE_CONV_2D takes {c} channels, found {cin}"
        )));
    }
    let g = win.geometry(h, wd)?;
    let mut out = vec![0.0f32; g.oh * g.ow * c];
    out.par_chunks_mut(g.ow * c)
        .enumerate()
        .for_each(|(oy, row)| {
            for (ox, acc) in row.chunks_exact_mut(c).enumerate() {
                acc.copy_from_slice(bias);
                for ky in 0..win.kh {
                    let iy = (oy * win.sh + ky).wrapping_sub(g.top);
                    if iy >= h {
                        continue;
                    }
                    for kx in 0..win.kw {
                        let ix = (ox * win.sw + kx).wrapping_sub(g.left);
                        if ix >= wd {
                            continue;
                        }
                        let src = &x.data[(iy * wd + ix) * c..][..c];
                        let tap = &w[(ky * win.kw + kx) * c..][..c];
                        for ((a, &s), &t) in acc.iter_mut().zip(src).zip(tap) {
                            *a += s * t;
                        }
                    }
                }
                if matches!(act, Act::Relu) {
                    for a in acc.iter_mut() {
                        *a = relu(*a);
                    }
                }
            }
        });
    Ok(Tensor {
        shape: vec![1, g.oh, g.ow, c],
        data: out,
    })
}

fn max_pool(x: &Tensor, win: Window) -> Result<Tensor> {
    let (h, wd, c) = nhwc(x)?;
    let g = win.geometry(h, wd)?;
    let mut out = vec![0.0f32; g.oh * g.ow * c];
    out.par_chunks_mut(g.ow * c)
        .enumerate()
        .for_each(|(oy, row)| {
            for (ox, acc) in row.chunks_exact_mut(c).enumerate() {
                acc.fill(f32::NEG_INFINITY);
                for ky in 0..win.kh {
                    let iy = (oy * win.sh + ky).wrapping_sub(g.top);
                    if iy >= h {
                        continue;
                    }
                    for kx in 0..win.kw {
                        let ix = (ox * win.sw + kx).wrapping_sub(g.left);
                        if ix >= wd {
                            continue;
                        }
                        for (a, &s) in acc.iter_mut().zip(&x.data[(iy * wd + ix) * c..][..c]) {
                            if s > *a {
                                *a = s;
                            }
                        }
                    }
                }
            }
        });
    Ok(Tensor {
        shape: vec![1, g.oh, g.ow, c],
        data: out,
    })
}

fn pad(x: &Tensor, pads: &[(usize, usize); 4]) -> Result<Tensor> {
    let [n, h, w, c] = x.shape[..] else {
        return Err(MlError::Shape("PAD of a tensor that is not 4-D".into()));
    };
    let shape = [
        n + pads[0].0 + pads[0].1,
        h + pads[1].0 + pads[1].1,
        w + pads[2].0 + pads[2].1,
        c + pads[3].0 + pads[3].1,
    ];
    let mut out = vec![0.0f32; shape.iter().product()];
    for b in 0..n {
        for y in 0..h {
            for xx in 0..w {
                let src = &x.data[((b * h + y) * w + xx) * c..][..c];
                let at = (((b + pads[0].0) * shape[1] + y + pads[1].0) * shape[2] + xx + pads[2].0)
                    * shape[3]
                    + pads[3].0;
                out[at..at + c].copy_from_slice(src);
            }
        }
    }
    Ok(Tensor {
        shape: shape.to_vec(),
        data: out,
    })
}

/// TFLite's source coordinate for one axis: the two taps and the weight of the second.
fn resize_axis(
    dst: usize,
    in_len: usize,
    out_len: usize,
    align: bool,
    half: bool,
) -> (usize, usize, f32) {
    let scale = if align && out_len > 1 {
        (in_len - 1) as f32 / (out_len - 1) as f32
    } else {
        in_len as f32 / out_len as f32
    };
    let s = if half {
        (dst as f32 + 0.5) * scale - 0.5
    } else {
        dst as f32 * scale
    };
    let fl = s.floor();
    let lo = fl.max(0.0) as usize;
    let hi = (s.ceil().max(0.0) as usize).min(in_len - 1);
    (lo.min(in_len - 1), hi, s - fl)
}

fn resize_bilinear(x: &Tensor, oh: usize, ow: usize, align: bool, half: bool) -> Result<Tensor> {
    let (h, w, c) = nhwc(x)?;
    let mut out = vec![0.0f32; oh * ow * c];
    out.par_chunks_mut(ow * c)
        .enumerate()
        .for_each(|(oy, row)| {
            let (y0, y1, fy) = resize_axis(oy, h, oh, align, half);
            for (ox, px) in row.chunks_exact_mut(c).enumerate() {
                let (x0, x1, fx) = resize_axis(ox, w, ow, align, half);
                let at = |y: usize, xx: usize, ch: usize| x.data[(y * w + xx) * c + ch];
                for (ch, v) in px.iter_mut().enumerate() {
                    *v = at(y0, x0, ch) * (1.0 - fy) * (1.0 - fx)
                        + at(y1, x0, ch) * fy * (1.0 - fx)
                        + at(y0, x1, ch) * (1.0 - fy) * fx
                        + at(y1, x1, ch) * fy * fx;
                }
            }
        });
    Ok(Tensor {
        shape: vec![1, oh, ow, c],
        data: out,
    })
}

fn reshape(x: &Tensor, shape: &[i64]) -> Result<Tensor> {
    let total = x.data.len();
    let known: i64 = shape.iter().filter(|&&d| d >= 0).product();
    let mut dims = Vec::with_capacity(shape.len());
    for &d in shape {
        if d >= 0 {
            dims.push(d as usize);
        } else if known > 0 && total.is_multiple_of(known as usize) {
            dims.push(total / known as usize);
        } else {
            return Err(MlError::Shape(format!(
                "cannot reshape {total} values to {shape:?}"
            )));
        }
    }
    if dims.iter().product::<usize>() != total {
        return Err(MlError::Shape(format!(
            "cannot reshape {total} values to {shape:?}"
        )));
    }
    Ok(Tensor {
        shape: dims,
        data: x.data.clone(),
    })
}

fn concat(parts: &[&Tensor], axis: usize) -> Result<Tensor> {
    let first = parts[0];
    if axis >= first.shape.len() {
        return Err(MlError::Shape(format!("CONCATENATION on axis {axis}")));
    }
    let outer: usize = first.shape[..axis].iter().product();
    let mut shape = first.shape.clone();
    shape[axis] = 0;
    for p in parts {
        let mut s = p.shape.clone();
        if s.len() != shape.len() {
            return Err(MlError::Shape("CONCATENATION of different ranks".into()));
        }
        s[axis] = 0;
        let mut want = first.shape.clone();
        want[axis] = 0;
        if s != want {
            return Err(MlError::Shape("CONCATENATION of different shapes".into()));
        }
        shape[axis] += p.shape[axis];
    }
    let inner: usize = first.shape[axis + 1..].iter().product();
    let mut data = Vec::with_capacity(shape.iter().product());
    for o in 0..outer {
        for p in parts {
            let chunk = p.shape[axis] * inner;
            data.extend_from_slice(&p.data[o * chunk..(o + 1) * chunk]);
        }
    }
    Ok(Tensor { shape, data })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safetensors::{Dtype, tensor::TensorView};

    use super::*;

    /// A converted file by hand: float constants `(index, shape, values)` and the graph JSON.
    fn file(consts: &[(usize, Vec<usize>, Vec<f32>)], graph: &str) -> Vec<u8> {
        let bytes: Vec<(String, Vec<usize>, Vec<u8>)> = consts
            .iter()
            .map(|(i, shape, v)| {
                (
                    format!("t{i}"),
                    shape.clone(),
                    v.iter().flat_map(|x| x.to_le_bytes()).collect(),
                )
            })
            .collect();
        let views: HashMap<String, TensorView<'_>> = bytes
            .iter()
            .map(|(n, s, b)| {
                (
                    n.clone(),
                    TensorView::new(Dtype::F32, s.clone(), b).expect("view"),
                )
            })
            .collect();
        let meta = HashMap::from([
            ("format".to_owned(), "yellowmoon-tflite-graph-1".to_owned()),
            ("graph".to_owned(), graph.to_owned()),
        ]);
        safetensors::serialize(&views, Some(meta)).expect("serialize")
    }

    fn io(index: usize, shape: &[usize]) -> String {
        format!(r#"{{"index":{index},"name":"t{index}","shape":{shape:?}}}"#)
    }

    /// One-operator graph `input 0 -> op -> output 1` with the extra inputs given.
    fn single(
        op: &str,
        extra_in: &[i64],
        opt: &str,
        in_shape: &[usize],
        out_shape: &[usize],
        ints: &str,
    ) -> String {
        let mut ins = vec![0i64];
        ins.extend_from_slice(extra_in);
        format!(
            r#"{{"version":1,"inputs":[{}],"outputs":[{}],"ints":{ints},"ops":[{{"op":"{op}","in":{ins:?},"out":[1],"opt":{opt}}}]}}"#,
            io(0, in_shape),
            io(1, out_shape)
        )
    }

    fn run(file_bytes: &[u8], input: Tensor) -> Result<Vec<Tensor>> {
        Graph::from_bytes(file_bytes)?.run(&[input])
    }

    fn ramp(shape: &[usize]) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::new(
            shape.to_vec(),
            (0..n)
                .map(|i| (i % 7) as f32 - 2.5 + i as f32 * 0.01)
                .collect(),
        )
        .expect("tensor")
    }

    #[test]
    fn a_1x1_convolution_is_a_matrix_product_with_bias_and_a_fused_relu() {
        // 2 -> 3 channels on a 1 x 2 x 2 map.
        let w = vec![1.0, 2.0, -1.0, 0.5, 0.25, -3.0];
        let b = vec![0.5, -0.5, 0.0];
        let graph = single(
            "CONV_2D",
            &[2, 3],
            r#"{"Padding":1,"StrideH":1,"StrideW":1,"FusedActivationFunction":1}"#,
            &[1, 2, 2, 2],
            &[1, 2, 2, 3],
            "{}",
        );
        let f = file(
            &[(2, vec![3, 1, 1, 2], w.clone()), (3, vec![3], b.clone())],
            &graph,
        );
        let x = ramp(&[1, 2, 2, 2]);
        let out = &run(&f, x.clone()).expect("run")[0];
        assert_eq!(out.shape, vec![1, 2, 2, 3]);
        for p in 0..4 {
            for o in 0..3 {
                let want =
                    (b[o] + x.data[2 * p] * w[2 * o] + x.data[2 * p + 1] * w[2 * o + 1]).max(0.0);
                assert!((out.data[3 * p + o] - want).abs() < 1e-6, "p{p} o{o}");
            }
        }
    }

    #[test]
    fn a_3x3_stride_2_same_convolution_pads_after_on_even_sizes() {
        // 1 channel, 4 x 4 in, 2 x 2 out. TFLite SAME with stride 2: total padding 1, none before, one after.
        let w: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let graph = single(
            "CONV_2D",
            &[2, -1],
            r#"{"Padding":0,"StrideH":2,"StrideW":2}"#,
            &[1, 4, 4, 1],
            &[1, 2, 2, 1],
            "{}",
        );
        let f = file(&[(2, vec![1, 3, 3, 1], w.clone())], &graph);
        let x = ramp(&[1, 4, 4, 1]);
        let out = &run(&f, x.clone()).expect("run")[0];
        for oy in 0..2usize {
            for ox in 0..2usize {
                let mut acc = 0.0f32;
                for ky in 0..3usize {
                    for kx in 0..3usize {
                        let (iy, ix) = (oy * 2 + ky, ox * 2 + kx);
                        if iy < 4 && ix < 4 {
                            acc += x.data[iy * 4 + ix] * w[ky * 3 + kx];
                        }
                    }
                }
                assert!(
                    (out.data[oy * 2 + ox] - acc).abs() < 1e-4,
                    "({oy},{ox}) {} against {acc}",
                    out.data[oy * 2 + ox]
                );
            }
        }
    }

    #[test]
    fn a_depthwise_convolution_filters_each_channel_alone() {
        let w: Vec<f32> = (0..18).map(|v| (v as f32 - 8.0) * 0.1).collect(); // [1][3][3][2]
        let graph = single(
            "DEPTHWISE_CONV_2D",
            &[2, 3],
            r#"{"Padding":0,"StrideH":1,"StrideW":1,"DepthMultiplier":1}"#,
            &[1, 3, 3, 2],
            &[1, 3, 3, 2],
            "{}",
        );
        let f = file(
            &[
                (2, vec![1, 3, 3, 2], w.clone()),
                (3, vec![2], vec![1.0, -1.0]),
            ],
            &graph,
        );
        let x = ramp(&[1, 3, 3, 2]);
        let out = &run(&f, x.clone()).expect("run")[0];
        // The centre pixel sees the whole map.
        for c in 0..2 {
            let mut acc = if c == 0 { 1.0 } else { -1.0 };
            for ky in 0..3 {
                for kx in 0..3 {
                    acc += x.data[(ky * 3 + kx) * 2 + c] * w[(ky * 3 + kx) * 2 + c];
                }
            }
            assert!((out.data[(3 + 1) * 2 + c] - acc).abs() < 1e-5);
        }
        // The corner sees four taps.
        let mut acc = 1.0;
        for ky in 1..3 {
            for kx in 1..3 {
                acc += x.data[((ky - 1) * 3 + kx - 1) * 2] * w[(ky * 3 + kx) * 2];
            }
        }
        assert!((out.data[0] - acc).abs() < 1e-5);
    }

    #[test]
    fn max_pool_pad_resize_reshape_and_activations() {
        // 2 x 2 max pool, stride 2.
        let g = single(
            "MAX_POOL_2D",
            &[],
            r#"{"Padding":1,"StrideH":2,"StrideW":2,"FilterHeight":2,"FilterWidth":2}"#,
            &[1, 4, 4, 1],
            &[1, 2, 2, 1],
            "{}",
        );
        let x = ramp(&[1, 4, 4, 1]);
        let out = &run(&file(&[], &g), x.clone()).expect("run")[0];
        let want = |oy: usize, ox: usize| {
            [(0, 0), (0, 1), (1, 0), (1, 1)]
                .iter()
                .map(|(dy, dx)| x.data[(oy * 2 + dy) * 4 + ox * 2 + dx])
                .fold(f32::NEG_INFINITY, f32::max)
        };
        assert_eq!(
            out.data,
            vec![want(0, 0), want(0, 1), want(1, 0), want(1, 1)]
        );

        // Channel padding 1 + 2 around a 2-channel map.
        let g = single(
            "PAD",
            &[2],
            "{}",
            &[1, 1, 2, 2],
            &[1, 1, 2, 5],
            r#"{"2":{"shape":[4,2],"data":[0,0,0,0,0,0,1,2]}}"#,
        );
        let x = Tensor::new(vec![1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).expect("tensor");
        let out = &run(&file(&[], &g), x).expect("run")[0];
        assert_eq!(
            out.data,
            vec![0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0]
        );

        // Bilinear 2 x 2 -> 4 x 4 with half-pixel centres: the first output row is 0, 0.25, 0.75, 1 of the first input row.
        let g = single(
            "RESIZE_BILINEAR",
            &[2],
            r#"{"AlignCorners":false,"HalfPixelCenters":true}"#,
            &[1, 2, 2, 1],
            &[1, 4, 4, 1],
            r#"{"2":{"shape":[2],"data":[4,4]}}"#,
        );
        let x = Tensor::new(vec![1, 2, 2, 1], vec![0.0, 1.0, 0.0, 1.0]).expect("tensor");
        let out = &run(&file(&[], &g), x).expect("run")[0];
        assert_eq!(out.data[..4], [0.0, 0.25, 0.75, 1.0]);

        // Reshape with -1, from the option.
        let g = single(
            "RESHAPE",
            &[],
            r#"{"NewShape":[1,-1,2]}"#,
            &[1, 2, 3, 2],
            &[1, 6, 2],
            "{}",
        );
        let out = &run(&file(&[], &g), ramp(&[1, 2, 3, 2])).expect("run")[0];
        assert_eq!(out.shape, vec![1, 6, 2]);

        // PReLU with a slope per channel, LOGISTIC, RELU.
        let g = single("PRELU", &[2], "{}", &[1, 1, 2, 2], &[1, 1, 2, 2], "{}");
        let x = Tensor::new(vec![1, 1, 2, 2], vec![-1.0, -2.0, 3.0, -4.0]).expect("tensor");
        let out =
            &run(&file(&[(2, vec![1, 1, 2], vec![0.5, 0.25])], &g), x.clone()).expect("run")[0];
        assert_eq!(out.data, vec![-0.5, -0.5, 3.0, -1.0]);
        let g = single("LOGISTIC", &[], "{}", &[1, 1, 2, 2], &[1, 1, 2, 2], "{}");
        let out = &run(&file(&[], &g), x.clone()).expect("run")[0];
        assert!((out.data[2] - 1.0 / (1.0 + (-3.0f32).exp())).abs() < 1e-6);
        let g = single("RELU", &[], "{}", &[1, 1, 2, 2], &[1, 1, 2, 2], "{}");
        assert_eq!(
            run(&file(&[], &g), x).expect("run")[0].data,
            vec![0.0, 0.0, 3.0, 0.0]
        );
    }

    #[test]
    fn add_and_concatenation_of_two_paths() {
        // 0 -> RELU -> 2; 0 + 2 -> 3 (ADD); concat(2, 3) on the channel axis -> 1.
        let graph = format!(
            r#"{{"version":1,"inputs":[{}],"outputs":[{}],"ops":[
                {{"op":"RELU","in":[0],"out":[2],"opt":{{}}}},
                {{"op":"ADD","in":[0,2],"out":[3],"opt":{{"FusedActivationFunction":0}}}},
                {{"op":"CONCATENATION","in":[2,3],"out":[1],"opt":{{"Axis":3,"FusedActivationFunction":0}}}}]}}"#,
            io(0, &[1, 1, 2, 1]),
            io(1, &[1, 1, 2, 2])
        );
        let x = Tensor::new(vec![1, 1, 2, 1], vec![-2.0, 3.0]).expect("tensor");
        let out = &run(&file(&[], &graph), x).expect("run")[0];
        // relu = [0, 3]; add = [-2, 6]; interleaved by pixel: [0, -2, 3, 6].
        assert_eq!(out.data, vec![0.0, -2.0, 3.0, 6.0]);
    }

    #[test]
    fn a_graph_with_an_operator_we_do_not_have_is_refused_when_loaded() {
        let g = single(
            "TRANSPOSE_CONV",
            &[],
            "{}",
            &[1, 2, 2, 1],
            &[1, 2, 2, 1],
            "{}",
        );
        let err = Graph::from_bytes(&file(&[], &g)).expect_err("unsupported");
        assert!(
            matches!(err, MlError::Unsupported(ref m) if m.contains("TRANSPOSE_CONV")),
            "{err}"
        );
    }

    #[test]
    fn bad_files_and_bad_inputs_are_errors() {
        assert!(Graph::from_bytes(b"nope").is_err());
        // A safetensors file that is not one of ours.
        let views: HashMap<String, TensorView<'_>> = HashMap::new();
        let plain = safetensors::serialize(&views, None).expect("serialize");
        assert!(matches!(
            Graph::from_bytes(&plain),
            Err(MlError::Weights(_))
        ));
        // An operator that reads a tensor nothing computes.
        let g = format!(
            r#"{{"version":1,"inputs":[{}],"outputs":[{}],"ops":[{{"op":"RELU","in":[5],"out":[1],"opt":{{}}}}]}}"#,
            io(0, &[1, 1, 1, 1]),
            io(1, &[1, 1, 1, 1])
        );
        assert!(Graph::from_bytes(&file(&[], &g)).is_err());
        // A wrong input shape and a value that is not finite.
        let g = single("RELU", &[], "{}", &[1, 1, 2, 1], &[1, 1, 2, 1], "{}");
        let f = file(&[], &g);
        let graph = Graph::from_bytes(&f).expect("graph");
        assert!(matches!(
            graph.run(&[ramp(&[1, 1, 3, 1])]),
            Err(MlError::Shape(_))
        ));
        let nan = Tensor::new(vec![1, 1, 2, 1], vec![f32::NAN, 0.0]).expect("tensor");
        assert!(matches!(graph.run(&[nan]), Err(MlError::Input(_))));
        assert!(graph.run(&[]).is_err());
        assert!(Tensor::new(vec![2, 2], vec![0.0; 3]).is_err());
    }
}
