//! burn backend behind [`TileModel`] (`docs/07-ml-models.md` section Runtime): the `jdd` network written
//! with burn 0.22 tensor operations and run on burn's wgpu backend (Vulkan, Metal, ...), weights loaded
//! from the same safetensors file as the CPU runtime.
//!
//! The network is expressed directly with burn's functional API (`conv2d`, `silu`, reshape/permute for the
//! pixel shuffle) instead of generating Rust from the ONNX file with `burn-onnx`: that tool is a build-time
//! code generator that needs the model file at compile time, which does not fit weights that are fetched or
//! replaced after the build. The kernels executed are the same ones.
//!
//! Sharing the application's wgpu device (docs/07: "no copies between the image pipeline and the network")
//! goes through the constructor: a host that owns a burn [`Device`] (created from its wgpu setup with
//! `Device::wgpu_options().setup(..).init()`) loads the weights onto it with [`BurnJdd::with_device`]. The
//! weights live on that one device for the lifetime of the model, so [`TileModel::run`] always runs there
//! and ignores the [`ExecContext`]: a different device in the context would put the input on a device the
//! weights are not on.

use burn::tensor::activation::silu;
use burn::tensor::module::conv2d;
use burn::tensor::ops::ConvOptions;
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};

use crate::cpu_runtime::{Header, read_float_tensors, read_header};
use crate::model::{
    ExecContext, IN_CHANNELS, JddConfig, MlError, ModelId, OUT_CHANNELS, OUT_SCALE, Result,
    TileInputs, TileModel, TileOutput, TileShape,
};

fn backend_err(e: impl std::fmt::Display) -> MlError {
    MlError::Backend {
        backend: "burn",
        detail: e.to_string(),
    }
}

struct Layer {
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    stride: usize,
    pad: usize,
}

impl Layer {
    fn forward(&self, x: Tensor<4>, act: bool) -> Tensor<4> {
        let opts = ConvOptions::new([self.stride, self.stride], [self.pad, self.pad], [1, 1], 1);
        let y = conv2d(x, self.weight.clone(), self.bias.clone(), opts);
        if act { silu(y) } else { y }
    }
}

/// The weights of the network as burn tensors on one device.
struct Net {
    enc0: Vec<Layer>,
    down1: Layer,
    enc1: Vec<Layer>,
    down2: Layer,
    bott: Vec<Layer>,
    up1: Layer,
    dec1: Vec<Layer>,
    up0: Layer,
    dec0: Vec<Layer>,
    head: Layer,
    bilinear: Layer,
    in_scale: Tensor<4>,
}

fn block(x: Tensor<4>, layers: &[Layer]) -> Tensor<4> {
    layers.iter().fold(x, |x, l| l.forward(x, true))
}

/// PyTorch `pixel_shuffle(x, 2)`: `[B, 4C, H, W] -> [B, C, 2H, 2W]` (channel `c*4 + a*2 + b` -> pixel `(2y+a, 2x+b)`).
fn pixel_shuffle(x: Tensor<4>) -> Tensor<4> {
    let [b, c4, h, w] = x.dims();
    let c = c4 / 4;
    x.reshape([b, c, 2, 2, h, w])
        .permute([0, 1, 4, 2, 5, 3])
        .reshape([b, c, 2 * h, 2 * w])
}

impl Net {
    fn forward(&self, x: Tensor<4>, cfg: &JddConfig) -> Tensor<4> {
        let planes = x.clone().narrow(1, 0, 4).mul_scalar(cfg.plane_scale);
        let x = x.mul(self.in_scale.clone());
        let e0 = block(x, &self.enc0);
        let e1 = block(self.down1.forward(e0.clone(), true), &self.enc1);
        let b = block(self.down2.forward(e1.clone(), true), &self.bott);
        let u1 = pixel_shuffle(self.up1.forward(b, true));
        let d1 = block(Tensor::cat(vec![u1, e1], 1), &self.dec1);
        let u0 = pixel_shuffle(self.up0.forward(d1, true));
        let d0 = block(Tensor::cat(vec![u0, e0], 1), &self.dec0);
        let y = self.head.forward(d0, false) + self.bilinear.forward(planes, false);
        pixel_shuffle(y).mul_scalar(1.0 / cfg.plane_scale)
    }
}

/// The `jdd` network on burn's wgpu backend.
pub struct BurnJdd {
    header: Header,
    tile: TileShape,
    device: Device,
    net: Net,
}

impl BurnJdd {
    /// Load from safetensors bytes on burn's default wgpu device (the best adapter wgpu finds; set
    /// `CUBECL_WGPU_DEFAULT_DEVICE` to choose, for example `Cpu` for a software adapter).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        Self::with_device(bytes, Device::wgpu(DeviceKind::DefaultDevice))
    }

    /// Load onto a device the host already created.
    pub fn with_device(bytes: &[u8], device: Device) -> Result<Self> {
        let header = read_header(bytes)?;
        let tensors = read_float_tensors(bytes)?;
        let cfg = header.cfg.clone();
        let t4 = |name: &str| -> Result<Tensor<4>> {
            let (shape, data) = tensors
                .get(name)
                .ok_or_else(|| MlError::Weights(format!("tensor `{name}` missing")))?;
            Ok(Tensor::<4>::from_data(
                TensorData::new(data.clone(), shape.clone()),
                &device,
            ))
        };
        let t1 = |name: &str| -> Result<Tensor<1>> {
            let (shape, data) = tensors
                .get(name)
                .ok_or_else(|| MlError::Weights(format!("tensor `{name}` missing")))?;
            Ok(Tensor::<1>::from_data(
                TensorData::new(data.clone(), shape.clone()),
                &device,
            ))
        };
        let layer = |prefix: &str, stride: usize, bias: bool| -> Result<Layer> {
            let weight = t4(&format!("{prefix}.weight"))?;
            let k = weight.dims()[2];
            Ok(Layer {
                weight,
                bias: if bias {
                    Some(t1(&format!("{prefix}.bias"))?)
                } else {
                    None
                },
                stride,
                pad: k / 2,
            })
        };
        let block_layers = |prefix: &str| -> Result<Vec<Layer>> {
            (1..=cfg.convs_per_level)
                .map(|i| layer(&format!("{prefix}.conv{i}"), 1, true))
                .collect()
        };
        let in_scale = Tensor::<4>::from_data(
            TensorData::new(cfg.input_scale().to_vec(), [1, IN_CHANNELS, 1, 1]),
            &device,
        );
        let net = Net {
            enc0: block_layers("enc0")?,
            down1: layer("down1", 2, true)?,
            enc1: block_layers("enc1")?,
            down2: layer("down2", 2, true)?,
            bott: block_layers("bott")?,
            up1: layer("up1", 1, true)?,
            dec1: block_layers("dec1")?,
            up0: layer("up0", 1, true)?,
            dec0: block_layers("dec0")?,
            head: layer("head", 1, true)?,
            bilinear: layer("bilinear", 1, false)?,
            in_scale,
        };
        Ok(Self {
            tile: TileShape::square(512 + 2 * header.halo_half_px),
            header,
            device,
            net,
        })
    }

    /// Set the exact tile input shape reported by [`TileModel::input_shape`].
    #[must_use]
    pub fn with_tile_shape(mut self, shape: TileShape) -> Self {
        self.tile = shape;
        self
    }

    /// Identity (name, version, SHA-256) of the loaded weights.
    pub fn id(&self) -> &ModelId {
        &self.header.id
    }

    fn forward_vec(&self, inputs: &TileInputs, device: &Device) -> Result<Vec<f32>> {
        let s = inputs.shape();
        let x = Tensor::<4>::from_data(
            TensorData::new(
                inputs.as_slice().to_vec(),
                [1, IN_CHANNELS, s.height, s.width],
            ),
            device,
        );
        let y = self.net.forward(x, &self.header.cfg);
        let data = y.into_data();
        data.try_to_vec::<f32>()
            .map_err(|e| backend_err(format!("{e:?}")))
    }
}

impl TileModel for BurnJdd {
    fn input_shape(&self) -> TileShape {
        self.tile
    }

    fn halo_half_px(&self) -> usize {
        self.header.halo_half_px
    }

    fn run(&self, inputs: &TileInputs, out: &mut TileOutput, _ctx: &ExecContext<'_>) -> Result<()> {
        if inputs.shape() != self.tile || out.shape() != self.tile {
            return Err(MlError::Shape(format!(
                "burn model expects tiles of {}, got input {} output {}",
                self.tile,
                inputs.shape(),
                out.shape()
            )));
        }
        let data = self.forward_vec(inputs, &self.device)?;
        if data.len() != OUT_CHANNELS * OUT_SCALE * self.tile.height * OUT_SCALE * self.tile.width {
            return Err(backend_err("unexpected output size"));
        }
        out.as_mut_slice().copy_from_slice(&data);
        Ok(())
    }

    fn name(&self) -> &str {
        "burn"
    }
}
