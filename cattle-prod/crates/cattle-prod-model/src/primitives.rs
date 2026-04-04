use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

pub fn sigmoid(xs: &Tensor) -> Result<Tensor> {
    (xs.neg()?.exp()? + 1.0)?.recip()
}

pub fn silu(xs: &Tensor) -> Result<Tensor> {
    xs.mul(&sigmoid(xs)?)
}

pub fn softmax(xs: &Tensor, dim: usize) -> Result<Tensor> {
    let max = xs.max_keepdim(dim)?;
    let exp = xs.broadcast_sub(&max)?.exp()?;
    let sum = exp.sum_keepdim(dim)?;
    exp.broadcast_div(&sum)
}

// ---------------------------------------------------------------------------
// Linear (with optional bias)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        let bias = vb.get(out_features, "bias").ok();
        Ok(Self { weight, bias })
    }

    pub fn new_with_bias(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        let bias = Some(vb.get(out_features, "bias")?);
        Ok(Self { weight, bias })
    }

    pub fn forward_t(&self, xs: &Tensor) -> Result<Tensor> {
        let wt = self.weight.t()?;
        let out = if xs.rank() > 2 {
            let dims = xs.dims();
            let last = *dims.last().unwrap();
            let batch_size: usize = dims[..dims.len() - 1].iter().product();
            let flat = xs.reshape((batch_size, last))?;
            let result = flat.matmul(&wt)?;
            let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
            out_shape.push(result.dim(D::Minus1)?);
            result.reshape(out_shape)?
        } else {
            xs.matmul(&wt)?
        };
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs)
    }
}

// ---------------------------------------------------------------------------
// LinearNoBias
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LinearNoBias {
    weight: Tensor,
}

impl LinearNoBias {
    pub fn new(in_features: usize, out_features: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_features, in_features), "weight")?;
        Ok(Self { weight })
    }
}

impl Module for LinearNoBias {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let wt = self.weight.t()?;
        if xs.rank() > 2 {
            let dims = xs.dims();
            let last = *dims.last().unwrap();
            let batch_size: usize = dims[..dims.len() - 1].iter().product();
            let flat = xs.reshape((batch_size, last))?;
            let result = flat.matmul(&wt)?;
            let mut out_shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
            out_shape.push(result.dim(D::Minus1)?);
            result.reshape(out_shape)
        } else {
            xs.matmul(&wt)
        }
    }
}

// ---------------------------------------------------------------------------
// LayerNorm
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
    normalized_shape: usize,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self {
            weight,
            bias,
            eps,
            normalized_shape: dim,
        })
    }

    pub fn new_no_bias(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = Tensor::zeros(dim, weight.dtype(), weight.device())?;
        Ok(Self {
            weight,
            bias,
            eps,
            normalized_shape: dim,
        })
    }

    pub fn forward_t(&self, xs: &Tensor) -> Result<Tensor> {
        let mean = xs.mean_keepdim(D::Minus1)?;
        let centered = xs.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let inv_std = (var + self.eps)?.sqrt()?.recip()?;
        let normed = centered.broadcast_mul(&inv_std)?;
        normed.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.forward_t(xs)
    }
}

// ---------------------------------------------------------------------------
// AdaptiveLayerNorm  –  output = sigmoid(gate(LN(s))) * LN(a) + value(LN(s))
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AdaptiveLayerNorm {
    a_norm: LayerNorm,
    s_norm: LayerNorm,
    gate_linear: Linear,
    value_linear: LinearNoBias,
}

impl AdaptiveLayerNorm {
    pub fn new(c_a: usize, c_s: usize, vb: VarBuilder) -> Result<Self> {
        let a_norm = LayerNorm::new(c_a, 1e-5, vb.pp("a_norm"))?;
        let s_norm = LayerNorm::new(c_s, 1e-5, vb.pp("s_norm"))?;
        let gate_linear = Linear::new_with_bias(c_s, c_a, vb.pp("gate_linear"))?;
        let value_linear = LinearNoBias::new(c_s, c_a, vb.pp("value_linear"))?;
        Ok(Self {
            a_norm,
            s_norm,
            gate_linear,
            value_linear,
        })
    }

    pub fn forward(&self, a: &Tensor, s: &Tensor) -> Result<Tensor> {
        let a_norm = self.a_norm.forward(a)?;
        let s_norm = self.s_norm.forward(s)?;
        let gate = sigmoid(&self.gate_linear.forward(&s_norm)?)?;
        let value = self.value_linear.forward(&s_norm)?;
        gate.mul(&a_norm)?.add(&value)
    }
}

// ---------------------------------------------------------------------------
// Transition  –  LN → expand → SiLU-gate → contract (with chunked eval)
// ---------------------------------------------------------------------------

const TRANSITION_CHUNK_SIZE: usize = 3200;

#[derive(Debug, Clone)]
pub struct Transition {
    layer_norm: LayerNorm,
    linear_in: LinearNoBias,
    linear_out: LinearNoBias,
    chunk_size: usize,
}

impl Transition {
    pub fn new(dim: usize, n: usize, vb: VarBuilder) -> Result<Self> {
        let layer_norm = LayerNorm::new(dim, 1e-5, vb.pp("layer_norm"))?;
        let linear_in = LinearNoBias::new(dim, dim * n * 2, vb.pp("linear_in"))?;
        let linear_out = LinearNoBias::new(dim * n, dim, vb.pp("linear_out"))?;
        Ok(Self {
            layer_norm,
            linear_in,
            linear_out,
            chunk_size: TRANSITION_CHUNK_SIZE,
        })
    }

    fn forward_chunk(&self, xs: &Tensor) -> Result<Tensor> {
        let normed = self.layer_norm.forward(xs)?;
        let expanded = self.linear_in.forward(&normed)?;
        let chunks = expanded.chunk(2, D::Minus1)?;
        let gate = silu(&chunks[0])?;
        let val = &chunks[1];
        let gated = gate.mul(val)?;
        self.linear_out.forward(&gated)
    }
}

impl Module for Transition {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims();
        let seq_len = if dims.len() >= 2 { dims[dims.len() - 2] } else { 1 };

        if seq_len <= self.chunk_size {
            return self.forward_chunk(xs);
        }

        let rank = dims.len();
        let seq_dim = rank - 2;
        let mut outputs: Vec<Tensor> = Vec::new();
        let mut start = 0;
        while start < seq_len {
            let end = (start + self.chunk_size).min(seq_len);
            let chunk = xs.narrow(seq_dim, start, end - start)?;
            outputs.push(self.forward_chunk(&chunk)?);
            start = end;
        }
        Tensor::cat(&outputs, seq_dim)
    }
}

// ---------------------------------------------------------------------------
// Scaled dot-product attention  –  q,k,v: [B, H, N, D]
// ---------------------------------------------------------------------------

pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bias: Option<&Tensor>,
    mask: Option<&Tensor>,
) -> Result<Tensor> {
    let d_k = q.dim(D::Minus1)?;
    let scale = 1.0 / (d_k as f64).sqrt();

    let rank = q.rank();
    let scores = q.matmul(&k.transpose(rank - 2, rank - 1)?)?;
    let scores = (scores * scale)?;

    let scores = match bias {
        Some(b) => scores.broadcast_add(b)?,
        None => scores,
    };

    let scores = match mask {
        Some(m) => {
            let large_neg = Tensor::new(f32::NEG_INFINITY, scores.device())?
                .to_dtype(scores.dtype())?
                .broadcast_as(scores.shape())?;
            m.where_cond(&scores, &large_neg)?
        }
        None => scores,
    };

    let attn_weights = softmax(&scores, rank - 1)?;
    attn_weights.matmul(v)
}

// ---------------------------------------------------------------------------
// DropPath  –  stochastic depth (passthrough during inference)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DropPath {
    _drop_prob: f64,
}

impl DropPath {
    pub fn new(drop_prob: f64) -> Self {
        Self {
            _drop_prob: drop_prob,
        }
    }
}

impl Module for DropPath {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Ok(xs.clone())
    }
}
