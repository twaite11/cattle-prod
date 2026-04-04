use candle_core::{Result, Tensor, D};

use crate::primitives::softmax;

/// Scaled dot-product attention with optional masking and causal support.
///
/// q, k, v: `[..., seq_len, head_dim]`  (arbitrary batch/head leading dims)
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attn_mask: Option<&Tensor>,
    is_causal: bool,
) -> Result<Tensor> {
    let d_k = q.dim(D::Minus1)?;
    let scale = 1.0 / (d_k as f64).sqrt();

    let rank = q.rank();
    let scores = q.matmul(&k.transpose(rank - 2, rank - 1)?)?;
    let scores = (scores * scale)?;

    let scores = if let Some(mask) = attn_mask {
        scores.broadcast_add(mask)?
    } else {
        scores
    };

    let scores = if is_causal {
        let seq_q = q.dim(rank - 2)?;
        let seq_k = k.dim(rank - 2)?;
        let causal = build_causal_mask(seq_q, seq_k, scores.dtype(), scores.device())?;
        scores.broadcast_add(&causal)?
    } else {
        scores
    };

    let attn_weights = softmax(&scores, rank - 1)?;
    attn_weights.matmul(v)
}

fn build_causal_mask(
    seq_q: usize,
    seq_k: usize,
    dtype: candle_core::DType,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let _mask = Tensor::zeros((seq_q, seq_k), dtype, device)?;
    let neg_inf = f32::NEG_INFINITY;
    let mut data = vec![0f32; seq_q * seq_k];
    for i in 0..seq_q {
        for j in 0..seq_k {
            if j > i + (seq_k - seq_q) {
                data[i * seq_k + j] = neg_inf;
            }
        }
    }
    Tensor::new(data, device)?
        .reshape((seq_q, seq_k))?
        .to_dtype(dtype)
}
