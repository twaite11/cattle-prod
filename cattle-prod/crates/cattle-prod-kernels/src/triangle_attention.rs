use candle_core::{DType, Device, Result, Tensor, D};

/// Scaled dot-product triangle attention with two additive biases.
///
/// This is the attention variant used in AlphaFold-style triangle updates,
/// where pair representations attend over one edge of the triangle.
///
/// # Shapes
/// - `q`, `k`, `v`: `[B, N, S, H, D]`  (batch, nodes, sequence, heads, head_dim)
/// - `bias1`: `[B, N, 1, S]`            (per-node, broadcast over heads)
/// - `bias2`: `[B, N, S, S, H]`         (full pair-wise attention bias per head)
/// - `scale`: typically `1.0 / sqrt(D)`
///
/// Returns: `[B, N, S, H, D]`
///
/// # CPU path
/// Standard scaled dot-product attention:
///   1. scores = einsum("bnshd,bnthd->bnsht", Q, K) * scale
///   2. scores += bias1.unsqueeze(-1)       — broadcast [B,N,1,S] → [B,N,S,S,H]
///   3. scores += bias2                     — [B,N,S,S,H]
///   4. attn = softmax(scores, dim=3)       — over the T (key) dimension
///   5. out  = einsum("bnsht,bnthd->bnshd", attn, V)
///
/// # CUDA path (future)
/// A fused flash-attention-style kernel that avoids materialising the full
/// [B, N, S, S, H] score matrix. Kernel signature:
/// ```text
/// __global__ void triangle_attention_fwd(
///     const half* __restrict__ Q,       // [B, N, S, H, D]
///     const half* __restrict__ K,       // [B, N, S, H, D]
///     const half* __restrict__ V,       // [B, N, S, H, D]
///     const half* __restrict__ bias1,   // [B, N, 1, S]
///     const half* __restrict__ bias2,   // [B, N, S, S, H]
///     half* __restrict__ O,             // [B, N, S, H, D]
///     float* __restrict__ L,            // [B, N, S, H] logsumexp for bwd
///     int B, int N, int S, int H, int D,
///     float scale
/// );
/// ```
/// Tiling: each thread-block handles one (b, n, s_q, h) query row, iterating
/// over key tiles of size `BLOCK_K` in shared memory.
pub fn triangle_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bias1: &Tensor,
    bias2: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    match q.device() {
        Device::Cpu => triangle_attention_cpu(q, k, v, bias1, bias2, scale),
        Device::Cuda(_) => {
            log::warn!(
                "CUDA triangle_attention not yet implemented, falling back to CPU-style ops"
            );
            triangle_attention_cpu(q, k, v, bias1, bias2, scale)
        }
        Device::Metal(_) => {
            log::warn!(
                "Metal triangle_attention not yet implemented, falling back to CPU-style ops"
            );
            triangle_attention_cpu(q, k, v, bias1, bias2, scale)
        }
    }
}

/// CPU implementation of triangle attention.
///
/// We reshape Q, K, V to merge the batch dims and compute per-head attention
/// using matmuls, then reshape back.
fn triangle_attention_cpu(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    bias1: &Tensor,
    bias2: &Tensor,
    scale: f64,
) -> Result<Tensor> {
    let (b, n, s, h, d) = {
        let dims = q.dims();
        assert_eq!(dims.len(), 5, "Q must be [B, N, S, H, D]");
        (dims[0], dims[1], dims[2], dims[3], dims[4])
    };

    let orig_dtype = q.dtype();
    let internal_dtype = match orig_dtype {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    };
    let q = q.to_dtype(internal_dtype)?;
    let k = k.to_dtype(internal_dtype)?;
    let v = v.to_dtype(internal_dtype)?;
    let bias1 = bias1.to_dtype(internal_dtype)?;
    let bias2 = bias2.to_dtype(internal_dtype)?;

    // Reshape to [B*N, S, H, D] then transpose to [B*N, H, S, D] for batched matmul.
    let q = q.reshape((b * n, s, h, d))?.transpose(1, 2)?.contiguous()?; // [BN, H, S, D]
    let k = k.reshape((b * n, s, h, d))?.transpose(1, 2)?.contiguous()?; // [BN, H, S, D]
    let v = v.reshape((b * n, s, h, d))?.transpose(1, 2)?.contiguous()?; // [BN, H, S, D]

    // scores = Q @ K^T => [BN, H, S, S]
    let k_t = k.transpose(D::Minus2, D::Minus1)?;
    let scores = q.matmul(&k_t)?;
    let scores = (scores * scale)?;

    // Add bias1: [B, N, 1, S] → [B*N, 1, 1, S] → broadcast to [BN, H, S, S]
    let bias1 = bias1
        .reshape((b * n, 1, 1, s))?
        .broadcast_as((b * n, h, s, s))?;
    let scores = scores.broadcast_add(&bias1)?;

    // Add bias2: [B, N, S, S, H] → [B*N, S, S, H] → transpose to [BN, H, S, S]
    let bias2 = bias2
        .reshape((b * n, s, s, h))?
        .permute((0, 3, 1, 2))?
        .contiguous()?;
    let scores = scores.broadcast_add(&bias2)?;

    // Softmax over the key dimension (last dim = S).
    let scores = softmax_last_dim(&scores)?;

    // out = attn @ V => [BN, H, S, D]
    let out = scores.matmul(&v)?;

    // Transpose back to [BN, S, H, D] then reshape to [B, N, S, H, D].
    let out = out.transpose(1, 2)?.contiguous()?.reshape((b, n, s, h, d))?;

    if out.dtype() != orig_dtype { out.to_dtype(orig_dtype) } else { Ok(out) }
}

/// Numerically-stable softmax over the last dimension.
fn softmax_last_dim(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let shifted = x.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(D::Minus1)?;
    exp.broadcast_div(&sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_triangle_attention_shapes() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, n, s, h, d) = (1, 2, 4, 3, 8);

        let q = Tensor::randn(0.0_f32, 1.0, &[b, n, s, h, d], dev)?;
        let k = Tensor::randn(0.0_f32, 1.0, &[b, n, s, h, d], dev)?;
        let v = Tensor::randn(0.0_f32, 1.0, &[b, n, s, h, d], dev)?;
        let bias1 = Tensor::zeros(&[b, n, 1, s], DType::F32, dev)?;
        let bias2 = Tensor::zeros(&[b, n, s, s, h], DType::F32, dev)?;

        let scale = 1.0 / (d as f64).sqrt();
        let out = triangle_attention(&q, &k, &v, &bias1, &bias2, scale)?;

        assert_eq!(out.dims(), &[b, n, s, h, d]);
        Ok(())
    }

    #[test]
    fn test_triangle_attention_softmax_sums_to_one() -> Result<()> {
        let dev = &Device::Cpu;
        let x = Tensor::new(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], dev)?;
        let sm = softmax_last_dim(&x)?;
        let sums: Vec<f32> = sm.sum(D::Minus1)?.to_vec1()?;
        for s in sums {
            assert!((s - 1.0).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_triangle_attention_with_biases() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, n, s, h, d) = (2, 1, 3, 2, 4);

        let q = Tensor::randn(0.0_f32, 0.5, &[b, n, s, h, d], dev)?;
        let k = Tensor::randn(0.0_f32, 0.5, &[b, n, s, h, d], dev)?;
        let v = Tensor::randn(0.0_f32, 0.5, &[b, n, s, h, d], dev)?;
        let bias1 = Tensor::randn(0.0_f32, 0.1, &[b, n, 1, s], dev)?;
        let bias2 = Tensor::randn(0.0_f32, 0.1, &[b, n, s, s, h], dev)?;

        let scale = 1.0 / (d as f64).sqrt();
        let out = triangle_attention(&q, &k, &v, &bias1, &bias2, scale)?;

        assert_eq!(out.dims(), &[b, n, s, h, d]);
        assert!(!out.flatten_all()?.to_vec1::<f32>()?.is_empty());
        Ok(())
    }
}
