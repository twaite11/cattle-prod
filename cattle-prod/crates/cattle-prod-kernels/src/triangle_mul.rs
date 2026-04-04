use candle_core::{DType, Device, Result, Tensor, D};

const DEFAULT_CHUNK_SIZE: usize = 128;

/// Triangular multiplicative update — outgoing edges.
///
/// For pair representation `z` of shape `[B, N, N, C]` and mask `[B, N, N]`:
///
///   out_{i,j} = Σ_k  a_{i,k} * b_{j,k}
///
/// where `a` and `b` are linear projections of `z` (here we use `z` directly
/// to implement the core contraction; the caller applies linear projections).
///
/// The masked version zeros out positions where `mask == 0`.
///
/// Processing is chunked along the `i` dimension for memory efficiency on
/// long sequences.
///
/// # CUDA path (future)
/// A custom kernel would tile the `k`-reduction and fuse the mask application:
/// ```text
/// __global__ void triangle_mul_outgoing_kernel(
///     const float* __restrict__ z,       // [B, N, N, C]
///     const bool*  __restrict__ mask,     // [B, N, N]
///     float* __restrict__ out,            // [B, N, N, C]
///     int B, int N, int C,
///     int chunk_start, int chunk_end
/// );
/// ```
pub fn triangle_mul_outgoing(z: &Tensor, mask: &Tensor) -> Result<Tensor> {
    match z.device() {
        Device::Cpu => triangle_mul_outgoing_cpu(z, mask, DEFAULT_CHUNK_SIZE),
        Device::Cuda(_) => {
            log::warn!(
                "CUDA triangle_mul_outgoing not yet implemented, falling back to CPU-style ops"
            );
            triangle_mul_outgoing_cpu(z, mask, DEFAULT_CHUNK_SIZE)
        }
        Device::Metal(_) => {
            log::warn!(
                "Metal triangle_mul_outgoing not yet implemented, falling back to CPU-style ops"
            );
            triangle_mul_outgoing_cpu(z, mask, DEFAULT_CHUNK_SIZE)
        }
    }
}

/// Triangular multiplicative update — incoming edges.
///
/// For pair representation `z` of shape `[B, N, N, C]` and mask `[B, N, N]`:
///
///   out_{i,j} = Σ_k  a_{k,i} * b_{k,j}
///
/// This is the transpose of the outgoing contraction (summing over the
/// first spatial index instead of the second).
///
/// # CUDA path (future)
/// ```text
/// __global__ void triangle_mul_incoming_kernel(
///     const float* __restrict__ z,       // [B, N, N, C]
///     const bool*  __restrict__ mask,     // [B, N, N]
///     float* __restrict__ out,            // [B, N, N, C]
///     int B, int N, int C,
///     int chunk_start, int chunk_end
/// );
/// ```
pub fn triangle_mul_incoming(z: &Tensor, mask: &Tensor) -> Result<Tensor> {
    match z.device() {
        Device::Cpu => triangle_mul_incoming_cpu(z, mask, DEFAULT_CHUNK_SIZE),
        Device::Cuda(_) => {
            log::warn!(
                "CUDA triangle_mul_incoming not yet implemented, falling back to CPU-style ops"
            );
            triangle_mul_incoming_cpu(z, mask, DEFAULT_CHUNK_SIZE)
        }
        Device::Metal(_) => {
            log::warn!(
                "Metal triangle_mul_incoming not yet implemented, falling back to CPU-style ops"
            );
            triangle_mul_incoming_cpu(z, mask, DEFAULT_CHUNK_SIZE)
        }
    }
}

/// Outgoing: out_{b,i,j,c} = Σ_k  (z_{b,i,k,c} * mask_{b,i,k}) * (z_{b,j,k,c} * mask_{b,j,k})
///
/// Implemented as a batched matmul:
///   a = z * mask.unsqueeze(-1)   → [B, N, N, C]
///   out = a @ a^T  (over k-dim)  → but we need per-channel, so we transpose to
///   [B, C, N, N] and do matmul, yielding [B, C, N, N], then transpose back.
fn triangle_mul_outgoing_cpu(z: &Tensor, mask: &Tensor, chunk_size: usize) -> Result<Tensor> {
    let (_b, n, _n2, _c) = dims4(z)?;

    let orig_dtype = z.dtype();
    let internal_dtype = promote_dtype(orig_dtype);
    let z = z.to_dtype(internal_dtype)?;
    let mask = mask.to_dtype(internal_dtype)?;

    // masked_z: [B, N, N, C]
    let mask_expanded = mask.unsqueeze(D::Minus1)?; // [B, N, N, 1]
    let masked_z = z.broadcast_mul(&mask_expanded)?; // [B, N, N, C]

    // Transpose to [B, C, N, N] for batched matmul over the k dimension.
    let a = masked_z.permute((0, 3, 1, 2))?.contiguous()?; // [B, C, N, N]

    if n <= chunk_size {
        // a_t = [B, C, N, N]^T over last two dims = [B, C, N, N]
        let a_t = a.transpose(D::Minus2, D::Minus1)?;
        // [B, C, N, N] @ [B, C, N, N] = [B, C, N, N]
        let out = a.matmul(&a_t)?;
        let out = out.permute((0, 2, 3, 1))?.contiguous()?;
        return if out.dtype() != orig_dtype { out.to_dtype(orig_dtype) } else { Ok(out) };
    }

    // Chunked processing along the i dimension.
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < n {
        let end = (start + chunk_size).min(n);
        let a_chunk = a.narrow(2, start, end - start)?; // [B, C, chunk, N]
        let a_t = a.transpose(D::Minus2, D::Minus1)?;   // [B, C, N, N]
        let out_chunk = a_chunk.matmul(&a_t)?;            // [B, C, chunk, N]
        chunks.push(out_chunk);
        start = end;
    }

    let out = Tensor::cat(&chunks, 2)?; // [B, C, N, N]
    let out = out.permute((0, 2, 3, 1))?.contiguous()?;
    if out.dtype() != orig_dtype { out.to_dtype(orig_dtype) } else { Ok(out) }
}

/// Incoming: out_{b,i,j,c} = Σ_k  (z_{b,k,i,c} * mask_{b,k,i}) * (z_{b,k,j,c} * mask_{b,k,j})
///
/// Implemented by transposing the spatial dims of the masked input, then
/// performing the same matmul pattern as outgoing.
fn triangle_mul_incoming_cpu(z: &Tensor, mask: &Tensor, chunk_size: usize) -> Result<Tensor> {
    let (_b, n, _n2, _c) = dims4(z)?;

    let orig_dtype = z.dtype();
    let internal_dtype = promote_dtype(orig_dtype);
    let z = z.to_dtype(internal_dtype)?;
    let mask = mask.to_dtype(internal_dtype)?;

    let mask_expanded = mask.unsqueeze(D::Minus1)?; // [B, N, N, 1]
    let masked_z = z.broadcast_mul(&mask_expanded)?; // [B, N, N, C]

    // For incoming, we need z_{b,k,i,c} — transpose spatial dims 1 and 2.
    // a_{b,c,k,i} from z_{b,k,i,c}
    let a = masked_z.permute((0, 3, 1, 2))?.contiguous()?; // [B, C, N(k), N(i)]

    if n <= chunk_size {
        // out_{b,c,i,j} = Σ_k a_{b,c,k,i} * a_{b,c,k,j} = a^T @ a
        let a_t = a.transpose(D::Minus2, D::Minus1)?; // [B, C, N(i), N(k)]
        let out = a_t.matmul(&a)?; // [B, C, N(i), N(j)]
        let out = out.permute((0, 2, 3, 1))?.contiguous()?;
        return if out.dtype() != orig_dtype { out.to_dtype(orig_dtype) } else { Ok(out) };
    }

    // Chunked processing along the i dimension.
    let a_t = a.transpose(D::Minus2, D::Minus1)?.contiguous()?; // [B, C, N(i), N(k)]
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < n {
        let end = (start + chunk_size).min(n);
        let a_t_chunk = a_t.narrow(2, start, end - start)?; // [B, C, chunk, N(k)]
        let out_chunk = a_t_chunk.matmul(&a)?;                // [B, C, chunk, N(j)]
        chunks.push(out_chunk);
        start = end;
    }

    let out = Tensor::cat(&chunks, 2)?; // [B, C, N, N]
    let out = out.permute((0, 2, 3, 1))?.contiguous()?;
    if out.dtype() != orig_dtype { out.to_dtype(orig_dtype) } else { Ok(out) }
}

fn dims4(t: &Tensor) -> Result<(usize, usize, usize, usize)> {
    let d = t.dims();
    if d.len() != 4 {
        candle_core::bail!("Expected 4-D tensor [B, N, N, C], got {:?}", d);
    }
    Ok((d[0], d[1], d[2], d[3]))
}

fn promote_dtype(dtype: DType) -> DType {
    match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_triangle_mul_outgoing_shapes() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, n, c) = (1, 5, 4);
        let z = Tensor::randn(0.0_f32, 1.0, &[b, n, n, c], dev)?;
        let mask = Tensor::ones(&[b, n, n], DType::F32, dev)?;

        let out = triangle_mul_outgoing(&z, &mask)?;
        assert_eq!(out.dims(), &[b, n, n, c]);
        Ok(())
    }

    #[test]
    fn test_triangle_mul_incoming_shapes() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, n, c) = (2, 4, 3);
        let z = Tensor::randn(0.0_f32, 1.0, &[b, n, n, c], dev)?;
        let mask = Tensor::ones(&[b, n, n], DType::F32, dev)?;

        let out = triangle_mul_incoming(&z, &mask)?;
        assert_eq!(out.dims(), &[b, n, n, c]);
        Ok(())
    }

    #[test]
    fn test_triangle_mul_mask_zeros() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, n, c) = (1, 3, 2);
        let z = Tensor::randn(0.0_f32, 1.0, &[b, n, n, c], dev)?;
        let mask = Tensor::zeros(&[b, n, n], DType::F32, dev)?;

        let out = triangle_mul_outgoing(&z, &mask)?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for v in &vals {
            assert!(v.abs() < 1e-10, "Expected zero with zero mask, got {v}");
        }
        Ok(())
    }

    #[test]
    fn test_triangle_mul_symmetry() -> Result<()> {
        // For outgoing with symmetric z and full mask, out should be symmetric in (i,j).
        let dev = &Device::Cpu;
        let (b, n, c) = (1, 4, 2);
        let z_raw = Tensor::randn(0.0_f32, 1.0, &[b, n, n, c], dev)?;
        let z_t = z_raw.transpose(1, 2)?;
        let z_sym = ((&z_raw + &z_t)? * 0.5)?;
        let mask = Tensor::ones(&[b, n, n], DType::F32, dev)?;

        let out = triangle_mul_outgoing(&z_sym, &mask)?;
        let out_t = out.transpose(1, 2)?;
        let diff = (&out - &out_t)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "Expected symmetric output, max diff = {diff}");
        Ok(())
    }

    #[test]
    fn test_triangle_mul_chunked_matches_unchunked() -> Result<()> {
        let dev = &Device::Cpu;
        let (b, n, c) = (1, 10, 4);
        let z = Tensor::randn(0.0_f32, 1.0, &[b, n, n, c], dev)?;
        let mask = Tensor::ones(&[b, n, n], DType::F32, dev)?;

        let full = triangle_mul_outgoing_cpu(&z, &mask, 1024)?;
        let chunked = triangle_mul_outgoing_cpu(&z, &mask, 3)?;

        let diff = (&full - &chunked)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(diff < 1e-4, "Chunked and full should match, diff = {diff}");
        Ok(())
    }
}
