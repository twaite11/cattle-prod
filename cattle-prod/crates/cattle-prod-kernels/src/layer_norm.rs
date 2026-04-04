use candle_core::{DType, Device, Result, Tensor};

/// Fused Layer Normalization over the last dimension of `input`.
///
/// Computes: output = weight * (input - mean) / sqrt(var + eps) + bias
///
/// # CPU path
/// Computes mean and variance in a single pass, normalizes, then applies
/// the affine scale (`weight`) and shift (`bias`).
///
/// # CUDA path (future)
/// The GPU implementation should load a single PTX/CUBIN kernel that:
///   1. Reads one row (last dim) per thread-block into shared memory.
///   2. Performs a parallel reduction for sum and sum-of-squares.
///   3. Normalizes and applies scale+shift in a single fused write-back.
///
/// Kernel signature (pseudocode):
/// ```text
/// __global__ void fused_layer_norm_kernel(
///     const float* __restrict__ input,   // [outer, D]
///     const float* __restrict__ weight,   // [D]
///     const float* __restrict__ bias,     // [D] (nullable)
///     float* __restrict__ output,         // [outer, D]
///     int D,
///     float eps
/// );
/// ```
/// Load via `cudarc::driver::LaunchConfig` with grid = outer, block = min(D, 1024).
pub fn fused_layer_norm(
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> Result<Tensor> {
    match input.device() {
        Device::Cpu => fused_layer_norm_cpu(input, weight, Some(bias), eps),
        Device::Cuda(_) => {
            log::warn!("CUDA fused_layer_norm not yet implemented, falling back to CPU-style ops");
            fused_layer_norm_cpu(input, weight, Some(bias), eps)
        }
        Device::Metal(_) => {
            log::warn!("Metal fused_layer_norm not yet implemented, falling back to CPU-style ops");
            fused_layer_norm_cpu(input, weight, Some(bias), eps)
        }
    }
}

/// Layer normalization without a bias term.
///
/// Computes: output = weight * (input - mean) / sqrt(var + eps)
pub fn fused_layer_norm_no_bias(input: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    match input.device() {
        Device::Cpu => fused_layer_norm_cpu(input, weight, None, eps),
        Device::Cuda(_) => {
            log::warn!(
                "CUDA fused_layer_norm_no_bias not yet implemented, falling back to CPU-style ops"
            );
            fused_layer_norm_cpu(input, weight, None, eps)
        }
        Device::Metal(_) => {
            log::warn!(
                "Metal fused_layer_norm_no_bias not yet implemented, falling back to CPU-style ops"
            );
            fused_layer_norm_cpu(input, weight, None, eps)
        }
    }
}

/// CPU implementation of fused layer norm using candle tensor operations.
///
/// For an input of shape [..., D], computes layer norm over the last dimension.
fn fused_layer_norm_cpu(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    eps: f64,
) -> Result<Tensor> {
    let dtype = input.dtype();
    let internal_dtype = match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    };

    let input = if dtype != internal_dtype {
        input.to_dtype(internal_dtype)?
    } else {
        input.clone()
    };

    let dim = input.dims().len() - 1;

    let mean = input.mean_keepdim(dim)?;
    let centered = input.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(dim)?;
    let inv_std = (var + eps)?.sqrt()?.recip()?;
    let normed = centered.broadcast_mul(&inv_std)?;

    let weight = if weight.dtype() != internal_dtype {
        weight.to_dtype(internal_dtype)?
    } else {
        weight.clone()
    };

    let mut out = normed.broadcast_mul(&weight)?;

    if let Some(b) = bias {
        let b = if b.dtype() != internal_dtype {
            b.to_dtype(internal_dtype)?
        } else {
            b.clone()
        };
        out = out.broadcast_add(&b)?;
    }

    if out.dtype() != dtype {
        out = out.to_dtype(dtype)?;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_layer_norm_basic() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::new(&[[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]], dev)?;
        let weight = Tensor::ones(3, DType::F32, dev)?;
        let bias = Tensor::zeros(3, DType::F32, dev)?;

        let out = fused_layer_norm(&input, &weight, &bias, 1e-5)?;
        assert_eq!(out.dims(), &[2, 3]);

        let vals: Vec<Vec<f32>> = out.to_vec2()?;
        for row in &vals {
            let mean: f32 = row.iter().sum::<f32>() / 3.0;
            assert!((mean).abs() < 1e-5, "mean should be ~0, got {mean}");
        }
        Ok(())
    }

    #[test]
    fn test_layer_norm_no_bias() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::new(&[[10.0_f32, 20.0, 30.0]], dev)?;
        let weight = Tensor::new(&[2.0_f32, 2.0, 2.0], dev)?;

        let out = fused_layer_norm_no_bias(&input, &weight, 1e-5)?;
        assert_eq!(out.dims(), &[1, 3]);
        Ok(())
    }

    #[test]
    fn test_layer_norm_higher_rank() -> Result<()> {
        let dev = &Device::Cpu;
        let input = Tensor::randn(0.0_f32, 1.0, &[2, 4, 8], dev)?;
        let weight = Tensor::ones(8, DType::F32, dev)?;
        let bias = Tensor::zeros(8, DType::F32, dev)?;

        let out = fused_layer_norm(&input, &weight, &bias, 1e-5)?;
        assert_eq!(out.dims(), &[2, 4, 8]);
        Ok(())
    }
}
