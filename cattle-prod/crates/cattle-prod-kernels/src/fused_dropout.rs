use candle_core::{DType, Device, Result, Tensor};

/// Fused dropout + residual addition.
///
/// Computes: `output = residual + dropout(x, rate, training)`
///
/// When `training` is false, this reduces to a simple addition (no dropout).
///
/// # Row-wise dropout
/// The dropout mask is shared along dimension -3 (i.e. the same mask row is
/// applied to all positions in that dimension). For a tensor of shape
/// `[..., A, B, C]`, the mask has shape `[..., 1, B, C]` and is broadcast.
///
/// # CPU path
/// Generates a Bernoulli mask using candle's random facilities, scales the
/// kept values by `1/(1-rate)`, then adds the residual.
///
/// # CUDA path (future)
/// A single kernel fuses mask generation (Philox RNG), scaling, and addition:
/// ```text
/// __global__ void fused_dropout_add_kernel(
///     const float* __restrict__ residual,  // [N]
///     const float* __restrict__ x,         // [N]
///     float* __restrict__ output,          // [N]
///     const unsigned long long seed,
///     const unsigned long long offset,
///     float p_keep,           // 1 - dropout_rate
///     float scale,            // 1 / p_keep
///     int N,
///     int row_stride,         // stride of dim -3 for row-wise mask sharing
///     bool training
/// );
/// ```
/// Each thread computes `philox4(seed, offset + tid)`, uses bit 0 per element
/// for the mask, then writes `residual[i] + (mask ? x[i] * scale : 0)`.
pub fn fused_dropout_add(
    residual: &Tensor,
    x: &Tensor,
    dropout_rate: f64,
    training: bool,
) -> Result<Tensor> {
    if !training || dropout_rate <= 0.0 {
        return residual + x;
    }
    if dropout_rate >= 1.0 {
        return Ok(residual.clone());
    }

    match x.device() {
        Device::Cpu => fused_dropout_add_cpu(residual, x, dropout_rate),
        Device::Cuda(_) => {
            log::warn!(
                "CUDA fused_dropout_add not yet implemented, falling back to CPU-style ops"
            );
            fused_dropout_add_cpu(residual, x, dropout_rate)
        }
        Device::Metal(_) => {
            log::warn!(
                "Metal fused_dropout_add not yet implemented, falling back to CPU-style ops"
            );
            fused_dropout_add_cpu(residual, x, dropout_rate)
        }
    }
}

/// CPU fallback for fused dropout-add with row-wise mask sharing.
fn fused_dropout_add_cpu(residual: &Tensor, x: &Tensor, dropout_rate: f64) -> Result<Tensor> {
    let dtype = x.dtype();
    let internal_dtype = match dtype {
        DType::F16 | DType::BF16 => DType::F32,
        other => other,
    };

    let x = x.to_dtype(internal_dtype)?;
    let residual = residual.to_dtype(internal_dtype)?;

    let dims = x.dims();
    let ndim = dims.len();

    // Build the mask shape: same as x but with dim -3 set to 1 (row-wise sharing).
    let mask_shape: Vec<usize> = if ndim >= 3 {
        dims.iter()
            .enumerate()
            .map(|(i, &d)| if i == ndim - 3 { 1 } else { d })
            .collect()
    } else {
        dims.to_vec()
    };

    // Bernoulli mask: 1 with probability (1 - dropout_rate), 0 otherwise.
    // We use uniform random in [0,1) and threshold at dropout_rate.
    let mask_shape_s: Vec<usize> = mask_shape;
    let uniform = Tensor::rand(0.0_f32, 1.0, mask_shape_s.as_slice(), x.device())?
        .to_dtype(internal_dtype)?;
    let keep_prob = 1.0 - dropout_rate;
    let threshold = Tensor::new(&[dropout_rate as f32], x.device())?
        .to_dtype(internal_dtype)?
        .broadcast_as(mask_shape_s.as_slice())?;

    // mask = (uniform >= dropout_rate) as float — using ge then to_dtype
    let mask = uniform.ge(&threshold)?.to_dtype(internal_dtype)?;

    let scale = 1.0 / keep_prob;
    let dropped = x
        .broadcast_mul(&mask)?
        .affine(scale, 0.0)?;

    let out = (residual + dropped)?;

    if out.dtype() != dtype {
        out.to_dtype(dtype)
    } else {
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_dropout_add_no_training() -> Result<()> {
        let dev = &Device::Cpu;
        let residual = Tensor::new(&[[1.0_f32, 2.0, 3.0]], dev)?;
        let x = Tensor::new(&[[0.1_f32, 0.2, 0.3]], dev)?;

        let out = fused_dropout_add(&residual, &x, 0.5, false)?;
        let expected: Vec<Vec<f32>> = vec![vec![1.1, 2.2, 3.3]];
        let vals: Vec<Vec<f32>> = out.to_vec2()?;
        for (a, b) in vals[0].iter().zip(expected[0].iter()) {
            assert!((a - b).abs() < 1e-5);
        }
        Ok(())
    }

    #[test]
    fn test_dropout_add_rate_zero() -> Result<()> {
        let dev = &Device::Cpu;
        let residual = Tensor::ones(&[2, 3], DType::F32, dev)?;
        let x = Tensor::ones(&[2, 3], DType::F32, dev)?;

        let out = fused_dropout_add(&residual, &x, 0.0, true)?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for v in vals {
            assert!((v - 2.0).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_dropout_add_rate_one() -> Result<()> {
        let dev = &Device::Cpu;
        let residual = Tensor::ones(&[2, 3], DType::F32, dev)?;
        let x = Tensor::ones(&[2, 3], DType::F32, dev)?;

        let out = fused_dropout_add(&residual, &x, 1.0, true)?;
        let vals: Vec<f32> = out.flatten_all()?.to_vec1()?;
        for v in vals {
            assert!((v - 1.0).abs() < 1e-6, "Rate=1 should drop everything, got {v}");
        }
        Ok(())
    }

    #[test]
    fn test_dropout_add_statistical() -> Result<()> {
        // With many elements, the mean of dropout(x) should ≈ x (unbiased).
        let dev = &Device::Cpu;
        let n = 10_000;
        let residual = Tensor::zeros(&[1, n], DType::F32, dev)?;
        let x = Tensor::ones(&[1, n], DType::F32, dev)?;

        let out = fused_dropout_add(&residual, &x, 0.3, true)?;
        let mean = out.mean_all()?.to_scalar::<f32>()?;
        assert!(
            (mean - 1.0).abs() < 0.15,
            "Scaled dropout mean should be ~1.0, got {mean}"
        );
        Ok(())
    }

    #[test]
    fn test_dropout_add_row_wise_mask() -> Result<()> {
        // Dim -3 should share the same mask. For shape [A, B, C], mask is [1, B, C].
        let dev = &Device::Cpu;
        let residual = Tensor::zeros(&[4, 3, 8], DType::F32, dev)?;
        let x = Tensor::ones(&[4, 3, 8], DType::F32, dev)?;

        let out = fused_dropout_add(&residual, &x, 0.5, true)?;
        assert_eq!(out.dims(), &[4, 3, 8]);

        // Check that dim 0 (which is dim -3 for 3-D) shares the mask:
        // rows along dim 0 at same (dim1, dim2) index should all be equal.
        let vals: Vec<Vec<Vec<f32>>> = out.to_vec3()?;
        for j in 0..3 {
            for k in 0..8 {
                let first = vals[0][j][k];
                for i in 1..4 {
                    assert!(
                        (vals[i][j][k] - first).abs() < 1e-6,
                        "Row-wise mask should be shared along dim -3"
                    );
                }
            }
        }
        Ok(())
    }
}
