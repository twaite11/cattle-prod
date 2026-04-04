use candle_core::{DType, Device, Result, Tensor};

use crate::diffusion::DiffusionModule;
use cattle_prod_core::config::{NoiseSchedulerConfig, SampleDiffusionConfig};

// ---------------------------------------------------------------------------
// InferenceNoiseScheduler  –  polynomial schedule σ_max → σ_min
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct InferenceNoiseScheduler {
    schedule: Vec<f64>,
}

impl InferenceNoiseScheduler {
    pub fn new(cfg: &NoiseSchedulerConfig, n_steps: usize) -> Self {
        let rho_inv = 1.0 / cfg.rho;
        let s_max_rho = cfg.s_max.powf(rho_inv);
        let s_min_rho = cfg.s_min.powf(rho_inv);

        let mut schedule = Vec::with_capacity(n_steps + 1);
        for i in 0..=n_steps {
            let t = i as f64 / n_steps as f64;
            let sigma = (s_max_rho + t * (s_min_rho - s_max_rho)).powf(cfg.rho);
            schedule.push(sigma);
        }
        Self { schedule }
    }

    pub fn sigmas(&self) -> &[f64] {
        &self.schedule
    }

    pub fn len(&self) -> usize {
        self.schedule.len()
    }

    pub fn is_empty(&self) -> bool {
        self.schedule.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Diffusion sampling loop
// ---------------------------------------------------------------------------

/// Run the iterative denoising loop.
///
/// Returns denoised coordinates `[B, N_atom, 3]`.
#[allow(clippy::too_many_arguments)]
pub fn sample_diffusion(
    diffusion_module: &DiffusionModule,
    scheduler: &InferenceNoiseScheduler,
    sample_cfg: &SampleDiffusionConfig,
    s_trunk: &Tensor,
    s_inputs: &Tensor,
    z_pair: &Tensor,
    atom_single: &Tensor,
    atom_pair: &Tensor,
    atom_to_token: &Tensor,
    token_to_atom: &Tensor,
    n_tokens: usize,
    n_atoms: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let batch = s_trunk.dim(0)?;
    let sigmas = scheduler.sigmas();

    let x = Tensor::randn(0f32, 1f32, (batch, n_atoms, 3), device)?
        .to_dtype(dtype)?;
    let mut x = (x * sigmas[0])?;

    for i in 0..(sigmas.len() - 1) {
        let sigma_cur = sigmas[i];
        let sigma_next = sigmas[i + 1];

        let gamma = if sigma_cur >= sample_cfg.gamma_min {
            sample_cfg.gamma0
        } else {
            0.0
        };

        let sigma_hat = sigma_cur * (1.0 + gamma);

        if gamma > 0.0 {
            let noise = Tensor::randn(0f32, 1f32, x.shape(), device)?.to_dtype(dtype)?;
            let noise_scale = (sigma_hat * sigma_hat - sigma_cur * sigma_cur)
                .max(0.0)
                .sqrt()
                * sample_cfg.noise_scale_lambda;
            x = x.add(&(noise * noise_scale)?)?;
        }

        x = centre_random_augmentation(&x, None)?;

        let d = diffusion_module.forward(
            &x,
            sigma_hat,
            s_trunk,
            s_inputs,
            z_pair,
            atom_single,
            atom_pair,
            atom_to_token,
            token_to_atom,
            n_tokens,
        )?;

        let dt = (sigma_next - sigma_hat) * sample_cfg.step_scale_eta;
        let dx = d.broadcast_sub(&x)?;
        let dx = (dx * (1.0 / sigma_hat))?;
        x = x.add(&(dx * dt)?)?;
    }

    Ok(x)
}

/// Centre the structure and optionally apply random rotation + translation.
///
/// During inference the rotation is identity (no noise), so this centres only.
pub fn centre_random_augmentation(
    coords: &Tensor,
    noise_std: Option<f64>,
) -> Result<Tensor> {
    let mean = coords.mean_keepdim(1)?;
    let centered = coords.broadcast_sub(&mean)?;

    match noise_std {
        Some(std) if std > 0.0 => {
            let device = coords.device();
            let dtype = coords.dtype();
            let batch = coords.dim(0)?;
            let noise = Tensor::randn(0f32, std as f32, (batch, 1, 3), device)?
                .to_dtype(dtype)?;
            centered.broadcast_add(&noise)
        }
        _ => Ok(centered),
    }
}
