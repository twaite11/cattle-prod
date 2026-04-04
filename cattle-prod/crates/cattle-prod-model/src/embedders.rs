use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::primitives::{Linear, LinearNoBias};
use cattle_prod_core::config::RelPosEncConfig;

// ---------------------------------------------------------------------------
// Fourier Embedding  –  cos(2π(t·w + b))
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct FourierEmbedding {
    w: Tensor,
    b: Tensor,
}

impl FourierEmbedding {
    pub fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let w = vb.get(dim, "w")?;
        let b = vb.get(dim, "b")?;
        Ok(Self { w, b })
    }
}

impl Module for FourierEmbedding {
    fn forward(&self, t: &Tensor) -> Result<Tensor> {
        let t_exp = if t.rank() == 0 || t.dim(D::Minus1)? == 1 {
            t.unsqueeze(D::Minus1)?
        } else {
            t.clone()
        };
        let inner = t_exp.broadcast_mul(&self.w)?.broadcast_add(&self.b)?;
        let two_pi = 2.0 * std::f64::consts::PI;
        (inner * two_pi)?.cos()
    }
}

// ---------------------------------------------------------------------------
// Relative Position Encoding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct RelativePositionEncoding {
    linear: LinearNoBias,
    r_max: usize,
    s_max: usize,
}

impl RelativePositionEncoding {
    pub fn new(cfg: &RelPosEncConfig, vb: VarBuilder) -> Result<Self> {
        let feat_dim = 2 * cfg.r_max + 2 + 2 * cfg.s_max + 2;
        let linear = LinearNoBias::new(feat_dim, cfg.c_z, vb.pp("linear"))?;
        Ok(Self {
            linear,
            r_max: cfg.r_max,
            s_max: cfg.s_max,
        })
    }

    /// Build one-hot relative position features.
    /// `residue_index`: `[N_token]` i64, `asym_id`: `[N_token]` i64.
    pub fn generate_relp(
        &self,
        residue_index: &Tensor,
        asym_id: &Tensor,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let _n = residue_index.dim(0)?;
        let ri_f = residue_index.to_dtype(DType::F32)?;

        let ri_i = ri_f.unsqueeze(1)?;
        let ri_j = ri_f.unsqueeze(0)?;
        let d_res = ri_i.broadcast_sub(&ri_j)?;

        let r_max = self.r_max as f64;
        let d_clamp = d_res.clamp(-r_max, r_max)?;
        let d_shifted = (d_clamp + r_max)?;
        let num_bins_r = 2 * self.r_max + 2;
        let relp_residue = one_hot(&d_shifted, num_bins_r, device)?;

        let ai_f = asym_id.to_dtype(DType::F32)?;
        let ai_i = ai_f.unsqueeze(1)?;
        let ai_j = ai_f.unsqueeze(0)?;
        let d_asym = ai_i.broadcast_sub(&ai_j)?;

        let s_max = self.s_max as f64;
        let d_clamp_s = d_asym.clamp(-s_max, s_max)?;
        let d_shifted_s = (d_clamp_s + s_max)?;
        let num_bins_s = 2 * self.s_max + 2;
        let relp_chain = one_hot(&d_shifted_s, num_bins_s, device)?;

        let relp = Tensor::cat(&[relp_residue, relp_chain], D::Minus1)?;
        relp.to_dtype(dtype)
    }

    pub fn forward(
        &self,
        residue_index: &Tensor,
        asym_id: &Tensor,
    ) -> Result<Tensor> {
        let device = residue_index.device();
        let relp = self.generate_relp(residue_index, asym_id, device, DType::F32)?;
        self.linear.forward(&relp)
    }
}

fn one_hot(indices: &Tensor, num_classes: usize, device: &Device) -> Result<Tensor> {
    let flat = indices.flatten_all()?;
    let _n = flat.dim(0)?;
    let idx_i64 = flat.to_dtype(DType::U32)?;
    let eye = Tensor::eye(num_classes, DType::F32, device)?;
    let oh = eye.index_select(&idx_i64, 0)?;
    let shape = indices.dims();
    let mut new_shape: Vec<usize> = shape.to_vec();
    new_shape.push(num_classes);
    oh.reshape(new_shape)
}

// ---------------------------------------------------------------------------
// InputFeatureEmbedder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct InputFeatureEmbedder {
    proj: Linear,
    c_token: usize,
}

impl InputFeatureEmbedder {
    pub fn new(c_s_inputs: usize, c_token: usize, vb: VarBuilder) -> Result<Self> {
        let proj = Linear::new(c_s_inputs, c_token, vb.pp("proj"))?;
        Ok(Self { proj, c_token })
    }

    /// Combine atom encoder output with per-token features to produce s_inputs.
    ///
    /// * `atom_encoder_out` – `[B, N_token, c_atom]`
    /// * `restype`          – `[B, N_token, 32]`  one-hot residue type
    /// * `profile`          – `[B, N_token, 32]`  sequence profile
    /// * `deletion_mean`    – `[B, N_token, 1]`
    pub fn forward(
        &self,
        atom_encoder_out: &Tensor,
        restype: &Tensor,
        profile: &Tensor,
        deletion_mean: &Tensor,
    ) -> Result<Tensor> {
        let cat = Tensor::cat(
            &[
                atom_encoder_out.clone(),
                restype.clone(),
                profile.clone(),
                deletion_mean.clone(),
            ],
            D::Minus1,
        )?;
        self.proj.forward(&cat)
    }
}
