use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::embedders::FourierEmbedding;
use crate::primitives::{LayerNorm, Linear, LinearNoBias, Transition};
use crate::transformer::{AtomAttentionDecoder, AtomAttentionEncoder, DiffusionTransformer};
use cattle_prod_core::config::DiffusionModuleConfig;

// ---------------------------------------------------------------------------
// DiffusionConditioning  –  prepare s/z conditioning for the diffusion net
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DiffusionConditioning {
    proj_s: Linear,
    layer_norm_s: LayerNorm,
    fourier: FourierEmbedding,
    proj_fourier: Linear,
    transition_1: Transition,
    transition_2: Transition,
    proj_z: LinearNoBias,
    layer_norm_z: LayerNorm,
    c_token: usize,
}

impl DiffusionConditioning {
    pub fn new(cfg: &DiffusionModuleConfig, vb: VarBuilder) -> Result<Self> {
        let c_s_in = cfg.c_s + cfg.c_s_inputs;
        let proj_s = Linear::new(c_s_in, cfg.c_token, vb.pp("proj_s"))?;
        let layer_norm_s = LayerNorm::new(cfg.c_token, 1e-5, vb.pp("layer_norm_s"))?;
        let fourier_dim = 256;
        let fourier = FourierEmbedding::new(fourier_dim, vb.pp("fourier"))?;
        let proj_fourier = Linear::new(fourier_dim, cfg.c_token, vb.pp("proj_fourier"))?;
        let transition_1 = Transition::new(cfg.c_token, 2, vb.pp("transition_1"))?;
        let transition_2 = Transition::new(cfg.c_token, 2, vb.pp("transition_2"))?;
        let proj_z = LinearNoBias::new(cfg.c_z, cfg.c_z, vb.pp("proj_z"))?;
        let layer_norm_z = LayerNorm::new(cfg.c_z, 1e-5, vb.pp("layer_norm_z"))?;

        Ok(Self {
            proj_s,
            layer_norm_s,
            fourier,
            proj_fourier,
            transition_1,
            transition_2,
            proj_z,
            layer_norm_z,
            c_token: cfg.c_token,
        })
    }

    /// Prepare pair conditioning (run once then cache).
    ///
    /// * `z_trunk` – `[B, N, N, c_z]` from pairformer
    /// * `relp`    – `[B, N, N, c_z]` relative position encoding
    pub fn prepare_pair_cache(
        &self,
        z_trunk: &Tensor,
        relp: &Tensor,
    ) -> Result<Tensor> {
        let z = z_trunk.add(relp)?;
        let z = self.layer_norm_z.forward(&z)?;
        self.proj_z.forward(&z)
    }

    /// Compute the single-track conditioning signal.
    ///
    /// * `s_trunk`   – `[B, N, c_s]`
    /// * `s_inputs`  – `[B, N, c_s_inputs]`
    /// * `noise_level` – scalar or `[B]`
    pub fn forward(
        &self,
        s_trunk: &Tensor,
        s_inputs: &Tensor,
        noise_level: &Tensor,
    ) -> Result<Tensor> {
        let s_cat = Tensor::cat(&[s_trunk, s_inputs], D::Minus1)?;
        let s = self.proj_s.forward(&s_cat)?;
        let s = self.layer_norm_s.forward(&s)?;

        let noise_emb = self.fourier.forward(noise_level)?;
        let noise_proj = self.proj_fourier.forward(&noise_emb)?;

        let noise_proj = if noise_proj.rank() < s.rank() {
            noise_proj.unsqueeze(1)?.broadcast_as(s.shape())?
        } else {
            noise_proj
        };

        let s = s.add(&noise_proj)?;
        let s = self.transition_1.forward(&s)?;
        self.transition_2.forward(&s)
    }
}

// ---------------------------------------------------------------------------
// DiffusionModule  –  EDM-style denoising network
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DiffusionModule {
    pub conditioning: DiffusionConditioning,
    atom_encoder: AtomAttentionEncoder,
    transformer: DiffusionTransformer,
    atom_decoder: AtomAttentionDecoder,
    sigma_data: f64,
    c_token: usize,
    c_atom: usize,
}

impl DiffusionModule {
    pub fn new(cfg: &DiffusionModuleConfig, vb: VarBuilder) -> Result<Self> {
        let conditioning = DiffusionConditioning::new(cfg, vb.pp("conditioning"))?;
        let atom_encoder = AtomAttentionEncoder::new(
            cfg.c_token,
            cfg.c_atom,
            cfg.c_atompair,
            cfg.atom_encoder_n_blocks,
            cfg.atom_encoder_n_heads,
            vb.pp("atom_encoder"),
        )?;
        let transformer = DiffusionTransformer::new(
            cfg.transformer_n_blocks,
            cfg.c_token,
            cfg.c_z,
            cfg.transformer_n_heads,
            vb.pp("transformer"),
        )?;
        let atom_decoder = AtomAttentionDecoder::new(
            cfg.c_token,
            cfg.c_atom,
            cfg.c_atompair,
            cfg.atom_decoder_n_blocks,
            cfg.atom_decoder_n_heads,
            vb.pp("atom_decoder"),
        )?;

        Ok(Self {
            conditioning,
            atom_encoder,
            transformer,
            atom_decoder,
            sigma_data: cfg.sigma_data,
            c_token: cfg.c_token,
            c_atom: cfg.c_atom,
        })
    }

    /// EDM preconditioning:
    ///   c_skip  = sigma_data^2 / (sigma^2 + sigma_data^2)
    ///   c_out   = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
    fn edm_precond(&self, sigma: f64) -> (f64, f64) {
        let sd2 = self.sigma_data * self.sigma_data;
        let s2 = sigma * sigma;
        let c_skip = sd2 / (s2 + sd2);
        let c_out = sigma * self.sigma_data / (s2 + sd2).sqrt();
        (c_skip, c_out)
    }

    /// Full denoising forward pass.
    ///
    /// * `x_noisy`       – `[B, N_atom, 3]` noised coordinates
    /// * `sigma`         – noise level (scalar)
    /// * `s_trunk`       – `[B, N_token, c_s]`
    /// * `s_inputs`      – `[B, N_token, c_s_inputs]`
    /// * `z_pair`        – `[B, N_token, N_token, c_z]` (cached pair conditioning)
    /// * `atom_single`   – `[B, N_atom, c_atom]`
    /// * `atom_pair`     – `[B, N_atom, N_atom, c_atompair]`
    /// * `atom_to_token` – `[B, N_atom]`
    /// * `token_to_atom` – `[B, N_atom]`
    /// * `n_tokens`      – number of tokens
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        x_noisy: &Tensor,
        sigma: f64,
        s_trunk: &Tensor,
        s_inputs: &Tensor,
        z_pair: &Tensor,
        atom_single: &Tensor,
        atom_pair: &Tensor,
        atom_to_token: &Tensor,
        token_to_atom: &Tensor,
        n_tokens: usize,
    ) -> Result<Tensor> {
        let (c_skip, c_out) = self.edm_precond(sigma);
        let device = x_noisy.device();
        let dtype = x_noisy.dtype();

        let noise_level = Tensor::new(&[sigma as f32], device)?.to_dtype(dtype)?;
        let s_cond = self.conditioning.forward(s_trunk, s_inputs, &noise_level)?;

        let token_feats = self.atom_encoder.forward(
            atom_single,
            atom_pair,
            atom_to_token,
            n_tokens,
        )?;

        let token_feats = self.transformer.forward(&token_feats, &s_cond, z_pair)?;

        let r_update = self.atom_decoder.forward(
            &token_feats,
            atom_single,
            atom_pair,
            token_to_atom,
        )?;

        let x_denoised = (x_noisy * c_skip)?.add(&(r_update * c_out)?)?;
        Ok(x_denoised)
    }
}
