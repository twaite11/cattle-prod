use std::path::Path;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::confidence::ConfidenceHead;
use crate::diffusion::DiffusionModule;
use crate::embedders::{InputFeatureEmbedder, RelativePositionEncoding};
use crate::generator::{sample_diffusion, InferenceNoiseScheduler};
use crate::heads::DistogramHead;
use crate::pairformer::{MSAModule, PairformerStack};
use crate::primitives::{LayerNorm, LinearNoBias};
use crate::sample_confidence::{compute_plddt, compute_ptm};
use cattle_prod_core::config::ModelConfig;

// ---------------------------------------------------------------------------
// CattleProd  –  full model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CattleProd {
    // Input embedders
    relpos: RelativePositionEncoding,
    input_embedder: InputFeatureEmbedder,

    // Initial projections
    proj_s: LinearNoBias,
    proj_z_i: LinearNoBias,
    proj_z_j: LinearNoBias,

    // Optional MSA module
    msa_module: Option<MSAModule>,

    // Trunk pairformer
    pairformer: PairformerStack,

    // Distogram head
    distogram_head: DistogramHead,

    // Diffusion module
    diffusion: DiffusionModule,

    // Confidence head
    confidence_head: ConfidenceHead,

    // Norm layers for recycling
    s_norm: LayerNorm,
    z_norm: LayerNorm,

    cfg: ModelConfig,
}

impl CattleProd {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let relpos = RelativePositionEncoding::new(
            &cfg.relative_position_encoding,
            vb.pp("relpos"),
        )?;
        let input_embedder = InputFeatureEmbedder::new(
            cfg.c_s_inputs,
            cfg.c_s,
            vb.pp("input_embedder"),
        )?;

        let proj_s = LinearNoBias::new(cfg.c_s, cfg.c_s, vb.pp("proj_s"))?;
        let proj_z_i = LinearNoBias::new(cfg.c_s, cfg.c_z, vb.pp("proj_z_i"))?;
        let proj_z_j = LinearNoBias::new(cfg.c_s, cfg.c_z, vb.pp("proj_z_j"))?;

        let msa_module = if cfg.data.use_msa {
            Some(MSAModule::new(&cfg.msa_module, vb.pp("msa_module"))?)
        } else {
            None
        };

        let pairformer = PairformerStack::new(&cfg.pairformer, true, vb.pp("pairformer"))?;

        let distogram_head =
            DistogramHead::new(cfg.c_z, cfg.no_bins, vb.pp("distogram_head"))?;

        let diffusion = DiffusionModule::new(&cfg.diffusion_module, vb.pp("diffusion"))?;

        let confidence_head =
            ConfidenceHead::new(&cfg.confidence_head, cfg.no_bins, vb.pp("confidence_head"))?;

        let s_norm = LayerNorm::new(cfg.c_s, 1e-5, vb.pp("s_norm"))?;
        let z_norm = LayerNorm::new(cfg.c_z, 1e-5, vb.pp("z_norm"))?;

        Ok(Self {
            relpos,
            input_embedder,
            proj_s,
            proj_z_i,
            proj_z_j,
            msa_module,
            pairformer,
            distogram_head,
            diffusion,
            confidence_head,
            s_norm,
            z_norm,
            cfg: cfg.clone(),
        })
    }

    /// Run the pairformer trunk with N_cycle recycling.
    ///
    /// Returns `(s_trunk, z_trunk)`.
    #[allow(clippy::too_many_arguments)]
    pub fn get_pairformer_output(
        &self,
        s_inputs: &Tensor,
        residue_index: &Tensor,
        asym_id: &Tensor,
        msa_tokens: Option<&Tensor>,
        msa_deletion: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let batch = s_inputs.dim(0)?;
        let n_tok = s_inputs.dim(1)?;
        let device = s_inputs.device();
        let dtype = s_inputs.dtype();

        let relp = self.relpos.forward(residue_index, asym_id)?;
        let relp = relp.unsqueeze(0)?.broadcast_as((batch, n_tok, n_tok, self.cfg.relative_position_encoding.c_z))?;

        let mut s = Tensor::zeros((batch, n_tok, self.cfg.c_s), dtype, device)?;
        let mut z = Tensor::zeros((batch, n_tok, n_tok, self.cfg.c_z), dtype, device)?;

        for _cycle in 0..self.cfg.n_cycle {
            let s_in = s_inputs.add(&self.proj_s.forward(&s)?)?;

            let z_i = self.proj_z_i.forward(&s_in)?.unsqueeze(2)?;
            let z_j = self.proj_z_j.forward(&s_in)?.unsqueeze(1)?;
            let z_init = z_i.broadcast_add(&z_j)?.add(&relp)?;
            let z_cur = z_init.add(&z)?;

            let z_cur = if let (Some(msa), Some(tok), Some(del)) =
                (&self.msa_module, msa_tokens, msa_deletion)
            {
                msa.forward(tok, del, &z_cur)?
            } else {
                z_cur
            };

            let (z_new, s_opt) = self.pairformer.forward(&z_cur, Some(&s_in))?;
            s = self.s_norm.forward(&s_opt.unwrap_or(s_in))?;
            z = self.z_norm.forward(&z_new)?;
        }

        Ok((s, z))
    }

    /// Full inference forward pass.
    ///
    /// Returns predicted coordinates and confidence metrics.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_inference(
        &self,
        s_inputs: &Tensor,
        residue_index: &Tensor,
        asym_id: &Tensor,
        atom_single: &Tensor,
        atom_pair: &Tensor,
        atom_to_token: &Tensor,
        token_to_atom: &Tensor,
        n_atoms: usize,
        msa_tokens: Option<&Tensor>,
        msa_deletion: Option<&Tensor>,
    ) -> Result<InferenceOutput> {
        let (s_trunk, z_trunk) = self.get_pairformer_output(
            s_inputs,
            residue_index,
            asym_id,
            msa_tokens,
            msa_deletion,
        )?;

        let distogram_logits = self.distogram_head.forward(&z_trunk)?;

        let batch = s_trunk.dim(0)?;
        let n_tokens = s_trunk.dim(1)?;
        let device = s_trunk.device();
        let dtype = s_trunk.dtype();

        let relp_z = self.relpos.forward(residue_index, asym_id)?;
        let relp_z = relp_z.unsqueeze(0)?.broadcast_as((batch, n_tokens, n_tokens, self.cfg.relative_position_encoding.c_z))?;
        let z_pair = self.diffusion.conditioning.prepare_pair_cache(&z_trunk, &relp_z)?;

        let scheduler = InferenceNoiseScheduler::new(
            &self.cfg.inference_noise_scheduler,
            self.cfg.sample_diffusion.n_step,
        );

        let pred_coords = sample_diffusion(
            &self.diffusion,
            &scheduler,
            &self.cfg.sample_diffusion,
            &s_trunk,
            s_inputs,
            &z_pair,
            atom_single,
            atom_pair,
            atom_to_token,
            token_to_atom,
            n_tokens,
            n_atoms,
            device,
            dtype,
        )?;

        let (plddt_logits, pae_logits, pde_logits, resolved_logits) =
            self.confidence_head.forward(
                &s_trunk,
                s_inputs,
                &z_trunk,
                &pred_coords,
                atom_to_token,
            )?;

        let plddt = compute_plddt(&plddt_logits)?;
        let ptm = compute_ptm(
            &pae_logits,
            self.cfg.confidence_head.distance_bin_end,
            self.cfg.no_bins,
        )?;

        Ok(InferenceOutput {
            pred_coords,
            distogram_logits,
            plddt_logits,
            plddt,
            pae_logits,
            pde_logits,
            resolved_logits,
            ptm,
        })
    }

    /// Load model weights from a safetensors file.
    pub fn load_weights(
        cfg: &ModelConfig,
        path: &Path,
        device: &Device,
        dtype: DType,
    ) -> anyhow::Result<Self> {
        let data = std::fs::read(path)?;
        let vb = VarBuilder::from_buffered_safetensors(
            data,
            dtype,
            device,
        )?;
        let model = Self::new(cfg, vb)?;
        Ok(model)
    }
}

#[derive(Debug, Clone)]
pub struct InferenceOutput {
    pub pred_coords: Tensor,
    pub distogram_logits: Tensor,
    pub plddt_logits: Tensor,
    pub plddt: Tensor,
    pub pae_logits: Tensor,
    pub pde_logits: Tensor,
    pub resolved_logits: Tensor,
    pub ptm: Tensor,
}
