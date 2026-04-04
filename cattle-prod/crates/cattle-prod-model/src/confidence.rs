use candle_core::{DType, Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::pairformer::PairformerStack;
use crate::primitives::{LayerNorm, Linear, LinearNoBias};
use cattle_prod_core::config::{ConfidenceHeadConfig, PairformerConfig};

// ---------------------------------------------------------------------------
// ConfidenceHead  –  pLDDT / PAE / PDE / resolved predictions
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConfidenceHead {
    proj_s: LinearNoBias,
    proj_z: LinearNoBias,
    dist_linear: LinearNoBias,
    layer_norm_s: LayerNorm,
    layer_norm_z: LayerNorm,
    pairformer: PairformerStack,
    plddt_head: Linear,
    pae_head: Linear,
    pde_head: Linear,
    resolved_head: Linear,
    no_bins: usize,
    dist_bins: usize,
}

impl ConfidenceHead {
    pub fn new(cfg: &ConfidenceHeadConfig, no_bins: usize, vb: VarBuilder) -> Result<Self> {
        let dist_bins = ((cfg.distance_bin_end - cfg.distance_bin_start) / cfg.distance_bin_step)
            .ceil() as usize
            + 1;

        let proj_s = LinearNoBias::new(cfg.c_s + cfg.c_s_inputs, cfg.c_s, vb.pp("proj_s"))?;
        let proj_z = LinearNoBias::new(cfg.c_z, cfg.c_z, vb.pp("proj_z"))?;
        let dist_linear = LinearNoBias::new(dist_bins, cfg.c_z, vb.pp("dist_linear"))?;
        let layer_norm_s = LayerNorm::new(cfg.c_s, 1e-5, vb.pp("layer_norm_s"))?;
        let layer_norm_z = LayerNorm::new(cfg.c_z, 1e-5, vb.pp("layer_norm_z"))?;

        let pf_cfg = PairformerConfig {
            n_blocks: cfg.n_blocks,
            c_z: cfg.c_z,
            c_s: cfg.c_s,
            n_heads: 16,
            dropout: cfg.pairformer_dropout,
        };
        let pairformer =
            PairformerStack::new(&pf_cfg, true, vb.pp("pairformer"))?;

        let plddt_head = Linear::new(cfg.c_s, 50, vb.pp("plddt_head"))?;
        let pae_head = Linear::new(cfg.c_z, no_bins, vb.pp("pae_head"))?;
        let pde_head = Linear::new(cfg.c_z, no_bins, vb.pp("pde_head"))?;
        let resolved_head = Linear::new(cfg.c_s, 2, vb.pp("resolved_head"))?;

        Ok(Self {
            proj_s,
            proj_z,
            dist_linear,
            layer_norm_s,
            layer_norm_z,
            pairformer,
            plddt_head,
            pae_head,
            pde_head,
            resolved_head,
            no_bins,
            dist_bins,
        })
    }

    /// Compute confidence metrics.
    ///
    /// * `s_trunk`  – `[B, N, c_s]`
    /// * `s_inputs` – `[B, N, c_s_inputs]`
    /// * `z_trunk`  – `[B, N, N, c_z]`
    /// * `pred_xyz` – `[B, N_atom, 3]`
    /// * `atom_to_token` – `[B, N_atom]`
    #[allow(clippy::type_complexity)]
    pub fn forward(
        &self,
        s_trunk: &Tensor,
        s_inputs: &Tensor,
        z_trunk: &Tensor,
        pred_xyz: &Tensor,
        atom_to_token: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor, Tensor)> {
        let s_cat = Tensor::cat(&[s_trunk, s_inputs], D::Minus1)?;
        let s = self.proj_s.forward(&s_cat)?;
        let s = self.layer_norm_s.forward(&s)?;

        let z = self.proj_z.forward(z_trunk)?;

        let dist_feats = self.compute_distance_bins(pred_xyz, atom_to_token)?;
        let z = z.add(&self.dist_linear.forward(&dist_feats)?)?;
        let z = self.layer_norm_z.forward(&z)?;

        let (z, s_opt) = self.pairformer.forward(&z, Some(&s))?;
        let s = s_opt.unwrap_or(s);

        let plddt_logits = self.plddt_head.forward(&s)?;
        let pae_logits = self.pae_head.forward(&z)?;
        let pde_logits = self.pde_head.forward(&z)?;
        let resolved_logits = self.resolved_head.forward(&s)?;

        Ok((plddt_logits, pae_logits, pde_logits, resolved_logits))
    }

    fn compute_distance_bins(
        &self,
        pred_xyz: &Tensor,
        _atom_to_token: &Tensor,
    ) -> Result<Tensor> {
        let dims = pred_xyz.dims();
        let (batch, n_atoms, _) = (dims[0], dims[1], dims[2]);
        let device = pred_xyz.device();
        let dtype = pred_xyz.dtype();

        let xi = pred_xyz.unsqueeze(2)?;
        let xj = pred_xyz.unsqueeze(1)?;
        let dist = xi.broadcast_sub(&xj)?.sqr()?.sum(D::Minus1)?.sqrt()?;

        let bin_edges = Tensor::arange(0u32, self.dist_bins as u32, device)?
            .to_dtype(DType::F32)?;
        let bin_edges = bin_edges.affine(1.25, 3.25)?;

        let dist_f = dist.to_dtype(DType::F32)?;
        let dist_exp = dist_f.unsqueeze(D::Minus1)?;
        let edges_exp = bin_edges.reshape((1, 1, 1, self.dist_bins))?;

        let diff = dist_exp.broadcast_sub(&edges_exp)?.abs()?;
        let bins = diff.argmin(D::Minus1)?;

        let eye = Tensor::eye(self.dist_bins, dtype, device)?;
        let flat = bins.flatten_all()?.to_dtype(DType::U32)?;
        let oh = eye.index_select(&flat, 0)?;
        oh.reshape((batch, n_atoms, n_atoms, self.dist_bins))
    }
}
