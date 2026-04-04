use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;

use crate::primitives::Linear;

// ---------------------------------------------------------------------------
// DistogramHead  –  symmetric pairwise logits [N, N, no_bins]
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DistogramHead {
    linear: Linear,
    no_bins: usize,
}

impl DistogramHead {
    pub fn new(c_z: usize, no_bins: usize, vb: VarBuilder) -> Result<Self> {
        let linear = Linear::new(c_z, no_bins, vb.pp("linear"))?;
        Ok(Self { linear, no_bins })
    }

    /// z: `[B, N, N, c_z]` -> logits: `[B, N, N, no_bins]` (symmetrised)
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let logits = self.linear.forward(z)?;
        let logits_t = logits.transpose(1, 2)?;
        let sum = (&logits + logits_t)?;
        sum * 0.5
    }
}
