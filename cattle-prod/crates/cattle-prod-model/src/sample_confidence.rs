use candle_core::{Result, Tensor, D};

use crate::primitives::softmax;

/// Compute contact probabilities from distogram logits.
///
/// logits: `[B, N, N, no_bins]` -> probs of distance < 8 Å.
/// Bins are assumed equally spaced starting at 2.3125 Å with step 0.3125 Å.
pub fn compute_contact_prob(
    distogram_logits: &Tensor,
    bin_start: f64,
    bin_step: f64,
    contact_threshold: f64,
) -> Result<Tensor> {
    let rank = distogram_logits.rank();
    let probs = softmax(distogram_logits, rank - 1)?;
    let no_bins = distogram_logits.dim(D::Minus1)?;

    let n_contact_bins = ((contact_threshold - bin_start) / bin_step).ceil() as usize;
    let n_contact_bins = n_contact_bins.min(no_bins);

    probs.narrow(D::Minus1, 0, n_contact_bins)?.sum(D::Minus1)
}

/// Compute predicted TM-score (pTM) from PAE logits.
///
/// pae_logits: `[B, N, N, no_bins]`, bin edges in Å.
pub fn compute_ptm(
    pae_logits: &Tensor,
    max_bin: f64,
    no_bins: usize,
) -> Result<Tensor> {
    let rank = pae_logits.rank();
    let pae_probs = softmax(pae_logits, rank - 1)?;

    let device = pae_logits.device();
    let dtype = pae_logits.dtype();
    let n_tok = pae_logits.dim(1)?;

    let d0 = 1.24 * ((n_tok as f64 - 15.0).max(1.0)).cbrt() - 1.8;
    let d0 = d0.max(0.02);

    let bin_width = max_bin / no_bins as f64;
    let mut bin_centers: Vec<f32> = Vec::with_capacity(no_bins);
    for i in 0..no_bins {
        bin_centers.push((i as f64 * bin_width + bin_width * 0.5) as f32);
    }
    let centers = Tensor::new(bin_centers, device)?.to_dtype(dtype)?;

    // TM-score weight: 1 / (1 + (d/d0)^2)
    let d_over_d0 = (&centers * (1.0 / d0 as f64))?;
    let tm_weights = (d_over_d0.sqr()? + 1.0)?.recip()?;

    let tm_per_pair = pae_probs.broadcast_mul(&tm_weights)?.sum(D::Minus1)?;

    // pTM = max_i mean_j tm(i,j)
    let tm_per_i = tm_per_pair.mean(D::Minus1)?;
    tm_per_i.max(D::Minus1)
}

/// Compute a composite ranking score.
///
/// ranking = w_ptm * ptm + w_iptm * iptm + w_disorder * (1 - disorder_frac)
pub fn ranking_score(
    ptm: f64,
    iptm: f64,
    disorder_fraction: f64,
    w_ptm: f64,
    w_iptm: f64,
    w_disorder: f64,
) -> f64 {
    w_ptm * ptm + w_iptm * iptm + w_disorder * (1.0 - disorder_fraction)
}

/// Compute pLDDT from per-residue logits.
///
/// logits: `[B, N, 50]` -> expected LDDT value in [0, 1]
pub fn compute_plddt(plddt_logits: &Tensor) -> Result<Tensor> {
    let rank = plddt_logits.rank();
    let probs = softmax(plddt_logits, rank - 1)?;
    let n_bins = plddt_logits.dim(D::Minus1)?;
    let device = plddt_logits.device();
    let dtype = plddt_logits.dtype();

    let step = 1.0 / n_bins as f64;
    let mut centers: Vec<f32> = Vec::with_capacity(n_bins);
    for i in 0..n_bins {
        centers.push((i as f64 * step + step * 0.5) as f32);
    }
    let centers = Tensor::new(centers, device)?.to_dtype(dtype)?;
    probs.broadcast_mul(&centers)?.sum(D::Minus1)
}
