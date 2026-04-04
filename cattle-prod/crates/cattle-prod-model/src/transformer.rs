use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::primitives::{
    attention, sigmoid, AdaptiveLayerNorm, LayerNorm, Linear, LinearNoBias,
};

// ---------------------------------------------------------------------------
// AttentionPairBias  –  single-track attention with pair-bias
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AttentionPairBias {
    adaptive: bool,
    ada_ln: Option<AdaptiveLayerNorm>,
    plain_ln: Option<LayerNorm>,
    linear_q: LinearNoBias,
    linear_k: LinearNoBias,
    linear_v: LinearNoBias,
    linear_g: Linear,
    linear_o: LinearNoBias,
    linear_b: LinearNoBias,
    n_heads: usize,
    head_dim: usize,
    c_s: usize,
}

impl AttentionPairBias {
    pub fn new(
        c_s: usize,
        c_z: usize,
        n_heads: usize,
        adaptive: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = c_s / n_heads;
        let ada_ln = if adaptive {
            Some(AdaptiveLayerNorm::new(c_s, c_s, vb.pp("ada_ln"))?)
        } else {
            None
        };
        let plain_ln = if !adaptive {
            Some(LayerNorm::new(c_s, 1e-5, vb.pp("layer_norm"))?)
        } else {
            None
        };

        let linear_q = LinearNoBias::new(c_s, c_s, vb.pp("linear_q"))?;
        let linear_k = LinearNoBias::new(c_s, c_s, vb.pp("linear_k"))?;
        let linear_v = LinearNoBias::new(c_s, c_s, vb.pp("linear_v"))?;
        let linear_g = Linear::new_with_bias(c_s, c_s, vb.pp("linear_g"))?;
        let linear_o = LinearNoBias::new(c_s, c_s, vb.pp("linear_o"))?;
        let linear_b = LinearNoBias::new(c_z, n_heads, vb.pp("linear_b"))?;

        Ok(Self {
            adaptive,
            ada_ln,
            plain_ln,
            linear_q,
            linear_k,
            linear_v,
            linear_g,
            linear_o,
            linear_b,
            n_heads,
            head_dim,
            c_s,
        })
    }

    /// * `x` – `[B, N, c_s]`
    /// * `s` – conditioning signal `[B, N, c_s]` (only used when adaptive)
    /// * `z` – pair representation `[B, N, N, c_z]`
    pub fn forward(
        &self,
        x: &Tensor,
        s: Option<&Tensor>,
        z: &Tensor,
    ) -> Result<Tensor> {
        let x_norm = if self.adaptive {
            self.ada_ln
                .as_ref()
                .unwrap()
                .forward(x, s.unwrap())?
        } else {
            self.plain_ln.as_ref().unwrap().forward(x)?
        };

        let dims = x_norm.dims();
        let (b, n, _c) = (dims[0], dims[1], dims[2]);

        let q = self.linear_q.forward(&x_norm)?;
        let k = self.linear_k.forward(&x_norm)?;
        let v = self.linear_v.forward(&x_norm)?;

        let q = q
            .reshape((b, n, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, n, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, n, self.n_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Pair bias: [B, N, N, c_z] -> [B, N, N, n_heads] -> [B, n_heads, N, N]
        let pair_bias = self
            .linear_b
            .forward(z)?
            .permute((0, 3, 1, 2))?;

        let attn = attention(&q, &k, &v, Some(&pair_bias), None)?;

        let attn = attn
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, self.c_s))?;

        let gate = sigmoid(&self.linear_g.forward(&x_norm)?)?;
        self.linear_o.forward(&attn.mul(&gate)?)
    }
}

// ---------------------------------------------------------------------------
// ConditionedTransitionBlock  –  AdaLN + SiLU-gated expansion
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ConditionedTransitionBlock {
    ada_ln: AdaptiveLayerNorm,
    linear_in: LinearNoBias,
    linear_out: LinearNoBias,
    n_mult: usize,
}

impl ConditionedTransitionBlock {
    pub fn new(c_s: usize, n: usize, vb: VarBuilder) -> Result<Self> {
        let ada_ln = AdaptiveLayerNorm::new(c_s, c_s, vb.pp("ada_ln"))?;
        let linear_in = LinearNoBias::new(c_s, c_s * n * 2, vb.pp("linear_in"))?;
        let linear_out = LinearNoBias::new(c_s * n, c_s, vb.pp("linear_out"))?;
        Ok(Self {
            ada_ln,
            linear_in,
            linear_out,
            n_mult: n,
        })
    }

    /// * `x` – `[B, N, c_s]`
    /// * `s` – conditioning `[B, N, c_s]`
    pub fn forward(&self, x: &Tensor, s: &Tensor) -> Result<Tensor> {
        let x_norm = self.ada_ln.forward(x, s)?;
        let expanded = self.linear_in.forward(&x_norm)?;
        let chunks = expanded.chunk(2, D::Minus1)?;
        let gate = crate::primitives::silu(&chunks[0])?;
        let val = &chunks[1];
        let gated = gate.mul(val)?;
        self.linear_out.forward(&gated)
    }
}

// ---------------------------------------------------------------------------
// DiffusionTransformerBlock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DiffusionTransformerBlock {
    attn: AttentionPairBias,
    transition: ConditionedTransitionBlock,
}

impl DiffusionTransformerBlock {
    pub fn new(
        c_s: usize,
        c_z: usize,
        n_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn = AttentionPairBias::new(c_s, c_z, n_heads, true, vb.pp("attn"))?;
        let transition = ConditionedTransitionBlock::new(c_s, 2, vb.pp("transition"))?;
        Ok(Self { attn, transition })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        s: &Tensor,
        z: &Tensor,
    ) -> Result<Tensor> {
        let x = (x + self.attn.forward(x, Some(s), z)?)?;
        let x = (&x + self.transition.forward(&x, s)?)?;
        Ok(x)
    }
}

// ---------------------------------------------------------------------------
// DiffusionTransformer  –  stack of DiffusionTransformerBlocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct DiffusionTransformer {
    blocks: Vec<DiffusionTransformerBlock>,
}

impl DiffusionTransformer {
    pub fn new(
        n_blocks: usize,
        c_s: usize,
        c_z: usize,
        n_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            blocks.push(DiffusionTransformerBlock::new(
                c_s,
                c_z,
                n_heads,
                vb.pp(format!("blocks.{i}")),
            )?);
        }
        Ok(Self { blocks })
    }

    pub fn forward(
        &self,
        x: &Tensor,
        s: &Tensor,
        z: &Tensor,
    ) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(&h, s, z)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// AtomTransformer  –  DiffusionTransformer with local cross-attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AtomTransformer {
    blocks: Vec<DiffusionTransformerBlock>,
    n_queries: usize,
    n_keys: usize,
}

impl AtomTransformer {
    pub fn new(
        n_blocks: usize,
        c_atom: usize,
        c_atompair: usize,
        n_heads: usize,
        n_queries: usize,
        n_keys: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut blocks = Vec::with_capacity(n_blocks);
        for i in 0..n_blocks {
            blocks.push(DiffusionTransformerBlock::new(
                c_atom,
                c_atompair,
                n_heads,
                vb.pp(format!("blocks.{i}")),
            )?);
        }
        Ok(Self {
            blocks,
            n_queries,
            n_keys,
        })
    }

    /// * `x`     – `[B, N_atoms, c_atom]`
    /// * `s`     – `[B, N_atoms, c_atom]`
    /// * `z`     – `[B, N_atoms, N_atoms, c_atompair]` (windowed / sparse)
    pub fn forward(
        &self,
        x: &Tensor,
        s: &Tensor,
        z: &Tensor,
    ) -> Result<Tensor> {
        let mut h = x.clone();
        for block in &self.blocks {
            h = block.forward(&h, s, z)?;
        }
        Ok(h)
    }
}

// ---------------------------------------------------------------------------
// AtomAttentionEncoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AtomAttentionEncoder {
    proj_atom_in: LinearNoBias,
    proj_atompair_in: LinearNoBias,
    proj_atompair_out: LinearNoBias,
    atom_transformer: AtomTransformer,
    proj_out: LinearNoBias,
    layer_norm: LayerNorm,
    c_atom: usize,
    c_atompair: usize,
    c_token: usize,
}

impl AtomAttentionEncoder {
    pub fn new(
        c_token: usize,
        c_atom: usize,
        c_atompair: usize,
        n_blocks: usize,
        n_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let proj_atom_in = LinearNoBias::new(c_atom, c_atom, vb.pp("proj_atom_in"))?;
        let proj_atompair_in =
            LinearNoBias::new(c_atompair, c_atompair, vb.pp("proj_atompair_in"))?;
        let proj_atompair_out =
            LinearNoBias::new(c_atompair, c_atompair, vb.pp("proj_atompair_out"))?;
        let atom_transformer =
            AtomTransformer::new(n_blocks, c_atom, c_atompair, n_heads, 32, 128, vb.pp("atom_transformer"))?;
        let proj_out = LinearNoBias::new(c_atom, c_token, vb.pp("proj_out"))?;
        let layer_norm = LayerNorm::new(c_atom, 1e-5, vb.pp("layer_norm"))?;

        Ok(Self {
            proj_atom_in,
            proj_atompair_in,
            proj_atompair_out,
            atom_transformer,
            proj_out,
            layer_norm,
            c_atom,
            c_atompair,
            c_token,
        })
    }

    /// Encode atom-level features into token-level features.
    ///
    /// * `atom_single`  – `[B, N_atom, c_atom]`
    /// * `atom_pair`    – `[B, N_atom, N_atom, c_atompair]`
    /// * `atom_to_token`– `[B, N_atom]`  (int mapping atom→token)
    /// * `n_tokens`     – number of tokens
    pub fn forward(
        &self,
        atom_single: &Tensor,
        atom_pair: &Tensor,
        atom_to_token: &Tensor,
        n_tokens: usize,
    ) -> Result<Tensor> {
        let a = self.proj_atom_in.forward(atom_single)?;
        let p = self.proj_atompair_in.forward(atom_pair)?;

        let h = self.atom_transformer.forward(&a, &a, &p)?;
        let h = self.layer_norm.forward(&h)?;

        let token_feats = self.proj_out.forward(&h)?;

        aggregate_atom_to_token(&token_feats, atom_to_token, n_tokens)
    }
}

/// Average-pool atom features into their owning tokens via scatter_add.
pub fn aggregate_atom_to_token(
    atom_feats: &Tensor,
    atom_to_token: &Tensor,
    n_tokens: usize,
) -> Result<Tensor> {
    let dims = atom_feats.dims();
    let (batch, n_atoms, c) = (dims[0], dims[1], dims[2]);
    let device = atom_feats.device();
    let dtype = atom_feats.dtype();

    let map = atom_to_token.to_dtype(candle_core::DType::U32)?;
    let mut batch_results = Vec::with_capacity(batch);

    for b_idx in 0..batch {
        let feats_b = atom_feats.narrow(0, b_idx, 1)?.squeeze(0)?;
        let map_b = map.narrow(0, b_idx, 1)?.squeeze(0)?;
        let map_exp = map_b
            .unsqueeze(1)?
            .broadcast_as((n_atoms, c))?
            .contiguous()?;

        let token_sum = Tensor::zeros((n_tokens, c), dtype, device)?
            .scatter_add(&map_exp, &feats_b, 0)?;

        let ones = Tensor::ones((n_atoms, c), dtype, device)?;
        let token_count = Tensor::zeros((n_tokens, c), dtype, device)?
            .scatter_add(&map_exp, &ones, 0)?;

        let avg = token_sum.broadcast_div(&(token_count + 1e-8)?)?;
        batch_results.push(avg.unsqueeze(0)?);
    }

    Tensor::cat(&batch_results, 0)
}

// ---------------------------------------------------------------------------
// AtomAttentionDecoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct AtomAttentionDecoder {
    proj_token_to_atom: LinearNoBias,
    atom_transformer: AtomTransformer,
    linear_out: LinearNoBias,
    layer_norm: LayerNorm,
    c_atom: usize,
}

impl AtomAttentionDecoder {
    pub fn new(
        c_token: usize,
        c_atom: usize,
        c_atompair: usize,
        n_blocks: usize,
        n_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let proj_token_to_atom =
            LinearNoBias::new(c_token, c_atom, vb.pp("proj_token_to_atom"))?;
        let atom_transformer =
            AtomTransformer::new(n_blocks, c_atom, c_atompair, n_heads, 32, 128, vb.pp("atom_transformer"))?;
        let linear_out = LinearNoBias::new(c_atom, 3, vb.pp("linear_out"))?;
        let layer_norm = LayerNorm::new(c_atom, 1e-5, vb.pp("layer_norm"))?;

        Ok(Self {
            proj_token_to_atom,
            atom_transformer,
            linear_out,
            layer_norm,
            c_atom,
        })
    }

    /// Decode token features back to per-atom 3D updates.
    ///
    /// * `token_feats` – `[B, N_token, c_token]`
    /// * `atom_single` – `[B, N_atom, c_atom]`   (skip connection)
    /// * `atom_pair`   – `[B, N_atom, N_atom, c_atompair]`
    /// * `token_to_atom` – `[B, N_atom]` (int mapping token→atom broadcast)
    pub fn forward(
        &self,
        token_feats: &Tensor,
        atom_single: &Tensor,
        atom_pair: &Tensor,
        token_to_atom: &Tensor,
    ) -> Result<Tensor> {
        let broadcasted = broadcast_token_to_atom(token_feats, token_to_atom)?;
        let atom_in = self.proj_token_to_atom.forward(&broadcasted)?;
        let atom_combined = atom_in.add(atom_single)?;

        let h = self.atom_transformer.forward(&atom_combined, &atom_combined, atom_pair)?;
        let h = self.layer_norm.forward(&h)?;
        self.linear_out.forward(&h)
    }
}

fn broadcast_token_to_atom(
    token_feats: &Tensor,
    token_to_atom: &Tensor,
) -> Result<Tensor> {
    let map = token_to_atom.to_dtype(candle_core::DType::U32)?;
    let dims = token_feats.dims();
    let (batch, _n_tok, _c) = (dims[0], dims[1], dims[2]);

    let mut slices: Vec<Tensor> = Vec::new();
    for b_idx in 0..batch {
        let feats_b = token_feats.narrow(0, b_idx, 1)?.squeeze(0)?;
        let map_b = map.narrow(0, b_idx, 1)?.squeeze(0)?;
        let gathered = feats_b.index_select(&map_b, 0)?;
        slices.push(gathered.unsqueeze(0)?);
    }
    Tensor::cat(&slices, 0)
}
