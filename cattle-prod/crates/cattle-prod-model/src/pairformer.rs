use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::primitives::{DropPath, LayerNorm, LinearNoBias, Transition};
use crate::transformer::AttentionPairBias;
use crate::triangular::{TriAttnKind, TriMulKind, TriangleAttention, TriangleMultiplication};
use cattle_prod_core::config::{MsaModuleConfig, PairformerConfig};

// ---------------------------------------------------------------------------
// OuterProductMean  –  m -> pair features via outer product
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct OuterProductMean {
    layer_norm: LayerNorm,
    linear_a: LinearNoBias,
    linear_b: LinearNoBias,
    linear_out: LinearNoBias,
    c_opm: usize,
    c_z: usize,
}

impl OuterProductMean {
    pub fn new(c_m: usize, c_z: usize, c_opm: usize, vb: VarBuilder) -> Result<Self> {
        let layer_norm = LayerNorm::new(c_m, 1e-5, vb.pp("layer_norm"))?;
        let linear_a = LinearNoBias::new(c_m, c_opm, vb.pp("linear_a"))?;
        let linear_b = LinearNoBias::new(c_m, c_opm, vb.pp("linear_b"))?;
        let linear_out = LinearNoBias::new(c_opm * c_opm, c_z, vb.pp("linear_out"))?;
        Ok(Self {
            layer_norm,
            linear_a,
            linear_b,
            linear_out,
            c_opm,
            c_z,
        })
    }

    /// m: `[B, N_seq, N_token, c_m]` -> z_update: `[B, N_token, N_token, c_z]`
    pub fn forward(&self, m: &Tensor) -> Result<Tensor> {
        let m_norm = self.layer_norm.forward(m)?;
        let a = self.linear_a.forward(&m_norm)?;
        let b = self.linear_b.forward(&m_norm)?;

        let dims = a.dims();
        let (batch, n_seq, n_tok, c_a) = (dims[0], dims[1], dims[2], dims[3]);
        let c_b = b.dim(D::Minus1)?;

        // Efficient outer-product-mean via reshape + matmul.
        //   a: [B, S, N, c_a] -> [B, S, N*c_a] -> transpose -> [B, N*c_a, S]
        //   b: [B, S, N, c_b] -> [B, S, N*c_b]
        //   matmul: [B, N*c_a, S] @ [B, S, N*c_b] -> [B, N*c_a, N*c_b]
        let a_flat = a.reshape((batch, n_seq, n_tok * c_a))?;
        let b_flat = b.reshape((batch, n_seq, n_tok * c_b))?;
        let a_t = a_flat.transpose(1, 2)?;
        let outer = a_t.matmul(&b_flat)?;
        let outer = (outer * (1.0 / n_seq as f64))?;

        // Reshape to [B, N, c_a, N, c_b]
        let outer = outer.reshape((batch, n_tok, c_a, n_tok, c_b))?;
        // Transpose to [B, N, N, c_a, c_b]
        let outer = outer.transpose(2, 3)?.contiguous()?;
        // Flatten channel dims: [B, N, N, c_a*c_b]
        let outer = outer.reshape((batch, n_tok, n_tok, c_a * c_b))?;

        self.linear_out.forward(&outer)
    }
}

// ---------------------------------------------------------------------------
// PairformerBlock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PairformerBlock {
    tri_mul_out: TriangleMultiplication,
    tri_mul_in: TriangleMultiplication,
    tri_attn_start: TriangleAttention,
    tri_attn_end: TriangleAttention,
    pair_transition: Transition,
    drop_path: DropPath,
    // Optional single track
    single_attn: Option<AttentionPairBias>,
    single_transition: Option<Transition>,
}

impl PairformerBlock {
    pub fn new(
        c_z: usize,
        c_s: usize,
        n_heads: usize,
        dropout: f64,
        with_single_track: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let c_hidden_mul = c_z;
        let tri_mul_out = TriangleMultiplication::new(
            c_z,
            c_hidden_mul,
            TriMulKind::Outgoing,
            vb.pp("tri_mul_out"),
        )?;
        let tri_mul_in = TriangleMultiplication::new(
            c_z,
            c_hidden_mul,
            TriMulKind::Incoming,
            vb.pp("tri_mul_in"),
        )?;
        let tri_attn_start = TriangleAttention::new(
            c_z,
            n_heads.min(c_z / 8).max(1),
            TriAttnKind::StartingNode,
            vb.pp("tri_attn_start"),
        )?;
        let tri_attn_end = TriangleAttention::new(
            c_z,
            n_heads.min(c_z / 8).max(1),
            TriAttnKind::EndingNode,
            vb.pp("tri_attn_end"),
        )?;
        let pair_transition = Transition::new(c_z, 2, vb.pp("pair_transition"))?;
        let drop_path = DropPath::new(dropout);

        let (single_attn, single_transition) = if with_single_track && c_s > 0 {
            let sa = AttentionPairBias::new(c_s, c_z, n_heads, false, vb.pp("single_attn"))?;
            let st = Transition::new(c_s, 2, vb.pp("single_transition"))?;
            (Some(sa), Some(st))
        } else {
            (None, None)
        };

        Ok(Self {
            tri_mul_out,
            tri_mul_in,
            tri_attn_start,
            tri_attn_end,
            pair_transition,
            drop_path,
            single_attn,
            single_transition,
        })
    }

    /// * `z` – pair rep `[B, N, N, c_z]`
    /// * `s` – single rep `[B, N, c_s]` (optional for single track)
    pub fn forward(
        &self,
        z: &Tensor,
        s: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let z = (z + self.drop_path.forward(&self.tri_mul_out.forward(z)?)?)?;
        let z = (&z + self.drop_path.forward(&self.tri_mul_in.forward(&z)?)?)?;
        let z = (&z + self.drop_path.forward(&self.tri_attn_start.forward(&z)?)?)?;
        let z = (&z + self.drop_path.forward(&self.tri_attn_end.forward(&z)?)?)?;
        let z = (&z + self.drop_path.forward(&self.pair_transition.forward(&z)?)?)?;

        let s_out = if let (Some(sa), Some(st), Some(s_in)) =
            (&self.single_attn, &self.single_transition, s)
        {
            let s_new = (s_in + sa.forward(s_in, None, &z)?)?;
            let s_new = (&s_new + st.forward(&s_new)?)?;
            Some(s_new)
        } else {
            s.map(|t| t.clone())
        };

        Ok((z, s_out))
    }
}

// ---------------------------------------------------------------------------
// PairformerStack
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct PairformerStack {
    blocks: Vec<PairformerBlock>,
}

impl PairformerStack {
    pub fn new(cfg: &PairformerConfig, with_single: bool, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::with_capacity(cfg.n_blocks);
        for i in 0..cfg.n_blocks {
            blocks.push(PairformerBlock::new(
                cfg.c_z,
                cfg.c_s,
                cfg.n_heads,
                cfg.dropout,
                with_single,
                vb.pp(format!("blocks.{i}")),
            )?);
        }
        Ok(Self { blocks })
    }

    pub fn forward(
        &self,
        z: &Tensor,
        s: Option<&Tensor>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let mut z = z.clone();
        let mut s = s.cloned();
        for block in &self.blocks {
            let (z_new, s_new) = block.forward(&z, s.as_ref())?;
            z = z_new;
            s = s_new;
        }
        Ok((z, s))
    }
}

// ---------------------------------------------------------------------------
// MSABlock  –  z += OPM(m); run PairformerBlock
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MSABlock {
    opm: OuterProductMean,
    pairformer_block: PairformerBlock,
}

impl MSABlock {
    pub fn new(
        c_m: usize,
        c_z: usize,
        n_heads: usize,
        dropout: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let c_opm = 32;
        let opm = OuterProductMean::new(c_m, c_z, c_opm, vb.pp("opm"))?;
        let pairformer_block =
            PairformerBlock::new(c_z, 0, n_heads, dropout, false, vb.pp("pairformer_block"))?;
        Ok(Self {
            opm,
            pairformer_block,
        })
    }

    /// * `m` – MSA rep `[B, N_seq, N_token, c_m]`
    /// * `z` – pair rep `[B, N_token, N_token, c_z]`
    pub fn forward(&self, m: &Tensor, z: &Tensor) -> Result<Tensor> {
        let z = (z + self.opm.forward(m)?)?;
        let (z, _) = self.pairformer_block.forward(&z, None)?;
        Ok(z)
    }
}

// ---------------------------------------------------------------------------
// MSAModule  –  embed MSA, stack MSABlocks
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MSAModule {
    proj_msa: LinearNoBias,
    proj_del: LinearNoBias,
    blocks: Vec<MSABlock>,
    c_m: usize,
}

impl MSAModule {
    pub fn new(cfg: &MsaModuleConfig, vb: VarBuilder) -> Result<Self> {
        let proj_msa = LinearNoBias::new(32, cfg.c_m, vb.pp("proj_msa"))?;
        let proj_del = LinearNoBias::new(1, cfg.c_m, vb.pp("proj_del"))?;
        let mut blocks = Vec::with_capacity(cfg.n_blocks);
        for i in 0..cfg.n_blocks {
            blocks.push(MSABlock::new(
                cfg.c_m,
                cfg.c_z,
                4,
                cfg.pair_dropout,
                vb.pp(format!("blocks.{i}")),
            )?);
        }
        Ok(Self {
            proj_msa,
            proj_del,
            blocks,
            c_m: cfg.c_m,
        })
    }

    /// * `msa_tokens` – `[B, N_seq, N_token, 32]`  one-hot MSA
    /// * `deletion`   – `[B, N_seq, N_token, 1]`    deletion features
    /// * `z`          – `[B, N_token, N_token, c_z]`
    pub fn forward(
        &self,
        msa_tokens: &Tensor,
        deletion: &Tensor,
        z: &Tensor,
    ) -> Result<Tensor> {
        let m = self.proj_msa.forward(msa_tokens)?;
        let m = m.add(&self.proj_del.forward(deletion)?)?;

        let mut z = z.clone();
        for block in &self.blocks {
            z = block.forward(&m, &z)?;
        }
        Ok(z)
    }
}
