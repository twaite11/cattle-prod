use candle_core::{Result, Tensor, D};
use candle_nn::{Module, VarBuilder};

use crate::primitives::{sigmoid, Linear, LinearNoBias, LayerNorm};

// ---------------------------------------------------------------------------
// TriangleMultiplication  –  outgoing / incoming variants
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriMulKind {
    Outgoing,
    Incoming,
}

#[derive(Debug, Clone)]
pub struct TriangleMultiplication {
    kind: TriMulKind,
    layer_norm_in: LayerNorm,
    linear_a_p: LinearNoBias,
    linear_a_g: LinearNoBias,
    linear_b_p: LinearNoBias,
    linear_b_g: LinearNoBias,
    layer_norm_out: LayerNorm,
    linear_z: LinearNoBias,
    linear_gate: Linear,
    c_z: usize,
}

impl TriangleMultiplication {
    pub fn new(
        c_z: usize,
        c_hidden: usize,
        kind: TriMulKind,
        vb: VarBuilder,
    ) -> Result<Self> {
        let layer_norm_in = LayerNorm::new(c_z, 1e-5, vb.pp("layer_norm_in"))?;
        let linear_a_p = LinearNoBias::new(c_z, c_hidden, vb.pp("linear_a_p"))?;
        let linear_a_g = LinearNoBias::new(c_z, c_hidden, vb.pp("linear_a_g"))?;
        let linear_b_p = LinearNoBias::new(c_z, c_hidden, vb.pp("linear_b_p"))?;
        let linear_b_g = LinearNoBias::new(c_z, c_hidden, vb.pp("linear_b_g"))?;
        let layer_norm_out = LayerNorm::new(c_hidden, 1e-5, vb.pp("layer_norm_out"))?;
        let linear_z = LinearNoBias::new(c_hidden, c_z, vb.pp("linear_z"))?;
        let linear_gate = Linear::new_with_bias(c_z, c_z, vb.pp("linear_gate"))?;

        Ok(Self {
            kind,
            layer_norm_in,
            linear_a_p,
            linear_a_g,
            linear_b_p,
            linear_b_g,
            layer_norm_out,
            linear_z,
            linear_gate,
            c_z,
        })
    }

    /// z: `[B, N, N, c_z]`
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let z_in = self.layer_norm_in.forward(z)?;

        let a = self.linear_a_p.forward(&z_in)?;
        let a_gate = sigmoid(&self.linear_a_g.forward(&z_in)?)?;
        let a = a.mul(&a_gate)?;

        let b = self.linear_b_p.forward(&z_in)?;
        let b_gate = sigmoid(&self.linear_b_g.forward(&z_in)?)?;
        let b = b.mul(&b_gate)?;

        // Contraction: outgoing contracts over j, incoming over i
        //   outgoing: z_ij = Σ_k a_ik * b_jk  =>  a @ b^T  (contract last-1 dim)
        //   incoming: z_ij = Σ_k a_ki * b_kj  =>  a^T @ b  (contract last-2 dim)
        let combined = match self.kind {
            TriMulKind::Outgoing => {
                let rank = a.rank();
                a.matmul(&b.transpose(rank - 2, rank - 1)?)?
            }
            TriMulKind::Incoming => {
                let rank = a.rank();
                a.transpose(rank - 2, rank - 1)?.matmul(&b)?
            }
        };

        let combined = self.layer_norm_out.forward(&combined)?;
        let out = self.linear_z.forward(&combined)?;

        let gate = sigmoid(&self.linear_gate.forward(&z_in)?)?;
        gate.mul(&out)
    }
}

// ---------------------------------------------------------------------------
// TriangleAttention  –  attention along one axis of the pair representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TriAttnKind {
    StartingNode,
    EndingNode,
}

#[derive(Debug, Clone)]
pub struct TriangleAttention {
    kind: TriAttnKind,
    layer_norm: LayerNorm,
    linear_q: LinearNoBias,
    linear_k: LinearNoBias,
    linear_v: LinearNoBias,
    linear_b: LinearNoBias,
    linear_g: Linear,
    linear_o: LinearNoBias,
    n_heads: usize,
    head_dim: usize,
    c_z: usize,
}

impl TriangleAttention {
    pub fn new(
        c_z: usize,
        n_heads: usize,
        kind: TriAttnKind,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = c_z / n_heads;
        let layer_norm = LayerNorm::new(c_z, 1e-5, vb.pp("layer_norm"))?;
        let linear_q = LinearNoBias::new(c_z, c_z, vb.pp("linear_q"))?;
        let linear_k = LinearNoBias::new(c_z, c_z, vb.pp("linear_k"))?;
        let linear_v = LinearNoBias::new(c_z, c_z, vb.pp("linear_v"))?;
        let linear_b = LinearNoBias::new(c_z, n_heads, vb.pp("linear_b"))?;
        let linear_g = Linear::new_with_bias(c_z, c_z, vb.pp("linear_g"))?;
        let linear_o = LinearNoBias::new(c_z, c_z, vb.pp("linear_o"))?;

        Ok(Self {
            kind,
            layer_norm,
            linear_q,
            linear_k,
            linear_v,
            linear_b,
            linear_g,
            linear_o,
            n_heads,
            head_dim,
            c_z,
        })
    }

    /// z: `[B, I, J, c_z]`
    pub fn forward(&self, z: &Tensor) -> Result<Tensor> {
        let dims = z.dims();
        let (_b, _i_dim, _j_dim, _c) = (dims[0], dims[1], dims[2], dims[3]);

        // For ending-node, transpose I/J so attention runs along the other axis
        let z_work = match self.kind {
            TriAttnKind::StartingNode => z.clone(),
            TriAttnKind::EndingNode => z.transpose(1, 2)?.contiguous()?,
        };

        let (b, n_i, n_j, _) = {
            let d = z_work.dims();
            (d[0], d[1], d[2], d[3])
        };

        let z_norm = self.layer_norm.forward(&z_work)?;

        let q = self.linear_q.forward(&z_norm)?;
        let k = self.linear_k.forward(&z_norm)?;
        let v = self.linear_v.forward(&z_norm)?;

        let q = q
            .reshape((b, n_i, n_j, self.n_heads, self.head_dim))?
            .transpose(2, 3)?;
        let k = k
            .reshape((b, n_i, n_j, self.n_heads, self.head_dim))?
            .transpose(2, 3)?;
        let v = v
            .reshape((b, n_i, n_j, self.n_heads, self.head_dim))?
            .transpose(2, 3)?;

        // Pair bias from the other axis: [B, I, J, n_heads] -> [B, I, n_heads, 1, J]
        let bias = self.linear_b.forward(&z_norm)?;
        let bias = bias
            .transpose(2, 3)?
            .unsqueeze(D::Minus2)?;

        let attn = crate::primitives::attention(&q, &k, &v, Some(&bias), None)?;

        let attn = attn
            .transpose(2, 3)?
            .contiguous()?
            .reshape((b, n_i, n_j, self.c_z))?;

        let gate = sigmoid(&self.linear_g.forward(&z_norm)?)?;
        let out = self.linear_o.forward(&attn.mul(&gate)?)?;

        match self.kind {
            TriAttnKind::StartingNode => Ok(out),
            TriAttnKind::EndingNode => out.transpose(1, 2)?.contiguous(),
        }
    }
}
