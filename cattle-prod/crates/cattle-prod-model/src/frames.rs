use candle_core::{Result, Tensor, D};

/// Orthonormal frame from three points via Gram–Schmidt.
///
/// Returns (e1, e2, e3) each of shape `[..., 3]`.
/// e1 = normalize(p2 - p1)
/// e2 = normalize(p3 - p1 - <p3-p1, e1> * e1)
/// e3 = e1 × e2
pub fn build_frame(
    p1: &Tensor,
    p2: &Tensor,
    p3: &Tensor,
) -> Result<(Tensor, Tensor, Tensor)> {
    let v1 = p2.sub(p1)?;
    let e1 = l2_normalize(&v1)?;

    let v2 = p3.sub(p1)?;
    let proj = dot_last(&v2, &e1)?;
    let v2_orth = v2.sub(&e1.broadcast_mul(&proj)?)?;
    let e2 = l2_normalize(&v2_orth)?;

    let e3 = cross(&e1, &e2)?;

    Ok((e1, e2, e3))
}

/// Transform coordinates into a frame-local representation.
///
/// coords: `[..., N, 3]`, frame = (origin, e1, e2, e3) where origin is `[..., 3]`.
/// Returns `[..., N, 3]` in the local coordinate system.
pub fn express_in_frame(
    coords: &Tensor,
    origin: &Tensor,
    e1: &Tensor,
    e2: &Tensor,
    e3: &Tensor,
) -> Result<Tensor> {
    let origin_exp = origin.unsqueeze(D::Minus2)?;
    let delta = coords.broadcast_sub(&origin_exp)?;

    let e1_exp = e1.unsqueeze(D::Minus2)?;
    let e2_exp = e2.unsqueeze(D::Minus2)?;
    let e3_exp = e3.unsqueeze(D::Minus2)?;

    let x = dot_last(&delta, &e1_exp)?.squeeze(D::Minus1)?;
    let y = dot_last(&delta, &e2_exp)?.squeeze(D::Minus1)?;
    let z = dot_last(&delta, &e3_exp)?.squeeze(D::Minus1)?;

    Tensor::stack(&[x, y, z], D::Minus1)
}

fn l2_normalize(xs: &Tensor) -> Result<Tensor> {
    let norm = xs
        .sqr()?
        .sum_keepdim(D::Minus1)?
        .sqrt()?;
    let norm = (norm + 1e-8)?;
    xs.broadcast_div(&norm)
}

fn dot_last(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    a.broadcast_mul(b)?.sum_keepdim(D::Minus1)
}

fn cross(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let a0 = a.narrow(D::Minus1, 0, 1)?;
    let a1 = a.narrow(D::Minus1, 1, 1)?;
    let a2 = a.narrow(D::Minus1, 2, 1)?;
    let b0 = b.narrow(D::Minus1, 0, 1)?;
    let b1 = b.narrow(D::Minus1, 1, 1)?;
    let b2 = b.narrow(D::Minus1, 2, 1)?;

    let c0 = a1.mul(&b2)?.sub(&a2.mul(&b1)?)?;
    let c1 = a2.mul(&b0)?.sub(&a0.mul(&b2)?)?;
    let c2 = a0.mul(&b1)?.sub(&a1.mul(&b0)?)?;

    Tensor::cat(&[c0, c1, c2], D::Minus1)
}
