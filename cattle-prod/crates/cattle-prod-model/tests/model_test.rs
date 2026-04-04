use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Module, VarBuilder, VarMap};

use cattle_prod_model::primitives::{
    self, sigmoid, silu, softmax, LayerNorm, Linear, LinearNoBias, Transition,
};
use cattle_prod_model::sample_confidence::ranking_score;

fn device() -> Device {
    Device::Cpu
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

#[test]
fn sigmoid_boundaries() {
    let dev = device();
    let xs = Tensor::new(&[-100.0f32, 0.0, 100.0], &dev).unwrap();
    let out = sigmoid(&xs).unwrap();
    let vals: Vec<f32> = out.to_vec1().unwrap();

    assert!(vals[0] < 1e-5, "sigmoid(-100) should be near 0");
    assert!((vals[1] - 0.5).abs() < 1e-5, "sigmoid(0) should be 0.5");
    assert!((vals[2] - 1.0).abs() < 1e-5, "sigmoid(100) should be near 1");
}

#[test]
fn silu_zero_is_zero() {
    let dev = device();
    let xs = Tensor::zeros((3,), DType::F32, &dev).unwrap();
    let out = silu(&xs).unwrap();
    let vals: Vec<f32> = out.to_vec1().unwrap();
    for v in vals {
        assert!(v.abs() < 1e-6, "silu(0) should be 0");
    }
}

#[test]
fn softmax_sums_to_one_1d() {
    let dev = device();
    let xs = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], &dev).unwrap();
    let out = softmax(&xs.unsqueeze(0).unwrap(), 1).unwrap().squeeze(0).unwrap();
    let sum: f32 = out.sum_all().unwrap().to_scalar().unwrap();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "softmax should sum to 1, got {sum}"
    );
}

#[test]
fn softmax_sums_to_one_2d() {
    let dev = device();
    let xs = Tensor::randn(0.0f32, 1.0, (4, 8), &dev).unwrap();
    let out = softmax(&xs, 1).unwrap();
    let sums = out.sum(1).unwrap();
    let vals: Vec<f32> = sums.to_vec1().unwrap();
    for (i, v) in vals.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-4,
            "softmax row {i} sums to {v}, expected 1.0"
        );
    }
}

#[test]
fn softmax_preserves_ordering() {
    let dev = device();
    let xs = Tensor::new(&[1.0f32, 3.0, 2.0], &dev).unwrap();
    let out = softmax(&xs.unsqueeze(0).unwrap(), 1).unwrap().squeeze(0).unwrap();
    let vals: Vec<f32> = out.to_vec1().unwrap();
    assert!(vals[1] > vals[2], "softmax should preserve ordering");
    assert!(vals[2] > vals[0], "softmax should preserve ordering");
}

// ---------------------------------------------------------------------------
// Linear forward pass shape
// ---------------------------------------------------------------------------

#[test]
fn linear_forward_shape() {
    let dev = device();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let linear = Linear::new(16, 32, vb).unwrap();

    let input = Tensor::randn(0.0f32, 1.0, (2, 5, 16), &dev).unwrap();
    let output = linear.forward(&input).unwrap();

    assert_eq!(output.dims(), &[2, 5, 32]);
}

#[test]
fn linear_with_bias_forward_shape() {
    let dev = device();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let linear = Linear::new_with_bias(8, 4, vb).unwrap();

    let input = Tensor::randn(0.0f32, 1.0, (3, 8), &dev).unwrap();
    let output = linear.forward(&input).unwrap();

    assert_eq!(output.dims(), &[3, 4]);
}

#[test]
fn linear_no_bias_forward_shape() {
    let dev = device();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let linear = LinearNoBias::new(10, 20, vb).unwrap();

    let input = Tensor::randn(0.0f32, 1.0, (4, 10), &dev).unwrap();
    let output = linear.forward(&input).unwrap();

    assert_eq!(output.dims(), &[4, 20]);
}

// ---------------------------------------------------------------------------
// LayerNorm normalization properties
// ---------------------------------------------------------------------------

#[test]
fn layer_norm_output_shape() {
    let dev = device();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let ln = LayerNorm::new(16, 1e-5, vb).unwrap();
    let input = Tensor::randn(0.0f32, 1.0, (2, 8, 16), &dev).unwrap();
    let output = ln.forward(&input).unwrap();

    assert_eq!(output.dims(), &[2, 8, 16]);
}

#[test]
fn layer_norm_normalization_properties() {
    let dev = device();
    let dim = 64;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let ln = LayerNorm::new_no_bias(dim, 1e-5, vb).unwrap();

    let input = Tensor::randn(5.0f32, 3.0, (1, dim), &dev).unwrap();
    let output = ln.forward(&input).unwrap();

    let mean: f32 = output
        .mean_keepdim(D::Minus1)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()[0];
    assert!(
        mean.abs() < 0.5,
        "after LN (no bias), mean should be near 0, got {mean}"
    );
}

// ---------------------------------------------------------------------------
// Transition shape
// ---------------------------------------------------------------------------

#[test]
fn transition_forward_shape() {
    let dev = device();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let transition = Transition::new(16, 2, vb).unwrap();
    let input = Tensor::randn(0.0f32, 1.0, (1, 10, 16), &dev).unwrap();
    let output = transition.forward(&input).unwrap();

    assert_eq!(output.dims(), &[1, 10, 16]);
}

// ---------------------------------------------------------------------------
// Attention: output shape matches input
// ---------------------------------------------------------------------------

#[test]
fn attention_output_shape() {
    let dev = device();
    let batch = 2;
    let heads = 4;
    let seq_len = 8;
    let head_dim = 16;

    let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &dev).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &dev).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &dev).unwrap();

    let out = primitives::attention(&q, &k, &v, None, None).unwrap();
    assert_eq!(out.dims(), &[batch, heads, seq_len, head_dim]);
}

#[test]
fn attention_with_bias_output_shape() {
    let dev = device();
    let batch = 1;
    let heads = 2;
    let seq_len = 6;
    let head_dim = 8;

    let q = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &dev).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &dev).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (batch, heads, seq_len, head_dim), &dev).unwrap();
    let bias = Tensor::zeros((batch, heads, seq_len, seq_len), DType::F32, &dev).unwrap();

    let out = primitives::attention(&q, &k, &v, Some(&bias), None).unwrap();
    assert_eq!(out.dims(), &[batch, heads, seq_len, head_dim]);
}

#[test]
fn attention_softmax_sums_to_one() {
    let dev = device();
    let seq_len = 5;
    let head_dim = 4;

    let q = Tensor::randn(0.0f32, 1.0, (1, 1, seq_len, head_dim), &dev).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, seq_len, head_dim), &dev).unwrap();

    let d_k = head_dim as f64;
    let scale = 1.0 / d_k.sqrt();
    let scores = q.matmul(&k.transpose(2, 3).unwrap()).unwrap();
    let scores = (scores * scale).unwrap();
    let attn_weights = softmax(&scores, 3).unwrap();

    let sums = attn_weights.sum(D::Minus1).unwrap();
    let vals: Vec<f32> = sums.flatten_all().unwrap().to_vec1().unwrap();
    for (i, &v) in vals.iter().enumerate() {
        assert!(
            (v - 1.0).abs() < 1e-4,
            "attention weights at pos {i} sum to {v}, expected 1.0"
        );
    }
}

// ---------------------------------------------------------------------------
// Scaled dot-product attention (attention module)
// ---------------------------------------------------------------------------

#[test]
fn scaled_dot_product_attention_shape() {
    use cattle_prod_model::attention::scaled_dot_product_attention;

    let dev = device();
    let seq = 10;
    let dim = 8;

    let q = Tensor::randn(0.0f32, 1.0, (1, 2, seq, dim), &dev).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 2, seq, dim), &dev).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 2, seq, dim), &dev).unwrap();

    let out = scaled_dot_product_attention(&q, &k, &v, None, false).unwrap();
    assert_eq!(out.dims(), &[1, 2, seq, dim]);
}

#[test]
fn scaled_dot_product_attention_causal() {
    use cattle_prod_model::attention::scaled_dot_product_attention;

    let dev = device();
    let seq = 6;
    let dim = 4;

    let q = Tensor::randn(0.0f32, 1.0, (1, 1, seq, dim), &dev).unwrap();
    let k = Tensor::randn(0.0f32, 1.0, (1, 1, seq, dim), &dev).unwrap();
    let v = Tensor::randn(0.0f32, 1.0, (1, 1, seq, dim), &dev).unwrap();

    let out = scaled_dot_product_attention(&q, &k, &v, None, true).unwrap();
    assert_eq!(out.dims(), &[1, 1, seq, dim]);
}

// ---------------------------------------------------------------------------
// Noise scheduler: schedule is monotonically decreasing
// ---------------------------------------------------------------------------

#[test]
fn noise_schedule_monotonically_decreasing() {
    use cattle_prod_core::config::NoiseSchedulerConfig;

    let cfg = NoiseSchedulerConfig::default();
    let n_steps = 200usize;

    let mut schedule = Vec::with_capacity(n_steps + 1);
    for i in 0..=n_steps {
        let t = i as f64 / n_steps as f64;
        let sigma = (cfg.s_max.powf(1.0 / cfg.rho) * (1.0 - t)
            + cfg.s_min.powf(1.0 / cfg.rho) * t)
            .powf(cfg.rho);
        schedule.push(sigma);
    }

    for i in 1..schedule.len() {
        assert!(
            schedule[i] <= schedule[i - 1] + 1e-10,
            "schedule not monotonically decreasing at step {i}: {} > {}",
            schedule[i],
            schedule[i - 1]
        );
    }

    assert!(
        (schedule[0] - cfg.s_max).abs() < 1e-6,
        "first sigma should be s_max"
    );
    assert!(
        (schedule[n_steps] - cfg.s_min).abs() < 1e-6,
        "last sigma should be s_min"
    );
}

#[test]
fn noise_schedule_sigma_data_range() {
    use cattle_prod_core::config::NoiseSchedulerConfig;

    let cfg = NoiseSchedulerConfig::default();
    assert!(cfg.s_max > cfg.s_min);
    assert!(cfg.sigma_data > 0.0);
    assert!(cfg.rho > 0.0);
}

// ---------------------------------------------------------------------------
// Sample diffusion config schedule
// ---------------------------------------------------------------------------

#[test]
fn sample_diffusion_n_step_positive() {
    use cattle_prod_core::config::SampleDiffusionConfig;
    let cfg = SampleDiffusionConfig::default();
    assert!(cfg.n_step > 0);
    assert!(cfg.n_sample > 0);
    assert!(cfg.step_scale_eta > 0.0);
    assert!(cfg.noise_scale_lambda > 0.0);
}

// ---------------------------------------------------------------------------
// ranking_score computation
// ---------------------------------------------------------------------------

#[test]
fn ranking_score_basic() {
    let score = ranking_score(0.8, 0.9, 0.1, 0.2, 0.5, 0.3);
    let expected = 0.2 * 0.8 + 0.5 * 0.9 + 0.3 * (1.0 - 0.1);
    assert!(
        (score - expected).abs() < 1e-10,
        "ranking_score mismatch: {score} vs {expected}"
    );
}

#[test]
fn ranking_score_perfect() {
    let score = ranking_score(1.0, 1.0, 0.0, 0.2, 0.5, 0.3);
    let expected = 0.2 + 0.5 + 0.3;
    assert!(
        (score - expected).abs() < 1e-10,
        "perfect ranking score should be sum of weights"
    );
}

#[test]
fn ranking_score_worst_case() {
    let score = ranking_score(0.0, 0.0, 1.0, 0.2, 0.5, 0.3);
    assert!(
        score.abs() < 1e-10,
        "worst-case ranking score should be ~0"
    );
}

#[test]
fn ranking_score_weights_matter() {
    let s1 = ranking_score(0.5, 0.5, 0.5, 1.0, 0.0, 0.0);
    let s2 = ranking_score(0.5, 0.5, 0.5, 0.0, 1.0, 0.0);
    assert!((s1 - 0.5).abs() < 1e-10);
    assert!((s2 - 0.5).abs() < 1e-10);

    let s3 = ranking_score(0.5, 0.5, 0.5, 0.0, 0.0, 1.0);
    assert!((s3 - 0.5).abs() < 1e-10);
}

// ---------------------------------------------------------------------------
// DropPath (passthrough during inference)
// ---------------------------------------------------------------------------

#[test]
fn droppath_is_identity_at_inference() {
    use cattle_prod_model::primitives::DropPath;

    let dev = device();
    let dp = DropPath::new(0.5);
    let input = Tensor::randn(0.0f32, 1.0, (2, 4), &dev).unwrap();
    let output = dp.forward(&input).unwrap();

    let diff = (input - output).unwrap().abs().unwrap().sum_all().unwrap();
    let diff_val: f32 = diff.to_scalar().unwrap();
    assert!(diff_val < 1e-10, "DropPath should be identity during inference");
}

// ---------------------------------------------------------------------------
// FourierEmbedding
// ---------------------------------------------------------------------------

#[test]
fn fourier_embedding_output_shape() {
    use cattle_prod_model::embedders::FourierEmbedding;

    let dev = device();
    let dim = 16;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let emb = FourierEmbedding::new(dim, vb).unwrap();
    let t = Tensor::new(0.5f32, &dev).unwrap();
    let out = emb.forward(&t).unwrap();

    assert_eq!(out.dims()[out.rank() - 1], dim);
}

#[test]
fn fourier_embedding_scalar_input() {
    use cattle_prod_model::embedders::FourierEmbedding;

    let dev = device();
    let dim = 8;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let emb = FourierEmbedding::new(dim, vb).unwrap();
    let t = Tensor::new(0.5f32, &dev).unwrap();
    let out = emb.forward(&t).unwrap();

    assert_eq!(out.dims()[out.rank() - 1], dim);
}

#[test]
fn fourier_embedding_bounded_output() {
    use cattle_prod_model::embedders::FourierEmbedding;

    let dev = device();
    let dim = 32;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let emb = FourierEmbedding::new(dim, vb).unwrap();
    let t = Tensor::new(0.5f32, &dev).unwrap();
    let out = emb.forward(&t).unwrap();

    let flat = out.flatten_all().unwrap();
    let vals: Vec<f32> = flat.to_vec1().unwrap();
    let max_abs = vals.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(
        max_abs <= 1.0 + 1e-5,
        "cos output should be in [-1, 1], got max {max_abs}"
    );
}

// ---------------------------------------------------------------------------
// RelativePositionEncoding
// ---------------------------------------------------------------------------

#[test]
fn relative_position_encoding_output_shape() {
    use cattle_prod_model::embedders::RelativePositionEncoding;
    use cattle_prod_core::config::RelPosEncConfig;

    let dev = device();
    let cfg = RelPosEncConfig::default();
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let rpe = RelativePositionEncoding::new(&cfg, vb).unwrap();
    let n_tok = 6;
    let res_idx = Tensor::arange(0i64, n_tok as i64, &dev).unwrap();
    let asym_id = Tensor::zeros(n_tok, DType::I64, &dev).unwrap();

    let out = rpe.forward(&res_idx, &asym_id).unwrap();

    assert_eq!(out.dims(), &[n_tok, n_tok, cfg.c_z]);
}

#[test]
fn relative_position_encoding_symmetric_asym() {
    use cattle_prod_model::embedders::RelativePositionEncoding;
    use cattle_prod_core::config::RelPosEncConfig;

    let dev = device();
    let cfg = RelPosEncConfig { r_max: 4, s_max: 1, c_z: 16 };
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let rpe = RelativePositionEncoding::new(&cfg, vb).unwrap();
    let n = 4;
    let res_idx = Tensor::arange(0i64, n as i64, &dev).unwrap();
    let asym = Tensor::zeros(n, DType::I64, &dev).unwrap();

    let out = rpe.forward(&res_idx, &asym).unwrap();
    assert_eq!(out.dims(), &[n, n, 16]);
}

// ---------------------------------------------------------------------------
// InputFeatureEmbedder
// ---------------------------------------------------------------------------

#[test]
fn input_feature_embedder_construction() {
    use cattle_prod_model::embedders::InputFeatureEmbedder;

    let dev = device();
    let c_s_inputs = 449;
    let c_token = 384;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let _embedder = InputFeatureEmbedder::new(c_s_inputs, c_token, vb).unwrap();
}

#[test]
fn input_feature_embedder_forward_shape() {
    use cattle_prod_model::embedders::InputFeatureEmbedder;

    let dev = device();
    let batch = 1;
    let n_tok = 10;
    let c_atom = 128;
    let c_token = 64;
    let c_s_inputs = c_atom + 32 + 32 + 1; // atom_encoder + restype + profile + del_mean

    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);
    let embedder = InputFeatureEmbedder::new(c_s_inputs, c_token, vb).unwrap();

    let atom_enc = Tensor::randn(0.0f32, 1.0, (batch, n_tok, c_atom), &dev).unwrap();
    let restype = Tensor::randn(0.0f32, 1.0, (batch, n_tok, 32), &dev).unwrap();
    let profile = Tensor::randn(0.0f32, 1.0, (batch, n_tok, 32), &dev).unwrap();
    let del_mean = Tensor::randn(0.0f32, 1.0, (batch, n_tok, 1), &dev).unwrap();

    let out = embedder.forward(&atom_enc, &restype, &profile, &del_mean).unwrap();
    assert_eq!(out.dims(), &[batch, n_tok, c_token]);
}

// ---------------------------------------------------------------------------
// DistogramHead
// ---------------------------------------------------------------------------

#[test]
fn distogram_head_output_shape() {
    use cattle_prod_model::heads::DistogramHead;

    let dev = device();
    let c_z = 32;
    let no_bins = 64;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let head = DistogramHead::new(c_z, no_bins, vb).unwrap();
    let batch = 1;
    let n = 8;
    let z = Tensor::randn(0.0f32, 1.0, (batch, n, n, c_z), &dev).unwrap();
    let out = head.forward(&z).unwrap();

    assert_eq!(out.dims(), &[batch, n, n, no_bins]);
}

#[test]
fn distogram_head_output_is_symmetric() {
    use cattle_prod_model::heads::DistogramHead;

    let dev = device();
    let c_z = 16;
    let no_bins = 10;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &dev);

    let head = DistogramHead::new(c_z, no_bins, vb).unwrap();
    let n = 5;
    let z = Tensor::randn(0.0f32, 1.0, (1, n, n, c_z), &dev).unwrap();
    let out = head.forward(&z).unwrap();

    let out_t = out.transpose(1, 2).unwrap();
    let diff = (out - out_t).unwrap().abs().unwrap().sum_all().unwrap();
    let diff_val: f32 = diff.to_scalar().unwrap();
    assert!(
        diff_val < 1e-4,
        "distogram output should be symmetric, total diff={diff_val}"
    );
}

// ---------------------------------------------------------------------------
// frames::build_frame produces orthonormal frame
// ---------------------------------------------------------------------------

#[test]
fn build_frame_orthonormal() {
    use cattle_prod_model::frames::build_frame;

    let dev = device();
    let p1 = Tensor::new(&[0.0f32, 0.0, 0.0], &dev).unwrap();
    let p2 = Tensor::new(&[1.0f32, 0.0, 0.0], &dev).unwrap();
    let p3 = Tensor::new(&[0.0f32, 1.0, 0.0], &dev).unwrap();

    let (e1, e2, e3) = build_frame(&p1, &p2, &p3).unwrap();

    let e1v: Vec<f32> = e1.to_vec1().unwrap();
    let e2v: Vec<f32> = e2.to_vec1().unwrap();
    let e3v: Vec<f32> = e3.to_vec1().unwrap();

    let dot_e1_e2: f32 = e1v.iter().zip(&e2v).map(|(a, b)| a * b).sum();
    let dot_e1_e3: f32 = e1v.iter().zip(&e3v).map(|(a, b)| a * b).sum();
    let dot_e2_e3: f32 = e2v.iter().zip(&e3v).map(|(a, b)| a * b).sum();

    assert!(dot_e1_e2.abs() < 1e-5, "e1·e2 should be 0, got {dot_e1_e2}");
    assert!(dot_e1_e3.abs() < 1e-5, "e1·e3 should be 0, got {dot_e1_e3}");
    assert!(dot_e2_e3.abs() < 1e-5, "e2·e3 should be 0, got {dot_e2_e3}");

    let norm_e1: f32 = e1v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_e2: f32 = e2v.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_e3: f32 = e3v.iter().map(|x| x * x).sum::<f32>().sqrt();

    assert!((norm_e1 - 1.0).abs() < 1e-5, "|e1| should be 1, got {norm_e1}");
    assert!((norm_e2 - 1.0).abs() < 1e-5, "|e2| should be 1, got {norm_e2}");
    assert!((norm_e3 - 1.0).abs() < 1e-5, "|e3| should be 1, got {norm_e3}");
}

#[test]
fn build_frame_e1_direction() {
    use cattle_prod_model::frames::build_frame;

    let dev = device();
    let p1 = Tensor::new(&[0.0f32, 0.0, 0.0], &dev).unwrap();
    let p2 = Tensor::new(&[3.0f32, 0.0, 0.0], &dev).unwrap();
    let p3 = Tensor::new(&[0.0f32, 0.0, 5.0], &dev).unwrap();

    let (e1, _e2, _e3) = build_frame(&p1, &p2, &p3).unwrap();
    let e1v: Vec<f32> = e1.to_vec1().unwrap();

    assert!((e1v[0] - 1.0).abs() < 1e-5, "e1 should point along x");
    assert!(e1v[1].abs() < 1e-5);
    assert!(e1v[2].abs() < 1e-5);
}

#[test]
fn build_frame_batched() {
    use cattle_prod_model::frames::build_frame;

    let dev = device();
    let p1 = Tensor::randn(0.0f32, 1.0, (3, 3), &dev).unwrap();
    let p2 = Tensor::randn(0.0f32, 1.0, (3, 3), &dev).unwrap();
    let p3 = Tensor::randn(0.0f32, 1.0, (3, 3), &dev).unwrap();

    let (e1, e2, e3) = build_frame(&p1, &p2, &p3).unwrap();

    assert_eq!(e1.dims(), &[3, 3]);
    assert_eq!(e2.dims(), &[3, 3]);
    assert_eq!(e3.dims(), &[3, 3]);
}

// ---------------------------------------------------------------------------
// frames::express_in_frame round-trip
// ---------------------------------------------------------------------------

#[test]
fn express_in_frame_round_trip() {
    use cattle_prod_model::frames::{build_frame, express_in_frame};

    let dev = device();
    let origin = Tensor::new(&[[1.0f32, 2.0, 3.0]], &dev).unwrap();
    let p2 = Tensor::new(&[[2.0f32, 2.0, 3.0]], &dev).unwrap();
    let p3 = Tensor::new(&[[1.0f32, 3.0, 3.0]], &dev).unwrap();

    let (e1, e2, e3) = build_frame(&origin, &p2, &p3).unwrap();

    let coords = Tensor::new(&[[[5.0f32, 6.0, 7.0], [1.0, 2.0, 3.0]]], &dev).unwrap();
    let local = express_in_frame(&coords, &origin, &e1, &e2, &e3).unwrap();
    assert_eq!(local.dims(), &[1, 2, 3]);

    let local_vals: Vec<Vec<Vec<f32>>> = local.to_vec3().unwrap();
    for &v in &local_vals[0][1] {
        assert!(v.abs() < 1e-4, "origin in local frame should be ~0, got {v}");
    }
}

#[test]
fn express_in_frame_preserves_distances() {
    use cattle_prod_model::frames::{build_frame, express_in_frame};

    let dev = device();
    let origin = Tensor::new(&[[0.0f32, 0.0, 0.0]], &dev).unwrap();
    let p2 = Tensor::new(&[[1.0f32, 0.0, 0.0]], &dev).unwrap();
    let p3 = Tensor::new(&[[0.0f32, 1.0, 0.0]], &dev).unwrap();

    let (e1, e2, e3) = build_frame(&origin, &p2, &p3).unwrap();

    let coords = Tensor::new(&[[[3.0f32, 4.0, 0.0], [0.0, 0.0, 0.0]]], &dev).unwrap();
    let local = express_in_frame(&coords, &origin, &e1, &e2, &e3).unwrap();

    let global_diff = coords.i((0, 0)).unwrap().sub(&coords.i((0, 1)).unwrap()).unwrap();
    let global_dist_sq: f32 = global_diff.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();

    let local_diff = local.i((0, 0)).unwrap().sub(&local.i((0, 1)).unwrap()).unwrap();
    let local_dist_sq: f32 = local_diff.sqr().unwrap().sum_all().unwrap().to_scalar().unwrap();

    assert!(
        (global_dist_sq - local_dist_sq).abs() < 1e-3,
        "frame transform should preserve distances: {global_dist_sq} vs {local_dist_sq}"
    );
}
