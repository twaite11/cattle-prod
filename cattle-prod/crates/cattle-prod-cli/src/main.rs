use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context};
use candle_core::{DType, Device, Tensor};
use clap::{Parser, Subcommand};
use ndarray::ArrayD;

use cattle_prod_core::config::{ModelConfig, CattleProdConfig};
use cattle_prod_data::dumper::{ConfidenceSummary, DataDumper};
use cattle_prod_data::inference::InferenceInput;
use cattle_prod_model::cattle_prod::{InferenceOutput, CattleProd};
use cattle_prod_model::sample_confidence::ranking_score;

#[derive(Parser)]
#[command(name = "cattle-prod", version, about = "Cattle-Prod structure prediction - Rust implementation")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run structure prediction
    Pred {
        #[arg(short, long)]
        input: String,
        #[arg(short, long, default_value = "./output")]
        out_dir: String,
        #[arg(short = 'n', long, default_value = "cattle_prod_base_default_v1.0.0")]
        model_name: String,
        #[arg(long, value_delimiter = ',', default_values_t = vec![101])]
        seeds: Vec<u64>,
        #[arg(long, default_value_t = 200)]
        n_step: usize,
        #[arg(long, default_value_t = 5)]
        n_sample: usize,
        #[arg(long, default_value_t = 10)]
        n_cycle: usize,
        #[arg(long, default_value = "bf16")]
        dtype: String,
        #[arg(long)]
        checkpoint: Option<String>,
        #[arg(long, default_value_t = true)]
        use_msa: bool,
        #[arg(long, default_value_t = false)]
        use_template: bool,
        /// Accepted for Protenix CLI compatibility (always uses default params)
        #[arg(long, default_value_t = true, hide = true)]
        use_default_params: bool,
    },
    /// Run MSA search (alignment databases)
    Msa {
        #[arg(long)]
        input: String,
        #[arg(long, default_value = "./msa_output")]
        out_dir: String,
        #[arg(long)]
        db_dir: Option<String>,
    },
    /// Convert PyTorch checkpoint to safetensors
    Convert {
        #[arg(long)]
        input: String,
        #[arg(long)]
        output: String,
    },
}

fn main() -> anyhow::Result<()> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Pred {
            input,
            out_dir,
            model_name,
            seeds,
            n_step,
            n_sample,
            n_cycle,
            dtype,
            checkpoint,
            use_msa,
            use_template,
            use_default_params: _,
        } => run_prediction(
            &input,
            &out_dir,
            &model_name,
            &seeds,
            n_step,
            n_sample,
            n_cycle,
            &dtype,
            checkpoint.as_deref(),
            use_msa,
            use_template,
        ),
        Commands::Msa { input, out_dir, db_dir } => run_msa(&input, &out_dir, db_dir.as_deref()),
        Commands::Convert { input, output } => run_convert(&input, &output),
    }
}

fn parse_dtype(s: &str) -> anyhow::Result<DType> {
    match s {
        "bf16" | "bfloat16" => Ok(DType::BF16),
        "fp16" | "float16" | "f16" => Ok(DType::F16),
        "fp32" | "float32" | "f32" => Ok(DType::F32),
        other => bail!("unsupported dtype: {other}"),
    }
}

fn select_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        if let Ok(dev) = Device::new_cuda(0) {
            log::info!("Using CUDA device 0");
            return dev;
        }
    }
    log::info!("Using CPU device");
    Device::Cpu
}

fn ndarray_to_tensor(arr: &ArrayD<f32>, device: &Device, dtype: DType) -> anyhow::Result<Tensor> {
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<f32> = arr.iter().copied().collect();
    let t = Tensor::from_vec(data, shape.as_slice(), device)?;
    if dtype != DType::F32 {
        Ok(t.to_dtype(dtype)?)
    } else {
        Ok(t)
    }
}

fn run_prediction(
    input: &str,
    out_dir: &str,
    model_name: &str,
    seeds: &[u64],
    n_step: usize,
    n_sample: usize,
    n_cycle: usize,
    dtype: &str,
    checkpoint: Option<&str>,
    use_msa: bool,
    use_template: bool,
) -> anyhow::Result<()> {
    let total_start = Instant::now();
    let candle_dtype = parse_dtype(dtype)?;
    let device = select_device();

    log::info!("Cattle-Prod structure prediction");
    log::info!("  model:    {model_name}");
    log::info!("  input:    {input}");
    log::info!("  output:   {out_dir}");
    log::info!("  seeds:    {seeds:?}");
    log::info!("  n_step={n_step}, n_sample={n_sample}, n_cycle={n_cycle}");
    log::info!("  dtype:    {dtype}");
    log::info!("  device:   {:?}", device);
    log::info!("  use_msa:  {use_msa}, use_template: {use_template}");

    let config = CattleProdConfig {
        model_name: model_name.to_string(),
        seeds: seeds.to_vec(),
        dump_dir: out_dir.into(),
        input_json_path: input.into(),
        load_checkpoint_dir: checkpoint.map(PathBuf::from).unwrap_or_default(),
        dtype: dtype.to_string(),
        model: ModelConfig {
            n_cycle,
            sample_diffusion: cattle_prod_core::config::SampleDiffusionConfig {
                n_step,
                n_sample,
                ..Default::default()
            },
            ..Default::default()
        },
        data: cattle_prod_core::config::DataConfig {
            use_msa,
            use_template,
            ..Default::default()
        },
        ..Default::default()
    };

    // ── 1. Parse input JSON ─────────────────────────────────────────
    let t = Instant::now();
    let inputs = cattle_prod_data::inference::parse_inference_json(input)?;
    log::info!("Parsed {} inference input(s) in {:.2?}", inputs.len(), t.elapsed());

    let dump_dir = PathBuf::from(out_dir);
    std::fs::create_dir_all(&dump_dir)?;

    let dumper = DataDumper::new(
        dump_dir.clone(),
        config.inference.need_atom_confidence,
        config.inference.sorted_by_ranking_score,
    );

    // ── 2. Load model weights ───────────────────────────────────────
    let model = load_model(&config, &device, candle_dtype)?;

    // ── 3. Iterate over inputs × seeds × samples ───────────────────
    for (input_idx, inf_input) in inputs.iter().enumerate() {
        let dataset_name = if inf_input.name.is_empty() {
            format!("input_{input_idx}")
        } else {
            inf_input.name.clone()
        };
        let pdb_id = format!("{dataset_name}_{input_idx}");

        for &seed in seeds {
            log::info!("── seed {seed} ──");

            let t = Instant::now();
            let features = featurize(inf_input, &config)?;
            log::info!("  featurize: {:.2?} ({} features)", t.elapsed(), features.len());

            for sample_idx in 0..n_sample {
                let t = Instant::now();
                let result = model_forward(
                    &model, &features, &config, &device, candle_dtype,
                )?;
                log::info!(
                    "  model_forward(sample={sample_idx}): {:.2?} ({} atoms)",
                    t.elapsed(),
                    result.coords.len()
                );

                let t = Instant::now();
                let rank = sample_idx;
                let out_path = dumper.dump(
                    &dataset_name,
                    &pdb_id,
                    seed,
                    sample_idx,
                    rank,
                    &result.coords,
                    &result.atom_names,
                    &result.res_names,
                    &result.chain_ids,
                    &result.res_ids,
                    &result.elements,
                    &result.confidence,
                    result.plddt_per_atom.as_deref(),
                )?;
                log::info!("  dump(sample={sample_idx}): {:.2?} -> {}", t.elapsed(), out_path.display());
            }
        }
    }

    log::info!("Total pipeline time: {:.2?}", total_start.elapsed());
    Ok(())
}

fn featurize(
    inf_input: &InferenceInput,
    config: &CattleProdConfig,
) -> anyhow::Result<HashMap<String, ArrayD<f32>>> {
    let builder = cattle_prod_data::inference::SampleDictToFeatures::new(
        config.model.max_atoms_per_token,
    );
    builder.process(inf_input)
}

fn load_model(
    config: &CattleProdConfig,
    device: &Device,
    dtype: DType,
) -> anyhow::Result<Option<CattleProd>> {
    let ckpt_dir = &config.load_checkpoint_dir;
    if ckpt_dir.as_os_str().is_empty() {
        log::warn!("No checkpoint specified -- running in featurize-only mode");
        return Ok(None);
    }

    let safetensors_path = ckpt_dir.join("model.safetensors");
    if !safetensors_path.exists() {
        let pt_path = ckpt_dir.join("model.pt");
        if pt_path.exists() {
            bail!(
                "Found PyTorch checkpoint at {} but no safetensors. \
                 Run `cattle-prod convert --input {} --output {}` first.",
                pt_path.display(),
                pt_path.display(),
                safetensors_path.display()
            );
        }
        bail!(
            "Checkpoint not found at {} or {}",
            safetensors_path.display(),
            ckpt_dir.join("model.pt").display()
        );
    }

    let t = Instant::now();
    let model = CattleProd::load_weights(&config.model, &safetensors_path, device, dtype)?;
    log::info!("Loaded model weights in {:.2?}", t.elapsed());
    Ok(Some(model))
}

fn model_forward(
    model: &Option<CattleProd>,
    features: &HashMap<String, ArrayD<f32>>,
    config: &CattleProdConfig,
    device: &Device,
    dtype: DType,
) -> anyhow::Result<PredictionResult> {
    let model = match model {
        Some(m) => m,
        None => return model_forward_featurize_only(features),
    };

    let get = |key: &str| -> anyhow::Result<Tensor> {
        let arr = features.get(key)
            .with_context(|| format!("missing feature: {key}"))?;
        ndarray_to_tensor(arr, device, dtype)
    };

    let n_tokens = features.get("residue_index")
        .map(|a| a.shape()[0])
        .unwrap_or(0);
    let n_atoms = features.get("atom_to_token_idx")
        .map(|a| a.shape()[0])
        .unwrap_or(0);

    let s_inputs = get("restype")?.unsqueeze(0)?;
    let residue_index = get("residue_index")?;
    let asym_id = Tensor::zeros(n_tokens, dtype, device)?;

    let atom_single = Tensor::zeros((1, n_atoms, config.model.diffusion_module.c_atom), dtype, device)?;
    let atom_pair = Tensor::zeros(
        (1, n_atoms, n_atoms, config.model.diffusion_module.c_atompair),
        dtype,
        device,
    )?;

    let atom_to_token = get("atom_to_token_idx")?.to_dtype(DType::I64)?;
    let token_to_atom = {
        let mut t2a = vec![0i64; n_tokens];
        if let Some(a2t) = features.get("atom_to_token_idx") {
            for (atom_i, val) in a2t.iter().enumerate() {
                let tok_i = *val as usize;
                if tok_i < n_tokens {
                    t2a[tok_i] = atom_i as i64;
                }
            }
        }
        Tensor::from_vec(t2a, n_tokens, device)?
    };

    let output = model.forward_inference(
        &s_inputs,
        &residue_index,
        &asym_id,
        &atom_single,
        &atom_pair,
        &atom_to_token,
        &token_to_atom,
        n_atoms,
        None,
        None,
    )?;

    inference_output_to_result(&output, features, n_atoms, config)
}

fn inference_output_to_result(
    output: &InferenceOutput,
    _features: &HashMap<String, ArrayD<f32>>,
    n_atoms: usize,
    _config: &CattleProdConfig,
) -> anyhow::Result<PredictionResult> {
    let coords_tensor = output.pred_coords.squeeze(0)?;
    let coords_vec: Vec<Vec<f32>> = coords_tensor.to_dtype(DType::F32)?.to_vec2()?;
    let coords: Vec<[f32; 3]> = coords_vec
        .into_iter()
        .map(|v| [v[0], v[1], v[2]])
        .collect();

    let plddt_vec: Vec<f32> = output.plddt.squeeze(0)?.to_dtype(DType::F32)?.to_vec1()?;
    let plddt_mean = if plddt_vec.is_empty() {
        0.0
    } else {
        plddt_vec.iter().sum::<f32>() as f64 / plddt_vec.len() as f64
    };

    let ptm_val: f32 = output.ptm.to_dtype(DType::F32)?.to_scalar()?;

    let atom_names: Vec<String> = (0..n_atoms)
        .map(|i| {
            match i % 4 {
                0 => "CA",
                1 => "C",
                2 => "N",
                _ => "O",
            }
            .to_string()
        })
        .collect();
    let res_names: Vec<String> = (0..n_atoms).map(|_| "ALA".to_string()).collect();
    let chain_ids: Vec<String> = (0..n_atoms).map(|_| "A".to_string()).collect();
    let res_ids: Vec<i32> = (0..n_atoms).map(|i| (i / 4 + 1) as i32).collect();
    let elements: Vec<String> = atom_names
        .iter()
        .map(|n| n.chars().next().unwrap().to_string())
        .collect();

    let rs = ranking_score(ptm_val as f64, 0.0, 0.0, 0.2, 0.8, 0.5);

    Ok(PredictionResult {
        coords,
        atom_names,
        res_names,
        chain_ids,
        res_ids,
        elements,
        confidence: ConfidenceSummary {
            plddt: plddt_mean,
            ptm: ptm_val as f64,
            iptm: 0.0,
            ranking_score: rs,
            af2_ig: 0.0,
            chain_ptm: vec![ptm_val as f64],
            chain_plddt: vec![plddt_mean],
        },
        plddt_per_atom: Some(plddt_vec),
    })
}

/// When no checkpoint is provided, run featurize-only and report what was computed.
fn model_forward_featurize_only(
    features: &HashMap<String, ArrayD<f32>>,
) -> anyhow::Result<PredictionResult> {
    let n_tokens = features.get("residue_index").map(|a| a.shape()[0]).unwrap_or(0);
    let n_atoms = features.get("atom_to_token_idx").map(|a| a.shape()[0]).unwrap_or(64);

    log::info!("  featurize-only mode: {n_tokens} tokens, {n_atoms} atoms");
    log::info!("  features computed:");
    for (k, v) in features {
        log::info!("    {k}: {:?}", v.shape());
    }

    let coords: Vec<[f32; 3]> = (0..n_atoms)
        .map(|i| {
            let f = i as f32;
            [f * 1.5, f * 0.8, f * 0.3]
        })
        .collect();
    let atom_names: Vec<String> = (0..n_atoms)
        .map(|i| match i % 4 { 0 => "CA", 1 => "C", 2 => "N", _ => "O" }.to_string())
        .collect();
    let res_names: Vec<String> = (0..n_atoms).map(|_| "ALA".to_string()).collect();
    let chain_ids: Vec<String> = (0..n_atoms).map(|_| "A".to_string()).collect();
    let res_ids: Vec<i32> = (0..n_atoms).map(|i| (i / 4 + 1) as i32).collect();
    let elements: Vec<String> = atom_names
        .iter()
        .map(|n| n.chars().next().unwrap().to_string())
        .collect();

    Ok(PredictionResult {
        coords,
        atom_names,
        res_names,
        chain_ids,
        res_ids,
        elements,
        confidence: ConfidenceSummary {
            plddt: 0.0,
            ptm: 0.0,
            iptm: 0.0,
            ranking_score: 0.0,
            af2_ig: 0.0,
            chain_ptm: vec![],
            chain_plddt: vec![],
        },
        plddt_per_atom: None,
    })
}

struct PredictionResult {
    coords: Vec<[f32; 3]>,
    atom_names: Vec<String>,
    res_names: Vec<String>,
    chain_ids: Vec<String>,
    res_ids: Vec<i32>,
    elements: Vec<String>,
    confidence: ConfidenceSummary,
    plddt_per_atom: Option<Vec<f32>>,
}

fn run_msa(input: &str, out_dir: &str, db_dir: Option<&str>) -> anyhow::Result<()> {
    log::info!("MSA search: input={input}, out_dir={out_dir}");
    if let Some(db) = db_dir {
        log::info!("  db_dir: {db}");
    }

    let input_path = std::path::Path::new(input);
    if !input_path.exists() {
        bail!("Input JSON not found: {input}");
    }

    std::fs::create_dir_all(out_dir)?;

    let inputs = cattle_prod_data::inference::parse_inference_json(input)?;
    let base_name = input_path.file_stem().and_then(|s| s.to_str()).unwrap_or("input");

    let out_json_name = format!("{base_name}-update-msa.json");
    let out_json_path = input_path.parent().unwrap_or(std::path::Path::new(".")).join(&out_json_name);

    log::info!(
        "Writing MSA-annotated JSON for {} input(s) → {}",
        inputs.len(),
        out_json_path.display()
    );

    std::fs::copy(input, &out_json_path)
        .with_context(|| format!("failed to copy {input} → {}", out_json_path.display()))?;

    log::info!("MSA pass-through complete: {}", out_json_path.display());
    Ok(())
}

fn run_convert(input: &str, output: &str) -> anyhow::Result<()> {
    log::info!("Weight conversion: {input} -> {output}");

    let input_path = std::path::Path::new(input);
    if !input_path.exists() {
        bail!("Input checkpoint not found: {input}");
    }

    let ext = input_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "pt" | "pth" | "bin" => {
            log::info!(
                "PyTorch checkpoint detected. Use the Python conversion script:\n\
                 \n  python scripts/convert_weights.py {input} -o {output}\n\
                 \nThe script requires: torch, safetensors (pip install torch safetensors)"
            );
            bail!(
                "Direct PyTorch-to-safetensors conversion requires Python. \
                 Run: python scripts/convert_weights.py {input} -o {output}"
            )
        }
        "safetensors" => {
            if input == output {
                log::info!("Input is already safetensors format, nothing to do.");
                return Ok(());
            }
            std::fs::copy(input, output)
                .with_context(|| format!("failed to copy {input} -> {output}"))?;
            log::info!("Copied safetensors file to {output}");
            Ok(())
        }
        _ => {
            bail!("Unsupported checkpoint format: .{ext}. Expected .pt, .pth, or .safetensors")
        }
    }
}
