use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CattleProdConfig {
    #[serde(default)]
    pub model_name: String,
    #[serde(default)]
    pub seeds: Vec<u64>,
    #[serde(default)]
    pub dump_dir: PathBuf,
    #[serde(default)]
    pub input_json_path: PathBuf,
    #[serde(default)]
    pub load_checkpoint_dir: PathBuf,
    #[serde(default)]
    pub dtype: String,
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub data: DataConfig,
    #[serde(default)]
    pub inference: InferenceConfig,
    #[serde(flatten)]
    pub extra: HashMap<String, serde_json::Value>,
}

impl Default for CattleProdConfig {
    fn default() -> Self {
        Self {
            model_name: "cattle_prod_base_default_v1.0.0".to_string(),
            seeds: vec![101],
            dump_dir: PathBuf::from("./output"),
            input_json_path: PathBuf::new(),
            load_checkpoint_dir: PathBuf::new(),
            dtype: "bf16".to_string(),
            model: ModelConfig::default(),
            data: DataConfig::default(),
            inference: InferenceConfig::default(),
            extra: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub c_s: usize,
    pub c_z: usize,
    pub c_s_inputs: usize,
    pub c_atom: usize,
    pub c_atompair: usize,
    pub c_token: usize,
    pub n_blocks: usize,
    pub max_atoms_per_token: usize,
    pub no_bins: usize,
    pub sigma_data: f64,
    pub n_cycle: usize,
    pub n_model_seed: usize,

    pub pairformer: PairformerConfig,
    pub diffusion_module: DiffusionModuleConfig,
    pub confidence_head: ConfidenceHeadConfig,
    pub template_embedder: TemplateEmbedderConfig,
    pub msa_module: MsaModuleConfig,
    pub relative_position_encoding: RelPosEncConfig,

    pub triangle_multiplicative: String,
    pub triangle_attention: String,
    pub enable_diffusion_shared_vars_cache: bool,
    pub enable_efficient_fusion: bool,
    pub enable_tf32: bool,

    pub sample_diffusion: SampleDiffusionConfig,
    pub inference_noise_scheduler: NoiseSchedulerConfig,
    pub data: DataConfig,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            c_s: 384,
            c_z: 128,
            c_s_inputs: 449,
            c_atom: 128,
            c_atompair: 16,
            c_token: 384,
            n_blocks: 48,
            max_atoms_per_token: 24,
            no_bins: 64,
            sigma_data: 16.0,
            n_cycle: 10,
            n_model_seed: 1,
            pairformer: PairformerConfig::default(),
            diffusion_module: DiffusionModuleConfig::default(),
            confidence_head: ConfidenceHeadConfig::default(),
            template_embedder: TemplateEmbedderConfig::default(),
            msa_module: MsaModuleConfig::default(),
            relative_position_encoding: RelPosEncConfig::default(),
            triangle_multiplicative: "torch".to_string(),
            triangle_attention: "torch".to_string(),
            enable_diffusion_shared_vars_cache: true,
            enable_efficient_fusion: true,
            enable_tf32: true,
            sample_diffusion: SampleDiffusionConfig::default(),
            inference_noise_scheduler: NoiseSchedulerConfig::default(),
            data: DataConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairformerConfig {
    pub n_blocks: usize,
    pub c_z: usize,
    pub c_s: usize,
    pub n_heads: usize,
    pub dropout: f64,
}

impl Default for PairformerConfig {
    fn default() -> Self {
        Self {
            n_blocks: 48,
            c_z: 128,
            c_s: 384,
            n_heads: 16,
            dropout: 0.25,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffusionModuleConfig {
    pub sigma_data: f64,
    pub c_token: usize,
    pub c_atom: usize,
    pub c_atompair: usize,
    pub c_z: usize,
    pub c_s: usize,
    pub c_s_inputs: usize,
    pub atom_encoder_n_blocks: usize,
    pub atom_encoder_n_heads: usize,
    pub transformer_n_blocks: usize,
    pub transformer_n_heads: usize,
    pub atom_decoder_n_blocks: usize,
    pub atom_decoder_n_heads: usize,
}

impl Default for DiffusionModuleConfig {
    fn default() -> Self {
        Self {
            sigma_data: 16.0,
            c_token: 768,
            c_atom: 128,
            c_atompair: 16,
            c_z: 128,
            c_s: 384,
            c_s_inputs: 449,
            atom_encoder_n_blocks: 3,
            atom_encoder_n_heads: 4,
            transformer_n_blocks: 24,
            transformer_n_heads: 16,
            atom_decoder_n_blocks: 3,
            atom_decoder_n_heads: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceHeadConfig {
    pub c_z: usize,
    pub c_s: usize,
    pub c_s_inputs: usize,
    pub n_blocks: usize,
    pub max_atoms_per_token: usize,
    pub pairformer_dropout: f64,
    pub distance_bin_start: f64,
    pub distance_bin_end: f64,
    pub distance_bin_step: f64,
    pub stop_gradient: bool,
}

impl Default for ConfidenceHeadConfig {
    fn default() -> Self {
        Self {
            c_z: 128,
            c_s: 384,
            c_s_inputs: 449,
            n_blocks: 4,
            max_atoms_per_token: 24,
            pairformer_dropout: 0.0,
            distance_bin_start: 3.25,
            distance_bin_end: 52.0,
            distance_bin_step: 1.25,
            stop_gradient: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemplateEmbedderConfig {
    pub c: usize,
    pub c_z: usize,
    pub n_blocks: usize,
    pub dropout: f64,
}

impl Default for TemplateEmbedderConfig {
    fn default() -> Self {
        Self {
            c: 64,
            c_z: 128,
            n_blocks: 2,
            dropout: 0.25,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MsaModuleConfig {
    pub c_m: usize,
    pub c_z: usize,
    pub c_s_inputs: usize,
    pub n_blocks: usize,
    pub msa_dropout: f64,
    pub pair_dropout: f64,
    pub msa_chunk_size: Option<usize>,
    pub msa_max_size: usize,
}

impl Default for MsaModuleConfig {
    fn default() -> Self {
        Self {
            c_m: 64,
            c_z: 128,
            c_s_inputs: 449,
            n_blocks: 4,
            msa_dropout: 0.15,
            pair_dropout: 0.25,
            msa_chunk_size: Some(2048),
            msa_max_size: 16384,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelPosEncConfig {
    pub r_max: usize,
    pub s_max: usize,
    pub c_z: usize,
}

impl Default for RelPosEncConfig {
    fn default() -> Self {
        Self {
            r_max: 32,
            s_max: 2,
            c_z: 128,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleDiffusionConfig {
    pub gamma0: f64,
    pub gamma_min: f64,
    pub noise_scale_lambda: f64,
    pub step_scale_eta: f64,
    pub n_step: usize,
    pub n_sample: usize,
}

impl Default for SampleDiffusionConfig {
    fn default() -> Self {
        Self {
            gamma0: 0.8,
            gamma_min: 1.0,
            noise_scale_lambda: 1.003,
            step_scale_eta: 1.5,
            n_step: 200,
            n_sample: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseSchedulerConfig {
    pub s_max: f64,
    pub s_min: f64,
    pub rho: f64,
    pub sigma_data: f64,
}

impl Default for NoiseSchedulerConfig {
    fn default() -> Self {
        Self {
            s_max: 160.0,
            s_min: 4e-4,
            rho: 7.0,
            sigma_data: 16.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataConfig {
    #[serde(default)]
    pub use_msa: bool,
    #[serde(default)]
    pub use_template: bool,
    #[serde(default)]
    pub use_rna_msa: bool,
    #[serde(default)]
    pub esm_enable: bool,
    #[serde(default)]
    pub esm_model_name: String,
    #[serde(default)]
    pub esm_embedding_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub need_atom_confidence: bool,
    pub sorted_by_ranking_score: bool,
    pub num_workers: usize,
    pub chunk_size: Option<usize>,
    pub dynamic_chunk_size: bool,
    pub diffusion_chunk_size: Option<usize>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            need_atom_confidence: false,
            sorted_by_ranking_score: true,
            num_workers: 0,
            chunk_size: Some(256),
            dynamic_chunk_size: true,
            diffusion_chunk_size: Some(5),
        }
    }
}

impl CattleProdConfig {
    pub fn from_yaml(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_yaml::from_str(&contents)?;
        Ok(config)
    }

    pub fn from_json(path: &str) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&contents)?;
        Ok(config)
    }

    pub fn to_yaml(&self, path: &str) -> anyhow::Result<()> {
        let contents = serde_yaml::to_string(self)?;
        std::fs::write(path, contents)?;
        Ok(())
    }
}
