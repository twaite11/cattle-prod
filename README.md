<div align="center">

# Cattle-Prod

**Biomolecular structure prediction engine — rewritten in Rust.**

High-performance protein structure prediction for therapeutic discovery.<br>
Drop-in replacement for Protenix. Built for the [CASCADE](https://github.com/twaite11/CASCADE-Cas-Collateral-Activation-Discovery-Engineering) pipeline.

---

[![Rust](https://img.shields.io/badge/Rust-1.78+-000000?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](./LICENSE)
[![Tests](https://img.shields.io/badge/Tests-186%20passing-brightgreen)](#testing)
[![CASCADE](https://img.shields.io/badge/CASCADE-Integrated-ff6b35)](#cascade-integration)

</div>

---

## What is Cattle-Prod?

Cattle-Prod is a **Rust-native reimplementation** of the [Protenix](https://github.com/bytedance/Protenix) biomolecular structure prediction engine. It predicts 3D protein, RNA, and complex structures from sequence — the same core capability behind AlphaFold3 — but compiled to a single static binary with zero Python runtime overhead.

Built as the **eval engine for the CASCADE pipeline** (Cas13 Collateral Activation Switch Discovery & Engineering), Cattle-Prod drives the structural screening and fitness scoring loop that engineers programmable RNA-targeting suicide switches for oncology therapeutics.

### Why Rust?

| | Python (Protenix) | Rust (Cattle-Prod) |
|---|---|---|
| **Startup** | ~4s (interpreter + imports) | ~5ms (native binary) |
| **CPU featurization** | GIL-bound, single-core | Rayon parallel, all cores |
| **Memory safety** | Runtime errors | Compile-time guarantees |
| **Deployment** | conda env + 2GB deps | Single 15MB binary |
| **GPU inference** | PyTorch/CUDA | Candle/CUDA (same kernels) |
| **Iteration speed** | Minutes per variant | Seconds per variant |

For the CASCADE evolution loop — which evaluates **hundreds of Cas13 variants** across OFF/ON/off-target structural predictions per generation — the compound speedup on CPU-bound stages (parsing, tokenization, featurization, scoring) directly translates to more generations explored per GPU-hour.

---

## Architecture

```
cattle-prod/
├── crates/
│   ├── cattle-prod-core       # Types, constants, config, residue & token system
│   │   ├── constants.rs       #   All residue/atom tables, element maps, glycans, ions
│   │   ├── config.rs          #   CattleProdConfig, ModelConfig, DataConfig
│   │   ├── residue.rs         #   Residue/StdResidue enums, mol_type classification
│   │   └── token.rs           #   Token/TokenArray for the featurization pipeline
│   │
│   ├── cattle-prod-data       # Full data pipeline: parse → tokenize → featurize
│   │   ├── parser.rs          #   mmCIF parser (atom_site, entity, resolution)
│   │   ├── tokenizer.rs       #   AtomArray → TokenArray conversion
│   │   ├── featurizer.rs      #   One-hot encoding, reference features, pair features
│   │   ├── template.rs        #   Template featurization, HHR/A3M parsing
│   │   ├── msa.rs             #   MSA parsing, profile computation
│   │   ├── inference.rs       #   JSON input parsing, SampleDictToFeatures
│   │   ├── dumper.rs          #   CIF + summary JSON output (CASCADE-compatible)
│   │   └── metrics.rs         #   lDDT, RMSD, GDT-TS, clash score
│   │
│   ├── cattle-prod-model      # Neural network (Candle ML framework)
│   │   ├── primitives.rs      #   Linear, LayerNorm, softmax, SiLU, DropPath
│   │   ├── attention.rs       #   Multi-head attention, scaled dot-product
│   │   ├── embedders.rs       #   InputFeatureEmbedder, FourierEmbedding, RelPosEnc
│   │   ├── pairformer.rs      #   PairformerBlock, MSABlock, OuterProductMean
│   │   ├── triangular.rs      #   TriangleMultiplication, TriangleAttention
│   │   ├── diffusion.rs       #   DiffusionTransformerBlock, AtomTransformer
│   │   ├── generator.rs       #   DiffusionModule, InferenceNoiseScheduler
│   │   ├── confidence.rs      #   ConfidenceHead (pLDDT, pTM, ipTM, pDE, pAE)
│   │   ├── heads.rs           #   DistogramHead
│   │   ├── frames.rs          #   Geometric frame ops (build_frame, express_in_frame)
│   │   └── cattle_prod.rs     #   Top-level CattleProd model, forward_inference
│   │
│   ├── cattle-prod-kernels    # Fused ops (CPU fallbacks, CUDA stubs ready)
│   │   ├── layer_norm.rs      #   Fused LayerNorm
│   │   ├── fused_dropout.rs   #   Fused dropout + residual add
│   │   ├── triangle_mul.rs    #   Chunked triangle multiplication
│   │   └── triangle_attention.rs  # Fused triangle attention
│   │
│   └── cattle-prod-cli        # CLI binary
│       └── main.rs            #   pred, msa, convert subcommands
│
└── scripts/
    └── convert_weights.py     # PyTorch → safetensors conversion
```

---

## Tech Stack

```
┌─────────────────────────────────────────────────────────┐
│                    cattle-prod CLI                       │
│              (clap argument parsing)                     │
├──────────────┬──────────────────┬───────────────────────┤
│  cattle-prod │  cattle-prod     │  cattle-prod          │
│  -data       │  -model          │  -kernels             │
│              │                  │                       │
│  mmCIF parse │  Pairformer      │  Fused LayerNorm      │
│  Tokenizer   │  Diffusion       │  Fused Dropout+Add    │
│  Featurizer  │  Confidence      │  Triangle Mul/Attn    │
│  MSA/Templ   │  Heads           │  (CPU + CUDA stubs)   │
│  Metrics     │                  │                       │
├──────────────┴──────────────────┴───────────────────────┤
│                   cattle-prod-core                       │
│     Constants · Config · Residue/Token system            │
├─────────────────────────────────────────────────────────┤
│                   Candle ML Framework                    │
│          Tensor ops · Safetensors · CUDA backend         │
├──────────────┬──────────────────┬───────────────────────┤
│    ndarray   │     rayon        │    serde/json/yaml    │
│  N-dim arrays│  Parallelism     │   Serialization       │
└──────────────┴──────────────────┴───────────────────────┘
```

---

## Quick Start

### Build from source

```bash
# Clone and build (release mode for full optimization)
git clone https://github.com/twaite11/model_rustprot.git
cd model_rustprot/cattle-prod
cargo build --release -p cattle-prod-cli

# Binary is at target/release/cattle-prod (or cattle-prod.exe on Windows)
# Add to PATH or copy to /usr/local/bin
```

### Predict a structure

```bash
# Run structure prediction from a JSON input
cattle-prod pred -i input.json -o ./output -n cattle_prod_base_default_v1.0.0

# With a specific checkpoint (safetensors format)
cattle-prod pred -i input.json -o ./output --checkpoint ./weights/
```

### Convert weights from PyTorch

```bash
# Protenix .pt checkpoints → safetensors (requires Python + torch)
python scripts/convert_weights.py model.pt -o model.safetensors

# Already safetensors? Just copy:
cattle-prod convert --input model.safetensors --output ./weights/model.safetensors
```

### Input format

Cattle-Prod accepts the same JSON format as Protenix:

```json
[{
  "name": "my_complex",
  "sequences": [
    {"proteinChain": {"sequence": "MKTAYIAKQ...", "count": 1}},
    {"rnaSequence": {"sequence": "GUCGACUG...", "count": 1}}
  ]
}]
```

---

## CASCADE Integration

Cattle-Prod is the default eval engine for the [CASCADE pipeline](https://github.com/twaite11/CASCADE-Cas-Collateral-Activation-Discovery-Engineering). The integration is automatic:

```
CASCADE Evolution Loop
        │
        ▼
┌─────────────────────┐     JSON payload      ┌──────────────────┐
│  evolution_          │ ──────────────────►   │                  │
│  orchestrator.py     │                       │   cattle-prod    │
│                      │  ◄──────────────────  │   pred / msa     │
│  PXDesign → mutate   │   .cif + summary.json │                  │
│  fitness scoring     │                       │   (Rust binary)  │
│  RL bias export      │                       │                  │
└─────────────────────┘                        └──────────────────┘
        │
        ▼
  HEPN distance · ipTM · AF2-IG · specificity
```

**Setup:**

```bash
# Option 1: cattle-prod on PATH (auto-detected)
export PATH="/path/to/cattle-prod/target/release:$PATH"

# Option 2: explicit env var
export EVAL_CMD=/path/to/cattle-prod

# Option 3: falls back to protenix if cattle-prod not found
# (no config needed — just have protenix installed)
```

CASCADE auto-detects cattle-prod and uses it for all structural predictions (Phase 1 screening, Phase 2 evolution OFF/ON/off-target evals, and high-fidelity base-model scoring).

---

## Testing

186 tests across all crates, covering the full pipeline from constants to model inference:

```bash
cargo test --workspace
```

```
cattle-prod-core      ···  62 tests passed
cattle-prod-data      ···  68 tests passed  (27 unit + 41 integration)
cattle-prod-model     ···  38 tests passed
cattle-prod-kernels   ···  16 tests passed
cattle-prod-cli       ···   2 tests passed
────────────────────────────────────────────
                          186 tests passed
```

---

## License

Cattle-Prod is a derivative work of [Protenix](https://github.com/bytedance/Protenix) by ByteDance, released under the **[Apache License 2.0](./LICENSE)**.

**Commercial use is explicitly permitted.** You may use, modify, and distribute this software — including for commercial therapeutic discovery and product development — under the terms of Apache 2.0. The only requirements are attribution retention and license inclusion.

Original copyright: ByteDance and/or its affiliates (2024). LayerNorm operators reference [OneFlow](https://github.com/Oneflow-Inc/oneflow) and [FastFold](https://github.com/hpcaitech/FastFold). Some module implementations from [OpenFold](https://github.com/aqlaboratory/openfold).

---

## Citation

If you use Cattle-Prod in published research, please cite the underlying Protenix work:

```bibtex
@article{Zhang2026.02.05.703733,
  author  = {Zhang, Yuxuan and Gong, Chengyue and Zhang, Hanyu and others},
  title   = {Protenix-v1: Toward High-Accuracy Open-Source Biomolecular Structure Prediction},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.02.05.703733},
}
```

---

<div align="center">

*Built for therapeutic discovery. Powered by Rust.*

</div>
