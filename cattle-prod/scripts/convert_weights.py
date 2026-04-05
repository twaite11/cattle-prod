#!/usr/bin/env python3
"""Convert Cattle-Prod PyTorch checkpoint to safetensors format for Rust inference."""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import torch
from safetensors.torch import save_file
from safetensors import safe_open

from checkpoint_keymap import (
    available_mapping_versions,
    convert_key,
    expected_ranks,
    required_keys,
    supported_profiles,
)


def write_diagnostic_report(
    report_path: str,
    mapping_version: str,
    source_keys: list[str],
    converted_keys: list[str],
    req_keys: tuple[str, ...],
    converted_shapes: dict[str, tuple[int, ...]],
    expected_key_ranks: dict[str, int],
) -> None:
    converted_set = set(converted_keys)
    missing_required = [k for k in req_keys if k not in converted_set]
    shape_mismatches = []
    transpose_candidates = []
    for key, rank in expected_key_ranks.items():
        shape = converted_shapes.get(key)
        if not shape:
            continue
        if len(shape) != rank:
            shape_mismatches.append({"key": key, "shape": list(shape), "expected_rank": rank})
        if len(shape) == 2 and key.endswith(".weight"):
            transpose_candidates.append({"key": key, "shape": list(shape), "transpose_shape": [shape[1], shape[0]]})
    report = {
        "mapping_version": mapping_version,
        "supported_profiles": supported_profiles(),
        "source_key_count": len(source_keys),
        "converted_key_count": len(converted_keys),
        "missing_required_keys": missing_required,
        "shape_mismatches": shape_mismatches,
        "transpose_candidates": transpose_candidates[:100],
        "sample_source_keys": source_keys[:200],
        "sample_converted_keys": converted_keys[:200],
    }
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")


def validate_tensor(name: str, tensor: torch.Tensor) -> list[str]:
    """Return a list of warnings for suspicious tensors."""
    warnings = []
    if tensor.numel() == 0:
        warnings.append(f"  WARN: {name} is empty (0 elements)")
    if any(d > 1_000_000 for d in tensor.shape):
        warnings.append(
            f"  WARN: {name} has a very large dimension: {list(tensor.shape)}"
        )
    if torch.isnan(tensor).any().item():
        warnings.append(f"  WARN: {name} contains NaN values")
    if torch.isinf(tensor).any().item():
        warnings.append(f"  WARN: {name} contains Inf values")
    return warnings


def convert_checkpoint(
    input_path: str,
    output_path: str,
    mapping_version: str = "v1",
    strict_validate: bool = True,
    diagnose_only: bool = False,
    report_path: str | None = None,
) -> None:
    """Load PyTorch checkpoint and save as safetensors."""
    print(f"Loading checkpoint: {input_path}")
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print("  Found 'model_state_dict' key — extracting model weights")
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        print("  Found 'state_dict' key — extracting model weights")
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and all(
        isinstance(v, torch.Tensor) for v in checkpoint.values()
    ):
        print("  Checkpoint is a plain state dict")
        state_dict = checkpoint
    else:
        candidates = [
            k for k, v in checkpoint.items() if isinstance(v, dict)
        ] if isinstance(checkpoint, dict) else []
        if candidates:
            print(
                f"  Could not find standard keys. Dict keys: {list(checkpoint.keys())}"
            )
            print(f"  Trying first dict-valued key: '{candidates[0]}'")
            state_dict = checkpoint[candidates[0]]
        else:
            print(
                "ERROR: Unable to locate a state dict in the checkpoint.",
                file=sys.stderr,
            )
            sys.exit(1)

    dtype_counts: Counter[str] = Counter()
    total_params = 0
    total_bytes = 0
    all_warnings: list[str] = []

    converted: dict[str, torch.Tensor] = {}
    source_keys: list[str] = []
    converted_keys: list[str] = []
    converted_shapes: dict[str, tuple[int, ...]] = {}
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        source_keys.append(key)
        new_key = convert_key(key, mapping_version=mapping_version)
        converted_keys.append(new_key)
        dtype_counts[str(tensor.dtype)] += 1
        total_params += tensor.numel()

        # safetensors only supports float32, float16, bfloat16, int types, etc.
        # Convert float16 / bfloat16 to float32 for maximum compatibility.
        if tensor.dtype in (torch.float16, torch.bfloat16):
            tensor = tensor.to(torch.float32)

        total_bytes += tensor.numel() * tensor.element_size()

        ws = validate_tensor(new_key, tensor)
        all_warnings.extend(ws)

        converted[new_key] = tensor
        converted_shapes[new_key] = tuple(int(d) for d in tensor.shape)

    print()
    print("=== Conversion Statistics ===")
    print(f"  Mapping version:      {mapping_version}")
    print(f"  Parameters (tensors): {len(converted)}")
    print(f"  Total elements:       {total_params:,}")
    print(f"  Output size (approx): {total_bytes / (1024 ** 2):.1f} MB")
    print(f"  Data types found:     {dict(dtype_counts)}")

    req_keys = required_keys(mapping_version)
    missing_required = [k for k in req_keys if k not in converted]
    if missing_required:
        print("  Required-key check:   FAILED")
        for k in missing_required[:20]:
            print(f"    missing: {k}")
    else:
        print("  Required-key check:   OK")

    if all_warnings:
        print()
        print("=== Warnings ===")
        for w in all_warnings:
            print(w)

    if report_path:
        exp_ranks = expected_ranks(mapping_version)
        write_diagnostic_report(
            report_path=report_path,
            mapping_version=mapping_version,
            source_keys=source_keys,
            converted_keys=converted_keys,
            req_keys=req_keys,
            converted_shapes=converted_shapes,
            expected_key_ranks=exp_ranks,
        )
        print(f"Wrote diagnostic report: {report_path}")

    if strict_validate and missing_required:
        print(
            "ERROR: Required tensor keys are missing after conversion. "
            "Run with --diagnose_only to inspect key mapping and --mapping_version to test alternatives.",
            file=sys.stderr,
        )
        sys.exit(2)

    if diagnose_only:
        print("Diagnose-only mode; skipping safetensors write.")
        return

    print()
    print(f"Saving safetensors: {output_path}")
    save_file(converted, output_path)
    print("Done.")


def verify_safetensors(path: str, mapping_version: str = "v1") -> int:
    req = required_keys(mapping_version)
    missing: list[str] = []
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
    for key in req:
        if key not in keys:
            missing.append(key)
    if missing:
        print("Verification FAILED: missing required keys:")
        for key in missing[:20]:
            print(f"  - {key}")
        return 2
    print("Verification OK: all required keys present.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Cattle-Prod PyTorch checkpoint to safetensors format."
    )
    parser.add_argument(
        "input",
        type=str,
        nargs="?",
        help="Path to input .pt / .pth checkpoint file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help=(
            "Path to output .safetensors file. "
            "Defaults to the input path with .safetensors extension."
        ),
    )
    parser.add_argument(
        "--mapping_version",
        type=str,
        default="v1",
        choices=sorted(available_mapping_versions()),
        help="Versioned key mapping ruleset.",
    )
    parser.add_argument(
        "--no_strict_validate",
        action="store_true",
        help="Allow conversion output even if required keys are missing.",
    )
    parser.add_argument(
        "--diagnose_only",
        action="store_true",
        help="Run key mapping diagnostics without writing safetensors.",
    )
    parser.add_argument(
        "--diagnostic_report",
        type=str,
        default=None,
        help="Write diagnostic JSON report to this path.",
    )
    parser.add_argument(
        "--verify_safetensors",
        type=str,
        default=None,
        help="Verify an existing safetensors file has required keys.",
    )
    args = parser.parse_args()

    if args.verify_safetensors:
        sys.exit(verify_safetensors(args.verify_safetensors, mapping_version=args.mapping_version))

    if not args.input:
        print("ERROR: input checkpoint path is required unless --verify_safetensors is used.", file=sys.stderr)
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is not None:
        output_path = args.output
    else:
        output_path = str(input_path.with_suffix(".safetensors"))

    convert_checkpoint(
        str(input_path),
        output_path,
        mapping_version=args.mapping_version,
        strict_validate=not args.no_strict_validate,
        diagnose_only=args.diagnose_only,
        report_path=args.diagnostic_report,
    )


if __name__ == "__main__":
    main()
