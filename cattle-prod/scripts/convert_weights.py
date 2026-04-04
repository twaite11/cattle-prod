#!/usr/bin/env python3
"""Convert Cattle-Prod PyTorch checkpoint to safetensors format for Rust inference."""

import argparse
import sys
from pathlib import Path
from collections import Counter

import torch
from safetensors.torch import save_file


def convert_key(key: str) -> str:
    """Convert PyTorch state dict key to Candle-compatible name."""
    if key.startswith("module."):
        key = key[len("module."):]
    return key


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


def convert_checkpoint(input_path: str, output_path: str) -> None:
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
    for key, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue

        new_key = convert_key(key)
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

    print()
    print("=== Conversion Statistics ===")
    print(f"  Parameters (tensors): {len(converted)}")
    print(f"  Total elements:       {total_params:,}")
    print(f"  Output size (approx): {total_bytes / (1024 ** 2):.1f} MB")
    print(f"  Data types found:     {dict(dtype_counts)}")

    if all_warnings:
        print()
        print("=== Warnings ===")
        for w in all_warnings:
            print(w)

    print()
    print(f"Saving safetensors: {output_path}")
    save_file(converted, output_path)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Cattle-Prod PyTorch checkpoint to safetensors format."
    )
    parser.add_argument(
        "input",
        type=str,
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
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    if args.output is not None:
        output_path = args.output
    else:
        output_path = str(input_path.with_suffix(".safetensors"))

    convert_checkpoint(str(input_path), output_path)


if __name__ == "__main__":
    main()
