#!/usr/bin/env python3
"""Checkpoint key mapping rules for Protenix -> cattle-prod."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class MappingSpec:
    version: str
    strip_prefixes: tuple[str, ...]
    replace_pairs: tuple[tuple[str, str], ...]
    required_keys: tuple[str, ...]
    expected_ranks: dict[str, int]


MAPPING_SPECS: dict[str, MappingSpec] = {
    "v1": MappingSpec(
        version="v1",
        strip_prefixes=(
            "module.",
            "_orig_mod.",
            "model.",
            "state_dict.",
        ),
        replace_pairs=(
            ("model.model.", ""),
            ("model.module.", ""),
        ),
        required_keys=(
            # Early stem keys where mismatches are common and fatal.
            "relpos.linear.weight",
            "input_embedder.linear.weight",
            "pairformer.blocks.0.tri_mul_out.linear_a_p.weight",
            "confidence_head.dist.linear.weight",
        ),
        expected_ranks={
            "relpos.linear.weight": 2,
            "input_embedder.linear.weight": 2,
            "pairformer.blocks.0.tri_mul_out.linear_a_p.weight": 2,
            "confidence_head.dist.linear.weight": 2,
        },
    ),
}

SUPPORTED_CHECKPOINT_PROFILES = {
    "v0.5.0-mini": {
        "model_name": "cattle_prod_mini_default_v0.5.0",
        "mapping_version": "v1",
    },
    "v0.5.0-base": {
        "model_name": "cattle_prod_base_default_v0.5.0",
        "mapping_version": "v1",
    },
    "v1.0.0-base": {
        "model_name": "cattle_prod_base_default_v1.0.0",
        "mapping_version": "v1",
    },
    "v1.0.0-20250630-base": {
        "model_name": "cattle_prod_base_default_v1.0.0_20250630",
        "mapping_version": "v1",
    },
}


def available_mapping_versions() -> Iterable[str]:
    return MAPPING_SPECS.keys()


def convert_key(key: str, mapping_version: str = "v1") -> str:
    spec = MAPPING_SPECS[mapping_version]
    out = key

    # Apply static substring replacements first.
    for src, dst in spec.replace_pairs:
        if src in out:
            out = out.replace(src, dst)

    # Then peel leading wrapper prefixes repeatedly.
    changed = True
    while changed:
        changed = False
        for prefix in spec.strip_prefixes:
            if out.startswith(prefix):
                out = out[len(prefix) :]
                changed = True

    return out


def required_keys(mapping_version: str = "v1") -> tuple[str, ...]:
    return MAPPING_SPECS[mapping_version].required_keys


def expected_ranks(mapping_version: str = "v1") -> dict[str, int]:
    return MAPPING_SPECS[mapping_version].expected_ranks.copy()


def supported_profiles() -> dict[str, dict[str, str]]:
    return SUPPORTED_CHECKPOINT_PROFILES.copy()
