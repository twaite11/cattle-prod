import tempfile
import unittest
from pathlib import Path

import sys

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

from checkpoint_keymap import convert_key

try:
    import torch
    from safetensors.torch import save_file
    from convert_weights import verify_safetensors
    HAS_TORCH = True
except Exception:  # pragma: no cover - environment-dependent import
    HAS_TORCH = False


class ConvertWeightsTests(unittest.TestCase):
    def test_convert_key_strips_wrappers(self):
        self.assertEqual(
            convert_key("module.model.relpos.linear.weight", mapping_version="v1"),
            "relpos.linear.weight",
        )
        self.assertEqual(
            convert_key("_orig_mod.state_dict.input_embedder.linear.weight", mapping_version="v1"),
            "input_embedder.linear.weight",
        )

    @unittest.skipUnless(HAS_TORCH, "torch/safetensors not available")
    def test_verify_safetensors_required_keys(self):
        with tempfile.TemporaryDirectory() as td:
            ok_path = Path(td) / "ok.safetensors"
            fail_path = Path(td) / "fail.safetensors"
            save_file(
                {
                    "relpos.linear.weight": torch.randn(4, 4),
                    "input_embedder.linear.weight": torch.randn(4, 4),
                    "pairformer.blocks.0.tri_mul_out.linear_a_p.weight": torch.randn(4, 4),
                    "confidence_head.dist.linear.weight": torch.randn(4, 4),
                },
                str(ok_path),
            )
            save_file(
                {
                    "input_embedder.linear.weight": torch.randn(4, 4),
                },
                str(fail_path),
            )
            self.assertEqual(verify_safetensors(str(ok_path), mapping_version="v1"), 0)
            self.assertEqual(verify_safetensors(str(fail_path), mapping_version="v1"), 2)


if __name__ == "__main__":
    unittest.main()
