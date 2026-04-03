import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._refs.nn.functional import prelu as prelu_decomp


class TestPreluMeta(TestCase):
    def test_prelu_decomp_dtype_mismatch_error(self):
        """prelu decomposition should reject mismatched dtypes."""
        x = torch.randn(3, 4, device="meta", dtype=torch.float32)
        weight = torch.randn(4, device="meta", dtype=torch.float16)
        with self.assertRaisesRegex(RuntimeError, "Type promoting not supported"):
            prelu_decomp(x, weight)

    def test_prelu_decomp_same_dtype_ok(self):
        """prelu decomposition should work when dtypes match."""
        x = torch.randn(3, 4, device="meta", dtype=torch.float32)
        weight = torch.randn(4, device="meta", dtype=torch.float32)
        result = prelu_decomp(x, weight)
        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(result.dtype, torch.float32)


if __name__ == "__main__":
    run_tests()
