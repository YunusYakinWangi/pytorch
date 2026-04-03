import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMetaRandintLike(TestCase):
    def test_randint_like_tensor_overload_dtype_kwarg(self):
        """aten.randint_like.Tensor meta should respect dtype kwarg."""
        x = torch.randn(3, 4, device="meta")
        high = torch.tensor(10, device="meta")
        y = torch.ops.aten.randint_like.Tensor(x, high, dtype=torch.int32)
        self.assertEqual(y.dtype, torch.int32)

    def test_randint_like_tensor_overload_preserves_default(self):
        """Without kwargs, should preserve self's properties."""
        x = torch.randn(3, 4, device="meta", dtype=torch.float16)
        high = torch.tensor(10, device="meta")
        y = torch.ops.aten.randint_like.Tensor(x, high)
        self.assertEqual(y.dtype, torch.float16)
        self.assertEqual(y.shape, (3, 4))


if __name__ == "__main__":
    run_tests()
