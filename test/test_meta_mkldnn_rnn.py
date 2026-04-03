import torch
from torch.testing._internal.common_utils import run_tests, TestCase


class TestMkldnnRnnBackwardMeta(TestCase):
    def test_mkldnn_rnn_backward_dtype(self):
        """Gradient tensors should always be float32, even if input is bfloat16."""
        hidden_size = 16
        seq_len, batch, input_size = 5, 2, 8

        input_t = torch.randn(seq_len, batch, input_size, device="meta", dtype=torch.bfloat16)
        weight1 = torch.randn(hidden_size * 4, input_size, device="meta", dtype=torch.bfloat16)
        weight2 = torch.randn(hidden_size * 4, device="meta", dtype=torch.bfloat16)
        weight3 = torch.randn(hidden_size * 4, device="meta", dtype=torch.bfloat16)
        weight4 = torch.randn(hidden_size * 4, hidden_size, device="meta", dtype=torch.bfloat16)
        hx = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.bfloat16)
        cx = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.bfloat16)
        output = torch.randn(seq_len, batch, hidden_size, device="meta", dtype=torch.bfloat16)
        hy = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.bfloat16)
        cy = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.bfloat16)
        grad_output = torch.randn(seq_len, batch, hidden_size, device="meta", dtype=torch.bfloat16)
        grad_hy = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.bfloat16)
        grad_cy = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.bfloat16)
        workspace = torch.randn(10, device="meta", dtype=torch.bfloat16)

        # mode=2 is LSTM
        result = torch.ops.aten.mkldnn_rnn_layer_backward(
            input_t, weight1, weight2, weight3, weight4,
            hx, cx, output, hy, cy,
            grad_output, grad_hy, grad_cy,
            False, 2, hidden_size, 1, True, True,
            False, [], False, workspace,
        )
        diff_x, diff_w1, diff_w2, diff_b1, diff_b2, diff_hx, diff_cx = result
        self.assertEqual(diff_x.dtype, torch.float32, "diff_x should be float32")
        self.assertEqual(diff_w1.dtype, torch.float32, "diff_w1 should be float32")
        self.assertEqual(diff_w2.dtype, torch.float32, "diff_w2 should be float32")
        self.assertEqual(diff_b1.dtype, torch.float32, "diff_b1 should be float32")
        self.assertEqual(diff_b2.dtype, torch.float32, "diff_b2 should be float32")
        self.assertEqual(diff_hx.dtype, torch.float32, "diff_hx should be float32")
        self.assertEqual(diff_cx.dtype, torch.float32, "diff_cx should be float32")

    def test_mkldnn_rnn_backward_gru_bias_shape(self):
        """GRU bias gradient shape should be [4*hidden_size], not [3*hidden_size]."""
        hidden_size = 16
        seq_len, batch, input_size = 5, 2, 8

        input_t = torch.randn(seq_len, batch, input_size, device="meta", dtype=torch.float32)
        weight1 = torch.randn(hidden_size * 3, input_size, device="meta", dtype=torch.float32)
        weight2 = torch.randn(hidden_size * 3, device="meta", dtype=torch.float32)
        weight3 = torch.randn(hidden_size * 3, device="meta", dtype=torch.float32)
        weight4 = torch.randn(hidden_size * 3, hidden_size, device="meta", dtype=torch.float32)
        hx = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.float32)
        cx = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.float32)
        output = torch.randn(seq_len, batch, hidden_size, device="meta", dtype=torch.float32)
        hy = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.float32)
        cy = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.float32)
        grad_output = torch.randn(seq_len, batch, hidden_size, device="meta", dtype=torch.float32)
        grad_hy = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.float32)
        grad_cy = torch.randn(1, batch, hidden_size, device="meta", dtype=torch.float32)
        workspace = torch.randn(10, device="meta", dtype=torch.float32)

        # mode=3 is GRU
        result = torch.ops.aten.mkldnn_rnn_layer_backward(
            input_t, weight1, weight2, weight3, weight4,
            hx, cx, output, hy, cy,
            grad_output, grad_hy, grad_cy,
            False, 3, hidden_size, 1, True, True,
            False, [], False, workspace,
        )
        diff_x, diff_w1, diff_w2, diff_b1, diff_b2, diff_hx, diff_cx = result
        # _shuffle_bias for GRU produces [4*hidden_size] from two [3*hidden_size] inputs
        expected_bias_shape = torch.Size([4 * hidden_size])
        self.assertEqual(diff_b1.shape, expected_bias_shape, "GRU bias grad should be [4*hidden_size]")
        self.assertEqual(diff_b2.shape, expected_bias_shape, "GRU bias grad should be [4*hidden_size]")


if __name__ == "__main__":
    run_tests()
