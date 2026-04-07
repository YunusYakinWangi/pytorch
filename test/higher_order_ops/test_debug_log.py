# Owner(s): ["module: higher order operators"]
"""Tests for torch.utils.debug_log.debug_grad_log."""

import logging

import torch
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils.debug_log import debug_grad_log


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records: list[str] = []

    def emit(self, record):
        self.records.append(self.format(record))


class TestDebugGradLog(TestCase):
    def _add_log_capture(self):
        capture = _LogCapture()
        logger = logging.getLogger("torch.utils.debug_log")
        logger.addHandler(capture)
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.removeHandler, capture)
        return capture

    def test_single_tensor(self):
        capture = self._add_log_capture()

        x = torch.randn(4, requires_grad=True)
        y = x * 2
        debug_grad_log(y)
        y.sum().backward()

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("t0_grad_norm=", bwd[0])

    def test_multi_tensor(self):
        capture = self._add_log_capture()

        x = torch.randn(4, requires_grad=True)
        y = torch.randn(4, requires_grad=True)
        z = x * 2 + y * 3
        debug_grad_log(x, y)
        z.sum().backward()

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("t0_grad_norm=", bwd[0])
        self.assertIn("t1_grad_norm=", bwd[0])

    def test_gradient_values(self):
        capture = self._add_log_capture()

        x = torch.tensor([1.0], requires_grad=True)
        y = torch.tensor([1.0], requires_grad=True)
        debug_grad_log(x, y)
        (x * 2 + y * 3).sum().backward()

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 1)
        self.assertIn("t0_grad_norm=2.0000", bwd[0])
        self.assertIn("t1_grad_norm=3.0000", bwd[0])

    def test_no_requires_grad_no_log(self):
        capture = self._add_log_capture()

        x = torch.randn(3, requires_grad=False)
        debug_grad_log(x)

        bwd = [r for r in capture.records if "[bwd]" in r]
        self.assertEqual(len(bwd), 0)

    def test_forward_is_noop(self):
        capture = self._add_log_capture()

        x = torch.randn(3, requires_grad=True)
        debug_grad_log(x)

        self.assertEqual(len(capture.records), 0)


if __name__ == "__main__":
    run_tests()
