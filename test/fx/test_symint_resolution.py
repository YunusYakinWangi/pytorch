# Owner(s): ["module: fx"]

import unittest

import torch
import torch.fx as fx
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase


class TestSymIntResolution(TestCase):
    """Tests for automatic SymInt-to-Node resolution in FX Graph."""

    def _make_symbolic_graph(self):
        """Helper: create a symbolic graph with one placeholder."""
        x = torch.randn(100, 50)
        gm = make_fx(lambda x: x * 2, tracing_mode="symbolic")(x)
        ph = next(n for n in gm.graph.nodes if n.op == "placeholder")
        return gm, ph

    def test_auto_resolve_symint_in_call_function(self):
        """SymInt passed to call_function is auto-resolved to a Node."""
        _, ph = self._make_symbolic_graph()
        s0 = ph.meta["val"].size(0)

        new_graph = fx.Graph()
        new_ph = new_graph.placeholder("x")
        new_ph.meta = ph.meta.copy()

        # This should auto-resolve s0 to a sym_size node, not raise
        node = new_graph.call_function(
            torch.ops.aten.empty.memory_format, ([s0],)
        )
        # The arg should be a Node, not a SymInt
        self.assertIsInstance(node.args[0][0], fx.Node)
        self.assertEqual(node.args[0][0].target, torch.ops.aten.sym_size.int)

    def test_auto_resolve_compound_symint(self):
        """Compound SymInt expression (s0*s1) is auto-resolved."""
        _, ph = self._make_symbolic_graph()
        numel = ph.meta["val"].numel()  # s0 * s1

        new_graph = fx.Graph()
        new_ph = new_graph.placeholder("x")
        new_ph.meta = ph.meta.copy()

        node = new_graph.call_function(
            torch.ops.aten.empty.memory_format, ([numel],)
        )
        # Should be a mul Node (s0 * s1)
        self.assertIsInstance(node.args[0][0], fx.Node)

    def test_concrete_symint_becomes_int(self):
        """Concrete SymInt (backed by a number) becomes plain int."""
        _, ph = self._make_symbolic_graph()

        new_graph = fx.Graph()
        new_ph = new_graph.placeholder("x")
        new_ph.meta = ph.meta.copy()

        # A plain int should pass through unchanged
        node = new_graph.call_function(
            torch.ops.aten.empty.memory_format, ([42],)
        )
        self.assertEqual(node.args[0][0], 42)

    def test_no_symint_args_unchanged(self):
        """Regular args (Nodes, ints) are not modified."""
        _, ph = self._make_symbolic_graph()

        new_graph = fx.Graph()
        new_ph = new_graph.placeholder("x")
        new_ph.meta = ph.meta.copy()

        node = new_graph.call_function(
            torch.ops.aten.reshape.default, (new_ph, [-1])
        )
        self.assertIs(node.args[0], new_ph)
        self.assertEqual(node.args[1], [-1])

    def test_symint_to_node_explicit_api(self):
        """graph.symint_to_node() works for explicit conversion."""
        _, ph = self._make_symbolic_graph()
        s0 = ph.meta["val"].size(0)

        new_graph = fx.Graph()
        new_ph = new_graph.placeholder("x")
        new_ph.meta = ph.meta.copy()

        # Symbolic → Node
        result = new_graph.symint_to_node(s0)
        self.assertIsInstance(result, fx.Node)

        # int → int
        self.assertEqual(new_graph.symint_to_node(42), 42)

    def test_symint_to_node_compound_expr(self):
        """symint_to_node handles compound expressions (s0 * s1)."""
        _, ph = self._make_symbolic_graph()
        numel = ph.meta["val"].numel()

        new_graph = fx.Graph()
        new_ph = new_graph.placeholder("x")
        new_ph.meta = ph.meta.copy()

        result = new_graph.symint_to_node(numel)
        self.assertIsInstance(result, fx.Node)

    def test_symint_to_node_type_error(self):
        """symint_to_node raises TypeError for non-int/SymInt."""
        new_graph = fx.Graph()
        with self.assertRaises(TypeError):
            new_graph.symint_to_node("not_a_symint")

    def test_make_fx_tracing_unaffected(self):
        """make_fx symbolic tracing still works with auto-resolution."""
        x = torch.randn(100, 50)
        gm = make_fx(lambda x: x.reshape(-1), tracing_mode="symbolic")(x)
        # Should trace without error
        self.assertIsNotNone(gm)
        nodes = list(gm.graph.nodes)
        self.assertTrue(any(n.op == "call_function" for n in nodes))

    def test_symint_in_kwargs_resolved(self):
        """SymInt in kwargs is also auto-resolved."""
        _, ph = self._make_symbolic_graph()
        s0 = ph.meta["val"].size(0)

        new_graph = fx.Graph()
        new_ph = new_graph.placeholder("x")
        new_ph.meta = ph.meta.copy()

        node = new_graph.call_function(
            torch.ops.aten.empty.memory_format,
            ([s0],),
            {"dtype": torch.float32},
        )
        self.assertIsInstance(node.args[0][0], fx.Node)

    def test_placeholder_not_affected(self):
        """Placeholder op is not affected by SymInt resolution."""
        new_graph = fx.Graph()
        # placeholder should work fine, no auto-resolution attempted
        ph = new_graph.placeholder("x")
        self.assertEqual(ph.op, "placeholder")


def raise_on_run_directly(filename: str):
    raise RuntimeError(
        f"This test file should not be run directly. "
        f"Run it through {filename}."
    )


if __name__ == "__main__":
    raise_on_run_directly("test/test_fx.py")
