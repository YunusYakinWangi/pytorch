# Owner(s): ["module: dynamo"]

import torch
import torch._dynamo.testing
from torch._dynamo.dynamic_spec import (
    _active_dynamic_shapes,
    get_active_spec_for_arg,
    get_active_spec_for_dim,
    IntSpec,
    IntSpecType,
    ModelSpec,
    TensorSpec,
)
from torch._dynamo.testing import EagerAndRecordGraphs
from torch.fx.experimental.symbolic_shapes import free_unbacked_symbols
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase


def _tensor_placeholder_shape(gm):
    """Return the shape of the first tensor-typed placeholder in ``gm``."""
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            ev = node.meta.get("example_value")
            if isinstance(ev, torch.Tensor):
                return ev.shape
    raise AssertionError("no tensor placeholder found")


class TestIntSpecConstruction(TestCase):
    """Construction via the classmethod factories."""

    def test_static(self):
        s = IntSpec.static("x", value=10)
        self.assertEqual(s.name, "x")
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertEqual(s.value, 10)

    def test_static_no_value(self):
        s = IntSpec.static()
        self.assertEqual(s.type, IntSpecType.STATIC)
        self.assertIsNone(s.value)

    def test_backed(self):
        s = IntSpec.backed("batch", min=1, max=64, guarding_hint=32)
        self.assertEqual(s.name, "batch")
        self.assertEqual(s.type, IntSpecType.BACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 64)
        self.assertEqual(s.guarding_hint, 32)

    def test_unbacked(self):
        s = IntSpec.unbacked("seq", min=1, max=2048, optimization_hint=512)
        self.assertEqual(s.type, IntSpecType.UNBACKED)
        self.assertEqual(s.min, 1)
        self.assertEqual(s.max, 2048)
        self.assertEqual(s.optimization_hint, 512)

    def test_type_required_on_init(self):
        with self.assertRaises(TypeError):
            IntSpec("x")  # no type kwarg

    def test_type_not_none(self):
        with self.assertRaises(TypeError):
            IntSpec("x", type=None)  # type: ignore[arg-type]


class TestIntSpecImmutable(TestCase):
    """Once constructed, the mode is fixed; per-mode accessors enforce it."""

    def test_value_only_on_static(self):
        s = IntSpec.backed("x")
        with self.assertRaisesRegex(AttributeError, "STATIC"):
            _ = s.value

    def test_guarding_hint_only_on_backed(self):
        s = IntSpec.unbacked("x")
        with self.assertRaisesRegex(AttributeError, "BACKED"):
            _ = s.guarding_hint
        s2 = IntSpec.static("x")
        with self.assertRaisesRegex(AttributeError, "BACKED"):
            _ = s2.guarding_hint

    def test_optimization_hint_only_on_unbacked(self):
        s = IntSpec.backed("x")
        with self.assertRaisesRegex(AttributeError, "UNBACKED"):
            _ = s.optimization_hint
        s2 = IntSpec.static("x")
        with self.assertRaisesRegex(AttributeError, "UNBACKED"):
            _ = s2.optimization_hint

    def test_type_is_read_only(self):
        s = IntSpec.static("x", value=10)
        with self.assertRaises(AttributeError):
            s.type = IntSpecType.BACKED  # type: ignore[misc]

    def test_no_fluent_type_reset(self):
        # IntSpec has no instance method that reassigns type. The mode-named
        # factories are classmethods: calling one "on an instance" returns a
        # fresh IntSpec and does not mutate the original.
        s = IntSpec.static("x")
        new = IntSpec.backed("x")
        self.assertIs(s.type, IntSpecType.STATIC)
        self.assertIs(new.type, IntSpecType.BACKED)
        self.assertIsNot(s, new)


class TestIntSpecValidation(TestCase):
    """Cross-parameter validation rejects bad combinations."""

    def test_static_rejects_min(self):
        with self.assertRaisesRegex(ValueError, "min/max.*STATIC"):
            IntSpec("x", type=IntSpecType.STATIC, min=1)

    def test_static_rejects_max(self):
        with self.assertRaisesRegex(ValueError, "min/max.*STATIC"):
            IntSpec("x", type=IntSpecType.STATIC, max=100)

    def test_static_rejects_guarding_hint(self):
        with self.assertRaisesRegex(ValueError, "guarding_hint.*BACKED"):
            IntSpec("x", type=IntSpecType.STATIC, guarding_hint=10)

    def test_static_rejects_optimization_hint(self):
        with self.assertRaisesRegex(ValueError, "optimization_hint.*UNBACKED"):
            IntSpec("x", type=IntSpecType.STATIC, optimization_hint=10)

    def test_backed_rejects_value(self):
        with self.assertRaisesRegex(ValueError, "value.*STATIC"):
            IntSpec("x", type=IntSpecType.BACKED, value=42)

    def test_backed_rejects_optimization_hint(self):
        with self.assertRaisesRegex(ValueError, "optimization_hint.*UNBACKED"):
            IntSpec("x", type=IntSpecType.BACKED, optimization_hint=10)

    def test_unbacked_rejects_value(self):
        with self.assertRaisesRegex(ValueError, "value.*STATIC"):
            IntSpec("x", type=IntSpecType.UNBACKED, value=42)

    def test_unbacked_rejects_guarding_hint(self):
        with self.assertRaisesRegex(ValueError, "guarding_hint.*BACKED"):
            IntSpec("x", type=IntSpecType.UNBACKED, guarding_hint=10)

    def test_backed_min_greater_than_max(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec.backed("x", min=100, max=1)

    def test_unbacked_min_greater_than_max(self):
        with self.assertRaisesRegex(ValueError, "min must be <= max"):
            IntSpec.unbacked("x", min=100, max=1)


class TestIntSpecEq(TestCase):
    """__eq__ and __hash__."""

    def test_eq(self):
        a = IntSpec.backed("x", min=1, max=64)
        b = IntSpec.backed("x", min=1, max=64)
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))

    def test_neq_different_type(self):
        self.assertNotEqual(IntSpec.backed("x"), IntSpec.static("x"))

    def test_neq_different_name(self):
        self.assertNotEqual(IntSpec.backed("x"), IntSpec.backed("y"))

    def test_eq_not_intspec(self):
        self.assertNotEqual(IntSpec.static("x", value=1), 1)


class TestTensorSpecConstruction(TestCase):
    """Construction and list-like interface."""

    def test_basic(self):
        ts = TensorSpec(3)
        self.assertEqual(ts.rank, 3)
        self.assertEqual(len(ts), 3)
        for spec in ts:
            self.assertIsNone(spec)

    def test_zero_rank(self):
        ts = TensorSpec(0)
        self.assertEqual(ts.rank, 0)
        self.assertEqual(len(ts), 0)

    def test_negative_rank(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            TensorSpec(-1)

    def test_from_list(self):
        specs = [IntSpec.static(value=10), None, IntSpec.backed(min=1)]
        ts = TensorSpec.from_list(specs)
        self.assertEqual(ts.rank, 3)
        self.assertEqual(ts[0], IntSpec.static(value=10))
        self.assertIsNone(ts[1])

    def test_getitem_setitem(self):
        ts = TensorSpec(2)
        spec = IntSpec.backed("batch", min=1)
        ts[0] = spec
        self.assertEqual(ts[0], spec)
        self.assertIsNone(ts[1])

    def test_set_fluent(self):
        ts = TensorSpec(3)
        result = ts.set(0, IntSpec.static(value=10))
        self.assertIs(result, ts)
        self.assertEqual(ts[0], IntSpec.static(value=10))

    def test_iter(self):
        ts = TensorSpec(2)
        ts[0] = IntSpec.static(value=5)
        items = list(ts)
        self.assertEqual(len(items), 2)
        self.assertEqual(items[0], IntSpec.static(value=5))
        self.assertIsNone(items[1])

    def test_index_out_of_range(self):
        ts = TensorSpec(2)
        with self.assertRaises(IndexError):
            ts[5]

    def test_sparse_set(self):
        ts = TensorSpec(4)
        ts.set(1, IntSpec.backed("h"))
        ts.set(3, IntSpec.backed("w"))
        self.assertIsNone(ts[0])
        self.assertIsNotNone(ts[1])
        self.assertIsNone(ts[2])
        self.assertIsNotNone(ts[3])


class TestTensorSpecEq(TestCase):
    """__eq__ and __hash__."""

    def test_eq(self):
        a = TensorSpec(2).set(0, IntSpec.static(value=10))
        b = TensorSpec(2).set(0, IntSpec.static(value=10))
        self.assertEqual(a, b)

    def test_neq_different_rank(self):
        self.assertNotEqual(TensorSpec(2), TensorSpec(3))

    def test_neq_different_specs(self):
        a = TensorSpec(2).set(0, IntSpec.static(value=10))
        b = TensorSpec(2).set(0, IntSpec.static(value=20))
        self.assertNotEqual(a, b)


class TestTensorSpecCompile(TestCase):
    """torch.compile(dynamic_shapes=...) with TensorSpec."""

    def test_tensorspec_backed_dim(self):
        torch._dynamo.reset()
        ts = TensorSpec(2).set(0, IntSpec.backed("batch"))
        fn = torch.compile(
            lambda x: x.sum(0),
            backend="eager",
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x.sum(0))

    def test_tensorspec_mixed_dims(self):
        torch._dynamo.reset()
        ts = TensorSpec(2).set(0, IntSpec.backed("batch")).set(1, IntSpec.static())
        fn = torch.compile(
            lambda x: x + 1,
            backend="eager",
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x + 1)

    def test_tensorspec_partial_spec(self):
        torch._dynamo.reset()
        ts = TensorSpec(2).set(0, IntSpec.backed("batch"))
        fn = torch.compile(
            lambda x: x.sum(0),
            backend="eager",
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x.sum(0))

    @skipIfTorchDynamo("graph capture unreliable when dynamo traces the test")
    def test_tensorspec_backed_graph_has_backed_symbol(self):
        """BACKED TensorSpec dim appears as a backed SymInt in the final graph."""
        torch._dynamo.reset()
        backend = EagerAndRecordGraphs()
        ts = TensorSpec(2).set(0, IntSpec.backed("batch"))
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            dynamic_shapes={"x": ts},
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))
        self.assertLessEqual(len(backend.graphs), 2)
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)


class TestIntSpecCompile(TestCase):
    """torch.compile(dynamic_shapes=...) with IntSpec — graph inspection
    and precedence tests."""

    @skipIfTorchDynamo("graph capture unreliable when dynamo traces the test")
    def test_static_graph_has_concrete_shape(self):
        """STATIC dim appears as a concrete int in the captured graph; each
        distinct shape yields a new graph."""
        torch._dynamo.reset()
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x + 1,
            backend=backend,
            dynamic_shapes={"x": {0: IntSpec.static()}},
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        fn(torch.randn(4, 3))  # cache hit

        self.assertEqual(len(backend.graphs), 2)
        for gm in backend.graphs:
            shape = _tensor_placeholder_shape(gm)
            self.assertIsInstance(shape[0], int)

    @skipIfTorchDynamo("graph capture unreliable when dynamo traces the test")
    def test_backed_graph_has_backed_symbol(self):
        """BACKED dim appears as a backed SymInt in the final graph."""
        torch._dynamo.reset()
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            dynamic_shapes={"x": {0: IntSpec.backed("batch")}},
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))

        # Scaffolding uses maybe_mark_dynamic: first call specializes static,
        # second promotes to dynamic, then cached. ≤ 2 graphs total.
        self.assertLessEqual(len(backend.graphs), 2)
        shape = _tensor_placeholder_shape(backend.graphs[-1])
        self.assertIsInstance(shape[0], torch.SymInt)
        # backed symbol: no free unbacked symbols
        self.assertEqual(len(free_unbacked_symbols(shape[0])), 0)

    @skipIfTorchDynamo("graph capture unreliable when dynamo traces the test")
    def test_unbacked_graph_has_unbacked_symbol(self):
        """UNBACKED dim appears as an unbacked SymInt; single compile covers all shapes."""
        torch._dynamo.reset()
        backend = EagerAndRecordGraphs()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=backend,
            dynamic_shapes={"x": {0: IntSpec.unbacked("batch")}},
        )
        for n in [4, 8, 16, 32]:
            fn(torch.randn(n, 3))

        self.assertEqual(len(backend.graphs), 1)
        shape = _tensor_placeholder_shape(backend.graphs[0])
        self.assertIsInstance(shape[0], torch.SymInt)
        self.assertGreater(len(free_unbacked_symbols(shape[0])), 0)

    def test_unbacked_raises_dde_on_branching(self):
        """A function that branches on size(0) must raise a data-dependent
        error when that dim is marked UNBACKED."""

        def fn(x):
            if x.size(0) > 5:
                return x + 1
            return x - 1

        torch._dynamo.reset()
        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={"x": {0: IntSpec.unbacked()}},
        )
        with self.assertRaisesRegex(Exception, "data.dependent|GuardOnDataDependent"):
            compiled(torch.randn(10, 3))

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_static_precedence_over_dynamic_true(self):
        """IntSpec.static() must win over compile(dynamic=True)."""
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x + 1,
            backend=cnt,
            dynamic=True,
            dynamic_shapes={"x": {0: IntSpec.static()}},
        )
        fn(torch.randn(4, 3))
        fn(torch.randn(8, 3))
        # static wins: recompiles per distinct shape
        self.assertEqual(cnt.frame_count, 2)

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_backed_precedence_over_dynamic_false(self):
        """IntSpec.backed() must win over compile(dynamic=False).

        With the compile-context integration, the spec selects
        DimDynamic.DYNAMIC directly — the first call is already backed, no
        initial specialization, so a single compile covers all shapes.
        """
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        fn = torch.compile(
            lambda x: x.sum(0),
            backend=cnt,
            dynamic=False,
            dynamic_shapes={"x": {0: IntSpec.backed("batch")}},
        )
        for n in [4, 8, 16, 32, 64]:
            fn(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 1)

    def test_list_form(self):
        torch._dynamo.reset()
        fn = torch.compile(
            lambda x: x + 1,
            backend="eager",
            dynamic_shapes={"x": [IntSpec.backed("batch"), IntSpec.static()]},
        )
        for n in [4, 8, 16]:
            x = torch.randn(n, 3)
            self.assertEqual(fn(x), x + 1)

    def test_none_entry_inherits_context(self):
        """A None entry in a list-form spec should not mark the dim."""
        torch._dynamo.reset()
        fn = torch.compile(
            lambda x: x + 1,
            backend="eager",
            dynamic_shapes={"x": [IntSpec.backed("batch"), None]},
        )
        x = torch.randn(4, 3)
        self.assertEqual(fn(x), x + 1)


class TestModelSpec(TestCase):
    """Construction and dict-like interface of ModelSpec."""

    def test_empty(self):
        ms = ModelSpec()
        self.assertEqual(len(ms), 0)
        self.assertNotIn("x", ms)

    def test_construct_from_dict(self):
        ms = ModelSpec(
            {
                "x": TensorSpec(2).set(0, IntSpec.backed("batch")),
                "n": IntSpec.backed("n"),
            }
        )
        self.assertEqual(len(ms), 2)
        self.assertIn("x", ms)
        self.assertIn("n", ms)

    def test_set_fluent(self):
        ms = ModelSpec()
        result = ms.set("x", IntSpec.static(value=10))
        self.assertIs(result, ms)
        self.assertEqual(ms["x"], IntSpec.static(value=10))

    def test_getitem_setitem(self):
        ms = ModelSpec()
        ms["n"] = IntSpec.backed("n")
        self.assertEqual(ms["n"], IntSpec.backed("n"))

    def test_get_with_default(self):
        ms = ModelSpec()
        self.assertIsNone(ms.get("missing"))
        self.assertEqual(ms.get("missing", "sentinel"), "sentinel")

    def test_iter(self):
        ms = ModelSpec({"a": IntSpec.static(), "b": IntSpec.backed()})
        self.assertEqual(set(ms), {"a", "b"})

    def test_items(self):
        ms = ModelSpec({"a": IntSpec.static()})
        items = list(ms.items())
        self.assertEqual(items, [("a", IntSpec.static())])

    def test_eq(self):
        a = ModelSpec({"x": IntSpec.backed("x")})
        b = ModelSpec({"x": IntSpec.backed("x")})
        self.assertEqual(a, b)

    def test_neq(self):
        self.assertNotEqual(
            ModelSpec({"x": IntSpec.backed("x")}),
            ModelSpec({"y": IntSpec.backed("y")}),
        )


class TestContextVar(TestCase):
    """The _active_dynamic_shapes ContextVar and its helpers."""

    def test_default_is_none(self):
        self.assertIsNone(_active_dynamic_shapes.get())

    def test_helpers_return_none_when_unset(self):
        self.assertIsNone(get_active_spec_for_arg("x"))
        self.assertIsNone(get_active_spec_for_dim("x", 0))

    def test_set_and_reset(self):
        spec = {"x": IntSpec.backed("batch")}
        token = _active_dynamic_shapes.set(spec)
        try:
            self.assertEqual(get_active_spec_for_arg("x"), IntSpec.backed("batch"))
            self.assertEqual(get_active_spec_for_arg("missing"), None)
        finally:
            _active_dynamic_shapes.reset(token)
        self.assertIsNone(_active_dynamic_shapes.get())

    def test_resolve_dim_from_dict(self):
        token = _active_dynamic_shapes.set({"x": {0: IntSpec.backed("batch")}})
        try:
            self.assertEqual(
                get_active_spec_for_dim("x", 0), IntSpec.backed("batch")
            )
            self.assertIsNone(get_active_spec_for_dim("x", 1))
        finally:
            _active_dynamic_shapes.reset(token)

    def test_resolve_dim_from_list(self):
        token = _active_dynamic_shapes.set(
            {"x": [IntSpec.backed("batch"), None]}
        )
        try:
            self.assertEqual(
                get_active_spec_for_dim("x", 0), IntSpec.backed("batch")
            )
            self.assertIsNone(get_active_spec_for_dim("x", 1))
            self.assertIsNone(get_active_spec_for_dim("x", 5))  # OOB
        finally:
            _active_dynamic_shapes.reset(token)

    def test_resolve_dim_from_tensorspec(self):
        ts = TensorSpec(2).set(0, IntSpec.backed("batch"))
        token = _active_dynamic_shapes.set({"x": ts})
        try:
            self.assertEqual(
                get_active_spec_for_dim("x", 0), IntSpec.backed("batch")
            )
            self.assertIsNone(get_active_spec_for_dim("x", 1))
        finally:
            _active_dynamic_shapes.reset(token)

    def test_resolve_dim_from_intspec_scalar(self):
        # For a scalar arg, the per-arg spec is an IntSpec directly.
        # get_active_spec_for_dim returns it regardless of dim index.
        token = _active_dynamic_shapes.set({"n": IntSpec.backed("n")})
        try:
            self.assertEqual(
                get_active_spec_for_dim("n", 0), IntSpec.backed("n")
            )
        finally:
            _active_dynamic_shapes.reset(token)

    def test_modelspec_as_active(self):
        ms = ModelSpec({"x": IntSpec.backed("batch")})
        token = _active_dynamic_shapes.set(ms)
        try:
            self.assertEqual(get_active_spec_for_arg("x"), IntSpec.backed("batch"))
        finally:
            _active_dynamic_shapes.reset(token)


class TestScalarIntCompile(TestCase):
    """torch.compile(dynamic_shapes=...) with scalar-int arguments."""

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_scalar_backed_no_recompile(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()

        def fn(x, n):
            return x + n

        compiled = torch.compile(
            fn,
            backend=cnt,
            dynamic_shapes={"n": IntSpec.backed("n")},
        )
        for n in [4, 8, 16, 32]:
            compiled(torch.randn(3), n)
        # BACKED scalar: single compile, dynamic symbol.
        self.assertEqual(cnt.frame_count, 1)

    def test_scalar_unbacked_dde_on_branching(self):
        def fn(x, n):
            if n > 5:
                return x + 1
            return x - 1

        torch._dynamo.reset()
        compiled = torch.compile(
            fn,
            backend="eager",
            fullgraph=True,
            dynamic_shapes={"n": IntSpec.unbacked("n")},
        )
        with self.assertRaisesRegex(Exception, "data.dependent|GuardOnDataDependent"):
            compiled(torch.randn(3), 10)

    def test_modelspec_mixed_tensor_and_scalar(self):
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()

        def fn(x, n):
            return x.sum(0) + n

        compiled = torch.compile(
            fn,
            backend=cnt,
            dynamic_shapes=ModelSpec(
                {
                    "x": TensorSpec(2).set(0, IntSpec.backed("batch")),
                    "n": IntSpec.backed("n"),
                }
            ),
        )
        for (shape0, n) in [(4, 8), (8, 16), (16, 32)]:
            compiled(torch.randn(shape0, 3), n)
        # Both dims are dynamic → single compile.
        self.assertEqual(cnt.frame_count, 1)


class TestTopLevelOnly(TestCase):
    """Spec keys only match top-level arguments; nested sources are ignored."""

    def test_nested_dict_arg_is_not_spec_target(self):
        """If the fn takes a dict ``d`` and the spec names ``d``, the nested
        tensor inside ``d`` must NOT receive the spec meant for the dict."""
        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()

        def fn(d):
            return d["x"].sum(0)

        compiled = torch.compile(
            fn,
            backend=cnt,
            # A BACKED IntSpec keyed by "d" — only makes sense at the dict
            # level; must not apply to d["x"].
            dynamic_shapes={"d": IntSpec.backed("d")},
        )
        # Varying d["x"]'s dim 0 would normally trigger recompiles (static
        # default). If the spec had wrongly been applied to the tensor,
        # we'd see a single compile.
        compiled({"x": torch.randn(4, 3)})
        compiled({"x": torch.randn(8, 3)})
        # at least two frames -> the spec did NOT silently apply to the tensor
        self.assertGreaterEqual(cnt.frame_count, 2)


class TestApplyDynamicShapes(TestCase):
    """Entry-point validation in _apply_dynamic_shapes."""

    def test_rejects_non_dict_non_modelspec(self):
        from torch._dynamo.dynamic_spec import _apply_dynamic_shapes

        def fn(x):
            return x + 1

        compiled = torch.compile(fn, backend="eager")
        with self.assertRaisesRegex(TypeError, "dict or ModelSpec"):
            _apply_dynamic_shapes(compiled, fn, [IntSpec.backed("x")])  # type: ignore[arg-type]


class TestKnownGaps(TestCase):
    """Tests demonstrating known unfixed bugs / feature gaps in the current
    integration. Each test here is expected to fail until the corresponding
    fix lands; see torch/_dynamo/DYNAMIC_SHAPES_INTEGRATION.md."""

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_nn_module_spec_applies_to_forward_args(self):
        """Bug: OptimizedModule may not dispatch through the wrapper that
        sets the ContextVar, so the spec never reaches the tracing path.
        Expected: single compile across varying shapes (backed). Bug
        symptom: per-shape recompiles because the ContextVar was None
        during tracing."""

        class M(torch.nn.Module):
            def forward(self, x):
                return x.sum(0)

        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(
            M(),
            backend=cnt,
            dynamic_shapes={"x": {0: IntSpec.backed("batch")}},
        )
        for n in [4, 8, 16, 32]:
            compiled(torch.randn(n, 3))
        self.assertEqual(cnt.frame_count, 1)

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_backed_respects_max_bound(self):
        """Bug: IntSpec.backed(min=N, max=M) bounds are silently dropped.
        Expected: a value outside the declared max should either raise or
        at least recompile. Symptom of the bug: the out-of-range value is
        accepted without any complaint."""

        def fn(x):
            return x.sum(0)

        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(
            fn,
            backend=cnt,
            dynamic_shapes={"x": {0: IntSpec.backed("batch", min=1, max=8)}},
        )
        compiled(torch.randn(4, 3))
        # A size above max should produce a constraint violation, guard
        # failure, or forced recompile. None of these will happen today
        # because min/max are dropped by the hook.
        with self.assertRaises(Exception):
            compiled(torch.randn(100, 3))

    @skipIfTorchDynamo("frame_count unreliable when dynamo traces the test")
    def test_static_shapes_shortcircuit_does_not_override_spec(self):
        """Bug: tensor_always_has_static_shape (e.g. nn.Parameter,
        specialized-nn-module sources) short-circuits _automatic_dynamic
        before the spec hook runs, silently ignoring the spec.

        Reproduces by compiling an nn.Module that stores an nn.Parameter
        and takes it via forward. With BACKED spec on the parameter,
        shape changes should be absorbed into one graph."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(4, 3))

            def forward(self, x):
                return x + self.w.sum(0)

        torch._dynamo.reset()
        cnt = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(
            M(),
            backend=cnt,
            dynamic_shapes={"x": {0: IntSpec.backed("batch")}},
        )
        for n in [4, 8, 16]:
            compiled(torch.randn(n, 3))
        # Spec is on `x` (not the parameter), so this should work even
        # today — but keeping a placeholder test that parameters don't
        # interfere via the static-shape shortcut.
        self.assertEqual(cnt.frame_count, 1)


if __name__ == "__main__":
    run_tests()
