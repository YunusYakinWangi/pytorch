# Owner(s): ["module: dynamo"]
import unittest
import weakref

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch._dynamo.testing
import torch._logging
from torch._dynamo.exc import FailOnRecompileLimitHit
from torch.testing._internal.logging_utils import kwargs_to_settings, log_settings


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)


class RecompileUxTests(torch._dynamo.test_case.TestCase):
    # TODO(whc) dynamo actually recompiles one more time than the cache limit
    cache_limit = 1

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._exit_stack.enter_context(
            torch._dynamo.config.patch("recompile_limit", cls.cache_limit)
        )

    def test_drop_cache_on_skip(self):
        def model(x, i):
            return x + i

        attached = False
        triggered = False

        def trigger():
            nonlocal triggered
            triggered = True

        def compiler(gm, input):
            nonlocal attached
            f = gm.forward
            if attached:
                raise AssertionError("Expected not attached")
            # NB: making this a weakref.ref causes the cycle to no
            # longer be promptly GC'ed
            weakref.finalize(f, trigger)
            attached = True
            return f

        x = torch.randn(2)
        for i in range(2):
            opt_model = torch.compile(model, backend=compiler)
            opt_model(x, i)

        self.assertTrue(triggered)

    def test_loop_torture(self):
        def loop_torture(input, iters):
            out = input
            # randint itself causes one graph break
            for _ in range(iters):
                out += input
            return out

        compile_counter = torch._dynamo.testing.CompileCounter()
        for _ in range(10):
            x = torch.randn(3)
            iters = torch.randint(low=0, high=1000, size=())
            opt_loop_torture = torch.compile(loop_torture, backend=compile_counter)
            opt_loop_torture(x, iters)

        # Currently, we recompile each time,
        # We'd probably like to bail out quickly and warn
        # TODO(whc) these checks fail on py37.  Why?
        # self.assertEqual(counters["frames"]["total"], 2 + self.cache_limit)
        # self.assertEqual(counters["frames"]["ok"], 1 + self.cache_limit)

        # compile_counter only sees frames that were fed to the backend compiler,
        # which is a subset of counters["frames"]["ok"] -- probably because
        # counters["frames"]["ok"] includes frames not containing torch ops?
        self.assertEqual(compile_counter.frame_count, self.cache_limit)

    @torch._dynamo.config.patch("automatic_dynamic_shapes", False)
    def test_dynamic_input(self):
        def model(input):
            return input + input

        expected_recompiles = 2
        compile_counter = torch._dynamo.testing.CompileCounter()
        with torch._dynamo.config.patch("recompile_limit", expected_recompiles):
            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                for _ in range(10):
                    bsz = torch.randint(low=0, high=1000, size=())
                    x = torch.randn((bsz, 3, 4))
                    opt_model = torch.compile(model, backend=compile_counter)
                    opt_model(x)

        self.assertEqual(compile_counter.frame_count, expected_recompiles)
        self.assertEqual(len(logs.records), 1)
        print(logs.records[0])
        self.assertTrue(
            logs.records[0]
            .getMessage()
            .startswith("torch._dynamo hit config.recompile_limit")
        )

    @unittest.skipIf(
        not torch.cuda.is_available() and not torch.xpu.is_available(),
        "requires cuda or xpu",
    )
    def test_nvfuser_guards(self):
        # we may want to model dynamo's guards sufficiently after nvfuser's ProfilingExecutor guards
        # such that we ensure dynamo is in charge of all the recompilations at the top level,
        # and we could thus simplify the underlying torchscript executor
        def func(a, b, c):
            return a + b * c

        a = torch.rand(3, 4, 5, device=device_type)
        b = torch.rand(3, 4, 5, device=device_type)
        b_v = torch.rand(3, 5, 4, device=device_type).view(3, 4, 5)
        b_p = torch.rand(3, 5, 4, device=device_type).permute(0, 2, 1)
        c = torch.rand(3, 4, 5, device=device_type)
        compile_counter = torch._dynamo.testing.CompileCounter()

        with torch._dynamo.config.patch("recompile_limit", 2):
            opt_func = torch.compile(func, backend=compile_counter)
            opt_func(a, b, c)  # warmup
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b, c)  # no guard fail or recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_v, c)  # a view should not cause nvfuser recompile
            self.assertEqual(compile_counter.frame_count, 1)

            opt_func(a, b_p, c)  # a permutation should cause recompile
            self.assertEqual(compile_counter.frame_count, 2)

    def assert_single_log_contains(self, logs, contains_str):
        self.assertEqual(len(logs.records), 1)
        self.assertTrue(
            logs.records[0].getMessage().find(contains_str) > 0,
            msg=f'Expected to find "{contains_str}" in log "{logs.records[0].getMessage()}"',
        )

    def test_verbose_tensor_check(self):
        def func(a):
            # Warning: choose a function here whose meta implementation lives
            # entirely in C++.  If you do a Python one, Dynamo will dive into
            # torch._refs which is OK but it will muddy up the warnings
            return torch.add(a, 4)

        def cache_fail_test(cached_input, missed_input, expected_failure):
            # TODO(whc) maybe its hacky to have a 'test within a test' but this seemed convenient
            torch._dynamo.reset()
            torch._dynamo.utils.counters.clear()
            opt_func = torch.compile(func, backend="eager")
            # warmup
            opt_func(cached_input)

            with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
                opt_func = torch.compile(func, backend="eager")
                opt_func(missed_input)
            self.assert_single_log_contains(logs, expected_failure)

        a = torch.rand(3, 4, 5)
        cache_fail_test(
            a,
            a[0:2, :, :],
            "tensor 'a' size mismatch at index 0. expected 3, actual 2",
        )
        cache_fail_test(
            a,
            a.clone().as_strided((3, 4, 5), stride=(1, 3, 12)),
            "tensor 'a' stride mismatch at index 0. expected 20, actual 1",
        )
        cache_fail_test(a, a[0, :, :], "tensor 'a' rank mismatch. expected 3, actual 2")
        cache_fail_test(a, a.to("meta"), "tensor 'a' dispatch key set mismatch.")
        cache_fail_test(
            a,
            a.to(torch.float16),
            "tensor 'a' dtype mismatch. expected Float, actual Half",
        )
        a_grad = a.clone()
        a_grad.requires_grad = True
        cache_fail_test(
            a,
            a_grad,
            "tensor 'a' requires_grad mismatch. expected requires_grad=0",
        )

    def test_mismatched_type(self):
        a = torch.rand(3, 4, 5)
        b = torch.rand(3, 4, 5)

        def func(a, b):
            return a + b

        opt_func = torch.compile(func, backend="eager")
        # warmup
        opt_func(a, b)

        with self.assertLogs(logger="torch._dynamo", level="WARNING") as logs:
            opt_func = torch.compile(func, backend="eager")
            opt_func(a, 1)
        self.assert_single_log_contains(
            logs,
            "expected type of 'b' to be a tensor type, ' but found <class 'int'>",
        )

    @torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True)
    def test_fail_on_recompile_limit_hit(self):
        @torch.compile(backend="eager")
        def func(b, a):
            if a:
                return b * 2
            else:
                return b + 1

        func(torch.randn(5), True)
        with self.assertRaises(FailOnRecompileLimitHit):
            func(torch.randn(5), False)

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails(self):
        failure_reasons = []

        def guard_fail_fn(failure):
            failure_reasons.append(failure[0])

        def f(x):
            return torch.relu(x)

        opt_f = torch._dynamo.optimize(
            backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
        )(f)

        for i in range(5):
            failure_reasons.clear()
            opt_f(torch.randn(8 + i))

        failure_str = "\n".join(failure_reasons)
        for line in [
            "tensor 'x' size mismatch at index 0. expected 11, actual 12",
            "tensor 'x' size mismatch at index 0. expected 10, actual 12",
            "tensor 'x' size mismatch at index 0. expected 9, actual 12",
            "tensor 'x' size mismatch at index 0. expected 8, actual 12",
        ]:
            self.assertIn(
                line,
                failure_str,
            )

    @torch._dynamo.config.patch("recompile_limit", 32)
    def test_multiple_guard_fails_report_all(self):
        with log_settings(kwargs_to_settings(recompiles_verbose=True)):
            failure_reasons = []

            def guard_fail_fn(failure):
                failure_reasons.append(failure[0])

            def f(x):
                return torch.ones(len(x), x[-1])

            opt_f = torch._dynamo.optimize(
                backend="eager", guard_fail_fn=guard_fail_fn, dynamic=False
            )(f)

            opt_f([4, 5, 6])

            def filter_reasons():
                return "\n".join(
                    [
                        line
                        for line in "\n".join(failure_reasons).splitlines()
                        if not line.startswith("___check_type_id")
                    ]
                )

            failure_reasons.clear()
            opt_f([7, 8])

            for line in ["len(x) == 3"]:
                self.assertIn(line, filter_reasons())

            failure_reasons.clear()
            opt_f([9])

            for line in ["len(x) == 2", "len(x) == 3"]:
                self.assertIn(line, filter_reasons())

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_recompile_child_run_only(self):
        def f(x, n):
            if torch.compiler.is_compiling():
                x = x + 1
            x = g(x)
            return h(x) + n

        def g(x):
            if torch.compiler.is_compiling():
                return x + 2
            return x

        def h(x):
            if torch.compiler.is_compiling():
                return x + 4
            return x

        torch.compile(g, backend="eager")(torch.randn(3))
        inp = torch.randn(3)
        opt_f = torch.compile(f, backend="eager")
        opt_f(inp, 0)

        # expect f to run eager, g compiled (from previous invocatino), h eager
        res = opt_f(inp, 1)

        self.assertEqual(res, inp + 3)


class IsolatedCacheTests(torch._dynamo.test_case.TestCase):
    """Tests for isolated_cache=True on torch.compile(). Each compile region
    gets its own isolated cache via C++ region_id filtering."""

    @staticmethod
    def _num_cache_entries(code):
        return len(torch._dynamo.eval_frame._debug_get_cache_entry_list(code))

    def test_isolated_cache_same_function_different_regions(self):
        """Two torch.compile() calls on the same function with isolated_cache
        get fully independent caches."""
        cnt1 = torch._dynamo.testing.CompileCounter()
        cnt2 = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            return x + y

        opt_f = torch.compile(f, backend=cnt1, isolated_cache=True)
        opt_g = torch.compile(f, backend=cnt2, isolated_cache=True)

        opt_f(torch.randn(3), torch.randn(3))
        self.assertEqual(cnt1.frame_count, 1)

        opt_f(
            torch.randn(3, dtype=torch.float64),
            torch.randn(3, dtype=torch.float64),
        )
        self.assertEqual(cnt1.frame_count, 2)

        # opt_g can compile despite f.__code__ having 2 entries from opt_f
        opt_g(torch.randn(3, dtype=torch.float16), torch.randn(3, dtype=torch.float16))
        self.assertEqual(cnt2.frame_count, 1)

    @torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True)
    def test_isolated_cache_factory_pattern(self):
        """Factory creates multiple torch.compile wrappers around the same
        inner function. Each gets its own isolated cache, so the global
        recompile_limit doesn't block them."""
        from functools import cache

        def core(x):
            return x.sum()

        @cache
        def factory(key):
            @torch.compile(fullgraph=True, dynamic=False, isolated_cache=True)
            def frontend(x, n):
                return core(x) + n

            return frontend

        factory("foo")(torch.ones(3), 3)
        factory("bar")(torch.ones(4), 3)
        factory("baz")(torch.ones(5), 3)

    def test_isolated_cache_static_and_dynamic(self):
        """Two compile regions on the same function: one static, one dynamic.
        Their cache entries don't interfere."""
        cnt_static = torch._dynamo.testing.CompileCounter()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def magic(x, y):
            return x.sum() - y.sum()

        magic_static = torch.compile(
            magic, backend=cnt_static, dynamic=False, isolated_cache=True
        )
        magic_dynamic = torch.compile(
            magic, backend=cnt_dynamic, dynamic=True, isolated_cache=True
        )

        magic_static(torch.randn(128, 32), torch.randn(128, 32))
        self.assertEqual(cnt_static.frame_count, 1)

        magic_dynamic(torch.randn(64, 16), torch.randn(64, 16))
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static: same shape -> cache hit
        magic_static(torch.randn(128, 32), torch.randn(128, 32))
        self.assertEqual(cnt_static.frame_count, 1)

        # Dynamic: different shape -> cache hit (dynamic graph)
        magic_dynamic(torch.randn(32, 8), torch.randn(32, 8))
        self.assertEqual(cnt_dynamic.frame_count, 1)

    @torch._dynamo.config.patch(recompile_limit=1)
    def test_isolated_cache_fullgraph_raises(self):
        """With fullgraph=True, hitting the recompile limit raises
        FailOnRecompileLimitHit."""
        cnt = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_f = torch.compile(f, backend=cnt, fullgraph=True, isolated_cache=True)

        opt_f(torch.randn(3))
        self.assertEqual(cnt.frame_count, 1)

        with self.assertRaisesRegex(
            FailOnRecompileLimitHit,
            "reached with fullgraph=True",
        ):
            opt_f(torch.randn(3, dtype=torch.float64))

    def test_isolated_cache_two_models_independent(self):
        """Two compile regions wrapping the same model. Recompilations in
        one region don't affect the other."""
        cnt1 = torch._dynamo.testing.CompileCounter()
        cnt2 = torch._dynamo.testing.CompileCounter()

        def helper(x):
            return x.sin()

        def model(x):
            return helper(x).cos()

        opt_a = torch.compile(model, backend=cnt1, isolated_cache=True)
        opt_b = torch.compile(model, backend=cnt2, isolated_cache=True)

        opt_a(torch.randn(3))
        frame_count_a = cnt1.frame_count

        opt_b(torch.randn(4))
        frame_count_b = cnt2.frame_count

        # Region A recompiles, doesn't affect region B
        opt_a(torch.randn(3, dtype=torch.float64))
        self.assertGreater(cnt1.frame_count, frame_count_a)

        opt_b(torch.randn(4))
        self.assertEqual(cnt2.frame_count, frame_count_b)

    @torch._dynamo.config.patch(recompile_limit=3, accumulated_recompile_limit=64)
    def test_global_recompile_limit_resume_function(self):
        """Documents existing behavior: global recompile_limit is per-code-object.
        Resume functions independently accumulate entries."""
        cnt = torch._dynamo.testing.CompileCounter()

        mode = {"value": "a"}

        def f(x):
            a = x.sin()
            print("graph break")
            if mode["value"] == "a":
                return a.cos()
            elif mode["value"] == "b":
                return a.tan()
            elif mode["value"] == "c":
                return a.exp()
            else:
                return a + 1

        opt_f = torch.compile(f, backend=cnt)

        opt_f(torch.randn(4, 8))
        frame_count_after_1 = cnt.frame_count

        mode["value"] = "b"
        opt_f(torch.randn(4, 8))
        self.assertGreater(cnt.frame_count, frame_count_after_1)

        mode["value"] = "c"
        opt_f(torch.randn(4, 8))
        frame_count_after_3 = cnt.frame_count

        # Resume hits limit=3, stops
        mode["value"] = "d"
        opt_f(torch.randn(4, 8))
        self.assertEqual(cnt.frame_count, frame_count_after_3)

    def test_isolated_cache_mark_dynamic_vs_static(self):
        """Two regions on the same function: one with mark_static, one with
        mark_dynamic. Their guards don't interfere."""
        cnt_static = torch._dynamo.testing.CompileCounter()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_static = torch.compile(f, backend=cnt_static, isolated_cache=True)
        opt_dynamic = torch.compile(f, backend=cnt_dynamic, isolated_cache=True)

        # Static region: mark dim0 as static
        x_static = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static, 0)
        opt_static(x_static)
        self.assertEqual(cnt_static.frame_count, 1)

        # Dynamic region: mark dim0 as dynamic
        x_dynamic = torch.randn(4, 8)
        torch._dynamo.mark_dynamic(x_dynamic, 0)
        opt_dynamic(x_dynamic)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static region: same shape -> cache hit
        x_static2 = torch.randn(4, 8)
        torch._dynamo.mark_static(x_static2, 0)
        opt_static(x_static2)
        self.assertEqual(cnt_static.frame_count, 1)

        # Dynamic region: different shape -> cache hit (dynamic graph)
        x_dynamic2 = torch.randn(7, 8)
        opt_dynamic(x_dynamic2)
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static region: different shape -> recompile (static guard fails)
        x_static3 = torch.randn(7, 8)
        torch._dynamo.mark_static(x_static3, 0)
        opt_static(x_static3)
        self.assertEqual(cnt_static.frame_count, 2)

    @torch._dynamo.config.patch(automatic_dynamic_shapes=True)
    def test_isolated_cache_auto_dynamic_shared_pgo(self):
        """With isolated_cache, PGO (frame_state) is shared. Region B
        benefits from region A's shape observations — goes directly to
        dynamic compilation instead of starting static."""
        cnt_a = torch._dynamo.testing.CompileCounter()
        cnt_b = torch._dynamo.testing.CompileCounter()

        def f(x):
            return x.sin()

        opt_a = torch.compile(f, backend=cnt_a, isolated_cache=True)
        opt_b = torch.compile(f, backend=cnt_b, isolated_cache=True)

        # Region A: two calls with different shapes -> PGO marks dim0 dynamic
        opt_a(torch.randn(3, 4))
        opt_a(torch.randn(5, 4))
        self.assertEqual(cnt_a.frame_count, 2)

        # Region B: first call. PGO already knows dim0 is dynamic (shared).
        # Should compile once with dynamic dim0.
        opt_b(torch.randn(7, 4))
        self.assertEqual(cnt_b.frame_count, 1)

        # Region B: different dim0 -> cache hit (dynamic graph)
        opt_b(torch.randn(9, 4))
        self.assertEqual(cnt_b.frame_count, 1)

    def test_isolated_cache_explicit_dispatch_static_vs_dynamic(self):
        """With isolated_cache, users can explicitly dispatch to static or
        dynamic compiled regions without relying on automatic dynamic
        transitions or exclusion guards. Each region has its own compilation
        strategy set once at compile time."""
        cnt_static = torch._dynamo.testing.CompileCounter()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()

        def magic(x, y):
            return x.sum() - y.sum()

        # Set strategy once at compile time — no per-call mark_dynamic needed
        magic_static = torch.compile(
            magic, backend=cnt_static, dynamic=False, isolated_cache=True
        )
        magic_dynamic = torch.compile(
            magic, backend=cnt_dynamic, dynamic=True, isolated_cache=True
        )

        # Static region: always uses fixed shapes, no recompilation
        static_shape = (128, 32)
        magic_static(torch.randn(*static_shape), torch.randn(*static_shape))
        magic_static(torch.randn(*static_shape), torch.randn(*static_shape))
        self.assertEqual(cnt_static.frame_count, 1)

        # Dynamic region: varying shapes, single dynamic graph handles all
        magic_dynamic(torch.randn(64, 16), torch.randn(64, 16))
        magic_dynamic(torch.randn(32, 8), torch.randn(32, 8))
        magic_dynamic(torch.randn(128, 64), torch.randn(128, 64))
        self.assertEqual(cnt_dynamic.frame_count, 1)

        # Static region still works — not polluted by dynamic region
        magic_static(torch.randn(*static_shape), torch.randn(*static_shape))
        self.assertEqual(cnt_static.frame_count, 1)

        # Static region with DIFFERENT shape -> recompiles (static = specialized)
        magic_static(torch.randn(64, 16), torch.randn(64, 16))
        self.assertEqual(cnt_static.frame_count, 2)

        # Dynamic region still serves any shape without recompilation
        magic_dynamic(torch.randn(256, 128), torch.randn(256, 128))
        self.assertEqual(cnt_dynamic.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
