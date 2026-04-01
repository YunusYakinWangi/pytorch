# Owner(s): ["module: dynamo"]
"""Tests for mp_subscript_impl: unified __getitem__ dispatch via vt_getitem in Dynamo."""

import collections
import operator
import types

import torch
import torch._dynamo.test_case


class GetItemTests(torch._dynamo.test_case.TestCase):
    def _compile(self, fn, *args):
        return torch.compile(fn, backend="eager", fullgraph=True)(*args)

    # --- Infra (base VT falls through to unimplemented) ---

    def test_object_without_getitem(self):
        class NoGetItem:
            pass

        def fn(x):
            obj = NoGetItem()
            return operator.getitem(obj, 0)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- BaseListVariable (ListVariable) ---

    def test_list_int_index(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_slice(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, slice(0, 2))

        x = torch.randn(4)
        ref = fn(x)
        res = self._compile(fn, x)
        for r, e in zip(res, ref):
            self.assertEqual(r, e)

    def test_list_negative_index(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_invalid_index_type(self):
        def fn(x):
            items = [x, x + 1, x + 2]
            return operator.getitem(items, "a")

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            self._compile(fn, x)

    # --- BaseListVariable (TupleVariable) ---

    def test_tuple_int_index(self):
        def fn(x):
            items = (x, x + 1, x + 2)
            return operator.getitem(items, 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- RangeVariable ---

    def test_range_int_index(self):
        def fn(x):
            r = range(0, 10, 2)
            return x + operator.getitem(r, 3)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_slice(self):
        def fn(x):
            r = range(0, 10, 2)
            result = operator.getitem(r, slice(1, 3))
            return x + result[0]

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- SizeVariable ---

    def test_size_int_index(self):
        def fn(x):
            s = x.size()
            return x + operator.getitem(s, 0)

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- ConstDictVariable ---

    def test_dict_str_key(self):
        def fn(x):
            d = {"a": x, "b": x + 1}
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_dict_int_key(self):
        def fn(x):
            d = {0: x, 1: x + 1}
            return operator.getitem(d, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- DefaultDictVariable ---

    def test_defaultdict_existing_key(self):
        def fn(x):
            d = collections.defaultdict(lambda: x + 99)
            d["a"] = x
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- MappingProxyVariable ---

    def test_mappingproxy_getitem(self):
        def fn(x):
            d = {"a": 1, "b": 2}
            proxy = types.MappingProxyType(d)
            return x + operator.getitem(proxy, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
