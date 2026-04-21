# Owner(s): ["module: dynamo"]
"""Tests for tp_str / generic_str: str() via PyObject_Str in Dynamo."""

import torch
import torch.nn
from torch.testing._internal.common_utils import make_dynamo_test, run_tests, TestCase


class _CustomStrObject:
    def __str__(self):
        return "custom_str"


class _DefaultStrObject:
    pass


class _CustomReprOnly:
    def __repr__(self):
        return "custom_repr"


class _BadReprObject:
    def __repr__(self):
        raise AttributeError("bad repr")


class _CustomStrWithArgs:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __str__(self):
        return f"{self.name}={self.value}"


class _BadStrObject:
    def __str__(self):
        return 1  # noqa: PLE0307


class _CustomStrException(ValueError):
    def __str__(self):
        return "custom error message"


class _InheritedStrException(ValueError):
    pass


class _ClassForStrTest:
    pass


class _FalseStrMeta(type):
    def __str__(cls):
        return "meta_str"


class _MetaStrClass(metaclass=_FalseStrMeta):
    pass


class _MetaStrAndClassDunderStr(metaclass=_FalseStrMeta):
    def __str__(self):
        return "instance_str"


class _OpaqueStrDescriptorObject:
    __str__ = str.upper


class TpStrTests(TestCase):
    @make_dynamo_test
    def test_str_int(self):
        assert str(42) == "42"  # noqa: S101
        assert str(-1) == "-1"  # noqa: S101
        assert str(0) == "0"  # noqa: S101

    @make_dynamo_test
    def test_str_float(self):
        assert str(3.14) == "3.14"  # noqa: S101
        assert str(0.0) == "0.0"  # noqa: S101
        assert str(-2.5) == "-2.5"  # noqa: S101

    @make_dynamo_test
    def test_str_bool(self):
        assert str(True) == "True"  # noqa: S101
        assert str(False) == "False"  # noqa: S101

    @make_dynamo_test
    def test_str_none(self):
        assert str(None) == "None"  # noqa: S101

    @make_dynamo_test
    def test_str_string_identity(self):
        s = "hello"
        empty = ""
        assert str(s) == "hello"  # noqa: S101
        assert str(empty) == ""  # noqa: S101

    @make_dynamo_test
    def test_str_dunder_constant(self):
        assert (42).__str__() == "42"  # noqa: S101
        assert (3.14).__str__() == "3.14"  # noqa: S101
        assert True.__str__() == "True"  # noqa: S101

    @make_dynamo_test
    def test_str_list(self):
        assert str([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101
        assert str([]) == "[]"  # noqa: S101

    @make_dynamo_test
    def test_str_tuple(self):
        assert str((1, 2, 3)) == "(1, 2, 3)"  # noqa: S101
        assert str(()) == "()"  # noqa: S101

    @make_dynamo_test
    def test_str_dict(self):
        assert str({"a": 1}) == "{'a': 1}"  # noqa: S101
        assert str({}) == "{}"  # noqa: S101

    @make_dynamo_test
    def test_str_dict_keys_view(self):
        result = str({"a": 1}.keys())
        assert "dict_keys" in result  # noqa: S101
        assert "'a'" in result  # noqa: S101

    @make_dynamo_test
    def test_str_set(self):
        assert str({42}) == "{42}"  # noqa: S101

    @make_dynamo_test
    def test_str_range(self):
        assert str(range(5)) == "range(0, 5)"  # noqa: S101
        assert str(range(1, 10, 2)) == "range(1, 10, 2)"  # noqa: S101

    @make_dynamo_test
    def test_str_exception_no_args(self):
        assert str(ValueError()) == ""  # noqa: S101

    @make_dynamo_test
    def test_str_exception_one_arg(self):
        assert str(ValueError("oops")) == "oops"  # noqa: S101

    @make_dynamo_test
    def test_str_exception_one_int_arg(self):
        assert str(ValueError(42)) == "42"  # noqa: S101

    @make_dynamo_test
    def test_str_exception_multiple_args(self):
        assert str(ValueError("error", 42)) == "('error', 42)"  # noqa: S101

    @make_dynamo_test
    def test_str_exception_dunder(self):
        assert TypeError("bad type").__str__() == "bad type"  # noqa: S101

    @make_dynamo_test
    def test_str_exception_unbound_dunder(self):
        assert ValueError.__str__(ValueError("oops")) == "oops"  # noqa: S101

    @make_dynamo_test
    def test_str_list_unbound_dunder(self):
        assert list.__str__([1, 2, 3]) == "[1, 2, 3]"  # noqa: S101

    @make_dynamo_test
    def test_str_type_unbound_dunder(self):
        result = type.__str__(_ClassForStrTest)
        assert "_ClassForStrTest" in result  # noqa: S101
        assert result.startswith("<class")  # noqa: S101

    @make_dynamo_test
    def test_str_runtime_error(self):
        assert str(RuntimeError("runtime failure")) == "runtime failure"  # noqa: S101

    def test_user_defined_custom_str(self):
        def fn(x, obj):
            s = str(obj)
            if s == "custom_str":
                return x + 1
            return x

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, _CustomStrObject()), compiled(x, _CustomStrObject()))

    def test_user_defined_custom_str_dunder(self):
        def fn(x, obj):
            s = obj.__str__()
            if s == "custom_str":
                return x + 1
            return x

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, _CustomStrObject()), compiled(x, _CustomStrObject()))

    def test_user_defined_with_args(self):
        def fn(x, obj):
            s = str(obj)
            if s == "x=10":
                return x + 1
            return x

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            fn(x, _CustomStrWithArgs("x", 10)),
            compiled(x, _CustomStrWithArgs("x", 10)),
        )

    def test_user_defined_default_str(self):
        def fn(x, obj):
            s = str(obj)
            if "_DefaultStrObject" in s:
                return x + 1
            return x

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, _DefaultStrObject()), compiled(x, _DefaultStrObject()))

    def test_user_defined_repr_fallback(self):
        def fn(x, obj):
            s = str(obj)
            if s == "custom_repr":
                return x + 1
            return x

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, _CustomReprOnly()), compiled(x, _CustomReprOnly()))

    def test_user_defined_default_str_attribute_error(self):
        def fn(x, obj):
            try:
                str(obj)
            except AttributeError as e:
                return str(e)
            return "no error"

        x = torch.randn(4)
        self.assertEqual(fn(x, _BadReprObject()), "bad repr")

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled(x, _BadReprObject()), "bad repr")

    def test_user_defined_bad_str_return_type(self):
        def fn(x, obj):
            try:
                str(obj)
            except TypeError as e:
                return str(e)
            return "no error"

        x = torch.randn(4)
        self.assertEqual(
            fn(x, _BadStrObject()),
            "__str__ returned non-string (type int)",
        )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(
            compiled(x, _BadStrObject()),
            "__str__ returned non-string (type int)",
        )

    def test_user_defined_opaque_str_descriptor_raises_type_error(self):
        def fn(x, obj):
            try:
                return str(obj)
            except TypeError as e:
                return str(e)

        x = torch.randn(4)
        eager_result = fn(x, _OpaqueStrDescriptorObject())
        self.assertIn(
            "descriptor 'upper' for 'str' objects doesn't apply", eager_result
        )

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        compiled_result = compiled(x, _OpaqueStrDescriptorObject())
        self.assertEqual(eager_result, compiled_result)

    def test_exception_subclass_custom_str(self):
        def fn(x):
            exc = _CustomStrException("ignored")
            s = str(exc)
            if s == "custom error message":
                return x + 1
            return x

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_exception_subclass_inherited_str(self):
        def fn(x):
            exc = _InheritedStrException("inherited message")
            s = str(exc)
            if s == "inherited message":
                return x + 1
            return x

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    @make_dynamo_test
    def test_str_class_default(self):
        result = str(_ClassForStrTest)
        assert "_ClassForStrTest" in result  # noqa: S101
        assert result.startswith("<class")  # noqa: S101

    @make_dynamo_test
    def test_str_class_with_metaclass_str(self):
        assert str(_MetaStrClass) == "meta_str"  # noqa: S101

    def test_metaclass_str_fullgraph(self):
        def fn(x):
            s = str(_MetaStrClass)
            if s == "meta_str":
                return x + 1
            return x

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_metaclass_str_wins_over_class_dunder_str_fullgraph(self):
        def fn(x):
            s = str(_MetaStrAndClassDunderStr)
            if s == "meta_str":
                return x + 1
            return x

        x = torch.randn(4)
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compiled(x))

    def test_str_nn_linear(self):
        mod = torch.nn.Linear(4, 4)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            s = str(mod)
            if "Linear" in s:
                return x + 1
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_str_nn_module_list_nonempty(self):
        mod = torch.nn.ModuleList([torch.nn.Linear(4, 4)])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            s = str(mod)
            if "Linear" in s:
                return x + 1
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_str_nn_module_list_empty(self):
        mod = torch.nn.ModuleList()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            s = str(mod)
            if "ModuleList" in s:
                return x + 1
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    @make_dynamo_test
    def test_str_function(self):
        def my_func():
            pass

        result = str(my_func)
        assert "my_func" in result  # noqa: S101

    @make_dynamo_test
    def test_str_lambda(self):
        f = lambda: None  # noqa: E731

        result = str(f)
        assert "<lambda>" in result  # noqa: S101

    def test_str_list_with_tensor_raises_unsupported(self):
        def fn(x):
            return str([x])

        x = torch.tensor(1)
        self.assertIn("tensor(", fn(x))

        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            compiled(x)


if __name__ == "__main__":
    run_tests()
