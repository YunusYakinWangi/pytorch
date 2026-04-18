"""
Python runtime bridge for extern (non-Triton) CUDA kernels in cpp_wrapper mode.

Called from C++ via the CPython API.  The two entry points mirror the Triton
lazy-compile bridge in triton_lazy_compile.py:

  start_kernel_compile  – eval the kernel source at module-load time
  run_kernel            – invoke the compiled kernel on every launch
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def _get_async_compile():
    from torch._inductor.async_compile import AsyncCompile

    return AsyncCompile()


def start_kernel_compile(
    pending_kernels: dict[str, Any], kernel_name: str, kernel_source_path: str
) -> None:
    """Start compilation of an extern kernel.

    Called from C++ at module initialisation time for each extern kernel.
    Reads the sidecar source file (whose content is an ``async_compile.cutedsl(...)``
    expression), evaluates it to obtain a kernel wrapper or future, and stores
    the result in *pending_kernels* for later retrieval by ``run_kernel``.
    """
    async_compile = _get_async_compile()  # noqa: F841 (used by eval below)

    with open(kernel_source_path) as f:
        kernel_source = f.read()
    kernel_obj = eval(kernel_source.strip())  # noqa: S307

    pending_kernels[kernel_name] = kernel_obj


def run_kernel(
    pending_kernels: dict[str, Any],
    kernel_name: str,
    stream: int,
    args: list[Any],
) -> None:
    """Invoke a compiled extern kernel.

    Called from C++ on every kernel launch.  On the first call the kernel
    object may still be a ``LambdaFuture``; if so it is resolved and the dict
    entry is replaced with the concrete wrapper so that subsequent calls skip
    the future resolution.
    """
    kernel_obj = pending_kernels[kernel_name]

    # Resolve future on first invocation
    if hasattr(kernel_obj, "result"):
        kernel_obj = kernel_obj.result()
        pending_kernels[kernel_name] = kernel_obj

    kernel_obj.run(*args, stream=stream)
