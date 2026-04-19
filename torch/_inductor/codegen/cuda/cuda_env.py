import functools
import logging
import os
import re
import shutil
import subprocess

import torch
from torch._inductor.utils import clear_on_fresh_cache

from ... import config


log = logging.getLogger(__name__)


def _get_cuda_version_from_nvcc() -> str | None:
    nvcc = config.cuda.cuda_cxx
    if nvcc is None or not nvcc_exist(nvcc):
        nvcc = os.getenv("CUDACXX")
    if nvcc is None or not nvcc_exist(nvcc):
        nvcc = os.path.join(os.getenv("CUDA_HOME", ""), "bin/nvcc")
    if nvcc is None or not nvcc_exist(nvcc):
        nvcc = shutil.which("nvcc")
    if nvcc is None:
        return None

    try:
        output = subprocess.check_output(
            [nvcc, "--version"], stderr=subprocess.STDOUT, text=True
        )
    except Exception:
        log.debug("Error getting cuda version from nvcc", exc_info=True)
        return None

    match = re.search(r"release .+ V(.*)", output)
    return match.group(1) if match else None


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_arch() -> str | None:
    try:
        cuda_arch = config.cuda.arch
        if cuda_arch is None:
            # Get Compute Capability of the first Visible device
            major, minor = torch.cuda.get_device_capability(0)
            return str(major * 10 + minor)
        return str(cuda_arch)
    except Exception:
        log.exception("Error getting cuda arch")
        return None


@clear_on_fresh_cache
@functools.lru_cache(1)
def is_datacenter_blackwell_arch() -> bool:
    arch = get_cuda_arch()
    if arch is None:
        return False
    arch_number = int(arch)
    return arch_number >= 100 and arch_number < 110


@clear_on_fresh_cache
@functools.lru_cache(1)
def get_cuda_version() -> str | None:
    try:
        cuda_version = config.cuda.version
        if cuda_version is None:
            cuda_version = torch.version.cuda
        if cuda_version is None:
            cuda_version = _get_cuda_version_from_nvcc()
        return cuda_version
    except Exception:
        log.exception("Error getting cuda version")
        return None


@functools.cache
def nvcc_exist(nvcc_path: str | None = "nvcc") -> bool:
    return nvcc_path is not None and shutil.which(nvcc_path) is not None
