# Copyright (c) 2026 Ratha SIV | Apache-2.0 License

"""
nvidia_arch
-----------
A lightweight tool for detecting and querying NVIDIA GPU architectures (SM/CC), 
and generating `-gencode` flags for CUDA builds
"""

__version__ = '0.0.3'

from .arches import (
    ALL_ARCHS,
    ALL_ARCHS_CONS,
    ALL_ARCHS_JETS,
    ALL_ARCHS_CONS_JETS,
    TYPE_FILTERS,
    CUDA_FILTERS_RANGES,
    CUDA_FILTERS,
)

from .core import (
    detect_ctk,
    normalize_cuda_ver,
    nvcc_list_arches,
    get_architectures,
    make_gencode_flags,
    print_summary,
)

__all__: list = [
    "ALL_ARCHS",
    "ALL_ARCHS_CONS",
    "ALL_ARCHS_JETS",
    "ALL_ARCHS_CONS_JETS",
    "TYPE_FILTERS",
    "CUDA_FILTERS_RANGES",
    "CUDA_FILTERS",
    "detect_ctk",
    "normalize_cuda_ver",
    "nvcc_list_arches",
    "get_architectures",
    "make_gencode_flags",
    "print_summary",
]
