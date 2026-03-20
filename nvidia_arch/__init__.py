# Copyright (c) 2026 Ratha SIV | Apache-2.0 License

__version__ = '0.0.2'

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

__all__ = [
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
