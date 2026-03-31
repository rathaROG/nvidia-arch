# Copyright (c) 2026 Ratha SIV | Apache-2.0 License

"""
nvidia_arch
-----------
A lightweight tool for detecting and querying NVIDIA GPU architectures (SM/CC), 
and generating `-gencode` flags for CUDA builds
"""

__version__ = '7.1.0'

from .arches import (
    ALL_ARCHS,
    ALL_ARCHS_CONS,
    ALL_ARCHS_JETS,
    ALL_ARCHS_DCEN,
    ALL_ARCHS_CONS_JETS,
    TYPE_FILTERS,
    CUDA_FILTERS_RANGES,
    CUDA_FILTERS,
    CUDA_EXCLUDES,
)

from .core import (
    # new API names:
    find_gpus,
    get_arches,
    get_compute_caps,
    validate_arch_string,
    normalize_arch_string,
    normalize_arches,
    # Other utilities (unchanged):
    detect_ctk,
    normalize_cuda_ver,
    nvcc_list_arches,
    make_gencode_flags,
    print_summary,
    # Aliases: Deprecated, to be removed in 10.0.0
    find_gpu,
    get_architectures,
    get_compute_cap,
    validate_cc_string,
)

__all__: list = [
    # Constants
    "ALL_ARCHS",
    "ALL_ARCHS_CONS",
    "ALL_ARCHS_JETS",
    "ALL_ARCHS_DCEN",
    "ALL_ARCHS_CONS_JETS",
    "TYPE_FILTERS",
    "CUDA_FILTERS_RANGES",
    "CUDA_FILTERS",
    "CUDA_EXCLUDES",
    # Main functions
    "find_gpus",              # New API name in v6
    "get_arches",             # New API name in v6
    "get_compute_caps",       # New API name in v6
    "validate_arch_string",   # New API name in v6
    "normalize_arch_string",  # New API name in v6
    "normalize_arches",       # New API name in v7
    "detect_ctk",
    "normalize_cuda_ver",
    "nvcc_list_arches",
    "make_gencode_flags",
    "print_summary",
    "find_gpu",               # Alias: Deprecated, to be removed in 10.0.0
    "get_architectures",      # Alias: Deprecated, to be removed in 10.0.0
    "get_compute_cap",        # Alias: Deprecated, to be removed in 10.0.0
    "validate_cc_string",     # Alias: Deprecated, to be removed in 10.0.0
]
