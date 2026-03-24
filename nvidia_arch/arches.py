# Copyright (c) 2026 Ratha SIV | Apache-2.0 License

"""
nvidia_arch.arches
------------------
Definitions and filters for NVIDIA GPU architecture codes (SM/Compute).
Supports all major CUDA/SM combinations for both consumer and Jetson GPUs.
"""

# All arches from 30 (CUDA 11.0+)
ALL_ARCHS = [
    "30", "32", "35", "37", "50", "52", "53",
    "60", "61", "62",
    "70", "72", "75",
    "80", "86", "87", "88", "89",
    "90",
    "100", "101", "103",
    "110",
    "120", "121",
]
"""list of str: All supported architecture codes."""

# All arches for Consumer/Workstation GPUs (CUDA 11.0+)
ALL_ARCHS_CONS = ["30", "35", "50", "52", "60", "61", "70", "75", "86", "89", "120", "121"]
"""list of str: Consumer/Workstation GPU architectures."""

# All arches for Jetson (CUDA 11.0+)
ALL_ARCHS_JETS = ["32", "53", "62", "72", "87", "110"]
"""list of str: Jetson/embedded GPU architectures."""

# All arches for Consumer/Workstation GPUs + Jetson
ALL_ARCHS_CONS_JETS = sorted(set(ALL_ARCHS_CONS + ALL_ARCHS_JETS), key=lambda x: int(x))
"""list of str: Union of Consumer/Workstation and Jetson architectures."""

# All arches for Datacenter GPUs only
ALL_ARCHS_DC = sorted(set(ALL_ARCHS) - set(ALL_ARCHS_JETS) - {"50", "88", "121"}, key=lambda x: int(x))
"""list of str: Datacenter GPU architectures."""

# Arch filters for different GPU types
TYPE_FILTERS = {
    "all": ALL_ARCHS,
    "cons": ALL_ARCHS_CONS,
    "jets": ALL_ARCHS_JETS,
    "cons+jets": ALL_ARCHS_CONS_JETS,
}
"""dict: Mapping of GPU type to valid architectures."""

# Arch filter ranges for different CUDA versions, mapping CUDA(START, END) to ARCH(MIN, MAX)
CUDA_FILTERS_RANGES = {
    (11.0, 11.0): (30, 80),
    (11.1, 11.3): (35, 86),
    (11.4, 11.7): (35, 87),
    (11.8, 11.8): (35, 90),
    (12.0, 12.6): (50, 90),
    (12.8, 12.8): (50, 120),
    (12.9, 12.9): (50, 121),
    (13.0, 13.2): (75, 121),
}
"""dict: Mapping CUDA version ranges to supported SM ranges."""

# Arches to exclude for each CUDA version (those that are valid but not supported by that CUDA version)
CUDA_EXCLUDES = {
    (11.8, 12.9): [88],   # mysterious, only in cuda 13.0 or later
    (13.0, 13.2): [101],  # only in cuda 12.8 and 12.9
    (12.8, 12.8): [103],  # only in cuda 12.9 or later
    (12.8, 12.9): [110],  # only in cuda 13.0 or later
}
"""dict: Mapping CUDA version ranges to lists of valid but unsupported SM codes."""

# CUDA filters for each CUDA version
CUDA_FILTERS = {}
for (vmin, vmax), (amin, amax) in CUDA_FILTERS_RANGES.items():
    v = vmin
    while v <= vmax + 1e-9:
        major = int(v)
        minor = int(round((v - major) * 10))
        key = f"{major}.{minor}"

        # Find excludes for this CUDA version
        excludes = set()
        for (ex_min, ex_max), ex_arches in CUDA_EXCLUDES.items():
            if ex_min <= v <= ex_max + 1e-9:
                excludes.update(str(a) for a in ex_arches)

        CUDA_FILTERS[key] = [
            arch for arch in ALL_ARCHS
            if amin <= int(arch) <= amax and arch not in excludes
        ]
        v = round(v + 0.1, 1)
"""dict: Mapping CUDA version strings ('major.minor') to lists of supported SM codes."""
