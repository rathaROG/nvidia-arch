# Copyright (c) 2026 Ratha SIV | Apache-2.0 License

# All arches from 35 (CUDA 11.0+)
ALL_ARCHS = [
    "35", "37", "50", "52", "53",
    "60", "61", "62",
    "70", "72", "75",
    "80", "86", "87", "88", "89",
    "90",
    "100", "103",
    "110",
    "120", "121",
]

# All arches for Consumer/Workstation GPUs (CUDA 11.0+)
ALL_ARCHS_CONS = ["35", "50", "52", "60", "61", "70", "75", "86", "89", "120", "121"]

# All arches for Jetson (CUDA 11.0+)
ALL_ARCHS_JETS = ["53", "62", "72", "87", "110"]

# All arches for Consumer/Workstation GPUs + Jetson
ALL_ARCHS_CONS_JETS = sorted(set(ALL_ARCHS_CONS + ALL_ARCHS_JETS))

# Arch filters for different GPU types
TYPE_FILTERS = {
    "all": ALL_ARCHS,
    "cons": ALL_ARCHS_CONS,
    "jets": ALL_ARCHS_JETS,
    "cons+jets": ALL_ARCHS_CONS_JETS,
}

# Arch filter ranges for different CUDA versions, mapping CUDA(START, END) to ARCH(MIN, MAX)
CUDA_FILTERS_RANGES = {
    (11.0, 11.0): (35, 80),
    (11.1, 11.4): (35, 86),
    (11.5, 11.7): (35, 87),
    (11.8, 11.8): (35, 90),
    (12.0, 12.6): (50, 90),
    (12.8, 12.8): (50, 120),
    (12.9, 12.9): (50, 121),
    (13.0, 13.2): (75, 121),
}

# CUDA filters for each CUDA version
CUDA_FILTERS = {}
for (vmin, vmax), (amin, amax) in CUDA_FILTERS_RANGES.items():
    v = vmin
    while v <= vmax + 1e-9:
        major = int(v)
        minor = int(round((v - major) * 10))
        key = f"{major}.{minor}"

        CUDA_FILTERS[key] = [
            arch for arch in ALL_ARCHS
            if amin <= int(arch) <= amax
        ]
        v = round(v + 0.1, 1)
