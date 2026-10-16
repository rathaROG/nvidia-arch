<div align="center">

[![GitHub release](https://img.shields.io/github/release/rathaROG/nvidia-arch.svg?logo=github&logoColor=lightgray)](https://github.com/rathaROG/nvidia-arch/releases)
[![Wheels](https://img.shields.io/pypi/wheel/nvidia-arch)](https://pypi.org/project/nvidia-arch/)
[![Explore](https://github.com/rathaROG/nvidia-arch/actions/workflows/explorer.yaml/badge.svg)](https://github.com/rathaROG/nvidia-arch/actions/workflows/explorer.yaml)
[![Test](https://github.com/rathaROG/nvidia-arch/actions/workflows/test.yaml/badge.svg)](https://github.com/rathaROG/nvidia-arch/actions/workflows/test.yaml)
[![Build](https://github.com/rathaROG/nvidia-arch/actions/workflows/build.yaml/badge.svg)](https://github.com/rathaROG/nvidia-arch/actions/workflows/build.yaml)
[![Publish](https://github.com/rathaROG/nvidia-arch/actions/workflows/publish.yaml/badge.svg)](https://github.com/rathaROG/nvidia-arch/actions/workflows/publish.yaml)

[![cover](https://raw.githubusercontent.com/rathaROG/nvidia-arch/refs/heads/main/assets/nvidia-arch.jpg)](https://github.com/rathaROG/nvidia-arch/blob/main/NVIDIA-ARCH.md)

***“ `nvidia-arch` removes the guesswork and keeps your CUDA builds future‑proof and reproducible. ”***

✅ All info has been verified by the [***Explorer***](https://github.com/rathaROG/nvidia-arch/actions/workflows/explorer.yaml) 🤖

---

</div>

A lightweight tool for detecting and querying NVIDIA GPU architectures (SM/CC), and generating `-gencode` flags for CUDA builds; ideal for integrating into Python `setup.py` and custom CUDA workflows.

> If you just want to see my note, see [README.md](https://github.com/rathaROG/nvidia-arch/blob/main/README.md).

## 💡 Why this exists

Working with CUDA toolchains is notoriously inconsistent across systems, CUDA versions, and GPU families. Different machines report different supported architectures, `nvcc` behaves differently depending on the installed CTK (CUDA Toolkit), and build scripts often end up hard‑coding SM versions that quickly become outdated.

### This package solves that by providing:

- A **single reliable source of truth** for supported SM and compute capabilities
- Automatic detection of architectures from the installed CUDA Toolkit
- Clean overrides for building against **specific CUDA versions**
- Correct and reproducible generation of `-gencode` flags
- Consistent behavior across Linux, Windows, WSL, and CI environments

### Key features:

- Detect installed CUDA Toolkit (CTK) and its include/lib paths
- Query supported SM/CC architectures for any CUDA version
- Generate correct `-gencode` flags for nvcc
- Opt into CUDA-version-aware `a` and `f` targets with `include_a` / `include_f`
- Handle PTX emission cleanly (`+PTX` suffix or highest‑SM policy)
- Filter architectures by GPU family (consumer, workstation, Jetson)
- Provide PyTorch‑style CC strings (`7.5;8.6;8.9+PTX`)
- Work reliably across heterogeneous environments (local, Docker, CI)

## 💽 Installation

### Install from [PyPI](https://pypi.org/project/nvidia-arch/):

[![PyPI version](https://badge.fury.io/py/nvidia-arch.svg)](https://badge.fury.io/py/nvidia-arch)
[![Downloads total](https://static.pepy.tech/badge/nvidia-arch)](https://pepy.tech/project/nvidia-arch)
[![Downloads monthly](https://static.pepy.tech/badge/nvidia-arch/month)](https://pepy.tech/project/nvidia-arch)

```bash
pip install nvidia-arch
```

### Install from GitHub repo:

```bash
pip install git+https://github.com/rathaROG/nvidia-arch.git
```

## 🧪 Usage

For all details of all available functions: see [`core.py`](https://github.com/rathaROG/nvidia-arch/blob/main/nvidia_arch/core.py) and [`arches.py`](https://github.com/rathaROG/nvidia-arch/blob/main/nvidia_arch/arches.py).

### Main highlights

#### Print a summary of supported architectures for each CUDA version

```python
from nvidia_arch import print_summary
print_summary(min_sm=30, include_a=True, include_f=True)
```

```text
CUDA Arch (min..max)   Consumer/Workstation (cons)                                        Jetson (jets)
=============================================================================================================================
11.0  3.5..8.0          3.5;5.0;5.2;6.0;6.1;7.0;7.5                                        5.3;6.2;7.2
11.1  3.5..8.6          3.5;5.0;5.2;6.0;6.1;7.0;7.5;8.6                                    5.3;6.2;7.2
11.2  3.5..8.6          3.5;5.0;5.2;6.0;6.1;7.0;7.5;8.6                                    5.3;6.2;7.2
11.3  3.5..8.6          3.5;5.0;5.2;6.0;6.1;7.0;7.5;8.6                                    5.3;6.2;7.2
11.4  3.5..8.7          3.5;5.0;5.2;6.0;6.1;7.0;7.5;8.6                                    5.3;6.2;7.2;8.7
11.5  3.5..8.7          3.5;5.0;5.2;6.0;6.1;7.0;7.5;8.6                                    5.3;6.2;7.2;8.7
11.6  3.5..8.7          3.5;5.0;5.2;6.0;6.1;7.0;7.5;8.6                                    5.3;6.2;7.2;8.7
11.7  3.5..8.7          3.5;5.0;5.2;6.0;6.1;7.0;7.5;8.6                                    5.3;6.2;7.2;8.7
11.8  3.5..9.0          3.5;5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9                                5.3;6.2;7.2;8.7
12.0  5.0..9.0          5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9                                    5.3;6.2;7.2;8.7
12.1  5.0..9.0          5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9                                    5.3;6.2;7.2;8.7
12.2  5.0..9.0          5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9                                    5.3;6.2;7.2;8.7
12.3  5.0..9.0          5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9                                    5.3;6.2;7.2;8.7
12.4  5.0..9.0          5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9                                    5.3;6.2;7.2;8.7
12.5  5.0..9.0          5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9                                    5.3;6.2;7.2;8.7
12.6  5.0..9.0          5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9                                    5.3;6.2;7.2;8.7
12.8  5.0..12.0         5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9;12.0;12.0a                         5.3;6.2;7.2;8.7;10.1;10.1a
12.9  5.0..12.1         5.0;5.2;6.0;6.1;7.0;7.5;8.6;8.9;12.0;12.0a;12.0f;12.1;12.1a;12.1f  5.3;6.2;7.2;8.7;10.1;10.1a;10.1f
13.0  7.5..12.1         7.5;8.6;8.9;12.0;12.0a;12.0f;12.1;12.1a;12.1f                      8.7;8.8;11.0;11.0a;11.0f
13.1  7.5..12.1         7.5;8.6;8.9;12.0;12.0a;12.0f;12.1;12.1a;12.1f                      8.7;8.8;11.0;11.0a;11.0f
13.2  7.5..12.1         7.5;8.6;8.9;12.0;12.0a;12.0f;12.1;12.1a;12.1f                      8.7;8.8;11.0;11.0a;11.0f
13.3  7.5..12.1         7.5;8.6;8.9;12.0;12.0a;12.0f;12.1;12.1a;12.1f                      8.7;8.8;11.0;11.0a;11.0f
13.4  7.5..12.1         7.5;8.6;8.9;12.0;12.0a;12.0f;12.1;12.1a;12.1f                      8.7;8.8;11.0;11.0a;11.0f
=============================================================================================================================

* All NVIDIA Architectures:
  3.0;3.2;3.5;3.7;5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.8;8.9;9.0;9.0a;10.0;10.0a;10.0f;10.1;10.1a;10.1f;10.3;10.3a;10.3f;10.7;10.7a;10.7f;11.0;11.0a;11.0f;12.0;12.0a;12.0f;12.1;12.1a;12.1f

* Requested variants by CUDA version (all GPU categories):
  12.0: 9.0a
  12.1: 9.0a
  12.2: 9.0a
  12.3: 9.0a
  12.4: 9.0a
  12.5: 9.0a
  12.6: 9.0a
  12.8: 9.0a;10.0a;10.1a;12.0a
  12.9: 9.0a;10.0a;10.0f;10.1a;10.1f;10.3a;10.3f;12.0a;12.0f;12.1a;12.1f
  13.0: 9.0a;10.0a;10.0f;10.3a;10.3f;11.0a;11.0f;12.0a;12.0f;12.1a;12.1f
  13.1: 9.0a;10.0a;10.0f;10.3a;10.3f;11.0a;11.0f;12.0a;12.0f;12.1a;12.1f
  13.2: 9.0a;10.0a;10.0f;10.3a;10.3f;11.0a;11.0f;12.0a;12.0f;12.1a;12.1f
  13.3: 9.0a;10.0a;10.0f;10.3a;10.3f;11.0a;11.0f;12.0a;12.0f;12.1a;12.1f
  13.4: 9.0a;10.0a;10.0f;10.3a;10.3f;10.7a;10.7f;11.0a;11.0f;12.0a;12.0f;12.1a;12.1f

* Other Notes:
  1. Architecture(s) 8.8 is not officially supported in CUDA 11.8–12.9.
  2. Architecture(s) 10.1 is not officially supported in CUDA 13.0–13.4.
  3. Architecture(s) 10.3 is not officially supported in CUDA 12.8–12.8.
  4. Architecture(s) 10.7 is not officially supported in CUDA 12.8–13.3.
  5. Architecture(s) 11.0 is not officially supported in CUDA 12.8–12.9.
```

CUDA 11.0 supports compute capabilities starting at `3.5`; `3.0` and `3.2` were removed in that release. They remain in the legacy architecture catalog, but CUDA-version queries and validation exclude them. Automatic detection on CUDA 11.0 reads only the allowed target values in compiler help, ignoring its outdated deprecation note.

The consumer and Jetson columns include their variants, while min/max ranges remain numeric. A separate list shows requested variants per CUDA version across all GPU categories, including datacenter targets such as `90a` and `107a`. The global architecture catalog also includes the requested suffixes. Both options default to `False`, preserving the original summary output.

#### Detect CTK (CUDA Toolkit) in your environment

```python
import json
from nvidia_arch import detect_ctk

cuda_info = detect_ctk()
print(json.dumps(cuda_info, indent=2))
```

```bash
{
  "version": "13.0",
  "root": "/usr/local/cuda",
  "include": {
    "root": "/usr/local/cuda/include",
    "cuda": "/usr/local/cuda/include/cccl/cuda",
    "cub": "/usr/local/cuda/include/cccl/cub",
    "thrust": "/usr/local/cuda/include/cccl/thrust"
  },
  "lib": "/usr/local/cuda/lib64"
}
```

#### Find all NVIDIA GPU(s) installed

```python
import json
from nvidia_arch import find_gpus

gpu_info = find_gpus(extra_query_gpu='serial,temperature.gpu')
print(json.dumps(gpu_info, indent=2))
```

```bash
[
  {
    "name": "NVIDIA RTX A6000",
    "compute_cap": "8.6",
    "memory.total": "49140",
    "serial": "1234567891011",
    "temperature.gpu": "44"
  },
  {
    "name": "NVIDIA RTX A6000",
    "compute_cap": "8.6",
    "memory.total": "49140",
    "serial": "1234567891012",
    "temperature.gpu": "39"
  }
]
```

#### Get compute cap of the GPU(s) installed

```python
from nvidia_arch import get_compute_caps
get_compute_caps(return_mode='cc_string', add_ptx=True)
```

```bash
'8.6;8.9+PTX'
```

#### Get supported SM architectures from installed CTK (CUDA Toolkit)

```python
from nvidia_arch import get_arches
get_arches(cuda_ver=None, min_sm=75)
```

```bash
['75', '80', ...]
```

#### Get architectures for a specific CTK (CUDA Toolkit) version

```python
from nvidia_arch import get_arches
get_arches(cuda_ver="13.4", min_sm=75)
```

```bash
['75', '80', '86', '87', '88', '89', '90', '100', '103', '107', '110', '120', '121']
```

#### Get architectures and filter by GPU type (Consumer, Jetson, etc.)

Supported inputs for `gpu_type`: 
- `"all"`: All supported GPUs
- `"cons"`: Only consumer/workstation GPUs
- `"jets"`: Only Jetson/embedded GPUs
- `"dcen"`: Only datacenter GPUs
- `"cons+jets"`: Only consumer/workstation + Jetson/embedded GPUs

NVIDIA's [Tile IR architecture table](https://docs.nvidia.com/cuda/tile-ir/13.4/sections/stability.html#supported-architectures) supports these classifications:

- `88` / `8.8`: Ampere, grouped with Jetson Orin; included in `jets` and `cons+jets` for CUDA 13.x. The exact device mapping remains unspecified: the [GPU capability list](https://developer.nvidia.com/cuda/gpus) lists Orin models at `8.7`.
- `107` / `10.7`: Rubin, associated with HGX Rubin NVL8 and Vera Rubin NVL72; included in `dcen` for CUDA 13.4. The Tile IR table marks Rubin support as developer preview.

```python
from nvidia_arch import get_arches
get_arches(gpu_type="cons", cuda_ver="13.0", min_sm=75)
```

```bash
['75', '86', '89', '120', '121']
```

#### Get compute capabilities instead of SM codes

```python
from nvidia_arch import get_arches
get_arches(gpu_type="cons", cuda_ver="13.0", min_sm=75, return_mode="cc_list")
```

```bash
['7.5', '8.6', '8.9', '12.0', '12.1']
```

#### Get PyTorch‑style architectures string with PTX

```python
from nvidia_arch import get_arches
get_arches(gpu_type="cons+jets", cuda_ver="13.0", min_sm=75, return_mode="cc_string", add_ptx=True)
```

```bash
'7.5;8.6;8.7;8.8;8.9;11.0;12.0;12.1+PTX'
```

#### Include architecture-specific (`a`) and family-specific (`f`) targets

Both options are keyword-only and default to `False`, preserving existing outputs and positional arguments. Enabled variants are added alongside the baseline targets, sorted by numeric capability and then suffix (baseline, `a`, `f`).

```python
from nvidia_arch import get_arches, make_gencode_flags, validate_arch_string

arches = get_arches(
    gpu_type="jets", cuda_ver="12.9", min_sm=100,
    include_a=True, include_f=True, return_mode="cc_string", add_ptx=True,
)
# '10.1+PTX;10.1a;10.1f'

validate_arch_string(arches, against_cuda_ver="12.9")
make_gencode_flags(arches, against_cuda_ver="12.9")
# ['-gencode=arch=compute_101,code=[sm_101,compute_101]',
#  '-gencode=arch=compute_101a,code=sm_101a',
#  '-gencode=arch=compute_101f,code=sm_101f']
```

Supported variants, checked against NVIDIA's NVCC documentation and the [Explorer compiler probes](https://github.com/rathaROG/nvidia-arch/actions/runs/38828749437):

| CUDA version | `a` targets | `f` targets |
|---|---|---|
| 11.0–11.8 | None | None |
| 12.0–12.6 | `90a` | None |
| 12.8 | `90a`, `100a`, `101a`, `120a` | None |
| 12.9 | `90a`, `100a`, `101a`, `103a`, `120a`, `121a` | `100f`, `101f`, `103f`, `120f`, `121f` |
| 13.0–13.3 | `90a`, `100a`, `103a`, `110a`, `120a`, `121a` | `100f`, `103f`, `110f`, `120f`, `121f` |
| 13.4 | `90a`, `100a`, `103a`, `107a`, `110a`, `120a`, `121a` | `100f`, `103f`, `107f`, `110f`, `120f`, `121f` |

Sources: NVCC [12.0](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-compiler-driver-nvcc/index.html#gpu-feature-list), [12.8](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-compiler-driver-nvcc/index.html#gpu-feature-list), [12.9](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-compiler-driver-nvcc/index.html#gpu-feature-list), and [13.4](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-compiler-driver-nvcc/index.html#gpu-feature-list). The probes cover the toolkit versions in that run; the table also covers intervening supported minor versions.

Variants inherit every category of their base: `101a/f` and `110a/f` are Jetson, `107a/f` are datacenter, and `120a/f` retain the existing consumer/datacenter overlap. `min_sm` compares the numeric base. Unsupported variants are omitted from queries; explicitly validating `10.0f` against CUDA 12.8 raises `ValueError`. A request for `include_f=True` on CUDA 12.8 therefore returns the baseline list without `f` targets.

When `cuda_ver` is omitted, `get_arches()` and `nvcc_list_arches(include_a=True, include_f=True)` inspect the installed compiler's allowed help values because the list commands omit variants. Discovery requires matching virtual and real targets. If discovery fails, `get_arches()` uses the detected CUDA version's table; it does not guess support for an unknown version.

`normalize_arches()`, `validate_arch_string()`, and `make_gencode_flags()` accept explicit suffixed targets without inclusion options. Exclusions match exact targets: excluding `10.0` leaves `10.0a` and `10.0f`; excluding `100a` removes `10.0a` and `10.0a+PTX`. `make_gencode_flags(..., against_cuda_ver=...)` adds a CUDA version check. Without it, flag generation checks the known architecture catalog and works offline, as before.

Automatic `add_ptx=True` and `force_highest_ptx=True` select the highest **baseline** target. With only variants, select PTX explicitly (`10.0a+PTX` or `10.0f+PTX`) instead of automatic PTX. Explicit `+PTX` entries take precedence over `make_gencode_flags(add_ptx=True)`. NVIDIA documents that `a` PTX targets the matching architecture, while `f` PTX is limited to compatible targets in its family; neither provides the general forward compatibility of baseline PTX. See [NVCC target compatibility](https://docs.nvidia.com/cuda/archive/13.4.2/cuda-compiler-driver-nvcc/index.html#gpu-name-gpuname-arch).

The baseline `ALL_ARCHS`, `CUDA_FILTERS`, and category lists remain numeric. `ALL_ARCHS_VARIANTS` and `CUDA_VARIANTS` expose the separate variant catalog and version map. Hardware detection through `get_compute_caps()` still reports numeric capabilities. Consumers of CC strings, such as build frameworks, must themselves accept suffixed targets; `make_gencode_flags()` generates the corresponding NVCC flags directly.

The inclusion options apply to functions that discover or list targets:

| Functions | Variant handling |
|---|---|
| `get_arches()`, `nvcc_list_arches()`, `print_summary()` | Accept keyword-only `include_a=False` and `include_f=False`. |
| `get_architectures()` | Deprecated alias; forwards both options to `get_arches()`. |
| `normalize_arch_string()`, `normalize_arches()`, `validate_arch_string()`, `make_gencode_flags()` | Preserve or validate explicitly supplied variants; they do not add targets to the input. |
| `validate_cc_string()` | Deprecated alias; forwards explicit variant strings to `validate_arch_string()`. |
| `get_compute_caps()`, `find_gpus()` and their deprecated aliases | Report detected GPU capabilities and properties; compiler variants are selected separately with `get_arches()`. |
| `detect_ctk()`, `normalize_cuda_ver()` | Handle toolkit metadata and version strings; inclusion options do not apply. |

#### Normalize architectures

```python
from nvidia_arch import normalize_arches, get_arches

normalize_arches(['75', '86', '89+PTX'], return_mode='cc_string')
normalize_arches('7.5;8.6;8.9+PTX', exclude='8.6', return_mode='cc_string')

arches = get_arches(cuda_ver=12.8, return_mode='cc_string', add_ptx=True)
normalize_arches(arches, exclude='10.1', return_mode='cc_string')
```

```bash
'7.5;8.6;8.9+PTX'
'7.5;8.9+PTX'
'5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.9;9.0;10.0;12.0+PTX'
```

#### Validate a PyTorch‑style architectures string

```python
from nvidia_arch import validate_arch_string
validate_arch_string(
    "6.1+PTX;Pascal;12.0;Lovelace",
    named_arches={"Pascal": "6.0;6.1+PTX", "Lovelace": "8.9+PTX"},
    force_highest_ptx=True,
    against_cuda_ver="12.8"
)

```

```bash
'6.0;6.1;8.9;12.0+PTX'
```

```python
from nvidia_arch import validate_arch_string
validate_arch_string(
    "6.1+PTX;Pascal;12.0;Lovelace;13.5;0.9",
    named_arches={"Pascal": "6.0;6.1+PTX", "Lovelace": "8.9+PTX"},
    force_highest_ptx=True,
    against_cuda_ver="13.2"
)
```

```bash
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "C:\dev\exc\python\p311\Lib\site-packages\nvidia_arch\core.py", line 483, in validate_arch_string
    raise ValueError(f"Unknown architecture(s): {', '.join(unknown_arch)}. ")
ValueError: Unknown architecture(s): 0.9, 13.5+PTX.
```

### Generate `nvcc` `-gencode` flags in `Setup.py`

```python
from nvidia_arch import get_arches, make_gencode_flags
arches = get_arches(gpu_type="jets", cuda_ver="13.0", min_sm=75)
make_gencode_flags(arches, add_ptx=True)
# extra_compile_args["nvcc"] += make_gencode_flags(arches)
```

```bash
['-gencode=arch=compute_87,code=sm_87', '-gencode=arch=compute_88,code=sm_88', '-gencode=arch=compute_110,code=[sm_110,compute_110]']
```

See a real example in [BEVFusionx](https://github.com/rathaumons/bevfusionx/blob/main/setup.py).

## Deprecation

**The following legacy function names are deprecated and will be removed in version 10.0.0:**

- `find_gpu()` → use `find_gpus()` instead  
- `get_compute_cap()` → use `get_compute_caps()` instead  
- `get_architectures()` → use `get_arches()` instead  
- `validate_cc_string()` → use `validate_arch_string()` instead  

You can continue using the old names until v10.0.0, but all new code and documentation now use the new, more Pythonic API. Importing or calling any deprecated function will issue a `DeprecationWarning`.

## 📝 License

[![LICENSE](https://img.shields.io/badge/LICENSE-Apache_2.0-blue)](https://github.com/rathaROG/nvidia-arch/blob/main/LICENSE)
