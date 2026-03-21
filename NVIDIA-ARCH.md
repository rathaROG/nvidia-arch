<div align="center">

[![GitHub release](https://img.shields.io/github/release/rathaROG/nvidia-arch.svg?logo=github&logoColor=lightgray)](https://github.com/rathaROG/nvidia-arch/releases)
[![Wheels](https://img.shields.io/pypi/wheel/nvidia-arch)](https://pypi.org/project/nvidia-arch/)
[![Test](https://github.com/rathaROG/nvidia-arch/actions/workflows/test.yaml/badge.svg)](https://github.com/rathaROG/nvidia-arch/actions/workflows/test.yaml)
[![Build](https://github.com/rathaROG/nvidia-arch/actions/workflows/build.yaml/badge.svg)](https://github.com/rathaROG/nvidia-arch/actions/workflows/build.yaml)
[![Publish](https://github.com/rathaROG/nvidia-arch/actions/workflows/publish.yaml/badge.svg)](https://github.com/rathaROG/nvidia-arch/actions/workflows/publish.yaml)

![cover](https://raw.githubusercontent.com/rathaROG/nvidia-arch/refs/heads/main/assets/nvidia-arch.jpg)

***“ `nvidia-arch` removes the guesswork and keeps your CUDA builds future‑proof and reproducible. ”***

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
print_summary(min_sm=60)
```

```bash
CUDA  Arch (min..max)   Consumer/Workstation (cons)        Jetson (jets)
=============================================================================
11.0  6.0..8.0          6.0;6.1;7.0;7.5                    6.2;7.2
11.1  6.0..8.6          6.0;6.1;7.0;7.5;8.6                6.2;7.2
11.2  6.0..8.6          6.0;6.1;7.0;7.5;8.6                6.2;7.2
11.3  6.0..8.6          6.0;6.1;7.0;7.5;8.6                6.2;7.2
11.4  6.0..8.6          6.0;6.1;7.0;7.5;8.6                6.2;7.2
11.5  6.0..8.7          6.0;6.1;7.0;7.5;8.6                6.2;7.2;8.7
11.6  6.0..8.7          6.0;6.1;7.0;7.5;8.6                6.2;7.2;8.7
11.7  6.0..8.7          6.0;6.1;7.0;7.5;8.6                6.2;7.2;8.7
11.8  6.0..9.0          6.0;6.1;7.0;7.5;8.6;8.9            6.2;7.2;8.7
12.0  6.0..9.0          6.0;6.1;7.0;7.5;8.6;8.9            6.2;7.2;8.7
12.1  6.0..9.0          6.0;6.1;7.0;7.5;8.6;8.9            6.2;7.2;8.7
12.2  6.0..9.0          6.0;6.1;7.0;7.5;8.6;8.9            6.2;7.2;8.7
12.3  6.0..9.0          6.0;6.1;7.0;7.5;8.6;8.9            6.2;7.2;8.7
12.4  6.0..9.0          6.0;6.1;7.0;7.5;8.6;8.9            6.2;7.2;8.7
12.5  6.0..9.0          6.0;6.1;7.0;7.5;8.6;8.9            6.2;7.2;8.7
12.6  6.0..9.0          6.0;6.1;7.0;7.5;8.6;8.9            6.2;7.2;8.7
12.8  6.0..12.0         6.0;6.1;7.0;7.5;8.6;8.9;12.0       6.2;7.2;8.7;11.0
12.9  6.0..12.1         6.0;6.1;7.0;7.5;8.6;8.9;12.0;12.1  6.2;7.2;8.7;11.0
13.0  7.5..12.1         7.5;8.6;8.9;12.0;12.1              8.7;11.0
13.1  7.5..12.1         7.5;8.6;8.9;12.0;12.1              8.7;11.0
13.2  7.5..12.1         7.5;8.6;8.9;12.0;12.1              8.7;11.0
=============================================================================
* All NVIDIA Architectures:
6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6;8.7;8.8;8.9;9.0;10.0;10.3;11.0;12.0;12.1
```

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
from nvidia_arch import find_gpu

gpu_info = find_gpu(extra_query_gpu='serial,temperature.gpu')
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
from nvidia_arch import get_compute_cap
get_compute_cap(return_mode='cc_string', add_ptx=True)
```

```bash
'8.6;8.9+PTX'
```

#### Get supported SM architectures from installed CTK (CUDA Toolkit)

```python
from nvidia_arch import get_architectures
get_architectures(cuda_ver=None, min_sm=75)
```

```bash
['75', '80', ...]
```

#### Get architectures for a specific CTK (CUDA Toolkit) version

```python
from nvidia_arch import get_architectures
get_architectures(cuda_ver="13.0", min_sm=75)
```

```bash
['75', '80', '86', '87', '88', '89', '90', '100', '103', '110', '120', '121']
```

#### Get architectures and filter by GPU type (Consumer, Jetson, etc.)

Supported inputs for `gpu_type`: 
- `"all"`: All supported GPUs
- `"cons"`: Only consumer/workstation GPUs
- `"jets"`: Only Jetson/embedded GPUs
- `"cons+jets"`: Only consumer/workstation + Jetson/embedded GPUs

```python
from nvidia_arch import get_architectures
get_architectures(gpu_type="cons", cuda_ver="13.0", min_sm=75)
```

```bash
['75', '86', '89', '120', '121']
```

#### Get compute capabilities instead of SM codes

```python
from nvidia_arch import get_architectures
get_architectures(gpu_type="cons", cuda_ver="13.0", min_sm=75, return_mode="cc_list")
```

```bash
['7.5', '8.6', '8.9', '12.0', '12.1']
```

#### Get PyTorch‑style architectures string with PTX

```python
from nvidia_arch import get_architectures
get_architectures(gpu_type="cons+jets", cuda_ver="13.0", min_sm=75, return_mode="cc_string", add_ptx=True)
```

```bash
'7.5;8.6;8.7;8.9;11.0;12.0;12.1+PTX'
```

### Generate `nvcc` `-gencode` flags in `Setup.py`

```python
from nvidia_arch import get_architectures, make_gencode_flags
arches = get_architectures(gpu_type="jets", cuda_ver="13.0", min_sm=75)
make_gencode_flags(arches, add_ptx=True)
# extra_compile_args["nvcc"] += make_gencode_flags(arches)
```

```bash
['-gencode=arch=compute_87,code=sm_87', '-gencode=arch=compute_110,code=[sm_110,compute_110]']
```

See a real example in [BEVFusionx](https://github.com/rathaumons/bevfusionx/blob/main/setup.py).

## 📝 License

[![LICENSE](https://img.shields.io/badge/LICENSE-Apache_2.0-blue)](https://github.com/rathaROG/nvidia-arch/blob/main/LICENSE)

