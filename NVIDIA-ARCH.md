[![GitHub release](https://img.shields.io/github/release/rathaROG/nvidia-arch.svg?logo=github&logoColor=lightgray)](https://github.com/rathaROG/nvidia-arch/releases)
[![PyPI version](https://badge.fury.io/py/nvidia-arch.svg)](https://badge.fury.io/py/nvidia-arch)
[![Downloads total](https://static.pepy.tech/badge/nvidia-arch)](https://pepy.tech/project/nvidia-arch)
[![Downloads monthly](https://static.pepy.tech/badge/nvidia-arch/month)](https://pepy.tech/project/nvidia-arch)

# `nvidia-arch`

A lightweight tool for detecting and querying NVIDIA GPU architectures (SM/CC), and generating `-gencode` flags for CUDA builds; ideal for integrating into Python `setup.py` and custom CUDA workflows.

> If you just want to see my note, see [README.md](https://github.com/rathaROG/nvidia-arch/blob/main/README.md).

## ❓ Why this exists

Working with CUDA toolchains is notoriously inconsistent across systems, CUDA versions, and GPU families.  
Different machines report different supported architectures, `nvcc` behaves differently depending on the installed CTK, and build scripts often end up hard‑coding SM versions that quickly become outdated.

This package solves that by providing:

- A **single reliable source of truth** for supported SM and compute capabilities  
- Automatic detection of architectures from the installed CUDA Toolkit  
- Clean overrides for building against **specific CUDA versions**  
- Easy generation of correct `-gencode` flags  
- A simple API that works the same on Linux, Windows, WSL, and CI environments  

> **In short: `nvidia-arch` removes the guesswork and keeps your CUDA builds future‑proof and reproducible.**

## 💽 Installation

### Install from PyPi:

```bash
pip install nvidia-arch
```

### Install from GitHub repo:

```bash
pip install git+https://github.com/rathaROG/nvidia-arch.git
```

## 🧪 Usage

For all details of all available functions: see [`core.py`](nvidia_arch/core.py) and [`arches.py`](nvidia_arch/arches.py).

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

#### Get PyTorch‑style architectures string

```python
from nvidia_arch import get_architectures
get_architectures(gpu_type="cons+jets", cuda_ver="13.0", min_sm=75, return_mode="cc_string")
```

```bash
'7.5;8.6;8.7;8.9;11.0;12.0;12.1'
```

### Generate `nvcc` `-gencode` flags in `Setup.py`

```python
from nvidia_arch import get_architectures, make_gencode_flags
arches = get_architectures(gpu_type="jets", cuda_ver="13.0", min_sm=75)
make_gencode_flags(arches)
# extra_compile_args["nvcc"] += make_gencode_flags(arches)
```

```bash
['-gencode=arch=compute_87,code=sm_87', '-gencode=arch=compute_110,code=sm_110']
```

See a real example in [BEVFusionx](https://github.com/rathaumons/bevfusionx/blob/main/setup.py).

## 📝 License

[![LICENSE](https://img.shields.io/badge/LICENSE-Apache_2.0-blue)](https://github.com/rathaROG/nvidia-arch/blob/main/LICENSE)

