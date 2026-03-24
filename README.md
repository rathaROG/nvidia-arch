<div align="center">

[![cover](https://raw.githubusercontent.com/rathaROG/nvidia-arch/refs/heads/main/assets/nvidia-arch.jpg)](https://github.com/rathaROG/nvidia-arch/blob/main/NVIDIA-ARCH.md)

**~~JUST A NOTE~~ 👉 A NOTE & A PACKAGE [`nvidia-arch`](https://github.com/rathaROG/nvidia-arch/blob/main/NVIDIA-ARCH.md)**

✅ All info has been verified by the [***Explorer***](https://github.com/rathaROG/nvidia-arch/actions/workflows/explorer.yaml) 🤖

***Updated on 2026-03-23***

</div>

---

### CUDA versions and supported compute capabilities:

| CUDA Version    | Supported Compute Capabilities     |
|-----------------|------------------------------------|
| 11.0            | `3.0` – `8.0`                      |
| 11.1 – 11.3     | `3.5` – `8.6`                      |
| 11.4 – 11.7     | `3.5` – `8.7`                      |
| 11.8            | `3.5` – `9.0`                      |
| 12.0 – 12.6     | `5.0` – `9.0`                      |
| 12.8            | `5.0` – `12.0`                     |
| 12.9            | `5.0` – `12.1`                     |
| 13.0 – 13.2     | `7.5` – `12.1`                     |

<sup>1. CUDA 12.8 does not include arch/compute `10.3` even though it already exposes arch/compute `12.0`. </sup><br>
<sup>2. CUDA 12.8 and 12.9 are the only versions that expose arch/compute `10.1` (Meant for Thor T4000/5000, and later replaced by `11.0` in CUDA 13.x). </sup><br>
<sup>3. CUDA 12.8 and 12.9 do not support arch/compute `11.0` (Former `10.1` as explained in 2). </sup><br>
<sup>4. CUDA 13.x includes a mysterious arch/compute `8.8` which is not documented or explained anywhere, see: [mysterious-88.png](https://github.com/rathaROG/nvidia-arch/blob/main/assets/mysterious-88.png). </sup><br>

---

### Compute / Architectures:

```python
# All archs from 30
ALL_ARCH_QE30 = ["30", "32", "35", "37", "50", "52", "53", "60", "61", "62", "70", "72", "75", "80", "86", "87", "88", "89", "90", "100", "101", "103", "110", "120", "121"]

# All archs from 30 for Consumer/Workstation GPUs
ALL_ARCH_QE30_CONS = ["30", "35", "50", "52", "60", "61", "70", "75", "86", "89", "120", "121"]

# All archs from 30 for Jetson/Embedded GPUs
ALL_ARCH_QE30_JETS = ["32", "53", "62", "72", "87", "101", "110"]
```

---

### Check CUDA version installed:

```bash
nvcc --version
```

```
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:18:05_PDT_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0
```

---

### Check your GPU compute cap:

```bash
nvidia-smi --query-gpu=name,compute_cap
```

```bash
$ nvidia-smi --query-gpu=name,compute_cap
name, compute_cap
NVIDIA RTX A6000, 8.6
NVIDIA RTX A6000, 8.6
```

---

### List all GPU architectures supported by CUDA Toolkit:

```bash
nvcc --list-gpu-arch
```

```bash
$ nvcc --list-gpu-arch
compute_50
compute_52
compute_53
compute_60
compute_61
compute_62
compute_70
compute_72
compute_75
compute_80
compute_86
compute_87
compute_89
compute_90
```

---

### 🆕 Must read for CUDA 13 🔥

- [Important changes](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/#wheel_package_changes_to_cuda_130)
- [Migration guide](https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html)

---

### Other useful links:

- GPU Compute Capability - [Official NVIDIA](https://developer.nvidia.com/cuda/gpus)
- Legacy GPU Compute Capability - [Official NVIDIA](https://developer.nvidia.com/cuda/gpus/legacy)
- CUDA Toolkit Archive - [Official NVIDIA](https://developer.nvidia.com/cuda-toolkit-archive)
- cuDNN Archive - [Official NVIDIA](https://developer.nvidia.com/cudnn-archive)
- cuDNN Tarbal (Zip) - [Official NVIDIA](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/)
- Compilation - [Official NVIDIA](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-compilation)
- CUDA - [Wikipedia](https://en.wikipedia.org/wiki/CUDA) (Some info there is not accurate)
