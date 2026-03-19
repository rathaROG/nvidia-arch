<div align="center">

![cuda-2026](https://github.com/user-attachments/assets/dea107b7-e987-4c15-9749-ce3c924b57b2)

**JUST A NOTE**

***Updated on 2026-03-19***

</div>

---

### 🆕 Must read for CUDA 13 🔥

- [Important changes](https://developer.nvidia.com/blog/whats-new-and-important-in-cuda-toolkit-13-0/#wheel_package_changes_to_cuda_130)
- [Migration guide](https://nvidia.github.io/cccl/unstable/cccl/3.0_migration_guide.html)

---

### CUDA versions and supported compute capabilities

| CUDA Version  | Supported Compute Capabilities |
|---------------|-------------------------------|
| 11.0          | 3.5 – 8.0                     |
| 11.1 – 11.4   | 3.5 – 8.6                     |
| 11.5 – 11.7   | 3.5 – 8.7                     |
| 11.8          | 3.5 – 9.0                     |
| 12.0 – 12.6   | 5.0 – 9.0                     |
| 12.8          | 5.0 – 12.0                    |
| 12.9          | 5.0 – 12.1                    |
| 13.0 – 13.2   | 7.5 – 12.1                    |

---

### Compute architectures

```python
# All archs from 35
ALL_ARCH_QE35 = ["35", "37", "50", "52", "53", "60", "61", "62", "70", "72", "75", "80", "86", "87", "88", "89", "90", "100", "103", "110", "120", "121"]

# All archs from 60
ALL_ARCH_QE60 = ["60", "61", "62", "70", "72", "75", "80", "86", "87", "88", "89", "90", "100", "103", "110", "120", "121"]

# All archs from 60 for Consumer/Workstation GPUs
ALL_ARCH_QE60_CONS = ["60", "61", "70", "75", "86", "89", "120", "121"]

# All archs from 60 for Jetson
ALL_ARCH_QE60_JETS = ["62", "72", "87", "110"]
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

### Other useful links:

- GPU Compute Capability - [Official NVIDIA](https://developer.nvidia.com/cuda/gpus)
- Legacy GPU Compute Capability - [Official NVIDIA](https://developer.nvidia.com/cuda/gpus/legacy)
- CUDA Toolkit Archive - [Official NVIDIA](https://developer.nvidia.com/cuda-toolkit-archive)
- cuDNN Archive - [Official NVIDIA](https://developer.nvidia.com/cudnn-archive)
- cuDNN Tarbal (Zip) - [Official NVIDIA](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/)
- Compilation - [Official NVIDIA](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/#gpu-compilation)
- CUDA - [Wikipedia](https://en.wikipedia.org/wiki/CUDA)
