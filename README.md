<div align="center">

![cuda-2026](https://github.com/user-attachments/assets/dea107b7-e987-4c15-9749-ce3c924b57b2)

**JUST A NOTE**

***Updated on 2026-03-11***

</div>

---

| CUDA Version  | Supported Compute Capabilities |
|---------------|-------------------------------|
| 11.0          | 3.5 – 8.0                     |
| 11.1 – 11.4   | 3.5 – 8.6                     |
| 11.5 – 11.7   | 3.5 – 8.7                     |
| 11.8          | 3.5 – 9.0                     |
| 12.0 – 12.6   | 5.0 – 9.0                     |
| 12.8          | 5.0 – 12.0                    |
| 12.9          | 5.0 – 12.1                    |
| 13.0 – 13.1   | 7.5 – 12.1                    |

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
- cuDNN Tarbal (Zip) - [Official NVIDIA](https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/windows-x86_64/)
- CUDA - [Wikipedia](https://en.wikipedia.org/wiki/CUDA)
