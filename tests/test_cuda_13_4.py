import pytest

import nvidia_arch as na
import nvidia_arch.core as core


# Raw nvcc --list-gpu-arch output from the CUDA 13.4.1 Explorer artifact.
# The accompanying JSON was filtered by the older PyPI package and omitted 107.
NVCC_CUDA_13_4_1_OUTPUT = b"""compute_75
compute_80
compute_86
compute_87
compute_88
compute_89
compute_90
compute_100
compute_110
compute_103
compute_120
compute_121
compute_107
"""

CUDA_13_4_ARCHES = [
    "75", "80", "86", "87", "88", "89", "90",
    "100", "103", "107", "110", "120", "121",
]


@pytest.mark.parametrize("cuda_ver", ["13.4", "13.4.1", 13.4])
def test_cuda_13_4_matches_explorer(cuda_ver):
    assert na.get_arches(cuda_ver=cuda_ver, raise_on_error=True) == CUDA_13_4_ARCHES


@pytest.mark.parametrize("gpu_type, expected", [
    ("all", CUDA_13_4_ARCHES),
    ("dcen", ["75", "80", "86", "89", "90", "100", "103", "107", "120"]),
    ("jets", ["87", "88", "110"]),
    ("cons", ["75", "86", "89", "120", "121"]),
    ("cons+jets", ["75", "86", "87", "88", "89", "110", "120", "121"]),
])
def test_cuda_13_4_gpu_categories(monkeypatch, gpu_type, expected):
    assert na.get_arches(gpu_type=gpu_type, cuda_ver="13.4", raise_on_error=True) == expected

    # Exercise the real nvcc parser and category filtering without needing CUDA.
    monkeypatch.setattr(core.subprocess, "check_output", lambda *args, **kwargs: NVCC_CUDA_13_4_1_OUTPUT)
    assert na.get_arches(gpu_type=gpu_type, raise_on_error=True) == expected


def test_cuda_13_4_version_detection_fallback(monkeypatch):
    monkeypatch.setattr(core, "nvcc_list_arches", lambda: None)
    monkeypatch.setattr(core, "detect_ctk", lambda **kwargs: {"version": "13.4"})
    assert na.get_arches(raise_on_error=True) == CUDA_13_4_ARCHES


@pytest.mark.parametrize("cuda_ver", [
    "11.0", "11.1", "11.2", "11.3", "11.4", "11.5", "11.6", "11.7", "11.8",
    "12.0", "12.1", "12.2", "12.3", "12.4", "12.5", "12.6", "12.8", "12.9",
    "13.0", "13.1", "13.2", "13.3",
])
def test_rubin_is_excluded_from_older_cuda(cuda_ver):
    assert "107" not in na.get_arches(cuda_ver=cuda_ver, raise_on_error=True)
    with pytest.raises(ValueError, match=r"Unsupported architecture.*10\.7"):
        na.validate_arch_string("10.7", against_cuda_ver=cuda_ver)


@pytest.mark.parametrize("cuda_ver, expected", [
    ("11.8", False), ("12.8", False), ("12.9", False),
    ("13.0", True), ("13.1", True), ("13.2", True), ("13.3", True), ("13.4", True),
])
def test_ampere_88_jetson_classification(cuda_ver, expected):
    for gpu_type in ("all", "jets", "cons+jets"):
        arches = na.get_arches(gpu_type=gpu_type, cuda_ver=cuda_ver, return_mode="cc_list")
        assert ("8.8" in arches) == expected
    for gpu_type in ("cons", "dcen"):
        assert "88" not in na.get_arches(gpu_type=gpu_type, cuda_ver=cuda_ver)


def test_cuda_13_4_validation_and_build_flags():
    assert na.validate_arch_string("10.7+PTX;8.8", against_cuda_ver="13.4.1") == "8.8;10.7+PTX"
    with pytest.raises(ValueError, match=r"Unsupported architecture.*10\.1"):
        na.validate_arch_string("10.1", against_cuda_ver="13.4")
    assert na.make_gencode_flags("8.8;10.7", add_ptx=True) == [
        "-gencode=arch=compute_88,code=sm_88",
        "-gencode=arch=compute_107,code=[sm_107,compute_107]",
    ]
    assert na.get_arches(gpu_type="jets", cuda_ver="13.4", return_mode="cc_string", add_ptx=True) == "8.7;8.8;11.0+PTX"
