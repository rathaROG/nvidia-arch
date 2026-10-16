from pathlib import Path
import subprocess

import pytest

import nvidia_arch as na
import nvidia_arch.core as core


# Relevant help excerpt from Explorer run 38828749437, info-11.0.3.
# Its compile-verified targets exclude the deprecated mentions of SM 30/32.
NVCC_HELP = (Path(__file__).parent / 'fixtures' / 'nvcc-help-cuda-11.0.3.txt').read_bytes()
VERIFIED_ARCHES = ['35', '37', '50', '52', '53', '60', '61', '62', '70', '72', '75', '80']


def mock_old_nvcc(monkeypatch, help_text=NVCC_HELP):
    def check_output(command, **kwargs):
        if command == ['nvcc', '--list-gpu-arch']:
            raise subprocess.CalledProcessError(
                1, command, output=b"nvcc fatal   : Unknown option '--list-gpu-arch'\n"
            )
        assert command == ['nvcc', '--help']
        return help_text
    monkeypatch.setattr(core.subprocess, 'check_output', check_output)


@pytest.mark.parametrize('cuda_ver', ['11.0', '11.0.3', 11.0])
def test_cuda_11_0_matches_compiler_verified_targets(cuda_ver):
    assert na.get_arches(cuda_ver=cuda_ver, raise_on_error=True) == VERIFIED_ARCHES


@pytest.mark.parametrize('gpu_type,expected', [
    ('all', VERIFIED_ARCHES),
    ('cons', ['35', '50', '52', '60', '61', '70', '75']),
    ('jets', ['53', '62', '72']),
    ('dcen', ['35', '37', '52', '60', '61', '70', '75', '80']),
    ('cons+jets', ['35', '50', '52', '53', '60', '61', '62', '70', '72', '75']),
])
def test_cuda_11_0_detection_and_lookup_agree(monkeypatch, gpu_type, expected):
    mock_old_nvcc(monkeypatch)
    assert na.nvcc_list_arches() == VERIFIED_ARCHES
    assert na.get_arches(gpu_type=gpu_type, raise_on_error=True) == expected
    assert na.get_arches(gpu_type=gpu_type, cuda_ver='11.0') == expected


@pytest.mark.parametrize('cc', ['3.0', '3.2'])
def test_cuda_11_0_rejects_removed_architectures(cc):
    with pytest.raises(ValueError, match='Unsupported architecture'):
        na.validate_arch_string(cc, against_cuda_ver='11.0.3')
    assert na.validate_arch_string('3.5;3.7', against_cuda_ver='11.0.3') == '3.5;3.7'


def test_cuda_11_0_formats_and_ptx(monkeypatch):
    mock_old_nvcc(monkeypatch)
    arches = na.get_arches(return_mode='cc_string', add_ptx=True)
    assert arches == '3.5;3.7;5.0;5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0+PTX'
    flags = na.make_gencode_flags(arches)
    assert flags[0] == '-gencode=arch=compute_35,code=sm_35'
    assert flags[-1] == '-gencode=arch=compute_80,code=[sm_80,compute_80]'
    assert len(flags) == len(VERIFIED_ARCHES)


def test_missing_allowed_values_falls_back_to_version_map(monkeypatch):
    mock_old_nvcc(monkeypatch, b'--gpu-architecture <arch>\nDeprecated: sm_30, sm_32.\n')
    assert na.nvcc_list_arches() is None
    monkeypatch.setattr(core, 'detect_ctk', lambda **kwargs: {'version': '11.0'})
    assert na.get_arches(raise_on_error=True) == VERIFIED_ARCHES
