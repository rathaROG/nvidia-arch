import json
from unittest.mock import patch

import pytest

from tools.probe_cuda_targets import has_code_images, main, parse_help_targets, probe_target


HELP = """--gpu-architecture <arch> (-arch)
        Historical example: compute_35 and sm_35.
        Allowed values for this option: 'compute_90','compute_90a',
        'compute_100','compute_100a','compute_100f','sm_90','sm_90a',
        'sm_100','sm_100a','sm_100f','lto_100f','native'.
--gpu-code <code> (-code)
        Example: compute_60 is incompatible with sm_52.
        Allowed values for this option: 'compute_90','compute_90a',
        'compute_100','compute_100a','compute_100f','sm_90','sm_90a',
        'sm_100','sm_100a','sm_100f'.
--generate-code <specification> (-gencode)
        Unrelated example: sm_70.
"""


def test_help_discovery_preserves_suffixes_and_ignores_examples():
    expected = ['90', '90a', '100', '100a', '100f']
    assert parse_help_targets(HELP) == {'compute': expected, 'sm': expected}


def test_exact_native_and_ptx_images_are_required():
    dump = 'Fatbin elf code:\narch = sm_100a\nFatbin ptx code:\narch = sm_100a\n.target sm_100a\n'
    assert has_code_images(dump, '100a') == {'elf': True, 'ptx': True}
    assert has_code_images(dump, '100') == {'elf': False, 'ptx': False}
    assert has_code_images(dump.replace('.target sm_100a', '.target sm_100'), '100a') == {'elf': True, 'ptx': False}


def test_compiler_failure_is_not_reported_as_verified(tmp_path):
    failure = {'command': [], 'returncode': 1, 'stdout': '', 'stderr': 'unsupported architecture'}
    with patch('tools.probe_cuda_targets.run_command', return_value=failure) as run:
        result = probe_target('nvcc', 'cuobjdump', tmp_path / 'probe.cu', tmp_path, tmp_path, '100f')
    assert not result['verified']
    assert result['compilation']['stderr'] == 'unsupported architecture'
    run.assert_called_once()


@pytest.mark.parametrize('nvcc_name,dump_name', [('nvcc', 'cuobjdump'), ('nvcc.EXE', 'cuobjdump.exe')])
def test_old_nvcc_list_commands_can_fail_and_partial_results_are_saved(tmp_path, nvcc_name, dump_name):
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    (bin_dir / dump_name).touch()

    def run(command):
        option = command[-1]
        return {
            'command': command,
            'returncode': 1 if option.startswith('--list-') else 0,
            'stdout': HELP if option == '--help' else 'CUDA test compiler',
            'stderr': '',
        }

    def probe(nvcc, cuobjdump, source, work_dir, output_dir, target):
        return {'target': target, 'verified': target != '100f'}

    output_dir = tmp_path / 'results'
    with patch('tools.probe_cuda_targets.shutil.which', return_value=str(bin_dir / nvcc_name)), \
         patch('tools.probe_cuda_targets.run_command', side_effect=run), \
         patch('tools.probe_cuda_targets.probe_target', side_effect=probe):
        assert main(['--output-dir', str(output_dir)]) == 1
    report = json.loads((output_dir / 'results.json').read_text())
    assert report['verified_targets'] == ['90', '90a', '100', '100a']
    assert len(report['probes']) == 5
    assert not report['verified']
    assert (output_dir / 'help.txt').exists()


def test_missing_compiler_still_writes_failure_report(tmp_path):
    with patch('tools.probe_cuda_targets.shutil.which', return_value=None):
        assert main(['--output-dir', str(tmp_path)]) == 1
    report = json.loads((tmp_path / 'results.json').read_text())
    assert report['error'] == 'nvcc was not found'
    assert not report['verified']
