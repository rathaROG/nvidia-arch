import json
from pathlib import Path

import pytest

import nvidia_arch.core as core
from tools.extract_cuda_arch_info import extract_info


FIXTURES = Path(__file__).parent / 'fixtures'
RECORD = next(record for record in json.loads((FIXTURES / 'explorer-targets.json').read_text())['versions']
              if record['cuda'] == '12.9.2')


@pytest.fixture
def compiler_report(monkeypatch, tmp_path):
    def check_output(command, **kwargs):
        if command[-1] == '--list-gpu-arch':
            return '\n'.join('compute_' + target for target in RECORD['baseline']).encode()
        assert command[-1] == '--help'
        return (FIXTURES / 'nvcc-help-cuda-12.9.2.txt').read_bytes()

    monkeypatch.setattr(core.subprocess, 'check_output', check_output)
    monkeypatch.setattr('tools.extract_cuda_arch_info.na.detect_ctk', lambda **kwargs: {'version': '12.9'})
    monkeypatch.chdir(tmp_path)
    report = tmp_path / 'results.json'
    report.write_text(json.dumps({'verified': True, 'verified_targets': RECORD['baseline'] + RECORD['a'] + RECORD['f']}))
    return report


def test_explorer_records_variants_and_categories(compiler_report):
    extract_info(compiler_report)
    info = json.loads(Path('info-cuda-12.9.json').read_text())
    assert info['architectures'] == RECORD['baseline']
    assert set(info['architectures_with_variants']) == set(RECORD['baseline'] + RECORD['a'] + RECORD['f'])
    assert info['architectures_by_gpu_type']['jets'] == ['53', '62', '72', '87', '101', '101a', '101f']


@pytest.mark.parametrize('failure', ['probe_failed', 'probe_mismatch', 'version_table_mismatch', 'category_mismatch'])
def test_explorer_rejects_package_or_probe_mismatches(compiler_report, monkeypatch, failure):
    report = json.loads(compiler_report.read_text())
    if failure == 'probe_failed':
        report['verified'] = False
    elif failure == 'probe_mismatch':
        report['verified_targets'].remove('100f')
    elif failure == 'version_table_mismatch':
        monkeypatch.setitem(core.CUDA_VARIANTS, '12.9', [])
    else:
        import tools.extract_cuda_arch_info as extractor
        original = extractor.na.get_arches

        def missing_jetson_variant(*args, **kwargs):
            targets = original(*args, **kwargs)
            return [target for target in targets if target != '101f'] if args == ('jets',) else targets

        monkeypatch.setattr(extractor.na, 'get_arches', missing_jetson_variant)
    compiler_report.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='probe|differs'):
        extract_info(compiler_report)
    assert not Path('info-cuda-12.9.json').exists()
