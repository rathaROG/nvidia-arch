import json
import subprocess
from pathlib import Path

import pytest

import nvidia_arch as na
import nvidia_arch.core as core


FIXTURES = Path(__file__).parent / 'fixtures'
EXPLORER = json.loads((FIXTURES / 'explorer-targets.json').read_text())['versions']


def target_key(target):
    return int(target.rstrip('af')), target[len(target.rstrip('af')):]


@pytest.mark.parametrize('record', EXPLORER, ids=lambda record: record['cuda'])
@pytest.mark.parametrize('include_a,include_f', [(False, False), (True, False), (False, True), (True, True)])
def test_targets_match_compiler_verified_explorer(record, include_a, include_f):
    expected = record['baseline'] + (record['a'] if include_a else []) + (record['f'] if include_f else [])
    expected = sorted(expected, key=target_key)
    targets = na.get_arches(cuda_ver=record['cuda'], include_a=include_a, include_f=include_f, raise_on_error=True)
    assert targets == expected
    cc_string = na.normalize_arches(targets, return_mode='cc_string')
    assert na.validate_arch_string(cc_string, against_cuda_ver=record['cuda']) == cc_string
    assert na.make_gencode_flags(targets, against_cuda_ver=record['cuda']) == [
        '-gencode=arch=compute_{0},code=sm_{0}'.format(target) for target in expected
    ]


@pytest.mark.parametrize('version', list(na.CUDA_FILTERS))
@pytest.mark.parametrize('category', ['all', 'cons', 'jets', 'dcen', 'cons+jets'])
def test_variant_category_inheritance_and_baseline_defaults(version, category):
    baseline = na.get_arches(category, version, None, 'sm_list', False, True)
    all_targets = na.get_arches(cuda_ver=version, include_a=True, include_f=True)
    expected = [target for target in all_targets if target.rstrip('af') in baseline]
    actual = na.get_arches(category, version, include_a=True, include_f=True)
    assert actual == expected
    assert [target for target in actual if target.isdigit()] == baseline
    assert na.get_arches(category, version, include_a=False, include_f=False) == baseline


@pytest.mark.parametrize('version,category,variants', [
    ('11.8', 'all', []),
    ('12.0', 'dcen', ['90a']),
    ('12.6', 'dcen', ['90a']),
    ('12.8', 'cons', ['120a']),
    ('12.8', 'jets', ['101a']),
    ('12.8', 'dcen', ['90a', '100a', '120a']),
    ('12.9', 'jets', ['101a', '101f']),
    ('13.0', 'jets', ['110a', '110f']),
    ('13.3', 'dcen', ['90a', '100a', '100f', '103a', '103f', '120a', '120f']),
    ('13.4', 'dcen', ['90a', '100a', '100f', '103a', '103f', '107a', '107f', '120a', '120f']),
    ('13.4', 'cons', ['120a', '120f', '121a', '121f']),
    ('13.4', 'cons+jets', ['110a', '110f', '120a', '120f', '121a', '121f']),
])
def test_category_and_version_boundaries(version, category, variants):
    result = na.get_arches(category, version, include_a=True, include_f=True)
    assert [target for target in result if not target.isdigit()] == variants


@pytest.mark.parametrize('version,target', [
    ('11.8', '9.0a'), ('12.6', '10.0a'), ('12.8', '10.0f'),
    ('12.8', '10.3a'), ('12.8', '12.1a'), ('12.9', '11.0a'),
    ('12.9', '11.0f'), ('13.0', '10.1a'), ('13.0', '10.1f'),
    ('13.3', '10.7a'), ('13.3', '10.7f'), ('13.4', '9.0f'),
    ('13.4', '8.8a'), ('13.4', '10.0m'), ('13.4', '10.0s'),
])
def test_invalid_or_unsupported_targets_are_rejected(version, target):
    with pytest.raises(ValueError):
        na.validate_arch_string(target + '+PTX', against_cuda_ver=version)
    with pytest.raises(ValueError):
        na.make_gencode_flags(target, against_cuda_ver=version)


@pytest.mark.parametrize('version', ['10.2', '12.7', '13.5', '12.99', '13.10', '13.11'])
def test_unknown_version_is_not_assigned_variants(version):
    with pytest.raises(RuntimeError, match='Unsupported CUDA version'):
        na.get_arches(cuda_ver=version, include_a=True, include_f=True, raise_on_error=True)
    assert na.get_arches(cuda_ver=version, include_a=True) == []
    with pytest.raises(ValueError, match='Unknown or unsupported CUDA version'):
        na.validate_arch_string('10.0a', against_cuda_ver=version)
    with pytest.raises(ValueError, match='Unknown or unsupported CUDA version'):
        na.make_gencode_flags('10.0a', against_cuda_ver=version)


def test_version_check_is_explicit_for_offline_flag_generation():
    assert na.make_gencode_flags('10.0f') == ['-gencode=arch=compute_100f,code=sm_100f']
    with pytest.raises(ValueError, match='Unsupported architecture'):
        na.make_gencode_flags('10.0f', verify_arch=False, against_cuda_ver='12.8')


@pytest.mark.parametrize('mode,expected', [
    ('sm_list', ['110+PTX', '110a', '110f']),
    ('cc_list', ['11.0+PTX', '11.0a', '11.0f']),
    ('cc_string', '11.0+PTX;11.0a;11.0f'),
])
def test_output_formats_minimum_and_baseline_ptx(mode, expected):
    assert na.get_arches('jets', '13.4', 100, mode, True, True, include_a=True, include_f=True) == expected


def test_normalization_conversion_and_exact_exclusions():
    assert na.normalize_arch_string('10.0a + PTX, 100f + PTX') == '10.0a+PTX;100f+PTX'
    assert na.normalize_arches('10.0;10.0a + PTX;10.0f', return_mode='sm_list') == ['100', '100a+PTX', '100f']
    assert na.normalize_arches(['100', '100a+PTX', '100f'], return_mode='cc_string') == '10.0;10.0a+PTX;10.0f'
    assert na.normalize_arches('10.0;10.0a+PTX;10.0f', exclude='100a', return_mode='sm_list') == ['100', '100f']
    assert na.normalize_arches(['100', '100a', '100f+PTX'], exclude='10.0f', return_mode='cc_list') == ['10.0', '10.0a']
    assert na.normalize_arches('10.0;10.0a;10.0f', exclude='10.0') == ['100a', '100f']
    assert na.normalize_arches(['100A + ptx', '100F'], return_mode='cc_string') == '10.0a+PTX;10.0f'
    assert na.normalize_arches(['100A + ptx', '100F'], return_mode='sm_list') == ['100a+PTX', '100f']


def test_case_spacing_names_and_validation_sorting():
    assert na.validate_arch_string(
        'Blackwell;9.0a;10.0F + ptx;10.0A', named_arches={'Blackwell': '10.0;10.0a'},
        against_cuda_ver='12.9.2',
    ) == '9.0a;10.0;10.0a;10.0f+PTX'


def test_flag_generation_preserves_distinct_targets_and_explicit_ptx():
    assert na.make_gencode_flags('10.0f;9.0a;10.0a;10.0+PTX', min_sm=100, against_cuda_ver='12.9') == [
        '-gencode=arch=compute_100,code=[sm_100,compute_100]',
        '-gencode=arch=compute_100a,code=sm_100a',
        '-gencode=arch=compute_100f,code=sm_100f',
    ]
    assert na.make_gencode_flags(['100F + ptx', '10.0a+PTX'], add_ptx=True) == [
        '-gencode=arch=compute_100a,code=[sm_100a,compute_100a]',
        '-gencode=arch=compute_100f,code=[sm_100f,compute_100f]',
    ]


def test_automatic_ptx_stays_on_highest_baseline_even_with_higher_variants():
    flags = na.make_gencode_flags('9.0;10.0a;10.0f', add_ptx=True)
    assert flags == [
        '-gencode=arch=compute_90,code=[sm_90,compute_90]',
        '-gencode=arch=compute_100a,code=sm_100a',
        '-gencode=arch=compute_100f,code=sm_100f',
    ]
    assert na.validate_arch_string(
        '10.0a+PTX;9.0+PTX;9.0;10.0f', force_highest_ptx=True, against_cuda_ver='12.9'
    ) == '9.0+PTX;10.0a;10.0f'
    with pytest.raises(ValueError, match='Automatic PTX requires a baseline'):
        na.make_gencode_flags('10.0a;10.0f', add_ptx=True)
    with pytest.raises(ValueError, match='Automatic PTX requires a baseline'):
        na.validate_arch_string('10.0a;10.0f', force_highest_ptx=True, against_cuda_ver='12.9')


@pytest.mark.parametrize('version', ['12.8.2', '12.9.2'])
@pytest.mark.parametrize('include_a,include_f', [(False, False), (True, False), (False, True), (True, True)])
def test_auto_detection_reads_actual_allowed_values(monkeypatch, version, include_a, include_f):
    record = next(record for record in EXPLORER if record['cuda'] == version)
    calls = []

    def check_output(command, **kwargs):
        calls.append(command[-1])
        if command[-1] == '--list-gpu-arch':
            return '\n'.join('compute_' + target for target in record['baseline']).encode()
        assert command[-1] == '--help'
        return (FIXTURES / ('nvcc-help-cuda-' + version + '.txt')).read_bytes()

    monkeypatch.setattr(core.subprocess, 'check_output', check_output)
    assert na.get_arches(include_a=include_a, include_f=include_f) == na.get_arches(
        cuda_ver=version, include_a=include_a, include_f=include_f,
    )
    assert ('--help' in calls) == (include_a or include_f)


def test_discovery_does_not_strip_suffixes_or_invent_missing_pairs(monkeypatch):
    help_text = b'''--gpu-architecture <arch>
        Historical example: compute_90f sm_90f.
        Allowed values for this option: 'compute_100','compute_100a','compute_100f','sm_100'.
--gpu-code <code>
        Allowed values for this option: 'sm_100','sm_100a','sm_110a'.
'''

    def check_output(command, **kwargs):
        return b'compute_100\ncompute_110a\n' if command[-1] == '--list-gpu-arch' else help_text

    monkeypatch.setattr(core.subprocess, 'check_output', check_output)
    assert na.nvcc_list_arches() == ['100']
    assert na.nvcc_list_arches(include_a=True, include_f=True) == ['100', '100a']


@pytest.mark.parametrize('version', ['11.0', '12.8', '12.9', '13.4'])
def test_failed_help_discovery_uses_version_table(monkeypatch, version):
    def check_output(command, **kwargs):
        if command[-1] == '--list-gpu-arch':
            return b'compute_90\n'
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(core.subprocess, 'check_output', check_output)
    monkeypatch.setattr(core, 'detect_ctk', lambda **kwargs: {'version': version})
    assert na.get_arches(include_a=True, include_f=True) == na.get_arches(
        cuda_ver=version, include_a=True, include_f=True,
    )


def test_variant_options_are_keyword_only_and_aliases_forward_them():
    with pytest.raises(TypeError):
        na.get_arches('all', '12.8', None, 'sm_list', False, True, True)
    with pytest.warns(DeprecationWarning):
        assert na.get_architectures(cuda_ver='12.8', include_a=True) == na.get_arches(cuda_ver='12.8', include_a=True)
    with pytest.warns(DeprecationWarning):
        assert na.validate_cc_string('10.0a', against_cuda_ver='12.8') == '10.0a'


def test_catalogs_and_hardware_queries_remain_baseline_only(monkeypatch):
    assert all(target.isdigit() for target in na.ALL_ARCHS)
    assert all(target.isdigit() for targets in na.TYPE_FILTERS.values() for target in targets)
    assert all(target.isdigit() for targets in na.CUDA_FILTERS.values() for target in targets)
    monkeypatch.setattr(core, '_run_nvidia_smi', lambda _: '10.0\n10.0\n12.0\n')
    assert na.get_compute_caps(add_ptx=True) == ['100', '120+PTX']
