import json
import re
from pathlib import Path

import pytest

from nvidia_arch import print_summary


FIXTURES = Path(__file__).parent / 'fixtures'
EXPLORER = json.loads((FIXTURES / 'explorer-targets.json').read_text())['versions']


def summary_rows(output):
    return {parts[0]: parts[1:] for line in output.splitlines()
            if re.match(r'^\d+\.\d+\s', line) for parts in [line.split()]}


def test_summary_default_output_is_unchanged(capsys):
    fixture = json.loads((FIXTURES / 'summary-baseline.json').read_text())
    print_summary(fixture['mode'], fixture['min_sm'])
    assert capsys.readouterr().out == fixture['output']
    print_summary(fixture['mode'], fixture['min_sm'], include_a=False, include_f=False)
    assert capsys.readouterr().out == fixture['output']


@pytest.mark.parametrize('mode', ['sm_list', 'cc_list', 'cc_string'])
@pytest.mark.parametrize('include_a,include_f', [(True, False), (False, True), (True, True)])
def test_summary_variants_match_compiler_verified_versions(capsys, mode, include_a, include_f):
    print_summary(mode, include_a=include_a, include_f=include_f)
    output = capsys.readouterr().out
    variants = dict(re.findall(r'^  (\d+\.\d+): (.+)$', output, re.MULTILINE))
    for record in EXPLORER:
        version = '.'.join(record['cuda'].split('.')[:2])
        targets = (record['a'] if include_a else []) + (record['f'] if include_f else [])
        targets.sort(key=lambda target: (int(target[:-1]), target[-1]))
        if not targets:
            assert version not in variants
            continue
        if mode.startswith('cc'):
            targets = [target[:-2] + '.' + target[-2:] for target in targets]
        assert variants[version] == ';'.join(targets)


@pytest.mark.parametrize('mode', ['sm_list', 'cc_list', 'cc_string'])
def test_summary_categories_range_and_numeric_minimum(capsys, mode):
    print_summary(mode, 100, include_a=True, include_f=True)
    output = capsys.readouterr().out
    rows = summary_rows(output)
    assert rows['11.8'] == ['-', '-', '-']
    if mode.startswith('cc'):
        assert rows['12.8'] == ['10.0..12.0', '12.0;12.0a', '10.1;10.1a']
        assert rows['12.9'] == ['10.0..12.1', '12.0;12.0a;12.0f;12.1;12.1a;12.1f', '10.1;10.1a;10.1f']
        assert rows['13.4'] == ['10.0..12.1', '12.0;12.0a;12.0f;12.1;12.1a;12.1f', '11.0;11.0a;11.0f']
        assert '10.7a;10.7f' in output
        assert '9.0a' not in output
    else:
        assert rows['12.8'] == ['100..120', '120;120a', '101;101a']
        assert rows['12.9'] == ['100..121', '120;120a;120f;121;121a;121f', '101;101a;101f']
        assert rows['13.4'] == ['100..121', '120;120a;120f;121;121a;121f', '110;110a;110f']
        assert '107a;107f' in output
        assert '90a' not in output


@pytest.mark.parametrize('include_a,include_f', [(True, False), (False, True), (True, True)])
def test_summary_global_catalog_honors_requested_suffixes(capsys, include_a, include_f):
    print_summary('sm_list', include_a=include_a, include_f=include_f)
    output = capsys.readouterr().out
    catalog = output.split('* All NVIDIA Architectures:\n', 1)[1].splitlines()[0].strip().split(';')
    assert ('107a' in catalog) == include_a
    assert ('107f' in catalog) == include_f
    assert ('90a' in catalog) == include_a
    assert '90f' not in catalog
    assert '107' in catalog


def test_summary_empty_filter_and_keyword_only_options(capsys):
    print_summary(min_sm=999, include_a=True, include_f=True)
    output = capsys.readouterr().out
    assert all(row == ['-', '-', '-'] for row in summary_rows(output).values())
    assert 'None at the selected minimum SM.' in output
    with pytest.raises(TypeError):
        print_summary('cc_string', 30, True, True)
