import pytest
from nvidia_arch import get_arches, make_gencode_flags, validate_arch_string, print_summary, detect_ctk, find_gpus, normalize_arches, normalize_cuda_ver

# ---- Utility & integration tests ----

def test_import_version():
    import nvidia_arch
    assert hasattr(nvidia_arch, '__version__')

def test_print_summary_runs():
    # We only check that it runs, no exceptions thrown
    print_summary(min_sm=60)
    print_summary(return_mode='sm_list', min_sm=30)
    print_summary(return_mode='cc_list', min_sm=75)

def test_detect_ctk_runs():
    # Should return None or dict
    res = detect_ctk()
    assert res is None or isinstance(res, dict)

def test_find_gpus_runs():
    # Should not throw even if no GPU is present
    res = find_gpus(extra_query_gpu='serial,temperature.gpu')
    assert res is None or isinstance(res, list)

# ---- get_arches tests ----

@pytest.mark.parametrize("kwargs,expected", [
    ({}, list),  # Type check
    ({"cuda_ver": '13.0', "min_sm": 75}, ['75', '80', '86', '87', '88', '89', '90', '100', '103', '110', '120', '121']),
])
def test_get_arches_basic(kwargs, expected):
    result = get_arches(**kwargs)
    if isinstance(expected, type):
        assert isinstance(result, expected) or isinstance(result, str)
    else:
        assert result == expected

@pytest.mark.parametrize("kwargs,expected", [
    ({"gpu_type": 'cons', "cuda_ver": '13.0', "min_sm": 75, "return_mode": 'cc_string', "add_ptx": True}, '7.5;8.6;8.9;12.0;12.1+PTX'),
    ({"gpu_type": 'jets', "cuda_ver": '12.8', "min_sm": 60, "return_mode": 'cc_list', "add_ptx": True}, ['6.2', '7.2', '8.7', '10.1+PTX']),
    ({"gpu_type": 'dcen', "cuda_ver": '12.9', "min_sm": 30, "return_mode": 'cc_list', "add_ptx": True}, ['5.2', '6.0', '6.1', '7.0', '7.5', '8.0', '8.6', '8.9', '9.0', '10.0', '10.3', '12.0+PTX']),
    ({"gpu_type": 'cons+jets', "cuda_ver": '13.2', "min_sm": 75, "return_mode": 'sm_list', "add_ptx": True}, ['75', '86', '87', '89', '110', '120', '121+PTX']),
])
def test_get_arches_full(kwargs, expected):
    result = get_arches(**kwargs)
    assert result == expected

# ---- make_gencode_flags test ----

@pytest.mark.parametrize("arches,expected", [
    (['87', '110'], ['-gencode=arch=compute_87,code=sm_87', '-gencode=arch=compute_110,code=[sm_110,compute_110]']),
])
def test_make_gencode_flags(arches, expected):
    flags = make_gencode_flags(arches, add_ptx=True)
    assert flags == expected

# ---- validate_arch_string tests ----

def test_validate_arch_string_success():
    result = validate_arch_string(
        "6.1+PTX;Pascal;12.0;Lovelace",
        named_arches={"Pascal": "6.0;6.1+PTX", "Lovelace": "8.9+PTX"},
        force_highest_ptx=True,
        against_cuda_ver="12.8"
    )
    assert result == "6.0;6.1;8.9;12.0+PTX"

def test_validate_arch_string_exception():
    with pytest.raises(ValueError) as excinfo:
        validate_arch_string(
            "6.1+PTX;Pascal;12.0;Lovelace;13.5;0.9",
            named_arches={"Pascal": "6.0;6.1+PTX", "Lovelace": "8.9+PTX"},
            force_highest_ptx=True,
            against_cuda_ver="13.2"
        )
    assert "Unknown architecture(s): 0.9, 13.5+PTX" in str(excinfo.value)

# ---- normalize_arches tests ----

@pytest.mark.parametrize("input_arches,exclude,return_mode,expected", [
    (['75', '86', '89+PTX'], '8.6;120', 'sm_list', ['75', '89+PTX']),
    (['7.5', '8.6', '8.9+PTX'], ['86', '89'], 'cc_string', '7.5'),
    ('7.5;8.6;8.9+PTX', '8.6', 'cc_list', ['7.5', '8.9+PTX']),
    ('7.5 8.6 8.9+PTX', ['8.6'], 'sm_list', ['75', '89+PTX']),
    ('7.5;8.6;8.9+PTX', ['8.6', '12.1'], 'cc_string', '7.5;8.9+PTX'),
    ('7.5,8.6,8.9+PTX', ['8.6', '12.1'], 'cc_string', '7.5;8.9+PTX'),
    (['8.6', '8.9+PTX', '12.1'], None, 'cc_string', '8.6;8.9+PTX;12.1'),
    (['86', '89+PTX', '121'], None, 'cc_string', '8.6;8.9+PTX;12.1'),
    (['75', '86', '89+PTX'], None, 'cc_list', ['7.5', '8.6', '8.9+PTX']),
    ('7.5;8.6;8.9+PTX', None, 'cc_list', ['7.5', '8.6', '8.9+PTX']),
    (['75', '86'], '7.5;8.6', 'sm_list', []),
])
def test_normalize_arches(input_arches, exclude, return_mode, expected):
    result = normalize_arches(input_arches, exclude, return_mode)
    assert result == expected

# ---- normalize_cuda_ver tests ----

@pytest.mark.parametrize("inp,force_full_minor,expected", [
    # Basic numerics
    (12, False, "12.0"),
    (12, True, "12.0"),
    (12.1, False, "12.1"),
    (12.19, False, "12.1"),
    (12.19, True, "12.19"),
    (12.199, True, "12.199"),
    (13.01, False, "13.0"),
    (13.01, True, "13.1"),
    ("12.199", False, "12.1"),
    ("12.199", True, "12.199"),
    # Strings
    ("13", False, "13.0"),
    ("13", True, "13.0"),
    ("13.11", False, "13.1"),
    ("13.11", True, "13.11"),
    ("  13.19  ", False, "13.1"),
    ("  13.19  ", True, "13.19"),
    ("12.0.5", False, "12.0"),  # Patch ignored
    ("12.0.5", True, "12.0"),
    ("13.", False, "13.0"),     # Trailing dot
    ("13.", True, "13.0"),
    ("", False, None),          # Empty string
    ("   ", False, None),       # Whitespace string
    ("\t \n", False, None),     # Mixed whitespace (tab, space, newline)
    (None, False, None),        # None input
])
def test_normalize_cuda_ver_good_cases(inp, force_full_minor, expected):
    assert normalize_cuda_ver(inp, force_full_minor) == expected

@pytest.mark.parametrize("inp", [
    "foo",          # not a number
    "12.bar",       # non-digit minor
    "x.y",          # both wrong
    "-12.1",        # minus not allowed (isdigit is False)
    "12.-1",        # minus not allowed (isdigit is False)
    object(),       # invalid type
    [12, 1],        # invalid type
])
def test_normalize_cuda_ver_invalid(inp):
    if isinstance(inp, str) or isinstance(inp, (float, int)) or inp is None:
        with pytest.raises(ValueError):
            normalize_cuda_ver(inp)
    else:
        with pytest.raises(TypeError):
            normalize_cuda_ver(inp)
