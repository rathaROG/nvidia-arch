def test_import_version():
    import nvidia_arch
    assert hasattr(nvidia_arch, '__version__')

def test_print_summary():
    from nvidia_arch import print_summary
    print_summary(min_sm=60)
    print_summary(return_mode='sm_list', min_sm=30)
    print_summary(return_mode='cc_list', min_sm=75)

def test_get_arches_basic():
    from nvidia_arch import get_arches
    result = get_arches()
    assert isinstance(result, list) or isinstance(result, str)
    result = get_arches(cuda_ver='13.0', min_sm=75)
    assert isinstance(result, list)

def test_get_arches_cons_jets():
    from nvidia_arch import get_arches
    result = get_arches(gpu_type='cons', cuda_ver='13.0', min_sm=75, return_mode='cc_string', add_ptx=True)
    assert isinstance(result, str)
    result = get_arches(gpu_type='jets', cuda_ver='12.8', min_sm=60, return_mode='cc_list', add_ptx=True)
    assert isinstance(result, list)
    result = get_arches(gpu_type='dcen', cuda_ver='12.9', min_sm=30, return_mode='cc_list', add_ptx=True)
    assert isinstance(result, list)
    result = get_arches(gpu_type='cons+jets', cuda_ver='13.2', min_sm=75, return_mode='sm_list', add_ptx=True)
    assert isinstance(result, list)

def test_get_arches_make_gencode_flags():
    from nvidia_arch import get_arches, make_gencode_flags
    arches = get_arches(gpu_type='jets', cuda_ver='13.0', min_sm=75)
    flags = make_gencode_flags(arches, add_ptx=True)
    assert isinstance(flags, list)

def test_detect_ctk():
    from nvidia_arch import detect_ctk
    res = detect_ctk()
    # expecting None on non-CTK environments like basic CI, or dict if ctk is installed
    assert res is None or isinstance(res, dict)

def test_find_gpus():
    from nvidia_arch import find_gpus
    # Should not throw even if no GPU is present
    res = find_gpus(extra_query_gpu='serial,temperature.gpu')
    # Expecting None if no GPU is found, or a list of dicts if GPUs are found
    assert res is None or isinstance(res, list)

def test_validate_arch_string_success():
    from nvidia_arch import validate_arch_string
    result = validate_arch_string(
        "6.1+PTX;Pascal;12.0;Lovelace",
        named_arches={"Pascal": "6.0;6.1+PTX", "Lovelace": "8.9+PTX"},
        force_highest_ptx=True,
        against_cuda_ver="12.8"
    )
    assert result == "6.0;6.1;8.9;12.0+PTX"

def test_validate_cc_string_exception():
    from nvidia_arch import validate_arch_string
    import pytest
    with pytest.raises(ValueError) as excinfo:
        validate_arch_string(
            "6.1+PTX;Pascal;12.0;Lovelace;13.5;0.9",
            named_arches={"Pascal": "6.0;6.1+PTX", "Lovelace": "8.9+PTX"},
            force_highest_ptx=True,
            against_cuda_ver="13.2"
        )
    assert "Unknown architecture(s): 0.9, 13.5+PTX" in str(excinfo.value)

if __name__ == "__main__":
    test_import_version()
    test_print_summary()
    test_get_arches_basic()
    test_get_arches_cons_jets()
    test_get_arches_make_gencode_flags()
    test_detect_ctk()
    test_find_gpus()
    test_validate_arch_string_success()
    test_validate_cc_string_exception()
    print("All basic workflow tests passed!")
