import pytest

from nvidia_arch import normalize_arch_string, validate_arch_string, make_gencode_flags

@pytest.mark.parametrize("input_str,expected", [
    ("8.6;8.9+PTX;12.0", "8.6;8.9+PTX;12.0"),
    ("8.6 8.9+PTX 12.0", "8.6;8.9+PTX;12.0"),
    ("8.6,8.9+PTX,12.0", "8.6;8.9+PTX;12.0"),
    ("8.6 ; 8.9+PTX , 12.0", "8.6;8.9+PTX;12.0"),
    ("8.6\t, 8.9+PTX  ;12.0 ", "8.6;8.9+PTX;12.0"),
    ("8.6;;; 8.9+PTX,,,;; 12.0", "8.6;8.9+PTX;12.0"),

    # Common PTX suffix spacing—should auto-fix to canonical
    ("8.6+ PTX 8.9 + PTX 12.0", "8.6+PTX;8.9+PTX;12.0"),
    ("8.6 + PTX ; 8.9+ PTX , 12.0", "8.6+PTX;8.9+PTX;12.0"),
    ("8.6 +PTX,8.9 + PTX ;12.0", "8.6+PTX;8.9+PTX;12.0"),
    (" 8.6\t+\tPTX , 8.9+ PTX  , 12.0 ", "8.6+PTX;8.9+PTX;12.0"),
])
def test_normalize_arch_string(input_str, expected):
    assert normalize_arch_string(input_str) == expected

@pytest.mark.parametrize("input_str,good_count", [
    ("8.6 +PTX 8.9 12.0 +PTX", 3),  # two "+PTX" separated, one plain
    ("8.6+PTX 8.9+PTX", 2),         # two PTX, no spaces
    ("8.6+ PTX,8.9 +PTX", 2),       # both with spaced
])
def test_validate_arch_string_accepts_all(input_str, good_count):
    result = validate_arch_string(input_str, against_cuda_ver="13.2")
    assert result.count(";") == (good_count - 1)
    assert "8.6+PTX" in result
    # No trailing or leading junk
    for x in result.split(";"):
        assert "+" not in x or x.endswith("+PTX")

@pytest.mark.parametrize("input_str", [
    "8.6 +PTX 8.9 12.0 +PTX",
    "8.6+PTX 8.9+PTX",
    "8.6+ PTX,8.9 +PTX",
    "8.6+ PTX, 8.9, 12.0 + PTX",
])
def test_make_gencode_flags_accepts_all(input_str):
    flags = make_gencode_flags(input_str)
    assert any("compute_86" in f for f in flags)
    assert any(",code=[" in f for f in flags)  # This indicates PTX is emitted
    assert len(flags) >= 2
    # All start with -gencode and contain code=
    for f in flags:
        assert f.startswith("-gencode=arch=compute_") and ",code=" in f

@pytest.mark.parametrize("input_str", [
    "", "   ", ";;; , , ,, ;;  "
])
def test_normalize_empty_and_whitespace(input_str):
    assert normalize_arch_string(input_str) == ""

def test_validate_arch_string_invalid_entry():
    # Not auto-fixable
    with pytest.raises(ValueError):
        validate_arch_string("86 8x.9+PTX ; 12.0")  # "8x.9" not valid

    # Malformed + without number
    with pytest.raises(ValueError):
        validate_arch_string("+PTX ; 8.6", against_cuda_ver="13.0")
