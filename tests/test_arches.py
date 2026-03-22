import nvidia_arch as na

CUDA_VERSIONS_STR = sorted(na.CUDA_FILTERS.keys())
CUDA_VERSIONS_FLOAT = [float(v) for v in CUDA_VERSIONS_STR]

def test_arch_88():
    """
    Test get_architectures function to see if it correctly lists the correct architectures.
    Architecture 88 or 8.8 is mysterious; it is not documented or explained anywhere prior 
    to CUDA 13.0, but it appears as supported starting from CUDA 13.0.
    """
    for cuda_ver in CUDA_VERSIONS_FLOAT:

        # cc_string mode
        arches = na.get_architectures(cuda_ver=cuda_ver, return_mode='cc_string')
        if cuda_ver < 13.0 and '8.8' in arches:
            raise AssertionError(f"CUDA {cuda_ver} should not support architecture 8.8, but it is listed in {arches}")
        if cuda_ver >= 13.0 and '8.8' not in arches:
            raise AssertionError(f"CUDA {cuda_ver} should support architecture 8.8, but it is not listed in {arches}")
        
        # sm_list mode
        arches = na.get_architectures(cuda_ver=cuda_ver, return_mode='sm_list')
        if cuda_ver < 13.0 and '88' in arches:
            raise AssertionError(f"CUDA {cuda_ver} should not support architecture 88, but it is listed in {arches}")
        if cuda_ver >= 13.0 and '88' not in arches:
            raise AssertionError(f"CUDA {cuda_ver} should support architecture 88, but it is not listed in {arches}")

    print("test_arch_88 passed successfully for all CUDA versions.")


def test_arch_103():
    """
    Test get_architectures function to see if it correctly lists the correct architectures.
    Architecture 103 or 10.3 is only available from CUDA 12.9 even though CUDA 12.8 already 
    supports architecture 120.
    """
    for cuda_ver in CUDA_VERSIONS_FLOAT:

        # cc_string mode
        arches = na.get_architectures(cuda_ver=cuda_ver, return_mode='cc_string')
        if cuda_ver < 12.9 and '10.3' in arches:
            raise AssertionError(f"CUDA {cuda_ver} should not support architecture 10.3, but it is listed in {arches}")
        if cuda_ver >= 12.9 and '10.3' not in arches:
            raise AssertionError(f"CUDA {cuda_ver} should support architecture 10.3, but it is not listed in {arches}")
        
        # sm_list mode
        arches = na.get_architectures(cuda_ver=cuda_ver, return_mode='sm_list')
        if cuda_ver < 12.9 and '103' in arches:
            raise AssertionError(f"CUDA {cuda_ver} should not support architecture 103, but it is listed in {arches}")
        if cuda_ver >= 12.9 and '103' not in arches:
            raise AssertionError(f"CUDA {cuda_ver} should support architecture 103, but it is not listed in {arches}")

    print("test_arch_103 passed successfully for all CUDA versions.")


if __name__ == "__main__":
    test_arch_88()
