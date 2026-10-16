import argparse
import json
from pathlib import Path

import nvidia_arch as na


def extract_info(probe_report=None):
    cuda_info = na.detect_ctk(raise_on_error=True)
    if cuda_info is None:
        raise RuntimeError('CUDA Toolkit could not be detected')
    cuda_info['architectures'] = na.get_arches(raise_on_error=True)
    targets = na.get_arches(include_a=True, include_f=True, raise_on_error=True)
    cuda_info['architectures_with_variants'] = targets
    # Each group includes baseline targets and their supported variants.
    cuda_info['architectures_by_gpu_type'] = {
        category: na.get_arches(category, include_a=True, include_f=True, raise_on_error=True)
        for category in ('cons', 'jets', 'dcen', 'cons+jets')
    }
    version = cuda_info.get("version", "unknown")
    if probe_report is not None:
        report = json.loads(Path(probe_report).read_text(encoding='utf-8'))
        if not report.get('verified'):
            raise ValueError('Compiler probe did not verify all advertised targets')
        verified = set(report['verified_targets'])
        lookup = na.get_arches(cuda_ver=version, include_a=True, include_f=True, raise_on_error=True)
        if set(targets) != verified or set(lookup) != verified:
            raise ValueError('Package detection or CUDA version table differs from compile-verified targets')
        for category, actual in cuda_info['architectures_by_gpu_type'].items():
            expected = {target for target in verified if target.rstrip('af') in na.TYPE_FILTERS[category]}
            if set(actual) != expected:
                raise ValueError('Package category {} differs from compile-verified targets'.format(category))
    filename = f"info-cuda-{version}.json"
    
    print("\nExtracted CUDA and architecture information: \n")
    print(json.dumps(cuda_info, indent=2))
    print("\nSaving information to file...")
    
    with open(filename, "w") as f:
        json.dump(cuda_info, f, indent=2)
        print(f"Saved CUDA architecture info to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract and optionally verify package architecture support')
    parser.add_argument('--probe-report', type=Path)
    extract_info(parser.parse_args().probe_report)
