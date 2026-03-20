# Copyright (c) 2026 Ratha SIV | Apache-2.0 License

import subprocess
import re

from .arches import ALL_ARCHS, TYPE_FILTERS, CUDA_FILTERS


def normalize_cuda_ver(cuda_ver):
    """
    Normalize CUDA version to strict 'major.minor' format.
    """
    if cuda_ver is None:
        return None

    if isinstance(cuda_ver, (float, int)):
        major = int(cuda_ver)
        minor = int(round((cuda_ver - major) * 10))
        return f"{major}.{minor}"

    if isinstance(cuda_ver, str):
        cuda_ver = cuda_ver.strip()
        parts = cuda_ver.split(".")
        if len(parts) == 0:
            raise ValueError(f"Invalid CUDA version: {cuda_ver}")
        major = parts[0]
        minor = parts[1] if len(parts) > 1 else "0"
        if not major.isdigit() or not minor.isdigit():
            raise ValueError(f"Invalid CUDA version: {cuda_ver}")
        minor = str(int(minor))
        return f"{major}.{minor}"

    raise TypeError("cuda_ver must be None, float, int, or string")


def detect_ctk(raise_on_error=False):
    """Detect the installed CUDA version using nvcc and return it as a string (e.g., '12.8')."""
    try:
        out = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        match = re.search(r"release (\d+\.\d+)", out)
        return match.group(1) if match else None
    except Exception:
        msg = "CUDA version could not be detected. Ensure nvcc is in PATH."
        if raise_on_error:
            raise RuntimeError(msg)
        print(msg)
        return None


def nvcc_list_arches():
    """Return list of Compute/SM versions supported by nvcc, or None if unavailable."""
    try:
        out = subprocess.check_output(["nvcc", "--list-gpu-arch"]).decode("utf-8")
        if "sm_" in out:
            arches = re.findall(r"sm_(\d+)", out)
        elif "compute_" in out:
            arches = re.findall(r"compute_(\d+)", out)
        else: 
            return None
        return sorted(set(arches))
    except Exception:
        return None


def _sm_to_cc(sm):
    """Convert SM version string (e.g., '75') to compute capability string (e.g., '7.5')."""
    return f"{int(sm) // 10}.{int(sm) % 10}"


def get_architectures(
    gpu_type="all",
    cuda_ver=None,
    min_sm=None,
    return_mode="sm_list",
    raise_on_error=False
):
    """
    Return the list of GPU architectures (SM versions) supported by the installed
    CUDA Toolkit (CTK) or by a manually specified CUDA version.

    Parameters
    ----------
    gpu_type : {"all", "cons", "jets", "cons+jets"}, optional
        Selects which GPU families to include in the result:

        - "all"        : return all supported architectures
        - "cons"       : consumer/workstation GPUs only
        - "jets"       : Jetson/embedded GPUs only
        - "cons+jets"  : union of consumer/workstation and Jetson GPUs

        The filter is applied after determining the base set of supported
        architectures.

    cuda_ver : str or float or int or None, optional
        CUDA version to use when determining supported architectures.
        If ``None`` (default), the function attempts to query the installed
        CTK using ``nvcc --list-gpu-arch``. If that is unavailable, it falls
        back to parsing ``nvcc --version``.

        Accepted formats include:
            - ``12.8`` (float)
            - ``"12.8"`` (string)
            - ``"12.8.1"`` (string; extra components ignored)
            - ``12`` or ``"12"`` (interpreted as ``"12.0"``)

        The version is normalized to ``"major.minor"`` and matched against
        the internal CUDA capability filter table.

    min_sm : str or int or None
        SM number with two/three digit strings, filtering architectures to 
        those >= min_sm (e.g. 60).

    return_mode : {"sm_list", "cc_list", "cc_string"}, optional
        Controls the output format:

        - "sm_list"   : return SM codes, e.g. ["75", "86", "121"]
        - "cc_list"   : return compute capabilities, e.g. ["7.5", "8.6", "12.1"]
        - "cc_string" : return compute capabilities as a semicolon-separated
                        string, e.g. "7.5;8.6;12.1"

    raise_on_error : bool, optional
        If ``True``, raise an exception when CUDA detection fails, when an
        unsupported CUDA version is provided, or when ``return_mode`` is not
        recognized. If ``False`` (default), the function prints a warning and
        returns an empty list.

    Returns
    -------
    list of str or str
        Depending on `return_mode`, returns a list of SM codes, a list of compute
        capability strings, or a semicolon-separated compute capability string.

    Notes
    -----
    The function determines supported architectures using the following 
    priority order:

    1. If ``cuda_ver`` is provided, use the internal CUDA capability table.
    2. Otherwise, attempt to query ``nvcc --list-gpu-arch``.
    3. If that fails, fall back to parsing ``nvcc --version`` and using the
       internal CUDA capability table.
    """
    gpu_type = str(gpu_type).strip().lower()
    return_mode = str(return_mode).strip().lower()

    # Step 1: Normalize CUDA version (may be None)
    cuda_ver_norm = normalize_cuda_ver(cuda_ver)

    # Step 2: Lookup arches by CUDA version (from filter); else detect
    if cuda_ver_norm is not None:
        if cuda_ver_norm not in CUDA_FILTERS:
            msg = f"Unsupported CUDA version: {cuda_ver_norm}"
            if raise_on_error:
                raise RuntimeError(msg)
            print(msg)
            return []

        sm_list = CUDA_FILTERS[cuda_ver_norm][:]
    else:
        # Try nvcc --list-gpu-arch, fallback to nvcc --version version map
        sm_list = nvcc_list_arches()
        if sm_list is None:
            detected = detect_ctk(raise_on_error=raise_on_error)
            if detected is None or detected not in CUDA_FILTERS:
                msg = f"Cannot detect a supported CUDA version (detected: {detected})"
                if raise_on_error:
                    raise RuntimeError(msg)
                print(msg)
                return []
            sm_list = CUDA_FILTERS[detected][:]

    # Step 3: Filter by GPU type (using TYPE_FILTERS)
    if gpu_type not in TYPE_FILTERS:
        msg = f"Unknown gpu_type '{gpu_type}'. Valid: {list(TYPE_FILTERS.keys())}"
        if raise_on_error:
            raise ValueError(msg)
        print(msg)
        return []

    # Filter the SM list based on the selected GPU type and minimum SM
    allowed = set(TYPE_FILTERS[gpu_type])
    filtered = [sm for sm in sm_list if sm in allowed]
    if min_sm is not None:
        min_sm = int(min_sm)
        filtered = [sm for sm in filtered if int(sm) >= min_sm]

    # Step 4: Output format
    filtered = sorted(filtered, key=int)
    if return_mode == "sm_list":
        return filtered
    elif return_mode == "cc_list":
        return [_sm_to_cc(sm) for sm in filtered]
    elif return_mode == "cc_string":
        return ";".join(_sm_to_cc(sm) for sm in filtered)
    else:
        msg = f"Unknown return_mode '{return_mode}'. Valid: 'sm_list', 'cc_list', 'cc_string'"
        if raise_on_error:
            raise ValueError(msg)
        print(msg)
        return []


def make_gencode_flags(arch_input, min_sm=None, verify_arch=True):
    """
    Convert SM architecture codes or compute capability strings into nvcc -gencode flags.

    Parameters
    ----------
    arch_input : list[str] or str
        - sm_list: ['86', '121', ...] (list of two/three digit strings)
        - cc_list: ['8.6', '12.1', ...] (list of float strings)
        - cc_string: '8.6;12.1;...' (semicolon-separated string)
    min_sm : str or int or None
        SM number with two/three digit strings, filtering architectures to 
        those >= min_sm (e.g. 60).
    verify_arch : bool, optional (default=True)
        If True, ensure all SM codes are valid and present in ALL_ARCHS.

    Returns
    -------
    list of str
        Each entry is like '-gencode=arch=compute_86,code=sm_86'

    Raises
    ------
    ValueError
        If an architecture is not recognized or is not valid.
    """
    if isinstance(arch_input, str):
        arch_list = [item.strip() for item in arch_input.split(";") if item.strip()]
    else:
        arch_list = [str(item).strip() for item in arch_input if str(item).strip()]

    if min_sm is not None:
        min_sm = int(min_sm)
    flags = []
    for item in arch_list:
        if item.isdigit() and len(item) in (2, 3):
            sm = item
        elif "." in item:
            major, minor = item.split(".")
            sm = f"{int(major)*10 + int(minor)}"
        else:
            raise ValueError(f"Unrecognized architecture string: '{item}'")
        if min_sm is not None and int(sm) < min_sm:
            continue
        if verify_arch and sm not in ALL_ARCHS:
            raise ValueError(f"Invalid SM code '{sm}'.")
        flags.append(f"-gencode=arch=compute_{sm},code=sm_{sm}")
    return flags


def print_summary(return_mode="cc_string", min_sm=None):
    """
    Print a compact, well-aligned summary table of CUDA versions and supported architectures:
    - Arch (min..max) : Only min..max shown.
    - Consumer/Workstation (cons) : Full list shown.
    - Jetson (jets) : Full list shown.

    Filter with min_sm for the minimum SM version to display.

    Parameters
    ----------
    return_mode : {'sm_list', 'cc_list', 'cc_string'}
        Format for architecture output (default: 'cc_string')
    min_sm : str or int or None
        SM number with two/three digit strings, filtering architectures to 
        those >= min_sm (e.g. 60).
    """
    return_mode = str(return_mode).strip().lower()

    col_names = [
        "Arch (min..max) ",
        "Consumer/Workstation (cons)",
        "Jetson (jets)"
    ]
    types = ["all", "cons", "jets"]
    versions = sorted(CUDA_FILTERS.keys(), key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1])))

    def sm_to_cc(sm):
        sm = int(sm)
        return f"{sm // 10}.{sm % 10}"

    sep = ";"  # no space

    # Utility for output conversion
    def format_list(l):
        if return_mode.startswith("cc"):
            return sep.join([sm_to_cc(sm) for sm in l])
        else:
            return sep.join(l)

    # Compute column widths
    col_widths = {
        "all": max(len(col_names[0]), 18),
        "cons": max(len(col_names[1]), 24),
        "jets": max(len(col_names[2]), 16)
    }
    # Preprocess width
    for ver in versions:
        # "all": only min..max
        sm_list_all = [sm for sm in CUDA_FILTERS[ver] if sm in TYPE_FILTERS["all"]]
        if min_sm is not None:
            sm_list_all = [sm for sm in sm_list_all if int(sm) >= int(min_sm)]
        if sm_list_all:
            min_val = sm_list_all[0]
            max_val = sm_list_all[-1]
            all_val = (sm_to_cc(min_val) if return_mode.startswith("cc") else min_val) + ".." + (sm_to_cc(max_val) if return_mode.startswith("cc") else max_val)
        else:
            all_val = "-"
        col_widths["all"] = max(col_widths["all"], len(all_val) + 2)

        # "cons"
        sm_list_cons = [sm for sm in CUDA_FILTERS[ver] if sm in TYPE_FILTERS["cons"]]
        if min_sm is not None:
            sm_list_cons = [sm for sm in sm_list_cons if int(sm) >= int(min_sm)]
        cons_val = format_list(sm_list_cons) if sm_list_cons else "-"
        col_widths["cons"] = max(col_widths["cons"], len(cons_val) + 2)

        # "jets"
        sm_list_jets = [sm for sm in CUDA_FILTERS[ver] if sm in TYPE_FILTERS["jets"]]
        if min_sm is not None:
            sm_list_jets = [sm for sm in sm_list_jets if int(sm) >= int(min_sm)]
        jets_val = format_list(sm_list_jets) if sm_list_jets else "-"
        col_widths["jets"] = max(col_widths["jets"], len(jets_val) + 2)

    cuda_width = 6
    # Header
    header = (
        "CUDA".ljust(cuda_width)
        + col_names[0].ljust(col_widths["all"])
        + col_names[1].ljust(col_widths["cons"])
        + col_names[2].ljust(col_widths["jets"])
    )
    print(header)
    print("=" * (cuda_width + sum(col_widths.values())))

    # Rows
    for ver in versions:
        row = ver.ljust(cuda_width)
        # "all": min..max
        sm_list_all = [sm for sm in CUDA_FILTERS[ver] if sm in TYPE_FILTERS["all"]]
        if min_sm is not None:
            sm_list_all = [sm for sm in sm_list_all if int(sm) >= int(min_sm)]
        if sm_list_all:
            min_val = sm_list_all[0]
            max_val = sm_list_all[-1]
            all_val = (sm_to_cc(min_val) if return_mode.startswith("cc") else min_val) + ".." + (sm_to_cc(max_val) if return_mode.startswith("cc") else max_val)
        else:
            all_val = "-"
        row += all_val.ljust(col_widths["all"])

        # cons
        sm_list_cons = [sm for sm in CUDA_FILTERS[ver] if sm in TYPE_FILTERS["cons"]]
        if min_sm is not None:
            sm_list_cons = [sm for sm in sm_list_cons if int(sm) >= int(min_sm)]
        cons_val = format_list(sm_list_cons) if sm_list_cons else "-"
        row += cons_val.ljust(col_widths["cons"])

        # jets
        sm_list_jets = [sm for sm in CUDA_FILTERS[ver] if sm in TYPE_FILTERS["jets"]]
        if min_sm is not None:
            sm_list_jets = [sm for sm in sm_list_jets if int(sm) >= int(min_sm)]
        jets_val = format_list(sm_list_jets) if sm_list_jets else "-"
        row += jets_val.ljust(col_widths["jets"])

        print(row)
    print("=" * (cuda_width + sum(col_widths.values())))
    
    # Footnote: all archs filtered by min_sm, returned in current mode
    all_archs = [sm for sm in ALL_ARCHS if min_sm is None or int(sm) >= int(min_sm)]
    footnote = format_list(all_archs)
    print(f"* All NVIDIA Architectures:\n{footnote}")

