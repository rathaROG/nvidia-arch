# Copyright (c) 2026 Ratha SIV | Apache-2.0 License

import re
import subprocess
from typing import Optional, Union, List, Dict
from .arches import ALL_ARCHS, TYPE_FILTERS, CUDA_FILTERS

def _run_nvidia_smi(query_args: str) -> Optional[str]:
    """
    Internal helper to run nvidia-smi with specified query arguments and return decoded output.
    Returns None if nvidia-smi isn't found or errors.
    """
    command = [
        "nvidia-smi",
        f"--query-gpu={query_args}",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.check_output(command, stderr=subprocess.STDOUT)
        return result.decode("utf-8").strip()
    except Exception:
        return None

def get_compute_cap(
    return_mode: str = "cc_list"
) -> Optional[Union[List[str], str]]:
    """
    Returns the compute capabilities of detected NVIDIA GPUs (unique, sorted, no duplicates).

    Args:
        return_mode : {'sm_list', 'cc_list', 'cc_string'}, optional
            Output type:
                - 'sm_list': list of SM codes as strings, e.g. ['86', '89', ...]
                - 'cc_list': list of compute capability strings, e.g. ['8.6', '8.9', ...]
                - 'cc_string': semicolon-delimited string, e.g. '8.6;8.9'

    Returns:
        List[str] or str: Requested format based on return_mode, or None if not available.

    Example:
        >>> get_compute_cap()
        ['8.6', '8.9', '12.0']
        >>> get_compute_cap(return_mode='cc_string')
        '8.6;8.9;12.0'
        >>> get_compute_cap(return_mode='sm_list')
        ['86', '89', '120']
    """
    return_mode = str(return_mode).strip().lower()
    output = _run_nvidia_smi("compute_cap")
    if not output:
        return None

    # Parse and deduplicate (preserve order)
    seen = set()
    cc_strs: List[str] = []
    for line in output.splitlines():
        value = line.strip()
        if value.replace('.', '', 1).isdigit() and value not in seen:
            cc_strs.append(value)
            seen.add(value)
    if not cc_strs:
        return None

    if return_mode == "cc_list":
        return cc_strs
    elif return_mode == "cc_string":
        return ";".join(cc_strs)
    elif return_mode == "sm_list":
        sm_codes = [cc.replace('.', '') for cc in cc_strs]  # '8.6' → '86'
        return sm_codes
    else:
        raise ValueError("Invalid return_mode: choose from {'sm_list', 'cc_list', 'cc_string'}")

def find_gpu(extra_query_gpu: Optional[str] = None) -> Optional[List[Dict[str, str]]]:
    """
    Detect attached NVIDIA GPUs and return info for each.

    Parameters
    ----------
    extra_query_gpu : str, optional
        Comma-separated string of additional query fields (e.g., "serial,temperature.gpu").
        Only unique fields not present in the defaults are included.
        By default, only 'name', 'compute_cap', 'memory.total' are included.
        If any query field is invalid, a ValueError is raised suggesting to run
        'nvidia-smi --help-query-gpu' for the full list.

    Returns
    -------
    List[dict] or None
        List of dictionaries, one per GPU.
        Each dict contains:
            - 'name' : str
            - 'compute_cap' : str
            - 'memory.total' : str
            - ...any extra query fields (with units stripped where appropriate)
        Returns None if nvidia-smi is unavailable or no GPUs are found.

    Examples
    --------
    >>> find_gpu()
    [{'name': 'NVIDIA RTX A6000', 'compute_cap': '8.6', 'memory.total': '49140'}]
    >>> find_gpu(extra_query_gpu="serial,temperature.gpu")
    [{'name': 'NVIDIA RTX A6000', 'compute_cap': '8.6', 'memory.total': '49140', 'serial': '1711424012069', 'temperature.gpu': '43'}]

    Notes
    -----
    The returned values for 'memory.total' and 'temperature.gpu' have units stripped. If
    invalid query fields are specified, a ValueError is raised instructing to check
    with `nvidia-smi --help-query-gpu`.
    """
    base_fields = ["name", "compute_cap", "memory.total"]

    extras: List[str] = []
    if extra_query_gpu:
        for f in [field.strip() for field in extra_query_gpu.split(",") if field.strip()]:
            if f not in extras and f not in base_fields:
                extras.append(f)

    fields = base_fields + extras
    query = ",".join(fields)
    try:
        result = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
        )
        output = result.decode("utf-8").strip()
    except subprocess.CalledProcessError:
        raise ValueError(
            f"Some query fields in '{extra_query_gpu}' are not recognized by nvidia-smi.\n"
            "See all available fields by running: nvidia-smi --help-query-gpu"
        )
    except Exception:
        return None

    entries = [line.split(",") for line in output.splitlines() if line.strip()]
    if not entries:
        return None

    results: List[Dict[str, str]] = []
    for values in entries:
        gpu = {}
        for i, field in enumerate(fields):
            if i < len(values):
                value = values[i].strip()
                # Strip units from memory.total and temperature.gpu if present
                if field == "memory.total":
                    value = value.replace(" MiB", "").replace("MB", "").strip()
                elif field == "temperature.gpu":
                    value = value.replace(" C", "").replace("°C", "").strip()
                gpu[field] = value
            else:
                gpu[field] = ""
        results.append(gpu)
    return results

def normalize_cuda_ver(cuda_ver: Optional[Union[str, float, int]]) -> Optional[str]:
    """
    Normalize CUDA version to strict 'major.minor' string format.

    Parameters
    ----------
    cuda_ver : str, float, int or None
        The CUDA version in any of:
        - float (e.g. 12.1)
        - int (e.g. 12)
        - string (e.g. '12.1', '12.0.1', etc)
        - None

    Returns
    -------
    str or None
        Normalized CUDA version string in 'major.minor' form, or None if input is None.

    Raises
    ------
    ValueError
        If the string format is not a valid version.
    TypeError
        If input is not one of str, float, int, or None.
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

def detect_ctk(raise_on_error: bool = False) -> Optional[Dict[str, Optional[str]]]:
    """
    Detect the installed CUDA version and path using nvcc.

    Parameters
    ----------
    raise_on_error : bool, optional
        Whether to raise RuntimeError if CUDA version cannot be detected. Default is False.

    Returns
    -------
    dict or None
        Dictionary with keys:
            - 'version': CUDA version in 'major.minor' format, or None if not detected.
            - 'path'   : CUDA installation path, or None if not detected.
        Returns None if detection fails and raise_on_error is False.

    Raises
    ------
    RuntimeError
        If raise_on_error is True and CUDA version cannot be detected.
    """
    try:
        # Get version
        out = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        match = re.search(r"release (\d+\.\d+)", out)
        version = match.group(1) if match else None

        # Get CUDA path
        if sys.platform.startswith("win"):
            nvcc_path_proc = subprocess.run(["where", "nvcc"], capture_output=True, text=True)
            nvcc_path = nvcc_path_proc.stdout.splitlines()[0] if nvcc_path_proc.stdout else None
        else:
            nvcc_path_proc = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
            nvcc_path = nvcc_path_proc.stdout.strip() if nvcc_path_proc.stdout else None

        cuda_path = os.path.dirname(os.path.dirname(nvcc_path)) if nvcc_path else None

        if version is None or cuda_path is None:
            msg = "CUDA version or path could not be detected. Ensure nvcc is in PATH."
            if raise_on_error:
                raise RuntimeError(msg)
            print(msg)
            return None

        return {"version": version, "path": cuda_path}

    except Exception:
        msg = "CUDA version or path could not be detected. Ensure nvcc is in PATH."
        if raise_on_error:
            raise RuntimeError(msg)
        print(msg)
        return None

def nvcc_list_arches() -> Optional[List[str]]:
    """
    Get list of Compute/SM versions supported by the current nvcc.

    Returns
    -------
    list of str or None
        List of SM version codes as strings, or None if unavailable.
    """
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

def _sm_to_cc(sm: Union[str, int]) -> str:
    """
    Convert SM version string/integer to compute capability string.

    Parameters
    ----------
    sm : str or int
        Streaming multiprocessor (SM) version code, e.g. '75'.

    Returns
    -------
    str
        Compute capability string, e.g. '7.5'.
    """
    sm = int(sm)
    return f"{sm // 10}.{sm % 10}"

def get_architectures(
    gpu_type: str = "all",
    cuda_ver: Optional[Union[str, float, int]] = None,
    min_sm: Optional[Union[str, int]] = None,
    return_mode: str = "sm_list",
    raise_on_error: bool = False
) -> Union[List[str], str]:
    """
    Return the list of GPU architectures (SM versions) supported by the installed
    CUDA Toolkit (CTK) or a manually specified CUDA version.

    Parameters
    ----------
    gpu_type : {'all', 'cons', 'jets', 'cons+jets'}, optional
        Selects which GPU families to include:
        - 'all'         : all supported architectures (default)
        - 'cons'        : consumer/workstation GPUs only
        - 'jets'        : Jetson/embedded GPUs only
        - 'cons+jets'   : union of consumer/workstation and Jetson GPUs
    cuda_ver : str, float, int or None, optional
        CUDA version to use when determining supported architectures.
        If None (default), will auto-detect from installed CTK.
    min_sm : str, int, or None, optional
        Minimum SM number (e.g., 60), filtering to those >= min_sm.
    return_mode : {'sm_list', 'cc_list', 'cc_string'}, optional
        Output type:
        - 'sm_list': list of SM codes as strings ['75', '86', ...]
        - 'cc_list': list of compute capability strings ['7.5', ...]
        - 'cc_string': semicolon-delimited string, e.g. '7.5;8.6'
    raise_on_error : bool, optional
        If True, raise exceptions on invalid versions/gpu_type; else print warning and return [].

    Returns
    -------
    list of str or str
        Available architectures in the chosen mode.

    Raises
    ------
    RuntimeError
        If CUDA version cannot be detected (with raise_on_error True).
    ValueError
        If unknown gpu_type or return_mode (with raise_on_error True).
    """
    gpu_type = str(gpu_type).strip().lower()
    return_mode = str(return_mode).strip().lower()

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
            ctk_info = detect_ctk(raise_on_error=raise_on_error)
            detected = ctk_info["version"] if ctk_info else None
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

    # Step 4: Filter the SM list based on GPU type and minimum SM
    allowed = set(TYPE_FILTERS[gpu_type])
    filtered = [sm for sm in sm_list if sm in allowed]
    if min_sm is not None:
        min_sm = int(min_sm)
        filtered = [sm for sm in filtered if int(sm) >= min_sm]

    # Step 5: Output format
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

def make_gencode_flags(
    arch_input: Union[List[str], str],
    min_sm: Optional[Union[str, int]] = None,
    verify_arch: bool = True
) -> List[str]:
    """
    Convert SM architecture codes or compute capability strings to nvcc -gencode flags.

    Parameters
    ----------
    arch_input : list of str or str
        Either:
        - sm_list: ['86', ...] (list of string SM codes)
        - cc_list: ['8.6', ...] (list of string compute capabilities)
        - cc_string: '8.6;8.9' (semicolon-delimited string)
    min_sm : str or int, optional
        Filter to architectures >= min_sm.
    verify_arch : bool, optional
        If True, ensure all SM codes are in ALL_ARCHS. Default is True.

    Returns
    -------
    list of str
        Flags like ['-gencode=arch=compute_86,code=sm_86', ...].

    Raises
    ------
    ValueError
        For unknown SM/CC code or unrecognized entry.
    """
    if isinstance(arch_input, str):
        arch_list = [item.strip() for item in arch_input.split(";") if item.strip()]
    else:
        arch_list = [str(item).strip() for item in arch_input if str(item).strip()]

    if min_sm is not None:
        min_sm = int(min_sm)
    flags: List[str] = []
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

def print_summary(
    return_mode: str = "cc_string",
    min_sm: Optional[Union[str, int]] = None
) -> None:
    """
    Print a formatted summary of CUDA versions and supported architectures.

    Parameters
    ----------
    return_mode : {'sm_list', 'cc_list', 'cc_string'}, optional
        Format for architectures (default: 'cc_string').
    min_sm : str or int, optional
        Minimum SM number to include.

    Returns
    -------
    None

    Notes
    -----
    Shows a compact summary table with per-version min/max, consumer and jetson architectures.
    """
    return_mode = str(return_mode).strip().lower()

    col_names = [
        "Arch (min..max) ",
        "Consumer/Workstation (cons)",
        "Jetson (jets)"
    ]
    types = ["all", "cons", "jets"]
    versions = sorted(CUDA_FILTERS.keys(), key=lambda x: (int(x.split(".")[0]), int(x.split(".")[1])))

    sep = ";"

    def format_list(l):
        if return_mode.startswith("cc"):
            return sep.join([_sm_to_cc(sm) for sm in l])
        else:
            return sep.join(l)

    # Compute column widths
    col_widths = {
        "all": max(len(col_names[0]), 18),
        "cons": max(len(col_names[1]), 24),
        "jets": max(len(col_names[2]), 16)
    }

    for ver in versions:
        # "all": min..max
        sm_list_all = [sm for sm in CUDA_FILTERS[ver] if sm in TYPE_FILTERS["all"]]
        if min_sm is not None:
            sm_list_all = [sm for sm in sm_list_all if int(sm) >= int(min_sm)]
        if sm_list_all:
            min_val = sm_list_all[0]
            max_val = sm_list_all[-1]
            all_val = (_sm_to_cc(min_val) if return_mode.startswith("cc") else min_val) + ".." + (_sm_to_cc(max_val) if return_mode.startswith("cc") else max_val)
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
            all_val = (_sm_to_cc(min_val) if return_mode.startswith("cc") else min_val) + ".." + (_sm_to_cc(max_val) if return_mode.startswith("cc") else max_val)
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

