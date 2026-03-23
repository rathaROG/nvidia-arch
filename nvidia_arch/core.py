# Copyright (c) 2026 Ratha SIV | Apache-2.0 License

import os
import re
import sys
import subprocess
from typing import Optional, Union, List, Dict, Tuple, Any
from .arches import ALL_ARCHS, TYPE_FILTERS, CUDA_FILTERS, CUDA_EXCLUDES


PTX_SUFFIX_RE = re.compile(r"\+ptx$", re.IGNORECASE)


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


def get_compute_cap(return_mode: str = "sm_list", add_ptx: bool = False) -> Optional[Union[List[str], str]]:
    """
    Returns the compute capabilities of detected NVIDIA GPUs (unique, sorted, no duplicates).

    PTX emission policy
    -------------------
    If ``add_ptx=True``, PTX is added **only for the highest SM architecture** in the
    filtered/validated list. This matches NVIDIA best practices and official CUDA wheels.

    Parameters
    ----------
    return_mode : {'sm_list', 'cc_list', 'cc_string'}, optional
        Output format:
            - 'sm_list': list of SM codes as strings, e.g. ['86', '89', ...]
            - 'cc_list': list of compute capability strings, e.g. ['8.6', '8.9', ...]
            - 'cc_string': semicolon-delimited string, e.g. '8.6;8.9'
    add_ptx : bool, optional
        If True, add '+PTX' suffix to the highest architecture.

    Returns
    -------
    list of str or str or None
        Requested format based on ``return_mode``, or None if GPUs not found.

    Examples
    --------
    >>> get_compute_cap()
    ['86', '89', '120']
    >>> get_compute_cap(return_mode='cc_list')
    ['8.6', '8.9', '12.0']
    >>> get_compute_cap(return_mode='cc_string', add_ptx=True)
    '8.6;8.9;12.0+PTX'
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

    # Optionally add PTX only to the highest arch
    cc_strs = sorted(cc_strs, key=lambda x: tuple(map(int, x.split('.'))))
    if add_ptx and cc_strs:
        cc_strs[-1] = cc_strs[-1] + "+PTX"

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


def _verify_dir(path: Optional[str]) -> Optional[str]:
    """Return path if it exists as a directory, else None."""
    return path if path and os.path.isdir(path) else None


def detect_ctk(raise_on_error: bool = False) -> Optional[Dict[str, Any]]:
    """
    Detect CUDA Toolkit version, root, include, and lib paths.

    Parameters
    ----------
    raise_on_error : bool, optional
        If True, raise RuntimeError if detection fails. Default: False.

    Returns
    -------
    dict or None
        The detected CUDA environment as a dictionary with keys:
            - 'version' : CUDA version (str or None)
            - 'root'    : CUDA Toolkit root directory (str or None)
            - 'include' : dict, with subpaths:
                - 'root'    : base include directory
                - 'cuda'    : cuda headers directory
                - 'cub'     : cub headers directory
                - 'thrust'  : thrust headers directory
            - 'lib'    : CUDA library path (str or None)
        If detection fails, returns None (or raises if raise_on_error=True).

    Raises
    ------
    RuntimeError
        If raise_on_error is True and detection fails.
    """
    try:
        out = subprocess.check_output(["nvcc", "--version"]).decode("utf-8")
        match = re.search(r"release (\d+\.\d+)", out)
        version = match.group(1) if match else None

        if sys.platform.startswith("win"):
            nvcc_path_proc = subprocess.run(["where", "nvcc"], capture_output=True, text=True)
            nvcc_path = nvcc_path_proc.stdout.splitlines()[0] if nvcc_path_proc.stdout else None
        else:
            nvcc_path_proc = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
            nvcc_path = nvcc_path_proc.stdout.strip() if nvcc_path_proc.stdout else None

        root = os.path.dirname(os.path.dirname(nvcc_path)) if nvcc_path else None
        include_root = os.path.join(root, "include") if root else None

        cuda_include = cub_include = thrust_include = None
        if include_root:
            if version and float(version) >= 13.0:
                cccl_path = os.path.join(include_root, "cccl")
                cuda_include = os.path.join(cccl_path, "cuda")
                cub_include = os.path.join(cccl_path, "cub")
                thrust_include = os.path.join(cccl_path, "thrust")
            else:
                cuda_include = os.path.join(include_root, "cuda")
                cub_include = os.path.join(include_root, "cub")
                thrust_include = os.path.join(include_root, "thrust")

        include_root_verified = _verify_dir(include_root)
        cuda_include = _verify_dir(cuda_include)
        cub_include = _verify_dir(cub_include)
        thrust_include = _verify_dir(thrust_include)

        if sys.platform.startswith("win"):
            lib_path = os.path.join(root, "lib", "x64") if root else None
        else:
            lib_path = os.path.join(root, "lib64") if root else None

        lib_path = lib_path if lib_path and os.path.isdir(lib_path) else None

        if version is None or root is None:
            msg = "CUDA Toolkit version or root directory could not be detected. Ensure nvcc is in PATH."
            if raise_on_error:
                raise RuntimeError(msg)
            print(msg)
            return None

        cuda_info = {
            "version": version,
            "root": root,
            "include": {
                "root": include_root_verified,
                "cuda": cuda_include,
                "cub": cub_include,
                "thrust": thrust_include,
            },
            "lib": lib_path,
        }

        return cuda_info

    except Exception:
        msg = "CUDA Toolkit version or root directory could not be detected. Ensure nvcc is in PATH."
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
    # Try modern CUDA API first (CUDA 11.1+)
    try:
        out = subprocess.check_output(
            ["nvcc", "--list-gpu-arch"],
            stderr=subprocess.STDOUT
        ).decode("utf-8")

        arches = re.findall(r"sm_(\d+)", out) or re.findall(r"compute_(\d+)", out)
        if arches:
            return sorted(set(arches))
    except Exception:
        pass  # Old CUDA → fallback

    # Fallback for CUDA 11.0 and older
    try:
        help_out = subprocess.check_output(
            ["nvcc", "--help"],
            stderr=subprocess.STDOUT
        ).decode("utf-8")

        arches = re.findall(r"sm_(\d+)", help_out)
        if arches:
            return sorted(set(arches))
    except Exception:
        pass

    return None


def _sm_to_cc(sm: Union[str, int]) -> str:
    """
    Convert an SM string (optionally with suffixes like '+PTX') into 
    a compute capability string, preserving all suffixes.

    Examples:
        '89'            -> '8.9'
        '89+PTX'        -> '8.9+PTX'
        '89+PTX+DEBUG'  -> '8.9+PTX+DEBUG'

    Parameters
    ----------
    sm : str or int
        Streaming multiprocessor (SM) version code, e.g. '75'.

    Returns
    -------
    str
        Compute capability string, e.g. '7.5'.
    """
    sm_str = str(sm).strip()
    i = 0
    while i < len(sm_str) and sm_str[i].isdigit():
        i += 1
    if i == 0:
        raise ValueError(f"Invalid SM string (no numeric prefix): '{sm}'")
    numeric = sm_str[:i]
    suffix = sm_str[i:]
    sm_int = int(numeric)
    cc = f"{sm_int // 10}.{sm_int % 10}"
    return cc + suffix


def validate_cc_string(
    cc_string: str,
    named_arches: Optional[Dict[str, str]] = None,
    force_highest_ptx: bool = False,
    against_cuda_ver: Optional[str] = None,
) -> str:
    """
    Validate and normalize a PyTorch-style cc_string, optionally ensuring all architectures are valid
    for a specific CUDA Toolkit version present or specified.

    Parameters
    ----------
    cc_string : str
        Semicolon-delimited architecture string, e.g. '7.5;8.6+PTX;12.0'
    force_highest_ptx : bool, optional
        If True, removes '+PTX' from all entries and add to the numerically highest architecture.
        This matches best practice for CUDA/PTX emission and prevents duplicates.
    named_arches : dict, optional
        Mapping of architecture names to numeric architecture strings, e.g., {'Pascal': '6.0;6.1+PTX'}
    against_cuda_ver : str or None, optional
        If None, will detect the installed CUDA Toolkit version.
        If provided, must be a string or float (e.g. '12.1').
        Architectures will be checked for support in this CUDA version.

    Raises
    ------
    ValueError
        If any architecture is not valid, is not supported by the specified CUDA version,
        or the input string is malformed.

    Returns
    -------
    str
        Normalized cc_string.

    Examples
    --------
    >>> validate_cc_string(
    ...    "6.1+PTX;Pascal;12.0;Lovelace",
    ...    named_arches={"Pascal": "6.0;6.1+PTX", "Lovelace": "8.9+PTX"},
    ...    force_highest_ptx=True,
    ...    against_cuda_ver="12.8"
    ... )
    '6.0;6.1;8.9;12.0+PTX'
    """

    def arch_sort_key(val):
        num = val.replace('+PTX', '').strip()
        major, minor = map(int, num.split('.', 1))
        return (major, minor)

    # Expand named architectures
    s = cc_string
    if named_arches:
        for named_arch, archval in named_arches.items():
            s = re.sub(r"\b{}\b(?!\+PTX)".format(re.escape(named_arch)), archval, s)

    # Flatten and clean entries
    items = [a.strip() for a in s.split(';') if a.strip()]
    clean_items = []
    for item in items:
        norm = item
        m = re.fullmatch(r"(\d+)\.(\d+)(\+PTX)?", norm, flags=re.IGNORECASE)
        if not m:
            raise ValueError(f"Invalid arch entry: '{item}'.")
        clean_items.append(norm)
    
    # Deduplicate and sort
    clean_items = sorted(set(clean_items), key=arch_sort_key)

    # Add PTX only to the highest architecture if force_highest_ptx is True
    if force_highest_ptx:
        items_no_ptx = [re.sub(r'\+PTX$', '', item, flags=re.IGNORECASE) for item in clean_items]
        if items_no_ptx:
            items_no_ptx[-1] += '+PTX'
        clean_items = items_no_ptx

    # Validate against CUDA version filters
    if against_cuda_ver is not None:
        cuda_ver = normalize_cuda_ver(against_cuda_ver)
    else:
        ctk = detect_ctk()
        cuda_ver = ctk["version"] if ctk else None
    if cuda_ver is None:
        raise ValueError("Could not detect a CUDA Toolkit version.")
    if cuda_ver not in CUDA_FILTERS:
        raise ValueError(f"Unknown or unsupported CUDA version '{cuda_ver}'.")

    # Check each cleaned entry against these sets (ignore +PTX for arch validity)
    ALL_ARCHS_CC = [_sm_to_cc(sm) for sm in ALL_ARCHS]
    valid_for_ver = set([_sm_to_cc(sm) for sm in CUDA_FILTERS[cuda_ver]])
    unknown_arch = []
    unsupported_arch = []
    for item in clean_items:
        base_arch = item.replace("+PTX", "")
        if base_arch not in ALL_ARCHS_CC:
            unknown_arch.append(item)
        if base_arch not in valid_for_ver:
            unsupported_arch.append(item)
    if unknown_arch:
        raise ValueError(f"Unknown architecture(s): {', '.join(unknown_arch)}. ")
    if unsupported_arch:
        raise ValueError(f"Unsupported architecture(s) for CUDA {cuda_ver}: {', '.join(unsupported_arch)}. ")

    return ';'.join(clean_items)


def get_architectures(
    gpu_type: str = "all",
    cuda_ver: Optional[Union[str, float, int]] = None,
    min_sm: Optional[Union[str, int]] = None,
    return_mode: str = "sm_list",
    add_ptx: bool = False,
    raise_on_error: bool = False
) -> Union[List[str], str]:
    """
    Return the list of GPU architectures (SM versions) supported by the installed
    CUDA Toolkit (CTK) or a manually specified CUDA version.

    PTX emission policy:
    --------------------
    If add_ptx=True, PTX is added **only for the highest SM architecture** in the
    filtered/validated list. This follows NVIDIA best practices and matches the
    strategy used by PyTorch, TensorFlow, and official CUDA wheels.

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
    add_ptx : bool, optional
        If True, include PTX for the highest SM architecture.
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

    # Lookup arches by CUDA version (from filter); else detect
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

    # Filter by GPU type (using TYPE_FILTERS)
    if gpu_type not in TYPE_FILTERS:
        msg = f"Unknown gpu_type '{gpu_type}'. Valid: {list(TYPE_FILTERS.keys())}"
        if raise_on_error:
            raise ValueError(msg)
        print(msg)
        return []

    # Filter the SM list based on GPU type and minimum SM
    allowed = set(TYPE_FILTERS[gpu_type])
    filtered = [sm for sm in sm_list if sm in allowed]
    if min_sm is not None:
        min_sm = int(min_sm)
        filtered = [sm for sm in filtered if int(sm) >= min_sm]

    # Output format + PTX
    filtered = sorted(filtered, key=int)
    if add_ptx and filtered:
        filtered[-1] = filtered[-1] + "+PTX"
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
    verify_arch: bool = True,
    add_ptx: bool = False,
) -> List[str]:
    """
    Convert SM architecture codes or compute capability strings to nvcc -gencode flags.

    PTX emission policy:
    --------------------
    (1) If add_ptx=True, PTX is added **only for the highest SM architecture** in the
    filtered/validated list. This follows NVIDIA best practices and matches the
    strategy used by PyTorch, TensorFlow, and official CUDA wheels.

    (2) If any architecture string ends with '+PTX', only those architectures receive 
    PTX code generation. In this case, the global add_ptx flag is ignored.

    Parameters
    ----------
    arch_input : list of str or str
        Either:
        - sm_list: ['86', '89', ...] (list of SM codes)
        - cc_list: ['8.6', '8.9', ...] (list of compute capabilities)
        - cc_string: '8.6;8.9+PTX' (semicolon-delimited string)
    min_sm : str or int, optional
        Filter to architectures >= min_sm.
    verify_arch : bool, optional
        If True, ensure all SM codes are in ALL_ARCHS. Default is True.
    add_ptx : bool, optional
        If True, include PTX only for the highest SM architecture.
        Example: arch_input='8.6;8.9', add_pt=True
            sm_86 → -gencode=arch=compute_86,code=sm_86
            sm_89 → -gencode=arch=compute_89,code=[sm_89,compute_89]  (highest)

    Returns
    -------
    list of str
        Flags like:
            ['-gencode=arch=compute_86,code=sm_86',
             '-gencode=arch=compute_89,code=[sm_89,compute_89]']

    Raises
    ------
    ValueError
        For unknown SM/CC code or unrecognized entry.
    """
        # Normalize input into a list of strings
    if isinstance(arch_input, str):
        raw_list = [item.strip() for item in arch_input.split(";") if item.strip()]
    else:
        raw_list = [str(item).strip() for item in arch_input if str(item).strip()]

    # Convert min_sm to int if provided
    if min_sm is not None:
        min_sm = int(min_sm)

    parsed = []  # list of (sm: str, wants_ptx: bool)
    for item in raw_list:
        # Detect +PTX suffix
        wants_ptx = bool(PTX_SUFFIX_RE.search(item))
        clean = PTX_SUFFIX_RE.sub("", item).strip()
        # Parse numeric part
        if clean.isdigit() and len(clean) in (2, 3):
            sm = clean
        elif "." in clean:
            major, minor = clean.split(".")
            sm = f"{int(major) * 10 + int(minor)}"
        else:
            raise ValueError(f"Unrecognized architecture string: '{item}'")
        # Apply min_sm filter
        if min_sm is not None and int(sm) < min_sm:
            continue
        # Validate
        if verify_arch and sm not in ALL_ARCHS:
            raise ValueError(f"Invalid SM code '{sm}'.")
        parsed.append((sm, wants_ptx))

    if not parsed:
        return []

    # Sort by SM
    parsed.sort(key=lambda x: int(x[0]))
    sms = [sm for sm, _ in parsed]

    # Determine PTX policy
    any_explicit_ptx = any(wants for _, wants in parsed)
    highest = sms[-1]

    flags = []
    for sm, wants_ptx in parsed:
        emit_ptx = False
        if wants_ptx:
            emit_ptx = True
        elif not any_explicit_ptx and add_ptx and sm == highest:
            emit_ptx = True
        if emit_ptx:
            code = f"[sm_{sm},compute_{sm}]"
        else:
            code = f"sm_{sm}"
        flags.append(f"-gencode=arch=compute_{sm},code={code}")

    return flags


def _cuda_excludes_footnotes(
    cuda_excludes: Dict[Tuple[float, float], List[Any]],
    return_mode: str = "cc_string"
) -> List[str]:
    """
    Generate explanatory footnotes regarding architectures that are excluded from
    support for specific CUDA version ranges.

    Parameters
    ----------
    cuda_excludes : dict
        Mapping of CUDA version (min, max) tuple to list of architecture numbers (SM codes).
    return_mode : {'sm_list', 'cc_list', 'cc_string'}, optional
        Controls formatting of architecture numbers.
        - If 'cc_' prefix, formats as compute capability (e.g., '8.8').
        - Otherwise, outputs as SM numbers (e.g., '88').

    Returns
    -------
    List[str]
        A list of formatted footnote strings, one per exclusion block.
    """
    # Import _sm_to_cc if it's in core
    from .core import _sm_to_cc  # or define here if not available for import

    notes = []
    for (vmin, vmax), arches in cuda_excludes.items():
        if return_mode.startswith("cc"):
            arch_list = ", ".join(_sm_to_cc(str(a)) for a in arches)
        else:
            arch_list = ", ".join(str(a) for a in arches)
        vmin_str = f"{vmin:.1f}"
        vmax_str = f"{vmax:.1f}"
        notes.append(
            f"Architecture(s) {arch_list} is not officially supported in CUDA {vmin_str}–{vmax_str}."
        )
    return notes


def print_summary(
    return_mode: str = "cc_string",
    min_sm: Optional[Union[str, int]] = None
) -> None:
    """
    Print a formatted summary of CUDA versions and supported architectures.

    Parameters
    ----------
    return_mode : {'sm_list', 'cc_list', 'cc_string'}, optional
        Controls formatting of architecture numbers.
        - If 'cc_' prefix, formats as compute capability (e.g., '8.8').
        - Otherwise, outputs as SM numbers (e.g., '88').
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
    print(f"\n* All NVIDIA Architectures:\n  {footnote}")

    # Footnote: CUDA versions and their unsupported architectures
    notes = _cuda_excludes_footnotes(CUDA_EXCLUDES, return_mode=return_mode)
    if notes:
        print("\n* Other Notes:")
        for idx, note in enumerate(notes, start=1):
            print(f"  {idx}. {note}")
    print("\n")

