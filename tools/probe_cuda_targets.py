"""Collect and compile-check nvcc targets without depending on nvidia_arch.

Uses only the Python 3.8 standard library so it also runs in the CUDA 11.x
Ubuntu 20.04 Explorer containers. No GPU is required.
"""

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def target_key(target):
    match = re.fullmatch(r"(\d+)([a-z]*)", target)
    return int(match.group(1)), match.group(2)


def parse_help_targets(help_text):
    """Read allowed values, excluding historical examples elsewhere in help."""
    targets = {"compute": set(), "sm": set()}
    for option in ("gpu-architecture", "gpu-code"):
        section = re.search(
            r"^--" + option + r"\b[^\n]*\n(.*?)(?=^--|\Z)",
            help_text,
            re.MULTILINE | re.DOTALL,
        )
        if section is None:
            raise ValueError("Missing --{} section in nvcc help".format(option))
        _, separator, allowed = section.group(1).partition("Allowed values for this option:")
        if not separator:
            raise ValueError("Missing allowed values for --{}".format(option))
        for kind, target in re.findall(r"\b(compute|sm)_(\d+[a-z]*)\b", allowed):
            targets[kind].add(target)
    if not all(targets.values()):
        raise ValueError("Could not discover both virtual and real targets")
    return {kind: sorted(values, key=target_key) for kind, values in targets.items()}


def run_command(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        return {
            "command": command,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "returncode": None, "stdout": "", "stderr": str(error)}


def has_code_images(dump_text, target):
    """Require the exact target in separate native and PTX image sections."""
    images = {"elf": False, "ptx": False}
    sections = re.split(r"(?m)^Fatbin (elf|ptx) code:", dump_text)
    for kind, section in zip(sections[1::2], sections[2::2]):
        if not re.search(r"(?m)^arch = sm_" + re.escape(target) + r"\s*$", section):
            continue
        if kind == "elf" or re.search(r"\.target\s+sm_" + re.escape(target) + r"\b", section):
            images[kind] = True
    return images


def probe_target(nvcc, cuobjdump, source, work_dir, output_dir, target):
    fatbin = work_dir / (target + ".fatbin")
    flag = "-gencode=arch=compute_{0},code=[sm_{0},compute_{0}]".format(target)
    compilation = run_command([nvcc, "--fatbin", str(source), "-o", str(fatbin), flag])
    result = {"target": target, "compilation": compilation, "verified": False}
    if compilation["returncode"] == 0 and fatbin.is_file() and fatbin.stat().st_size:
        inspection = run_command([cuobjdump, "--dump-elf", "--dump-ptx", str(fatbin)])
        dump_name = target + ".dump.txt"
        (output_dir / dump_name).write_text(
            inspection["stdout"] + inspection["stderr"], encoding="utf-8"
        )
        images = has_code_images(inspection["stdout"], target)
        result.update({
            "inspection": {key: value for key, value in inspection.items() if key != "stdout"},
            "dump_file": dump_name,
            "code_images": images,
            "fatbin_bytes": fatbin.stat().st_size,
            "verified": inspection["returncode"] == 0 and all(images.values()),
        })
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "commands": {}, "probes": [], "verified": False}
    try:
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            raise RuntimeError("nvcc was not found")
        cuobjdump_path = Path(nvcc).with_name(
            "cuobjdump.exe" if Path(nvcc).suffix.lower() == ".exe" else "cuobjdump"
        )
        if not cuobjdump_path.is_file():
            raise RuntimeError("cuobjdump was not found beside nvcc")
        for name, option in (
            ("version", "--version"), ("help", "--help"),
            ("list-gpu-arch", "--list-gpu-arch"), ("list-gpu-code", "--list-gpu-code"),
        ):
            result = run_command([nvcc, option])
            report["commands"][name] = result
            (output_dir / (name + ".txt")).write_text(
                result["stdout"] + result["stderr"], encoding="utf-8"
            )
        # Older nvcc versions lack the list commands; their allowed help values
        # remain the source for discovery. Modern list commands omit a/f targets.
        for name in ("version", "help"):
            if report["commands"][name]["returncode"] != 0:
                raise RuntimeError("nvcc --{} failed".format(name))
        targets = parse_help_targets(report["commands"]["help"]["stdout"])
        report["advertised_targets"] = targets
        paired = sorted(set(targets["compute"]) & set(targets["sm"]), key=target_key)
        report["unpaired_targets"] = {
            kind: [target for target in values if target not in paired]
            for kind, values in targets.items()
        }
        if not paired:
            raise RuntimeError("No matching compute/SM targets were discovered")
        source_text = (
            'extern "C" __global__ void arch_probe(int *output) {\n'
            '    output[0] = __CUDA_ARCH__;\n'
            '}\n'
        )
        source = output_dir / "probe.cu"
        source.write_text(source_text, encoding="utf-8")
        with tempfile.TemporaryDirectory(prefix="nvidia-arch-probe-") as temporary:
            for target in paired:
                result = probe_target(nvcc, str(cuobjdump_path), source, Path(temporary), output_dir, target)
                report["probes"].append(result)
                print("{}: {}".format(target, "PASS" if result["verified"] else "FAIL"), flush=True)
        report["verified_targets"] = [result["target"] for result in report["probes"] if result["verified"]]
        report["verified"] = (
            all(result["verified"] for result in report["probes"])
            and not any(report["unpaired_targets"].values())
        )
    except (OSError, ValueError, RuntimeError) as error:
        report["error"] = str(error)
        print("Probe failed: {}".format(error), flush=True)
    finally:
        (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
