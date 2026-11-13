"""Run reproducible AscendC acceptance checks and an independent PTO audit.

Example:
    python scripts/verify_npu.py

By default, select the first idle, healthy device in the inherited allocation
and create a unique directory beside the checkout under torch-tilelang-artifacts.
Explicit options override these defaults. The script does not reserve/reset
devices, change toolchains, or stop other processes. Run stages serially.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def capture(command, cwd=ROOT):
    try:
        run = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=60)
        return {"command": command, "exit_code": run.returncode, "stdout": run.stdout, "stderr": run.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": str(error)}


def select_device(requested, allocated):
    """Choose only a confirmed idle, healthy device inside the allocation."""
    allowed = None
    if allocated is not None:
        try:
            allowed = [int(value.strip()) for value in allocated.split(",")]
        except ValueError:
            raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain comma-separated physical device IDs") from None
        if any(value < 0 for value in allowed):
            raise ValueError("ASCEND_RT_VISIBLE_DEVICES must contain nonnegative physical device IDs")
    if requested is not None:
        if requested < 0 or (allowed is not None and requested not in allowed):
            raise ValueError("physical-device must be inside the inherited ASCEND_RT_VISIBLE_DEVICES allocation")
        return requested
    if allowed is None:
        mapping = capture(["npu-smi", "info", "-m"])
        if mapping.get("exit_code") != 0:
            raise ValueError("Cannot discover NPU devices with npu-smi; specify --physical-device")
        allowed = [int(match.group(1)) for match in re.finditer(r"^\s*(\d+)\s+\d+\s+\d+\s+\d+\s+", mapping["stdout"], re.MULTILINE)]
    for device in allowed:
        health = capture(["npu-smi", "info", "-t", "health", "-i", str(device)])
        if health.get("exit_code") != 0 or not re.search(r"Health Status\s*:\s*OK\b", health["stdout"]):
            continue
        activity = capture(["npu-smi", "info", "-t", "proc-mem", "-i", str(device)])
        if activity.get("exit_code") == 0 and "No process in device." in activity["stdout"]:
            return device
    raise ValueError("No idle, healthy NPU found inside the visible allocation; retry when a device is available")


def prepare_output(requested):
    if requested is None:
        parent = ROOT.parent / f"{ROOT.name}-artifacts"
        parent.mkdir(parents=True, exist_ok=True)
        prefix = datetime.now(timezone.utc).strftime("npu-%Y%m%d-%H%M%S-")
        output = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    else:
        output = requested.resolve()
        if output.is_relative_to(ROOT):
            raise ValueError("output must be outside the source checkout")
        output.mkdir(parents=True, exist_ok=True)
    if not os.environ.get("TMPDIR"):
        temporary = output / "tmp"
        temporary.mkdir(exist_ok=True)
        os.environ["TMPDIR"] = str(temporary)
    return output


def environment(output, physical_device):
    import torch
    import torch_npu  # noqa: F401
    import tilelang
    from torch_tilelang import GraphConfig
    from torch_tilelang.compat.tilelang import resolve_graph_context

    torch.npu.set_device(0)
    dependency = Path(tilelang.__file__).resolve().parents[1]
    result = {
        "recorded_at": now(), "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "physical_device": physical_device,
        "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"],
        "device_name": torch.npu.get_device_name(0), "device_properties": str(torch.npu.get_device_properties(0)),
        "versions": {}, "toolchain": {}, "native_libraries": {}, "backend_contexts": {},
        "environment": {key: os.environ.get(key) for key in (
            "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "TILELANG_CACHE_DIR", "TILELANG_DISABLE_CACHE", "TMPDIR",
        )},
        "repository": capture(["git", "rev-parse", "HEAD"]),
        "repository_status": capture(["git", "status", "--short"]),
        "source_diff": capture(["git", "diff", "--", "src"]),
        "tilelang_path": str(dependency), "tilelang_version": tilelang.__version__,
        "tilelang_revision": capture(["git", "rev-parse", "HEAD"], dependency),
        "tilelang_status": capture(["git", "status", "--short"], dependency),
        "tvm_revision": capture(["git", "rev-parse", "HEAD"], dependency / "3rdparty/tvm"),
    }
    for package in ("torch", "torch-npu", "tilelang", "torch-tilelang", "pytest"):
        result["versions"][package] = metadata.version(package)
    for name, command in (("driver", ["npu-smi", "info"]), ("bisheng", ["bisheng", "--version"]),
                          ("ptoas", ["ptoas", "--version"])):
        result["toolchain"][name] = capture(command)
    bisheng = shutil.which("bisheng")
    if bisheng:
        install = Path(bisheng).parent.parent / "x86_64-linux/ascend_toolkit_install.info"
        if install.exists():
            result["toolchain"]["cann_install_info"] = {"path": str(install), "contents": install.read_text()}
    harness = output / "harness"
    harness.mkdir(exist_ok=True)
    result["harness_sha256"] = {}
    for name in ("verify_npu.py", "npu_soak.py"):
        source = ROOT / "scripts" / name
        shutil.copyfile(source, harness / name)
        result["harness_sha256"][name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for target, execution in ((GraphConfig().target, "tvm_ffi"), ("pto", "pto")):
        try:
            context = resolve_graph_context(GraphConfig(target=target, execution_backend=execution))
            result["backend_contexts"][execution] = {
                "target": str(context.target), "backend": context.module.name, "execution": context.execution_backend.name,
            }
        except Exception as error:
            result["backend_contexts"][execution] = {"error": f"{type(error).__name__}: {error}"}
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if any(name in line for name in ("libtilelang", "libtvm")) and line.split()[-1].startswith("/")}
    for name in sorted(paths):
        digest = hashlib.sha256()
        with open(name, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        result["native_libraries"][name] = digest.hexdigest()
    (output / "environment.json").write_text(json.dumps(result, indent=2) + "\n")


def run_command(output, name, command, timeout=1800, device_snapshots=False):
    start = time.monotonic()
    record = {"name": name, "command": command, "cwd": str(ROOT), "started_at": now(),
              "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"]}
    if device_snapshots:
        record["device_before"] = capture(["npu-smi", "info", "-t", "proc-mem", "-i", record["visible_devices"]])
    print(f"{now()} starting {name}", flush=True)
    with (output / f"{name}.log").open("w") as log:
        try:
            with subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True) as process:
                record["pid"] = process.pid
                try:
                    record["exit_code"] = process.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    # Stop only this command and its descendants, including
                    # compiler/cache subprocesses, never unrelated NPU users.
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
        except (OSError, subprocess.TimeoutExpired) as error:
            record.update(exit_code=-1, error=str(error))
    record.update(finished_at=now(), elapsed_seconds=time.monotonic() - start)
    if device_snapshots:
        record["device_after"] = capture(["npu-smi", "info", "-t", "proc-mem", "-i", record["visible_devices"]])
    xml = output / f"{name}.xml"
    if xml.exists():
        suites = ET.parse(xml).getroot().iter("testsuite")
        counts = {key: 0 for key in ("tests", "failures", "errors", "skipped")}
        for suite in suites:
            for key in counts:
                counts[key] += int(suite.get(key, 0))
        counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
        record["pytest"] = counts
        record["skips"] = [dict(module=case.get("classname"), case=case.get("name"), reason=case.find("skipped").get("message"))
                           for case in ET.parse(xml).getroot().iter("testcase") if case.find("skipped") is not None]
        record["acceptance_passed"] = record["exit_code"] == 0 and counts["passed"] > 0 and not any(
            row["module"].startswith(("tests.ascend.", "examples.ascend.")) for row in record["skips"]
        )
    with (output / "commands.jsonl").open("a") as manifest:
        manifest.write(json.dumps(record) + "\n")
    print(json.dumps(record), flush=True)
    return record


def summarize_benchmarks(output):
    """Require stable samples within and across all three fresh processes."""
    result = {}
    for workload in ("elementwise", "gemm_epilogue"):
        paths = [output / f"{workload}-{index}.json" for index in range(3)]
        if not all(path.exists() for path in paths):
            result[workload] = {"status": "incomplete", "missing": [path.name for path in paths if not path.exists()]}
            continue
        runs = [json.loads(path.read_text()) for path in paths]
        metrics = {}
        for name in runs[0]["variants"]:
            metrics[name] = {}
            for boundary in ("event", "wall"):
                groups = [run["variants"][name][boundary + "_ms"] for run in runs]
                complete = all(len(group) == 5 and all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0
                                                     for x in group) for group in groups)
                medians = [statistics.median(group) for group in groups] if complete else []
                stable = complete and all((max(group) - min(group)) / statistics.median(group) <= 0.02 for group in groups)
                stable = stable and (max(medians) - min(medians)) / statistics.median(medians) <= 0.02
                metrics[name][boundary] = {"valid": stable, "process_medians_ms": medians,
                                           "median_ms": statistics.median(medians) if medians else None}
        comparisons = {}
        for boundary in ("event", "wall"):
            def ratio(left, right):
                lhs, rhs = metrics[left][boundary], metrics[right][boundary]
                return lhs["median_ms"] / rhs["median_ms"] if lhs["valid"] and rhs["valid"] else None

            comparisons[boundary] = {
                "baseline_over_fused": ratio("B_candidate_tile_unfused", "C_candidate_tile_fused"),
                "fused_over_torch_latency": ratio("C_candidate_tile_fused", "Torch_NPU_eager"),
            }
        result[workload] = {"status": "measured", "fusion_isolated": workload == "elementwise",
                            "metrics": metrics, "comparisons": comparisons}
    (output / "benchmarks-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-device", type=int, help="Physical card; default: first idle, healthy visible card")
    parser.add_argument("--output", type=Path, help="Artifact directory; default: unique directory beside the checkout")
    parser.add_argument("--stage", choices=("all", "environment", "regression", "examples", "soak", "benchmarks", "pto"), default="all")
    parser.add_argument("--include-pto", action="store_true")
    parser.add_argument("--soak-seconds", type=float, default=1800)
    parser.add_argument("--min-calls", type=int, default=10000)
    args = parser.parse_args()
    if args.soak_seconds < 0 or args.min_calls < 1:
        parser.error("soak-seconds must be nonnegative and min-calls must be positive")
    try:
        args.physical_device = select_device(args.physical_device, os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
        output = prepare_output(args.output)
    except ValueError as error:
        parser.error(str(error))
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(args.physical_device)
    print(f"Physical NPU: {args.physical_device}; logical device: npu:0; stage: {args.stage}", flush=True)
    print(f"Artifacts: {output}", flush=True)
    python = sys.executable
    records = []

    def selected(stage):
        return args.stage in ("all", stage)

    def pytest(name, paths, *options):
        return run_command(output, name, [python, "-m", "pytest", *paths, "-q", "--tb=short", "-ra",
                                          f"--junitxml={output / (name + '.xml')}", *options])

    if selected("environment"):
        environment(output, args.physical_device)
    if selected("regression"):
        records.append(pytest("ascendc", ["tests"], "-m", "not pto"))
    if selected("examples"):
        records.append(pytest("examples", ["examples/ascend"], "--device", "npu:0", "--seed", "0"))
    if selected("soak"):
        records.append(run_command(output, "soak", [python, "scripts/npu_soak.py", "--seconds", str(args.soak_seconds),
                       "--min-calls", str(args.min_calls), "--output", str(output / "soak.json")],
                       timeout=max(1800, args.soak_seconds + 900)))
    if selected("benchmarks"):
        for workload, dtype, dimensions in (
            ("elementwise", "float16", ["--rows", "1024", "--cols", "1024"]),
            ("gemm_epilogue", "bfloat16", ["--rows", "128", "--cols", "192", "--k", "256"]),
        ):
            for process in range(3):
                name = f"{workload}-{process}"
                records.append(run_command(output, name, [python, "examples/ascend/torch_compile.py", "--workload", workload,
                               "--dtype", dtype, *dimensions, "--device", "npu:0", "--seed", "0", "--bench", "--repeats", "5",
                               "--output", str(output / f"{name}.json")], device_snapshots=True))
        summarize_benchmarks(output)
    if args.stage == "pto" or (args.stage == "all" and args.include_pto):
        records.append(pytest("pto-codegen", ["tests/ascend"], "-m", "pto and not ascend"))
        runtime = pytest("pto-runtime", ["tests/ascend"], "-m", "pto and ascend", "--ignore=tests/ascend/test_pto_cache.py")
        records.append(runtime)
        if runtime["acceptance_passed"]:
            records.append(pytest("pto-cache", ["tests/ascend/test_pto_cache.py"], "-m", "pto"))
        else:
            blocked = {"name": "pto-cache", "status": "blocked", "exit_code": -1,
                       "reason": "PTO runtime acceptance did not pass; cache restoration was not attempted"}
            records.append(blocked)
            with (output / "commands.jsonl").open("a") as manifest:
                manifest.write(json.dumps(blocked) + "\n")
    (output / f"{args.stage}-commands.json").write_text(json.dumps(records, indent=2) + "\n")
    # PTO is independently reported and never changes the AscendC result.
    required = records if args.stage == "pto" else [row for row in records if not row["name"].startswith("pto")]
    sys.exit(int(any(row["exit_code"] != 0 or row.get("acceptance_passed") is False for row in required)))


if __name__ == "__main__":
    main()
