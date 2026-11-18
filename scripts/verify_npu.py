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
            "ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "TILELANG_DEFAULT_TARGET", "TILELANG_CACHE_DIR", "TILELANG_DISABLE_CACHE", "TMPDIR",
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
    for target, execution in (("ascendc", "tvm_ffi"), ("pto", "pto")):
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


def device_problem(activity, health):
    if health.get("exit_code") != 0 or not re.search(r"Health Status\s*:\s*OK\b", health.get("stdout", "")):
        return "Device health could not be confirmed OK"
    pids = {int(pid) for pid in re.findall(r"Process id:\s*(\d+)", activity.get("stdout", ""))}
    if activity.get("exit_code") != 0 or (not pids and "No process in device." not in activity.get("stdout", "")):
        return "Device activity could not be confirmed"
    if pids - {os.getpid()}:
        return "Other processes detected on the benchmark device"


def run_command(output, name, command, timeout=1800, device_snapshots=False, validator=None):
    start = time.monotonic()
    # Stages exercise the package default (AscendC) or select PTO explicitly.
    # Keep the caller's default target from redirecting a named hardware check.
    child_env = os.environ.copy()
    child_env.pop("TILELANG_DEFAULT_TARGET", None)
    record = {"name": name, "command": command, "cwd": str(ROOT), "started_at": now(),
              "visible_devices": os.environ["ASCEND_RT_VISIBLE_DEVICES"]}
    if device_snapshots:
        record["device_before"] = capture(["npu-smi", "info", "-t", "proc-mem", "-i", record["visible_devices"]])
        record["health_before"] = capture(["npu-smi", "info", "-t", "health", "-i", record["visible_devices"]])
    print(f"{now()} starting {name}", flush=True)
    with (output / f"{name}.log").open("w") as log:
        try:
            if device_snapshots and (problem := device_problem(record["device_before"], record["health_before"])):
                raise RuntimeError(problem)
            with subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=child_env
            ) as process:
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
        except (OSError, subprocess.TimeoutExpired, RuntimeError) as error:
            record.update(exit_code=-1, error=str(error))
            log.write(str(error) + "\n")
    record.update(finished_at=now(), elapsed_seconds=time.monotonic() - start)
    if device_snapshots:
        record["device_after"] = capture(["npu-smi", "info", "-t", "proc-mem", "-i", record["visible_devices"]])
        record["health_after"] = capture(["npu-smi", "info", "-t", "health", "-i", record["visible_devices"]])
        if problem := device_problem(record["device_after"], record["health_after"]):
            record.update(acceptance_passed=False, error=problem)
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
    if validator is not None:
        try:
            if record["exit_code"] == 0 and record.get("acceptance_passed") is not False:
                record.update(validator())
            else:
                record["acceptance_passed"] = False
        except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
            record.update(acceptance_passed=False, error=f"Result validation failed: {error}")
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


def command_passed(record):
    return record["exit_code"] == 0 and record.get("acceptance_passed") is not False


def validate_soak(path, backend, execution, seconds, min_calls):
    result = json.loads(path.read_text())
    assert result["status"] == "passed", "Soak did not pass"
    assert (result["target_backend"], result["execution_backend"]) == (backend, execution), "Wrong soak backend"
    assert math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] >= seconds, "Short soak duration"
    assert result["calls"] >= min_calls, "Insufficient measured calls"
    cases = result["cases"]
    expected_names = {f"{model}_torch.{dtype}" for model in ("elementwise", "softmax", "rmsnorm", "gemm")
                      for dtype in ("float16", "bfloat16", "float32") if model != "gemm" or dtype != "float32"}
    assert len(cases) == 11 and {case["name"] for case in cases} == expected_names, "Incomplete soak cases"
    for case in cases:
        partitions = case["partitions"]
        assert partitions and all(p["compiled"] and not p["fallback_calls"] and
                                  (p["target_backend"], p["execution_backend"]) == (backend, execution)
                                  for p in partitions), "Fallback or backend mismatch"
        assert sum(p["compiled_calls"] for p in partitions) == case["calls_including_warmup"], "Call count mismatch"
    assert sum(case["calls_including_warmup"] for case in cases) == result["calls"] + 110, "Warmup accounting mismatch"
    samples = result["memory_samples"]
    assert len(samples) >= 2, "Missing memory samples"
    baseline = samples[0]["allocated_bytes"]
    assert all(s["allocated_bytes"] <= baseline + 1024 * 1024 for s in samples), "Live memory growth"
    return {"acceptance_passed": True, "soak": {
        "elapsed_seconds": result["elapsed_seconds"], "calls": result["calls"],
        "long_soak_accepted": result["elapsed_seconds"] >= 1800 and result["calls"] >= 10000,
        "baseline_allocated_bytes": baseline, "peak_allocated_bytes": max(s["allocated_bytes"] for s in samples),
    }}


def validate_benchmark(path, backend, execution):
    result = json.loads(path.read_text())
    assert result["torch_compile_correct"], "Dynamo validation failed"
    expected_adapter = "PTOKernelAdapter" if backend == "pto" else "TVMFFIKernelAdapter"
    for name in ("A_default_unfused", "B_candidate_tile_unfused", "C_candidate_tile_fused", "Torch_NPU_eager"):
        variant = result["variants"][name]
        assert variant["correct"], "Numerical validation failed"
        if name != "Torch_NPU_eager":
            kernels = variant["kernels"]
            assert kernels and all((k["target_backend"], k["execution_backend"], k["adapter"]) ==
                                   (backend, execution, expected_adapter) for k in kernels.values()), "Wrong benchmark adapter"
    partitions = result["dynamo_partitions"]
    assert partitions and all(p["compiled"] and p["compiled_calls"] > 0 and not p["fallback_calls"] and
                              (p["target_backend"], p["execution_backend"]) == (backend, execution)
                              for p in partitions), "Dynamo fallback or backend mismatch"
    return {"acceptance_passed": True}


def run_soak(output, name, python, args, pto=False):
    options = ["--target", "pto", "--execution-backend", "pto"] if pto else []
    backend, execution = ("pto", "pto") if pto else ("ascend", "tvm_ffi")
    path = output / "soak.json"
    return run_command(output, name, [python, "scripts/npu_soak.py", "--seconds", str(args.soak_seconds),
                       "--min-calls", str(args.min_calls), "--output", str(path), *options],
                       timeout=max(1800, args.soak_seconds + 900),
                       validator=lambda: validate_soak(path, backend, execution, args.soak_seconds, args.min_calls))


def run_benchmarks(output, python, pto=False):
    records = []
    options = ["--target", "pto", "--execution-backend", "pto"] if pto else []
    backend, execution = ("pto", "pto") if pto else ("ascend", "tvm_ffi")
    for workload, dtype, dimensions in (
        ("elementwise", "float16", ["--rows", "1024", "--cols", "1024"]),
        ("gemm_epilogue", "bfloat16", ["--rows", "128", "--cols", "192", "--k", "256"]),
    ):
        for process in range(3):
            stem = f"{workload}-{process}"
            path = output / f"{stem}.json"
            name = f"pto-{stem}" if pto else stem
            record = run_command(output, name, [python, "examples/ascend/torch_compile.py", "--workload", workload,
                                 "--dtype", dtype, *dimensions, "--device", "npu:0", "--seed", "0", *options,
                                 "--bench", "--repeats", "5", "--output", str(path)], device_snapshots=True,
                                 validator=lambda path=path: validate_benchmark(path, backend, execution))
            records.append(record)
            if not command_passed(record):
                summary = {w: {"status": "incomplete", "reason": f"{name} failed; remaining sampling stopped"}
                           for w in ("elementwise", "gemm_epilogue")}
                (output / "benchmarks-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
                return records
    summarize_benchmarks(output)
    return records


def backend_report(output, records, scope):
    tests = [r for r in records if r["name"] in ("ascendc", "examples", "pto-codegen", "pto-runtime", "pto-cache")]
    soak = next((r for r in records if r["name"] in ("soak", "pto-soak")), None)
    summary_path = output / "benchmarks-summary.json"
    has_benchmarks = any("elementwise-" in r["name"] or "gemm_epilogue-" in r["name"] for r in records)
    timing = json.loads(summary_path.read_text()) if has_benchmarks and summary_path.exists() else {}
    comparable = set(timing) == {"elementwise", "gemm_epilogue"} and all(row.get("status") == "measured" and
                                     all(row["comparisons"][boundary]["fused_over_torch_latency"] is not None
                                         for boundary in ("event", "wall")) for row in timing.values())
    return {"scope": scope, "execution_passed": bool(records) and all(command_passed(r) for r in records),
            "correctness": "passed" if tests and all(command_passed(r) for r in tests) else ("failed" if tests else "not_run"),
            "soak": {"status": "passed" if command_passed(soak) else soak.get("status", "failed"),
                     **soak.get("soak", {})} if soak else {"status": "not_run"},
            "long_soak_accepted": bool(soak and command_passed(soak) and soak.get("soak", {}).get("long_soak_accepted")),
            "timing_comparisons": "available" if comparable else ("inconclusive" if has_benchmarks else
                                  ("blocked" if any(r["name"] == "pto-benchmarks" for r in records) else "not_run")),
            "benchmarks": timing, "commands": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-device", type=int, help="Physical card; default: first idle, healthy visible card")
    parser.add_argument("--output", type=Path, help="Artifact directory; default: unique directory beside the checkout")
    parser.add_argument("--stage", choices=("all", "environment", "regression", "examples", "soak", "benchmarks", "pto",
                                          "pto-soak", "pto-benchmarks"), default="all")
    parser.add_argument("--include-pto", action="store_true", help="Add the complete, independent PTO workflow")
    parser.add_argument("--soak-seconds", type=float, default=1800)
    parser.add_argument("--min-calls", type=int, default=10000)
    args = parser.parse_args()
    if not math.isfinite(args.soak_seconds) or args.soak_seconds < 0 or args.min_calls < 1:
        parser.error("soak-seconds must be finite and nonnegative and min-calls must be positive")
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
    reports = {}
    # Record metadata for partial invocations too; every artifact can identify
    # its Python package, compiler revision and actual loaded native libraries.
    environment(output, args.physical_device)

    def selected(stage):
        return args.stage in ("all", stage)

    def pytest(directory, name, paths, *options):
        return run_command(directory, name, [python, "-m", "pytest", *paths, "-q", "--tb=short", "-ra",
                                             f"--junitxml={directory / (name + '.xml')}", *options])

    if args.stage in ("all", "regression", "examples", "soak", "benchmarks"):
        directory = output / "ascendc"
        directory.mkdir(exist_ok=True)
        ascendc = []
        if selected("regression"):
            ascendc.append(pytest(directory, "ascendc", ["tests"], "-m", "not pto"))
        if selected("examples"):
            ascendc.append(pytest(directory, "examples", ["examples/ascend"], "--device", "npu:0", "--seed", "0"))
        if selected("soak"):
            ascendc.append(run_soak(directory, "soak", python, args))
        if selected("benchmarks"):
            ascendc.extend(run_benchmarks(directory, python))
        records.extend(ascendc)
        reports["ascendc"] = backend_report(directory, ascendc, "full" if args.stage == "all" else "partial")

    if args.stage.startswith("pto") or (args.stage == "all" and args.include_pto):
        directory = output / "pto"
        directory.mkdir(exist_ok=True)
        pto = []
        full = args.stage in ("all", "pto")
        blocked_reason = None

        def stage(name, action):
            nonlocal blocked_reason
            if blocked_reason:
                record = {"name": name, "status": "blocked", "exit_code": -1, "reason": blocked_reason}
                with (directory / "commands.jsonl").open("a") as manifest:
                    manifest.write(json.dumps(record) + "\n")
                pto.append(record)
                return
            record = action()
            pto.append(record)
            if not command_passed(record):
                blocked_reason = f"{name} did not pass; dependent PTO stages were not attempted"

        if full:
            stage("pto-codegen", lambda: pytest(directory, "pto-codegen", ["tests/ascend"], "-m", "pto and not ascend"))
            stage("pto-runtime", lambda: pytest(directory, "pto-runtime", ["tests/ascend"], "-m", "pto and ascend",
                                               "--ignore=tests/ascend/test_pto_cache.py"))
            stage("pto-cache", lambda: pytest(directory, "pto-cache", ["tests/ascend/test_pto_cache.py"], "-m", "pto"))
        if full or args.stage == "pto-soak":
            stage("pto-soak", lambda: run_soak(directory, "pto-soak", python, args, pto=True))
        if full or args.stage == "pto-benchmarks":
            if blocked_reason:
                stage("pto-benchmarks", lambda: None)
            else:
                pto.extend(run_benchmarks(directory, python, pto=True))
        records.extend(pto)
        reports["pto"] = backend_report(directory, pto, "full" if full else "partial")

    report = {"recorded_at": now(), "stage": args.stage, "environment": "environment.json", "backends": reports}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / f"{args.stage}-commands.json").write_text(json.dumps(records, indent=2) + "\n")
    # Optional PTO does not change the AscendC exit status. Timing availability
    # is reported independently; noisy comparisons never become speedups.
    required = records if args.stage.startswith("pto") else [r for r in records if not r["name"].startswith("pto-")]
    sys.exit(int(any(not command_passed(r) for r in required)))


if __name__ == "__main__":
    main()
