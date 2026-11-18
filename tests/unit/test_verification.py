"""Acceptance reporting must not turn noisy or incomplete timing into a win."""

import json
import sys

import pytest

from scripts import verify_npu
from scripts.verify_npu import run_command, summarize_benchmarks


def test_hardware_checks_run_with_their_own_target(tmp_path, monkeypatch):
    monkeypatch.setenv("TILELANG_DEFAULT_TARGET", "cuda")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    command = [sys.executable, "-c", "import os; assert 'TILELANG_DEFAULT_TARGET' not in os.environ"]
    assert run_command(tmp_path, "target", command)["exit_code"] == 0


@pytest.mark.parametrize("fault", [None, "noise", "nonfinite", "missing"])
def test_benchmark_evidence_requires_three_complete_stable_processes(tmp_path, fault):
    for workload in ("elementwise", "gemm_epilogue"):
        for index in range(3):
            if fault == "missing" and index == 2:
                continue
            variants = {name: {"event_ms": [latency] * 5, "wall_ms": [latency] * 5} for name, latency in (
                ("A_default_unfused", 2), ("B_candidate_tile_unfused", 2),
                ("C_candidate_tile_fused", 1), ("Torch_NPU_eager", 0.5),
            )}
            if fault == "noise" and index == 2:
                variants["C_candidate_tile_fused"]["event_ms"] = [1.1] * 5
            if fault == "nonfinite":
                variants["C_candidate_tile_fused"]["event_ms"][0] = float("nan")
            (tmp_path / f"{workload}-{index}.json").write_text(json.dumps({"variants": variants}))
    summary = summarize_benchmarks(tmp_path)
    for workload, row in summary.items():
        if fault == "missing":
            assert row["status"] == "incomplete"
            continue
        assert row["fusion_isolated"] == (workload == "elementwise")
        event = row["comparisons"]["event"]
        if fault is None:
            assert event["baseline_over_fused"] == 2
            # A fusion win still loses to Torch in this fixture.
            assert event["fused_over_torch_latency"] == 2
        else:
            assert event["baseline_over_fused"] is None
            assert event["fused_over_torch_latency"] is None


@pytest.mark.parametrize("scenario,accepted", [("hardware_skip", False), ("unit_skip", True), ("failure", False), ("empty", False)])
def test_command_acceptance_does_not_hide_hardware_skips(tmp_path, monkeypatch, scenario, accepted):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "6")
    if scenario == "empty":
        xml = '<testsuites><testsuite tests="0" failures="0" errors="0" skipped="0" /></testsuites>'
    elif scenario == "failure":
        xml = '<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0" /></testsuites>'
    else:
        module = "tests.ascend.test_backend" if scenario == "hardware_skip" else "tests.unit.test_compat"
        xml = (f'<testsuites><testsuite tests="2" failures="0" errors="0" skipped="1">'
               f'<testcase classname="{module}" name="optional"><skipped message="unavailable" /></testcase>'
               '<testcase name="executed" /></testsuite></testsuites>')
    (tmp_path / "probe.xml").write_text(xml)
    result = run_command(tmp_path, "probe", [sys.executable, "-c", "print('completed')"])
    assert result["exit_code"] == 0
    assert result["acceptance_passed"] is accepted


def test_default_device_stays_in_allocation_and_skips_busy_cards(monkeypatch):
    queries = []

    def capture(command):
        queries.append(command)
        kind, device = command[3], command[-1]
        text = "Health Status : OK" if kind == "health" else (
            "Process id:123 Process name:python" if device == "5" else "No process in device."
        )
        return {"exit_code": 0, "stdout": text}

    monkeypatch.setattr(verify_npu, "capture", capture)
    assert verify_npu.select_device(None, "5, 6,7") == 6
    assert all(command[-1] in ("5", "6") for command in queries)


def test_default_device_discovers_cards_when_visibility_is_unset(monkeypatch):
    def capture(command):
        if command[-1] == "-m":
            text = "NPU ID Slot ID Chip ID Chip Phy-ID Chip Name\n2 3 0 2 Ascend950PR\n5 14 0 5 Ascend950PR\n"
        elif command[3] == "health":
            text = "Health Status : Alarm" if command[-1] == "2" else "Health Status : OK"
        else:
            text = "No process in device."
        return {"exit_code": 0, "stdout": text}

    monkeypatch.setattr(verify_npu, "capture", capture)
    assert verify_npu.select_device(None, None) == 5


@pytest.mark.parametrize("probe", [
    {"exit_code": 0, "stdout": "Health Status : Alarm"},
    {"exit_code": 1, "stdout": ""},
    {"error": "npu-smi unavailable"},
])
def test_default_device_requires_confirmed_availability(monkeypatch, probe):
    monkeypatch.setattr(verify_npu, "capture", lambda command: probe)
    with pytest.raises(ValueError, match="No idle, healthy NPU"):
        verify_npu.select_device(None, "5,6")


def test_explicit_device_preserves_override_and_allocation(monkeypatch):
    monkeypatch.setattr(verify_npu, "capture", lambda command: pytest.fail("Explicit device should bypass auto-selection"))
    assert verify_npu.select_device(6, "5,6,7") == 6
    with pytest.raises(ValueError, match="inherited"):
        verify_npu.select_device(0, "5,6,7")


def test_default_output_is_unique_and_outside_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_npu, "ROOT", tmp_path / "torch-tilelang")
    monkeypatch.delenv("TMPDIR", raising=False)
    first, second = verify_npu.prepare_output(None), verify_npu.prepare_output(None)
    assert first != second
    assert first.parent == second.parent == tmp_path / "torch-tilelang-artifacts"
    assert (first / "tmp").is_dir()


def test_no_arguments_runs_complete_default_workflow(tmp_path, monkeypatch):
    monkeypatch.setattr(verify_npu, "ROOT", tmp_path / "torch-tilelang")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "6")
    monkeypatch.delenv("TMPDIR", raising=False)
    monkeypatch.setattr(sys, "argv", ["verify_npu.py"])
    monkeypatch.setattr(verify_npu, "capture", lambda command: {
        "exit_code": 0, "stdout": "Health Status : OK" if command[3] == "health" else "No process in device.",
    })
    phases = []
    monkeypatch.setattr(verify_npu, "environment", lambda output, device: phases.append(("environment", device)))

    def run_command(output, name, command, **kwargs):
        phases.append((name, command))
        return {"name": name, "exit_code": 0}

    monkeypatch.setattr(verify_npu, "run_command", run_command)
    monkeypatch.setattr(verify_npu, "summarize_benchmarks", lambda output: None)
    with pytest.raises(SystemExit) as result:
        verify_npu.main()
    assert result.value.code == 0
    assert [name for name, _ in phases] == [
        "environment", "ascendc", "examples", "soak", "elementwise-0", "elementwise-1", "elementwise-2",
        "gemm_epilogue-0", "gemm_epilogue-1", "gemm_epilogue-2",
    ]
    soak = phases[3][1]
    assert float(soak[soak.index("--seconds") + 1]) == 1800
    assert int(soak[soak.index("--min-calls") + 1]) == 10000
    output = next((tmp_path / "torch-tilelang-artifacts").iterdir())
    assert len(json.loads((output / "all-commands.json").read_text())) == 9


@pytest.fixture
def driver(monkeypatch, tmp_path):
    output = tmp_path / "results"
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    monkeypatch.setattr(verify_npu, "select_device", lambda requested, allocated: 2)
    monkeypatch.setattr(verify_npu, "environment", lambda *args: None)
    monkeypatch.setattr(verify_npu, "summarize_benchmarks", lambda *args: {})
    calls = []

    def invoke(stage, fail=None, include_pto=False):
        def command(directory, name, argv, **kwargs):
            calls.append((name, argv, directory))
            return {"name": name, "exit_code": int(name == fail), "acceptance_passed": name != fail}
        monkeypatch.setattr(verify_npu, "run_command", command)
        monkeypatch.setattr(sys, "argv", ["verify_npu.py", "--stage", stage, "--output", str(output),
                                         *(["--include-pto"] if include_pto else [])])
        with pytest.raises(SystemExit) as exit:
            verify_npu.main()
        return exit.value.code, json.loads((output / "report.json").read_text()), calls
    return invoke


def test_complete_pto_workflow_uses_pto_for_soak_and_six_benchmarks(driver):
    code, report, calls = driver("pto")
    names = [name for name, _, _ in calls]
    assert code == 0
    assert names[:4] == ["pto-codegen", "pto-runtime", "pto-cache", "pto-soak"]
    assert names[4:] == [f"pto-{work}-{i}" for work in ("elementwise", "gemm_epilogue") for i in range(3)]
    assert report["backends"]["pto"]["scope"] == "full"
    assert "ascendc" not in report["backends"]
    for _, argv, directory in calls[3:]:
        assert directory.name == "pto"
        assert argv[argv.index("--target") + 1] == "pto"
        assert argv[argv.index("--execution-backend") + 1] == "pto"
    soak = calls[3][1]
    assert soak[soak.index("--seconds") + 1] == "1800"
    assert soak[soak.index("--min-calls") + 1] == "10000"


@pytest.mark.parametrize("failed", ["pto-codegen", "pto-runtime", "pto-cache", "pto-soak"])
def test_failed_pto_stage_blocks_every_dependent_stage(driver, failed):
    code, report, calls = driver("pto", fail=failed)
    names = [name for name, _, _ in calls]
    assert code == 1 and names[-1] == failed
    records = report["backends"]["pto"]["commands"]
    assert not report["backends"]["pto"]["execution_passed"]
    assert all(r["status"] == "blocked" for r in records[len(calls):])
    assert records[-1]["name"] == "pto-benchmarks"


@pytest.mark.parametrize("stage,count", [("pto-soak", 1), ("pto-benchmarks", 6)])
def test_partial_pto_runs_never_claim_complete_gate(driver, stage, count):
    code, report, calls = driver(stage)
    assert code == 0 and len(calls) == count
    assert report["backends"]["pto"]["scope"] == "partial"
    assert report["backends"]["pto"]["correctness"] == "not_run"


def test_optional_pto_failure_does_not_change_ascendc_exit_status(driver):
    code, report, calls = driver("all", fail="pto-runtime", include_pto=True)
    assert code == 0
    assert report["backends"]["ascendc"]["execution_passed"]
    assert not report["backends"]["pto"]["execution_passed"]
    assert {directory.name for _, _, directory in calls} == {"ascendc", "pto"}


@pytest.fixture
def soak_result(tmp_path):
    cases = []
    for model in ("elementwise", "softmax", "rmsnorm", "gemm"):
        for dtype in ("float16", "bfloat16", "float32"):
            if model == "gemm" and dtype == "float32":
                continue
            cases.append({"name": f"{model}_torch.{dtype}", "calls_including_warmup": 1010, "partitions": [{
                "compiled": True, "fallback_calls": 0, "compiled_calls": 1010, "target_backend": "pto", "execution_backend": "pto",
            }]})
    data = {"status": "passed", "target_backend": "pto", "execution_backend": "pto", "calls": 11000,
            "elapsed_seconds": 1800, "cases": cases, "memory_samples": [{"allocated_bytes": 128}, {"allocated_bytes": 128}]}
    return tmp_path / "soak.json", data


@pytest.mark.parametrize("fault", [None, "backend", "fallback", "count", "memory", "duration", "cases", "warmup", "status"])
def test_soak_artifact_requires_real_backend_full_coverage_and_bounded_memory(soak_result, fault):
    path, data = soak_result
    if fault == "backend":
        data["cases"][0]["partitions"][0]["execution_backend"] = "tvm_ffi"
    elif fault == "fallback":
        data["cases"][0]["partitions"][0]["fallback_calls"] = 1
    elif fault == "count":
        data["cases"][0]["partitions"][0]["compiled_calls"] -= 1
    elif fault == "memory":
        data["memory_samples"][1]["allocated_bytes"] += 1024 * 1024 + 1
    elif fault == "duration":
        data["elapsed_seconds"] = 1799
    elif fault == "cases":
        data["cases"].pop()
    elif fault == "warmup":
        data["calls"] += 1
    elif fault == "status":
        data["status"] = "failed"
    path.write_text(json.dumps(data))
    if fault:
        with pytest.raises(AssertionError):
            verify_npu.validate_soak(path, "pto", "pto", 1800, 10000)
    else:
        assert verify_npu.validate_soak(path, "pto", "pto", 1800, 10000)["soak"]["long_soak_accepted"]


def test_short_soak_is_not_accepted_as_thirty_minutes(soak_result):
    path, data = soak_result
    data["elapsed_seconds"] = 5
    path.write_text(json.dumps(data))
    result = verify_npu.validate_soak(path, "pto", "pto", 5, 100)
    assert result["acceptance_passed"] and not result["soak"]["long_soak_accepted"]


def test_missing_soak_artifact_fails_even_when_command_exits_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    result = run_command(tmp_path, "soak", [sys.executable, "-c", "pass"],
                         validator=lambda: verify_npu.validate_soak(tmp_path / "missing.json", "pto", "pto", 1800, 10000))
    assert result["exit_code"] == 0 and not result["acceptance_passed"]


@pytest.mark.parametrize("when", ["before", "after"])
def test_benchmark_device_interference_invalidates_command(tmp_path, monkeypatch, when):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    queries = []
    sentinel = tmp_path / "launched"

    def capture(command):
        if command[3] == "health":
            return {"exit_code": 0, "stdout": "Health Status : OK"}
        queries.append(command)
        busy = len(queries) == (1 if when == "before" else 2)
        return {"exit_code": 0, "stdout": "Process id:999999999" if busy else "No process in device."}

    monkeypatch.setattr(verify_npu, "capture", capture)
    result = run_command(tmp_path, "benchmark", [sys.executable, "-c", f"open({str(sentinel)!r},'w').close()"], device_snapshots=True)
    assert not verify_npu.command_passed(result)
    assert sentinel.exists() == (when == "after")



@pytest.mark.parametrize("fault", [None, "adapter", "fallback", "empty", "numeric"])
def test_benchmark_artifact_requires_pto_kernels_and_actual_execution(tmp_path, fault):
    import copy

    kernel = {"target_backend": "pto", "execution_backend": "pto", "adapter": "PTOKernelAdapter"}
    variant = {"correct": True, "kernels": {"main": kernel}}
    data = {"torch_compile_correct": True, "variants": {name: copy.deepcopy(variant) for name in (
        "A_default_unfused", "B_candidate_tile_unfused", "C_candidate_tile_fused",
    )}, "dynamo_partitions": [{"compiled": True, "compiled_calls": 1, "fallback_calls": 0,
                              "target_backend": "pto", "execution_backend": "pto"}]}
    data["variants"]["Torch_NPU_eager"] = {"correct": True}
    if fault == "adapter":
        data["variants"]["C_candidate_tile_fused"]["kernels"]["main"]["adapter"] = "TVMFFIKernelAdapter"
    elif fault == "fallback":
        data["dynamo_partitions"][0]["fallback_calls"] = 1
    elif fault == "empty":
        data["variants"]["C_candidate_tile_fused"]["kernels"] = {}
    elif fault == "numeric":
        data["variants"]["Torch_NPU_eager"]["correct"] = False
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(data))
    if fault:
        with pytest.raises(AssertionError):
            verify_npu.validate_benchmark(path, "pto", "pto")
    else:
        assert verify_npu.validate_benchmark(path, "pto", "pto")["acceptance_passed"]


def test_benchmark_failure_stops_sampling_and_rejects_stale_summary(driver):
    code, report, calls = driver("pto-benchmarks", fail="pto-elementwise-0")
    assert code == 1 and len(calls) == 1
    row = report["backends"]["pto"]
    assert row["timing_comparisons"] == "inconclusive"
    assert all(work["status"] == "incomplete" for work in row["benchmarks"].values())


@pytest.mark.parametrize("stage", ["pto", "pto-soak", "pto-benchmarks"])
@pytest.mark.parametrize("seconds", ["nan", "inf", "-1"])
def test_invalid_duration_fails_before_environment_probe(monkeypatch, stage, seconds):
    monkeypatch.setattr(sys, "argv", ["verify_npu.py", "--stage", stage, "--soak-seconds", seconds])
    monkeypatch.setattr(verify_npu, "select_device", lambda *args: pytest.fail("must validate arguments first"))
    with pytest.raises(SystemExit) as result:
        verify_npu.main()
    assert result.value.code == 2


def test_soak_case_checks_resolved_backend_and_memory():
    from scripts import npu_soak
    from torch_tilelang.backend import GraphReport, PartitionReport

    partition = PartitionReport("main", compiled=True, compiled_calls=11, target_backend="pto", execution_backend="pto")
    case = {"name": "elementwise", "calls": 11, "reports": [GraphReport(partitions=[partition])]}
    assert npu_soak.validate_case(case, "pto", "pto")["calls_including_warmup"] == 11
    with pytest.raises(AssertionError, match="consistently"):
        npu_soak.validate_case(case, "ascend", "tvm_ffi")
    partition.fallback_calls = 1
    with pytest.raises(AssertionError, match="consistently"):
        npu_soak.validate_case(case, "pto", "pto")
    npu_soak.validate_memory_samples([{"allocated_bytes": 0}, {"allocated_bytes": 1024 * 1024}])
    with pytest.raises(AssertionError, match="grew"):
        npu_soak.validate_memory_samples([{"allocated_bytes": 0}, {"allocated_bytes": 1024 * 1024 + 1}])
