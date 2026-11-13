"""Acceptance reporting must not turn noisy or incomplete timing into a win."""

import json
import sys

import pytest

from scripts import verify_npu
from scripts.verify_npu import run_command, summarize_benchmarks


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
