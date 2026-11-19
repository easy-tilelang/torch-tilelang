"""Validate repeated compiled inference and stream ordering on one visible NPU.

Run with ASCEND_RT_VISIBLE_DEVICES set by scripts/verify_npu.py. Compilation
and warmup are excluded from the requested duration. No device error is retried.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import time
import traceback

import torch

from torch_tilelang import GraphConfig, make_backend
from torch_tilelang.fusion import _assert_outputs_close
from torch_tilelang.compat.tilelang import resolve_graph_context, resolved_config


def elementwise(x, bias, scale):
    return torch.relu(x + bias) * scale


def softmax(x):
    return torch.softmax(x, dim=-1)


def rmsnorm(x, weight):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)).to(x.dtype) * weight


def gemm(x, weight, bias):
    return torch.relu(x @ weight + bias)


def make_cases(device, config):
    cases = []
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        def rand(*shape):
            return torch.randn(*shape, device=device, dtype=dtype) * 0.125

        definitions = [
            (elementwise, [rand(7, 65), rand(65), rand(65)]),
            (softmax, [rand(7, 128)]),
            (rmsnorm, [rand(7, 128), rand(128) + 1]),
        ]
        if dtype != torch.float32:
            definitions.append((gemm, [rand(32, 64), rand(64, 64), rand(64)]))
        for model, inputs in definitions:
            reports = []
            backend = make_backend(config)

            def capture(graph, example_inputs, reports=reports, backend=backend):
                runner = backend(graph, example_inputs)
                reports.append(runner.graph_report)
                return runner

            cases.append({
                "name": f"{model.__name__}_{dtype}", "model": model, "inputs": inputs,
                "compiled": torch.compile(model, backend=capture, fullgraph=True, dynamic=False),
                "reports": reports, "previous": None, "calls": 0,
                "reduction": model is not elementwise,
            })
    return cases


def update_input(case, step):
    value = case["inputs"][0]
    mode = step % (4 if case["model"] is gemm else 5)
    if mode == 0:
        value.normal_().mul_(0.125)
    elif mode == 1:
        value.zero_()
    elif mode == 2:
        value.normal_().mul_(2)
    elif mode == 3:
        value.fill_(32 if case["model"] is gemm else 1024)
        value.flatten()[::2].neg_()
    else:
        value.normal_()
        value.flatten()[:3].copy_(torch.tensor([float("nan"), float("inf"), -float("inf")], device=value.device))


def exercise(case, stream, step):
    default = torch.npu.current_stream()
    stream.wait_stream(default)
    with torch.npu.stream(stream):
        update_input(case, step)
        expected = case["model"](*case["inputs"])
        actual = case["compiled"](*case["inputs"])
        _assert_outputs_close(actual, expected, reduction=case["reduction"])
        if case["previous"] is not None:
            old, snapshot = case["previous"]
            torch.testing.assert_close(old, snapshot, rtol=0, atol=0, equal_nan=True)
        case["previous"] = (actual, actual.clone())
    default.wait_stream(stream)
    case["calls"] += 1


def memory_sample(start, calls):
    gc.collect()
    torch.npu.synchronize()
    return {
        "elapsed_seconds": time.monotonic() - start, "calls": calls,
        "allocated_bytes": torch.npu.memory_allocated(),
        "reserved_bytes": torch.npu.memory_reserved(),
    }


def validate_memory_samples(samples):
    baseline = samples[0]["allocated_bytes"]
    if any(sample["allocated_bytes"] > baseline + 1024 * 1024 for sample in samples[1:]):
        raise AssertionError("Live NPU allocations grew by more than 1 MiB after warmup")


def validate_case(case, expected_backend, expected_execution):
    partitions = [partition for report in case["reports"] for partition in report.partitions]
    if not partitions or any(
        not p.compiled or p.fallback_calls or (p.target_backend, p.execution_backend) != (expected_backend, expected_execution)
        for p in partitions
    ):
        raise AssertionError(f"{case['name']} did not consistently use {expected_backend}/{expected_execution} kernels")
    if sum(p.compiled_calls for p in partitions) != case["calls"]:
        raise AssertionError(f"{case['name']} has unexpected compiled call counts")
    return {"name": case["name"], "calls_including_warmup": case["calls"], "partitions": [asdict(p) for p in partitions]}


def run(args, result):
    import torch_npu  # noqa: F401

    config = GraphConfig(target=args.target, execution_backend=args.execution_backend, strict=True, fusion="off")
    context = resolve_graph_context(config)
    config = resolved_config(config, context)
    result.update(target=config.target, target_backend=context.module.name, execution_backend=context.execution_backend.name)
    device = torch.device("npu:0")
    torch.npu.set_device(device)
    torch.manual_seed(args.seed)
    default = torch.npu.current_stream()
    alternate = torch.npu.Stream()
    with torch.no_grad():
        cases = make_cases(device, config)
        # Warm every input mode and both streams before measuring memory/time.
        for step in range(10):
            for case in cases:
                exercise(case, alternate if step % 2 else default, step)
        torch.npu.synchronize()
        start = time.monotonic()
        calls = 0
        samples = [memory_sample(start, calls)]
        next_sample = 30.0
        step = 0
        while time.monotonic() - start < args.seconds or calls < args.min_calls:
            for case in cases:
                exercise(case, alternate if step % 2 else default, step)
                calls += 1
            step += 1
            if time.monotonic() - start >= next_sample:
                sample = memory_sample(start, calls)
                samples.append(sample)
                print(json.dumps(sample), flush=True)
                args.output.with_suffix(".progress.json").write_text(json.dumps(sample, indent=2) + "\n")
                next_sample = sample["elapsed_seconds"] + 30
        samples.append(memory_sample(start, calls))
        # Compare live allocations against warmup. Reserved memory includes allocator caches.
        validate_memory_samples(samples)
        result.update(elapsed_seconds=time.monotonic() - start, calls=calls, memory_samples=samples, cases=[])
        for case in cases:
            result["cases"].append(validate_case(case, context.module.name, context.execution_backend.name))
        result["long_soak_accepted"] = result["elapsed_seconds"] >= 1800 and calls >= 10000
        result["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("--min-calls", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target", default=GraphConfig().target)
    parser.add_argument("--execution-backend", choices=("auto", "tvm_ffi", "pto"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 0 or args.min_calls < 1:
        parser.error("seconds must be finite and nonnegative and min-calls must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = {"status": "failed", "requested_seconds": args.seconds, "minimum_calls": args.min_calls, "seed": args.seed}
    try:
        run(args, result)
    except Exception:
        result["error"] = traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
