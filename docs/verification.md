# Verification

Run the commands below from the repository root with the package and its test
dependencies installed. Use a TileLang build with Ascend code generation and
the bundled TVM/Relax. Device execution also requires a compatible Torch,
torch_npu and CANN installation. PTO execution additionally requires PTODSL,
ptoas and Bisheng. [Compiler and runtime requirements](compatibility.md)

## Checks without an NPU

```bash
python -m pytest tests/unit -q -ra
python -m pytest tests/ascend -q -ra -m "not ascend and not pto"
```

The unit tests exercise FX conversion, Relax legalization and Torch-owned
tensor execution using CPU kernels. The second command checks scheduling and
AscendC source generation. To check PTO source generation, use
`-m "pto and not ascend"` on the same directory. Collection under
`tests/ascend` requires an Ascend-enabled TileLang build.

To check a packaged wheel, install it in a separate environment with the same
compiler dependencies and run `scripts/check_installed.py` from outside the
checkout. It checks fresh-process backend discovery and CPU fallback.
[Wheel smoke test](../scripts/check_installed.py)

## Hardware workflows

| What to verify | Command |
| --- | --- |
| AscendC regression suite, examples, soak and benchmarks | `python scripts/verify_npu.py` |
| PTO code generation, runtime, fresh-process cache restoration, soak and benchmarks | `python scripts/verify_npu.py --stage pto` |
| Both workflows in one invocation | `python scripts/verify_npu.py --include-pto` |
| Rerun one stage | `python scripts/verify_npu.py --stage examples` (see `--help` for all stages) |

The driver uses `npu-smi` to select an idle, healthy device within
`ASCEND_RT_VISIBLE_DEVICES`, or among discovered devices if the variable is
unset. Child processes see the selected device as `npu:0`. To select a card
explicitly, check that it is available and pass its physical ID with
`--physical-device`. Each invocation writes artifacts to a new directory beside
the checkout by default. Use `--output` to choose a directory outside the
checkout. [Hardware verification driver](../scripts/verify_npu.py)

## Reading a result

Open `report.json` in the output directory. Each backend has a requested
`scope` (`full` or `partial`) and separate fields for execution, correctness,
long-soak acceptance and timing. With `--include-pto`, the process exit code
reflects AscendC. Use `--stage pto` when PTO must determine the exit code.

- **Correctness.** For each pytest stage, the driver requires a successful
  exit, at least one passing test, and zero reported skips from Ascend tests
  or functional examples. The stage log and JUnit XML identify failures and
  skip reasons.
- **Soak.** Strict compilation with fusion disabled exercises elementwise,
  Softmax, RMSNorm and GEMM across their supported dtypes. The soak checks
  eager results, previous outputs on successive calls, explicit stream
  dependencies and execution on the requested backend with zero fallbacks.
  Sampled live allocations may grow by at most 1 MiB over the warmed baseline.
  `long_soak_accepted` requires at least 1,800 measured seconds and 10,000
  measured calls after warmup. [Soak checks](../scripts/npu_soak.py)
- **Performance.** Inspect `benchmarks-summary.json` under the backend's
  directory. The driver measures device-event and wall time for each workload
  in three fresh processes, with five samples per process. A comparison needs
  finite positive samples and a relative range of at most 2% within each
  process and across process medians. A null ratio means the comparison is
  unavailable. `baseline_over_fused` compares the two candidate-tile variants.
  `fused_over_torch_latency` compares the fused variant to eager Torch, with
  values below one favoring the fused variant. For GEMM the former includes
  both fusion and epilogue scheduling. [Benchmark variants](../scripts/benchmark_graph.py)

Use the generated environment metadata and raw samples when publishing a
compatibility finding or performance comparison in an issue, PR or release note.
