# Verification records

## CPU verification (2026-11-10)

Verified on 2026-11-10 in Ubuntu 26.04 under WSL, using Python 3.14.4 and
Torch 2.15.0+cpu. Dependency versions are recorded in
`constraints/cpu-wsl-py314.txt`.

### Native compiler build

Built TileLang at `72d986e1c45ba8e81f7611eb6835c9bece097d60` with its
unmodified compiler/runtime sources, paired with TVM
`0215a96cba91eabac903afa4ec1e42c99257e4c3`. The build used Release, Ascend
enabled, Ascend runtime stubs, CUDA/ROCm/LLVM disabled, and the PyPI Z3
installation. Native libraries were rebuilt for this baseline. This dependency
contains no source-tree graph packages.

### Installed external package

Built the Python wheel and installed it into a separate virtual environment.
Ran `scripts/check_installed.py` from outside the repository. It verified:

- `torch_tilelang` resolves to the wheel's site-packages directory;
- a fresh process finds `torch_tilelang_ascend` before importing the package;
- a CPU inference graph reaches the backend and correctly uses its Torch fallback;
- the dependency has no `tilelang.graph` or `tilelang.ascend.graph`.

The AscendC selection, using that installed wheel and fresh native build:

```text
python -m pytest tests -q -m "not pto" --tb=short -ra
146 passed, 46 skipped, 70 deselected
```

The passes include real FX/Relax conversion, injected CPU execution, schedule
legality, AscendC source generation, dtype/alias/layout regressions, profiler
error handling, and fresh-process package discovery. Of the skips, 45 need real
NPU execution and one needs TileLang's optional typed msprof-error interface.

### Optional PTO selection

The full suite was also run against the same compiler baseline:

```text
python -m pytest tests -q --tb=short -ra
1 failed, 168 passed, 93 skipped
```

The failure is
`test_pto_rectangular_gemm_epilogue_uses_nz_source_stride[wide]`: TileLang's
PTO codegen rejects the NZ source stride for the `M=32, N=64` epilogue.
The regression remains enabled under the `pto` marker. It is not an AscendC
failure. This baseline is not claimed as a working PTO combination.

### Artifacts and remaining device validation

The repository passes `git diff --check`; Python code passes ruff's configured
checks. Wheel and source-distribution builds succeeded. Both archives were
checked for unwanted compiler-fix payloads and contain only this project's
code, tests, examples, documentation and packaging metadata.

This verification does not establish NPU numerical performance, stream
execution, fresh-process PTO device-cache restoration, or PTODSL assembly.
Run the AscendC hardware tests with a matching Torch/torch_npu pair and CANN
compiler environment before claiming device compatibility. PTO additionally
needs PTODSL, ptoas and Bisheng for its assembly and device-cache checks.

## NPU verification (2026-11-13)

The hardware acceptance run used an Ascend950PR_957b, physical card 6 isolated
as logical `npu:0` with `ASCEND_RT_VISIBLE_DEVICES=6`. Card 5 had other users'
processes when implementation started; no processes or device settings were
changed to acquire it. The following environment differs from the CPU record
above:

| Component | Recorded value |
| --- | --- |
| Python | 3.12.3 |
| Torch / torch_npu | `2.12.0+cpu` / `2.12.0.post2`; NPU execution available |
| Device / driver | Ascend950PR_957b / `npu-smi 25.7.rc1` |
| CANN | 9.3.0, `V100R001C13B023` |
| Bisheng | 2026-10-20 build, clang 15.0.5 |
| ptoas | 0.69 |
| TileLang checkout | `6c9232103a9cba8177d7ac766f2465f1230f0a19`, clean |
| Bundled TVM | `0215a96cba91eabac903afa4ec1e42c99257e4c3` |

TileLang's loaded version is `0.1.15+cuda.git6c923210`; its distribution metadata
still says `0.1.15+cuda.git1cf5a1f0`. Use the checkout and native-library hashes,
not the distribution label alone, to identify this environment. The loaded
`libtilelang.so` SHA-256 was
`d1d236495c26224300c2957931c866972c8eb0c3414b200a97adb2242c92eb8d`.
The environment artifact also records TVM and FFI library hashes.

### AscendC correctness and repeated execution

The initial full non-PTO regression completed with **250 passed, 7 skipped,
122 deselected** in 53.07 seconds. All 149 AscendC schedule/backend tests ran,
including the 97 hardware-marked cases; none of those tests skipped. One unit
skip requires the optional typed profiler-error interface. The other six unit
skips concern PTO target/execution combinations, described below. The eight
new CPU tests for acceptance reporting subsequently passed separately.

The four standalone functional examples passed all **11** dtype cases with
seed 0, covering 33 requests. The strict, unfused soak then passed
**1800.22 measured seconds and 2,134,539 calls**, excluding compilation and
warmup. Each of its 11 model/dtype cases reports 194,059 compiled calls including
warmup, with zero fallback calls. Every measured call checked numerical results,
the preceding output's lifetime, and stream ordering using the explicit
producer/consumer waits in the harness.

All 61 synchronized memory samples reported **104,960 live allocated bytes**
and **41,943,040 reserved bytes**. No device error, numerical failure or live
allocation growth was observed. Device health remained OK; a late-run sample
reported 70 degrees Celsius. This bounds a serial fixed-shape inference soak;
it does not establish concurrent service behavior or arbitrary model coverage.

### Performance measurements

The first card-6 timing batch was excluded. Its first elementwise process
exited with code 120 and a truncated stdout JSON, without creating its result
file; the cause was not established. During the subsequent GEMM run, another
process entered card 6. Our benchmark and driver were stopped without resetting
the card or interfering with that process. Sustained utilization in that batch
is not evidence of a compiler/kernel hang.

Performance was repeated on the then-idle physical card 5, using `/data` for
artifacts and temporary files because the root filesystem had little free
space. All six fresh processes completed successfully. The table gives the
median of the three process medians, in microseconds; a number alone does not
mean its timing evidence met the acceptance boundary.

| Workload | A device | B device | C device | Torch device | Stable B/C device comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| Elementwise fp16 | 61.474 | 64.398 | 38.116 | 15.150 | Unavailable: C varied across processes |
| GEMM epilogue bf16 | 9.278 | 9.920 | 5.614 | 8.166 | 1.767x, combined fusion and epilogue scheduling |

All wall comparisons were unstable under the 2% boundary. The elementwise C
device result and GEMM Torch device result were also unstable. Consequently
neither workload has an accepted fused-versus-Torch comparison, and there is
no accepted end-to-end speedup claim. Keep the raw device and wall samples;
do not discard noisy rounds to manufacture an accepted result.

### PTO compatibility findings

All 23 existing source-only PTO tests failed at the package's target/execution
preflight: this TileLang checkout resolves `target="pto"` with
`execution_backend="auto"` to the PTO codegen variant but the `tvm_ffi` adapter.
An explicit `execution_backend="pto"` resolves correctly. This is also the cause
of the six PTO-related unit skips; the build does contain PTO native codegen.

A separate source-only probe with the explicit PTO execution backend reached
native codegen. The `M=32,N=64,K=16` bf16 GEMM epilogue still failed with
`PTO L0C->UB MTE source stride must be at least n_size`; the `M=64,N=32` case
passed its source-stride assertions. Thus the older wide-rectangle failure is
still reproducible independently of the automatic adapter selection issue.
These probes do not by themselves establish assembly or device execution.

The independent hardware PTO suite, with explicit PTO execution, completed
with **85 passed, 13 failed, no skips**. Nine failures occur during PTODSL IR
construction: `pto.select` receives a Python `False` operand instead of an MLIR
value. Four GEMM failures hit the native L0C-to-UB stride check above. These are
compilation/toolchain failures; they are not evidence of numerical failure in
the 85 passing cases. Since the full PTO runtime gate failed, the fresh-process
cache restoration stage was explicitly marked blocked and was not run. This
environment is not accepted as a production PTO combination.

### Acceptance result and artifacts

AscendC's existing correctness suite and the bounded repeated-execution soak
passed. End-to-end performance acceptance remains open because wall samples
were unstable and there is no accepted fused-versus-Torch comparison. PTO
acceptance failed independently for the issues above. No compiler implementation
or public package API was changed to obtain these results.

The local artifact root is
`/data/caojian/workspace/git/github/easy-tilelang/torch-tilelang-artifacts/npu-20261113.imM1MW`.
`card6/` retains the initial regression, examples, soak, excluded timing attempt
and explicit PTO source probe. `card5/` contains the complete replacement timing
batch and independent PTO audit. `report.json` summarizes the gates, while the
logs, pytest XML and raw timing JSON retain the evidence and skip reasons.

## Reproducing hardware acceptance

No arguments are required:

```bash
python scripts/verify_npu.py
```

The driver selects the first idle, healthy physical card inside the inherited
`ASCEND_RT_VISIBLE_DEVICES` allocation. If visibility is unset it discovers
physical IDs using `npu-smi info -m`. If no card can be confirmed available, it
exits with an explanation. The selected card is exposed as logical `npu:0`.
Selection does not reserve or reset the device, upgrade the shared toolchain,
or terminate other users' processes. Keep other workloads off that card while
collecting timings.

Results go to a unique timestamped directory under
`<checkout-parent>/torch-tilelang-artifacts/`; the selected card and absolute
output path are printed at startup. Without an inherited `TMPDIR`, temporary
files use that result directory's `tmp/` subdirectory. Explicit
`--physical-device` and `--output` override the defaults. When supplying a device,
check its availability first; explicit selection bypasses automatic probing.

The default run records the environment, runs the non-PTO regression suite and
functional examples, executes a 30-minute soak with at least 10000 measured
calls, and benchmarks two workloads. `--include-pto` adds an independent PTO
audit. A failing PTO audit is recorded without changing the AscendC exit status;
`--stage pto` returns failure when the standalone PTO audit fails. Select one
stage with `--stage environment|regression|examples|soak|benchmarks|pto`.

The output directory contains exact commands, timestamps and exit codes in
`commands.jsonl`, full logs, pytest XML with individual skip reasons, environment
and toolchain metadata, loaded native-library SHA-256 digests, and copies of the
verification scripts. Benchmark commands also record device process snapshots
before and after execution; these checks do not reserve the card. Hardware-test
skips prevent acceptance even when pytest
exits successfully. Skipped unit tests remain explicitly reported.

`scripts/npu_soak.py` tests elementwise, Softmax and RMSNorm in fp16/bf16/fp32,
and GEMM+bias+ReLU in fp16/bf16. After warming all input modes and both streams,
it alternates default and non-default streams with explicit dependencies.
Every call checks the Torch reference and preservation of the preceding output.
Inputs cycle through random, zero, large finite and, outside GEMM, NaN/Inf
values. Compilation is strict, fusion is disabled, and final reports require
actual AscendC calls with no fallback. Every 30 seconds the script synchronizes
and records live allocated and reserved NPU bytes. Live allocations may not
exceed the warmed baseline by more than 1 MiB; reserved allocator cache is
reported separately. Compilation/warmup do not count toward the 30 minutes.

Performance runs use three fresh processes per workload, five rounds per
process, seed 0, and the existing A/B/C benchmark:

- Elementwise: `1024x1024`, fp16.
- GEMM epilogue: `M=128, N=192, K=256`, bf16.

`benchmarks-summary.json` reports device and wall metrics separately. A ratio
requires complete positive finite samples, at most 2% relative range within
each process, and at most 2% relative range across process medians. Unstable
comparisons are null; their raw samples remain available. GEMM B/C measures
combined fusion and epilogue scheduling, not isolated fusion. Passing numerical
checks or beating the independent TileLang kernels does not establish a speedup
over Torch eager, nor does this workload acceptance establish a service SLA.
