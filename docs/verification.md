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

## PTO adapter follow-up (2026-11-13)

After the baseline acceptance above, the graph compatibility layer was updated
to explicitly select the PTO adapter when a PTO target's automatic choice is
TVM-FFI. An explicit PTO/TVM-FFI pair is rejected. This changes the graph-local
selection policy, preserves target attributes and immutable user options, and
does not alter TileLang's registry or compiler sources.

The focused compiler/profiler compatibility and runtime tests completed with
**91 passed, 1 skipped**. The only skip is the optional typed profiler-error
interface; the six former PTO-related skips now execute and pass. Source-only
PTO tests improved from 23 preflight failures to **22 passed, 1 failed**. The
remaining wide GEMM failure reaches the known native L0C-to-UB stride check.
The complete non-PTO regression then passed with **278 passed, 1 skipped,
124 deselected** in 41.93 seconds, including the hardware cases. Ruff and
`git diff --check` also passed.

On physical card 5, all **four** targeted hardware cases passed: the public
`torch.compile` options select actual PTO adapters with both `auto` and explicit
`pto`, and both settings pass a fresh-process disk-cache test. Each cache case
populates an isolated directory, then starts another process with compilation
forbidden and requires both kernels to restore as `PTOKernelAdapter`. It also
checks changed inputs on a non-default stream and preservation of old outputs.
These are fp32 elementwise cache checks; they do not accept the broader PTO
configuration or resolve the PTODSL/rectangular-GEMM failures above. The
acceptance driver's full-runtime prerequisite for its cache stage is unchanged.

The follow-up logs and pytest XML are stored separately in
`/data/caojian/workspace/git/github/easy-tilelang/torch-tilelang-artifacts/pto-adapter-fix.dlfhqr`.

## PTO compiler fixes (2026-11-14)

Both compiler fixes are available on
[`erhsh/tilelang:fix/ascend-pto-4013-4014`](https://github.com/erhsh/tilelang/tree/fix/ascend-pto-4013-4014),
based on upstream `d2dbf20cb279b3021ada636ffa8026706e0a0bcb`. No pull request
has been opened. The commits are:

- [`07563fc6`](https://github.com/erhsh/tilelang/commit/07563fc6): validate the NZ
  L0C source stride against M, keeping the ND UB destination check against N.
  This fixes [#4013](https://github.com/tile-ai/tilelang/issues/4013), and also
  rejects insufficient source strides for tall rectangles.
- [`de9c5126`](https://github.com/erhsh/tilelang/commit/de9c512609dffe7956b2e0a1a395fed6c1e7e923):
  materialize scalar `pto.select` predicates and boolean branches as IR `i1`,
  including expressions that Python evaluates to `True` or `False`. Control-flow
  handling retains its constexpr behavior. This fixes TileLang's generated code
  for [#4014](https://github.com/tile-ai/tilelang/issues/4014); PTODSL was not patched.

The compiler was rebuilt from this clean worktree with Ascend enabled, using
the same bundled TVM revision as above. Torch 2.12.0, torch_npu 2.12.0.post2,
CANN 9.3 and ptoas 0.69 were used on Ascend950PR_957b. An isolated Python
environment loads both the Python packages and native libraries from the fixed
compiler checkout; subprocesses use that same environment.

In the TileLang checkout, the following regression set passed **148 tests**:

```bash
python -m pytest -q \
  testing/ascend/language/pto/test_pto_l0c_ub_stride.py \
  testing/ascend/language/pto/test_pto_select_predicates.py \
  testing/ascend/language/pto/test_pto_vmi_value_provenance.py \
  testing/ascend/language/pto/test_pto_vmi_value_provenance_run.py
```

The new tests cover five wide/tall/square GEMM shapes with exact device
references, rejection of undersized pitches, fp16/bf16/fp32 NaN-safe ReLU with
constant and runtime conditions, boolean casts/branches and nonzero byte
predicates. NaN, infinities and both signed zeros are included. Numerical
comparisons allow equivalent zero signs: the existing PTOAS path also
canonicalizes negative zero in the constant-folded ReLU control. Bitwise
zero-sign preservation is not established by this acceptance.

Against the fixed compiler, the standalone graph PTO gate passed source
generation (**23 passed**), device runtime (**99 passed**) and fresh-process
cache restoration (**2 passed**), with no skips or failures. Cache restoration
was tested with both automatic and explicit PTO adapter selection, forbidding
compilation in the second process. The original nine select-compilation and
four stride-check failures no longer occur. These results establish the tested
numerical and cache cases; PTO's 30-minute soak and performance acceptance
remain separate requirements.

The same fixed compiler also passed the full non-PTO regression with
**278 passed, 1 skipped, 124 deselected**. The skip requires the optional
`tilelang.profiler.errors` interface; no AscendC hardware test skipped. The
standalone AscendC examples also passed all **11** dtype cases.

PTO performance was then measured with the existing A/B/C example, adding
`--target pto --execution-backend pto`. Both workloads used the original shapes,
three fresh processes, five rounds per process and the unchanged 2% stability
boundary. All six processes passed numerical validation and used actual
`PTOKernelAdapter` kernels, with no Dynamo fallback. Device process snapshots
before and after every process found no other workload on the selected card.
Device health remained OK after the run.

The values below are medians of process medians, in microseconds. Unstable
measurements remain descriptive samples and do not establish a comparison.

| Workload | A device | B device | C device | Torch device | Accepted device comparison |
| --- | ---: | ---: | ---: | ---: | --- |
| Elementwise fp16 | 54.575 | 60.498 | 32.780 | 14.552 | Unavailable: C varied across processes |
| GEMM epilogue bf16 | 9.052 | 9.478 | 4.834 | 8.034 | Torch/C = 1.662x |

GEMM's C and Torch device samples passed stability checks within and across
processes. Its A/B event samples did not, so B/C remains unavailable; this
does not establish an isolated fusion speedup. Every wall comparison failed
the stability boundary. The GEMM device result is therefore a bounded device
timing result, and no end-to-end speedup is accepted. PTO's 30-minute soak was
not run in this follow-up; the earlier AscendC soak does not validate PTO.

To reproduce the graph audit, build the fixed TileLang revision with Ascend
enabled and install its matching Python package into the environment used to
run this checkout. Keep the matching Torch/torch_npu and CANN/PTO toolchain
available, then run:

```bash
python scripts/verify_npu.py --stage environment
python scripts/verify_npu.py --stage pto
```

The first command records the loaded source revision and native-library hashes.
The second runs source generation, device correctness, and then fresh-process
cache restoration when the full runtime gate passes. Neither command changes
the compiler checkout or installs a dependency automatically.

## Unified PTO acceptance (2026-11-16)

The repository now owns the complete PTO workflow, including its soak and
performance stages. This run used the same fixed TileLang revision
`de9c512609dffe7956b2e0a1a395fed6c1e7e923` and matching Python/native libraries
as the compiler-fix record above. The saved harness SHA-256 values match the
scripts used for this acceptance.

The script unit tests passed **166 tests**, with one skip for the optional
typed profiler-error interface. The focused verification set passed **53
tests**, covering unchanged no-argument defaults, PTO parameter forwarding,
prerequisite failure/skip handling, partial-stage reporting, adapter/fallback
validation, memory growth, incomplete timing and interference before/after a
benchmark process. Ruff and `git diff --check` passed.

Both real-device smoke checks passed all 11 model/dtype cases: PTO completed
**2310 measured calls in 5.30 seconds**, and default AscendC completed
**2596 calls in 5.41 seconds**. Both report `long_soak_accepted=false`.

The complete PTO invocation then passed source generation (**23 tests**),
device runtime (**99 tests**) and fresh-process cache restoration (**2 tests**),
with no skips. Its strict, unfused soak passed **1800.26 measured seconds and
1,031,855 graph calls**, excluding compilation and warmup. Each of the 11
cases reports 93,815 compiled calls including its ten warmup calls. Every
partition resolved to PTO/PTO, with zero fallback calls. Every measured call
checked numerical results, preservation of the previous output and explicit
default/non-default stream dependencies.

All **61** synchronized memory samples reported **104,960 live allocated
bytes** and **41,943,040 reserved bytes**. The long-soak requirement passed.
The run used an explicitly selected, user-authorized card with other resident
processes; it does not establish exclusive-device performance. Health probes
reported `Health Status: OK`; the pre-existing HBM RAS configuration status
code was retained in the health records.

Before the first performance process, the occupancy check detected other
processes and stopped sampling without launching that process. The other
authorized card also had a resident process. Neither workload has new timing
samples from this run, and no old samples were reused to fill them. The full
workflow exited nonzero, with the report explicitly recording:

- `correctness: passed`;
- `long_soak_accepted: true`;
- `execution_passed: false` and `timing_comparisons: inconclusive`;
- both workload timing summaries marked `incomplete`, with the command's
  device snapshots and interference reason retained.

The next hardware action is a partial performance rerun on an idle authorized
card, using `python scripts/verify_npu.py --stage pto-benchmarks`. This runs the
same six-process protocol and does not repeat the 30-minute soak. Until valid
wall comparisons exist, the earlier 1.662x GEMM device result remains the only
accepted fused-versus-Torch comparison; it is not an end-to-end claim.

Local logs, XML, environment/library fingerprints, script snapshots and JSON
reports are retained under the artifact directory printed by the driver:
`pto-acceptance-20261116.gWW7PJ`, with `smoke/`, `ascendc-smoke/` and `full/`
subdirectories. The artifact directory is outside the source checkout.

## Card 4 performance retry (2026-11-16)

The pending partial PTO benchmark stage was retried on user-authorized card 4
using repository revision `9ae21ca` and the same fixed TileLang environment.
Initial probes found no device process, zero compute utilization and
`Health Status: OK`, with the existing HBM RAS configuration status retained.

The first elementwise process completed successfully. The second process also
completed numerical validation and used actual PTO adapters, but its post-run
snapshot detected a new foreign process. Sampling then stopped as designed:
**two of six processes executed, one accepted and one invalidated by device
interference; the remaining four were not launched**. Subsequent probes found
additional new processes on the card.

Both executed processes' JSON artifacts passed the numerical/backend audit.
These checks establish neither a complete three-process timing comparison nor
new speedup evidence. The invocation exited nonzero; `report.json` records
`scope: partial`, `execution_passed: false`, `timing_comparisons: inconclusive`,
and both workload timing summaries remain `incomplete`. The earlier correctness
and 30-minute soak results are unchanged. Performance acceptance still needs a
complete batch on a card that remains free of competing processes.

The artifact directory is `pto-card4-20261116.JqGZNY`, outside the checkout.
It retains the exact commands, environment and native-library fingerprints,
raw JSON and logs for both processes, and all command-level device snapshots.

## Card 5 recovery and toolchain comparison (2026-11-18)

NPU management queries returned successfully on this attempt. Cards 4, 5 and
6 initially reported OK health, zero compute utilization and no device
processes. Card 5 was selected from the inherited allocation for the pending
PTO performance stage; no device reset or management-service change was made.

The inherited toolchain had changed from CANN `V100R001C13B023` / ptoas 0.69
to `V100R001C13B046` / ptoas 0.72. Torch remained 2.12.0+cpu and torch_npu
2.12.0.post2. The fixed TileLang source revision and loaded native-library
hashes matched the earlier runs; the installed distribution's version label
had changed, reinforcing the need to record actual sources and libraries.

The first performance attempt failed the graph's signed-zero check before
timing. A fresh, isolated cache reproduced the failure through the same
acceptance entry point, so stale cache alone does not explain it. Direct
graph outputs and reference snapshots matched when compared on the CPU, while
device-side sign-bit comparisons disagreed. The following pure Torch probe,
without importing TileLang, reproduced a discrepancy in the B046 environment:

```python
import torch
import torch_npu

x = torch.tensor([-0.0, 0.0, -1.0, 1.0], dtype=torch.float16)
x = x.repeat(262144).reshape(1024, 1024).npu()
expected = torch.signbit(x.cpu())
for trial in range(3):
    actual = torch.signbit(x).cpu()
    print(trial, (actual != expected).sum().item())
```

All three B046 trials reported **370,779 mismatched bits out of 1,048,576**;
all three trials under B023 reported **zero**. This isolates a problem with
the current Torch/torch_npu/CANN combination independently of TileLang; it
does not identify the faulty component within that combination. The graph's
correctness checks were retained. The B046 validation failures cannot be used
as accepted performance evidence.

The six-process benchmark was then completed with B023 / ptoas 0.69 selected
only in the child process environment, with a separate cache. Shared shell,
installed packages and repository runtime/scheduling code were unchanged.
All **six processes passed** numerical validation, actual PTO-adapter checks
and before/after device-occupancy checks. Health snapshots remained OK with no
reported error code. This partial rerun exited successfully and reported
`execution_passed: true`; it did not rerun the earlier 30-minute soak.

The table gives medians of process medians in microseconds. These are
descriptive samples: both workloads' C device measurements and every wall
measurement failed the unchanged 2% stability boundary.

| Workload | A device | B device | C device | Torch device | C wall | Torch wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Elementwise fp16 | 54.792 | 61.038 | 33.578 | 14.656 | 56.485 | 38.949 |
| GEMM epilogue bf16 | 9.127 | 9.584 | 4.942 | 8.053 | 39.600 | 23.305 |

Every B/C and C/Torch comparison in this batch remains null, and
`timing_comparisons` is `inconclusive`. No new device or end-to-end speedup
claim is accepted. The earlier 1.662x GEMM device result remains a historical
result from its own validated batch. The sampling blockage is resolved;
toolchain compatibility and timing stability remain separate open items.

Artifacts outside the checkout are retained in:

- `pto-card5-20261118.IfhnFk`: inherited-toolchain failure, pure Torch and
  graph-validation probes, and old/new toolchain comparison logs;
- `pto-card5-fresh-20261118.EnOxGd`: same-entry-point failure with a fresh cache;
- `pto-card5-b023-20261118.mMumHL`: complete six-process results, environment
  and native-library fingerprints, occupancy snapshots and timing summaries.

## Static execution plan (2026-11-18)

The graph runner now prepares Python operations, integer value slots and
last-consumer release positions during construction. Request execution no
longer traverses Relax IR, reads TVM containers or resolves IR symbols. The
graph/kernel inspection metadata, input validation, fresh Torch-owned outputs,
per-request constant transfers, input aliases and stream handling are retained.
Slot storage belongs to each invocation, including overlapping invocations.
The public configuration and kernel ABI are unchanged.

The before revision is `8778569`; its source is preserved in a detached
worktree. Both versions use the fixed TileLang revision `de9c5126` and B023 /
ptoas 0.69 selected in child processes with a separate cache. The shared CANN
environment and installed packages were not changed. Each experiment logs the
imported package path and runtime source SHA-256 to distinguish the two versions.

The final full regression passed **444 tests, with 1 optional profiler-interface
skip**. Runtime tests cover execution without access to graph metadata, unused
outputs and last-consumer release, repeated inputs, multi-output projection,
nested output aliases, constant transfers and overlapping request slot storage.
The runtime subset passed all **88 tests**. Ruff and `git diff --check` passed.

A small CPU experiment used injected arithmetic kernels for the same three-op
graph, seven rounds of 5000 requests, and a separate 2000-request cProfile run:

| Measurement | Before | Static plan |
| --- | ---: | ---: |
| Median request time, CPU microseconds | 46.662 | 19.477 |
| Profiled function calls, 2000 requests | 708001 | 144001 |

These are host-side diagnostic samples, not NPU latency or an accepted
end-to-end speedup. The new profile contains no execution-time Relax container
accesses or recursive IR evaluator calls. The before/after profile logs retain
all seven time samples and the function-call breakdown.

Both backend smoke runs passed on real hardware. PTO then passed its full
**1800.20-second soak with 2,203,135 measured calls**. The first AscendC soak
stopped updating after **1482.08 seconds and 1,915,243 calls** while its card
reported sustained 100% utilization and another process was present. Only this
task's waiting process group was terminated. That interrupted run remains
explicitly failed/incomplete and is not counted as a 30-minute pass.

AscendC was rerun on the freed card and passed **1800.23 measured seconds and
2,374,900 calls**. Both completed soaks cover all 11 model/dtype cases with
strict compilation and fusion disabled. Every partition resolved to the
requested backend and reported zero fallbacks. Each completed soak has **61
memory samples**, all at **104,960 allocated bytes** and **41,943,040 reserved
bytes**. Numerical results, old-output lifetime and explicit stream dependencies
were checked on every measured call.

After both soaks completed, the paired performance workflow was started on the
freed card. It was configured to alternate before/after versions for three
processes per workload/version, five rounds per process, across both backends
(24 processes total), with the existing 2% timing boundary and occupancy checks.
During environment preflight, `npu-smi` entered uninterruptible driver
communication wait (`devdrv_sync_non_trans_msg_se`). Other management queries
also became blocked. **No timing process was launched**. Only this task's
waiting workflow was cancelled; no device reset or other-task termination was
performed.

There are therefore no accepted NPU before/after latency numbers for this
change. The lower CPU diagnostic time is not substituted for device or graph
wall-time evidence. The paired workflow can be rerun after device management
queries recover and a card can be confirmed available.

Artifacts are retained outside the checkout in `static-plan.xkPKtJ`: the
baseline worktree, runtime before/after source snapshots, final regression XML,
CPU profile logs, both completed soak reports, the interrupted AscendC run and
its interruption record, and `paired-bench/blocked.json`. The local `run.py`
selects B023 and the requested checkout only in child processes;
`paired_bench.py` records imported source paths and retains every command and
sample from a completed rerun. The saved runtime snapshot matches the tested
implementation's SHA-256.

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
workflow with source generation, device runtime, fresh-process cache restoration,
a separate 30-minute soak and performance measurements. Failure of a prerequisite
blocks the dependent PTO stages. A failing optional PTO workflow is recorded
without changing the AscendC exit status.

Run the complete standalone PTO workflow with `--stage pto`; it returns failure
if any required command, correctness check, soak or benchmark execution fails.
Timing noise is reported separately and does not change command success into a
failure or a speedup claim. Select a partial rerun with
`--stage environment|regression|examples|soak|benchmarks|pto-soak|pto-benchmarks`.
The last two run only the selected PTO stage; neither establishes a complete
PTO gate. For a quick PTO soak probe, use:

```bash
python scripts/verify_npu.py --stage pto-soak --soak-seconds 5 --min-calls 100
```

A short probe cannot set `long_soak_accepted`: that requires at least 1800
measured seconds and 10000 calls, regardless of a successful command exit.

Every invocation records environment and toolchain metadata, source revisions,
loaded native-library SHA-256 digests and copies of the verification scripts at
the output root. `ascendc/` and `pto/` contain their own exact commands,
timestamps and exit codes in `commands.jsonl`, full logs, pytest XML with skip
reasons, soak results and timing JSON. `<stage>-commands.json` retains the
invocation's command list at the root.

`report.json` separates each backend's requested scope (`full` workflow or
`partial` rerun), execution success, correctness, long-soak acceptance and timing
comparison availability. A full workflow may still fail or have inconclusive
timing; scope describes coverage requested, not a production-readiness claim.
Hardware-test skips prevent acceptance even when pytest exits successfully.
Skipped unit tests remain explicitly reported.

Benchmark commands record health and process snapshots before and after each
process. An unconfirmed health/activity probe or another process on the selected
card stops further sampling and invalidates that batch. The driver's own idle
metadata context is allowed. These checks do not reserve the card and cannot
exclude interference that starts and ends between snapshots. No other user's
process is terminated.

`scripts/npu_soak.py` tests elementwise, Softmax and RMSNorm in fp16/bf16/fp32,
and GEMM+bias+ReLU in fp16/bf16. After warming all input modes and both streams,
it alternates default and non-default streams with explicit dependencies.
Every call checks the Torch reference and preservation of the preceding output.
Inputs cycle through random, zero, large finite and, outside GEMM, NaN/Inf
values. Compilation is strict, fusion is disabled, and final reports require
actual calls through the requested backend with no fallback and consistent
compiled-call counts. The standalone soak accepts `--target` and
`--execution-backend`; defaults retain AscendC/automatic selection, while the PTO
driver passes `--target pto --execution-backend pto`. Every 30 seconds it synchronizes
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
