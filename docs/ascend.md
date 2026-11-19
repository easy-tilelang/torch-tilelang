# Ascend Torch graph compilation

The `tilelang` Dynamo backend compiles supported FX partitions into Ascend
TileLang kernels. [NPU quick start](../README.md#quick-start) ·
[Compiler and runtime requirements](compatibility.md)

Pass `GraphConfig` fields through `options` or use
`torch_tilelang.make_backend(config)`. Set `target="pto"` for PTO.
An omitted target uses `TILELANG_DEFAULT_TARGET`, then AscendC on `dav-3510`.
The default execution adapter is `tvm_ffi` for AscendC and `pto` for PTO.
To specify the architecture for PTO, pass an Ascend JSON target string with
the desired `mcpu` and set `execution_backend="pto"`.
[Configuration](../src/torch_tilelang/config.py) · [Target resolution](../src/torch_tilelang/compat/tilelang.py)

## Execution and fallback

The backend resolves the compiler target and checks for its code generator
before partitioning. Invalid target settings and missing code generators raise
errors at this stage. A known pure operator such as `torch.sin` can run in
Torch between compiled partitions. An in-place mutation or opaque module
keeps the whole captured graph in Torch by default. [Partitioning](../src/torch_tilelang/backend.py)

Each partition compiles on its first call with real inputs. Inputs must be
contiguous fp16, bf16 or fp32 tensors with zero storage offset, positive static
dimensions and a common NPU device. A tensor with `requires_grad=True` is
accepted when gradient tracking is disabled. The partition caches the runner
for its most recent input signature, including layout, device and gradient
state. The runner resolves the Relax graph into a sequence of kernel calls
when it is built. It allocates each kernel's outputs in Torch and releases its
references to intermediates after their last use. Graph outputs can also
refer directly to inputs. Kernels run on the current NPU stream.
Callers using multiple streams order their producers and consumers
with Torch events or stream waits. [Input validation](../src/torch_tilelang/frontend.py) ·
[Execution plan](../src/torch_tilelang/runtime.py) · [中文编译流程](ascend-backend.zh.md)

By default, a partition that fails compilation or numerical validation runs
in Torch. `strict=True` raises for those failures, unsafe graphs and graphs
with no supported partition. Pure Torch operators between supported partitions
can still run in strict mode. Kernel execution and synchronization errors
propagate to the caller.

To inspect execution through Dynamo, wrap `make_backend` and save the
`graph_report` on the callable it returns. Whole-graph fallback sets
`graph_report.fallback_reason`. Each partition records its target and execution
backend, the number of kernel call sites in the selected plan, and how many
calls took the compiled or fallback path. [Recording backend example](../tests/ascend/test_backend.py)

## Scheduling and fusion

| Schedule | Supported computation |
| --- | --- |
| Elementwise | Add/subtract with `alpha=1`, multiply, divide, ReLU, exp/sqrt/rsqrt, square, dtype casts and static singleton broadcasting. Inlined producers share a kernel. `vector_width=1` forces scalar transfers. Both `0` (default) and `4` allow four-element transfers when alignment and the index maps permit them. |
| Last-axis reduction | Sum, mean and maximum across a full last axis of at most 16,384 elements. Softmax lowers to max, subtract, exp, sum and divide with fp32 statistics. An RMSNorm expression with fp32 statistics can use the same schedules. |
| 2-D GEMM | Matching fp16 or bf16 operands, with M, N and K divisible by 16 and fp32 accumulation. A supported bias/add/ReLU/cast epilogue can consume the accumulator inside the same kernel. |

The scheduler tries each family against the generated TIR. The elementwise rule
checks input bounds before lowering the computation to a one-dimensional core
grid with local `SimtVF` threads. The GEMM rule transposes a logical NN right
operand on the way from global memory to L1 for the NT lowering. A fused
epilogue reads the result via an L0C-to-UB copy.
[Elementwise schedule](../src/torch_tilelang/schedules/ascend/elementwise.py) ·
[Reduction and GEMM schedules](../src/torch_tilelang/schedules/ascend/operators.py)

Fusion is off by default. With `options={"fusion": "profile"}`, Relax
`FuseOps` and `FuseTIR` construct a candidate before Ascend scheduling.
The backend checks both plans against the original Torch partition, including
output structure, dtype, shape, device, values and the sign of zero. Numerical
tolerances depend on dtype and reduction flags propagated through kernel calls.

The selector uses device events by default. Set `benchmark_backend="msprof"`
to use the optional profiler. The selector alternates measurement order for at
least three rounds. By default, the candidate must be faster on device in every pair, reduce median
device latency by at least 5%, and increase median wall time by at most 2%.
The relative range of each sample series must also stay within 2%. A failed
candidate compilation, numerical mismatch or unavailable measurement retains
the validated baseline. The partition's `fusion_decision` records the reason
and the timings from completed comparisons. [Validation and selection](../src/torch_tilelang/fusion.py)

## Benchmarking

```bash
python scripts/benchmark_graph.py --bench --output /tmp/graph.json
python scripts/benchmark_graph.py --workload gemm_epilogue \
    --rows 128 --cols 192 --k 256 --dtype bfloat16 --bench
python scripts/benchmark_graph.py --target pto --bench \
    --output /tmp/pto-graph.json --source-dir /tmp/pto-graph-sources
```

The benchmark builds A (unfused, default tile), B (unfused, candidate tile)
and C (fused, candidate tile). A/B compares unfused tiling. For the elementwise
workload, B and C use the same output tile and core mapping, so B/C isolates
fusion. For GEMM, B uses linear epilogue tiles and C uses the Cube's 2-D output
tiles. The report sets `fusion_isolated=false` and records their combined
effect in `combined_epilogue_speedup_B_over_C`.

`--bench` records event and wall samples for all three variants and eager
Torch. `--source-dir` exports each variant's Relax graph and generated kernel
sources. The JSON records the resolved compiler target and adapter for each
kernel. [Benchmark script](../scripts/benchmark_graph.py) · [Hardware verification](verification.md)
