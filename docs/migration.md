# Extraction provenance

Extracted from [tile-ai/tilelang PR #3892](https://github.com/tile-ai/tilelang/pull/3892)
at `7eb3edc8ba7b183c63681aa149a2e7512057d5ec`.

| Original file | External package file |
| --- | --- |
| `tilelang/graph/backend.py` | `src/torch_tilelang/backend.py` |
| `tilelang/graph/config.py` | `src/torch_tilelang/config.py` |
| `tilelang/graph/frontend.py` | `src/torch_tilelang/frontend.py` |
| `tilelang/graph/fusion.py` | `src/torch_tilelang/fusion.py` |
| `tilelang/graph/runtime.py` | `src/torch_tilelang/runtime.py` |
| `tilelang/graph/__init__.py` | `src/torch_tilelang/__init__.py` |
| `tilelang/ascend/graph/elementwise.py` | `src/torch_tilelang/schedules/ascend/elementwise.py` |
| `tilelang/ascend/graph/operators.py` | `src/torch_tilelang/schedules/ascend/operators.py` |
| `tilelang/ascend/graph/schedule.py` | `src/torch_tilelang/schedules/ascend/schedule.py` |
| `tilelang/ascend/graph/__init__.py` | `src/torch_tilelang/schedules/ascend/__init__.py` |
| `testing/python/graph/test_tilelang_graph_runtime.py` | `tests/unit/test_runtime.py` |
| `testing/python/graph/test_tilelang_graph_fusion.py` | `tests/unit/test_fusion.py` |
| `testing/ascend/graph/test_graph_backend.py` | `tests/ascend/test_backend.py` |
| `testing/ascend/graph/test_graph_operators.py` | `tests/ascend/test_operators.py` |
| `testing/ascend/graph/test_graph_schedule.py` | `tests/ascend/test_schedule.py` |
| `testing/ascend/graph/test_graph_pto_cache.py` | `tests/ascend/test_pto_cache.py` |
| `examples/ascend/graph/torch_compile.py` | `examples/ascend/torch_compile.py` |
| `docs/developer_guide/ascend_graph_compile.md` | `docs/ascend.md` |
| `LICENSE` | `LICENSE` |

## Changes made during extraction

- The Python namespace is `torch_tilelang`; Dynamo discovery uses the installed
  `torch_tilelang_ascend` entry point. Import-time registration was removed so
  first-use entry-point loading cannot register the same name twice.
- Target/compiler calls and profiler-error translation live in `compat/`.
  Kernels use TileLang's existing cache and receive caller-owned output buffers.
- Standalone tests own their IR inspection helpers and environment gates. The
  PTO cache subprocess runs from a temporary directory and imports the installed
  package, without relying on TileLang's source tree.
- Explicit sum/softmax dtype conversion occurs before fp32 statistics;
  same-dtype casts preserve input identity; layout-changing dtype methods stay
  in Torch. Focused regressions cover these review findings.

The graph compiler depends on TileLang's compiler and runtime interfaces.
Generic TileLang fixes are maintained in that project rather than distributed
by this package. The installed-wheel smoke test verifies discovery with a
TileLang build that contains neither `tilelang.graph` nor `tilelang.ascend.graph`.
