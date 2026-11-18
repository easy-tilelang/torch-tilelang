"""Hardware-free tests for graph conversion, allocation, partitioning and fallback.

The injected compiler below replaces device kernels only. Tests still exercise
the real FX importer, Relax legalization, call_tir ABI and Torch graph executor.
Native Ascend compilation and stream integration have separate NPU coverage.
"""

from __future__ import annotations

import importlib
import operator
import weakref
from types import SimpleNamespace

import pytest
import torch

from torch_tilelang import GraphConfig, build_runner, fx_to_relax, make_backend
from torch_tilelang.frontend import UnsupportedGraphError, validate_inputs
from tvm import relax, tirx


def _cpu_kernel_compiler(func, _config):
    name = str(func.attrs["global_symbol"]).removeprefix("tir_")
    constants = []
    tirx.stmt_functor.post_order_visit(
        func.body, lambda node: constants.append(float(node.value)) if isinstance(node, tirx.FloatImm) else None
    )

    def kernel(*args):
        inputs, output = args[:-1], args[-1]
        if name.startswith("add"):
            value = inputs[0] + inputs[1] if len(inputs) > 1 else inputs[0] + constants[0]
        elif name.startswith("multiply"):
            value = inputs[0] * inputs[1] if len(inputs) > 1 else inputs[0] * (constants[0] if constants else inputs[0])
        elif name.startswith("square"):
            value = inputs[0].square()
        elif name.startswith("subtract"):
            value = inputs[0] - inputs[1] if len(inputs) > 1 else inputs[0] - constants[0]
        elif name.startswith("divide"):
            value = inputs[0] / inputs[1] if len(inputs) > 1 else inputs[0] / constants[0]
        elif name.startswith("relu"):
            value = torch.relu(inputs[0])
        elif name.startswith(("astype", "cast")):
            value = inputs[0].to(output.dtype)
        elif name.startswith("matmul"):
            value = inputs[0].float() @ inputs[1].float()
        elif name.startswith("sum"):
            value = inputs[0].sum(-1).reshape(output.shape)
        elif name.startswith("mean"):
            value = inputs[0].mean(-1).reshape(output.shape)
        elif name.startswith("max"):
            value = inputs[0].amax(-1).reshape(output.shape)
        elif name.startswith("exp"):
            value = inputs[0].exp()
        elif name.startswith("rsqrt"):
            value = inputs[0].rsqrt()
        else:
            raise AssertionError(f"CPU test compiler has no implementation for {name}")
        output.copy_(value)

    return kernel


def _cpu_backend(monkeypatch, *, config=None, compiler=_cpu_kernel_compiler):
    backend = importlib.import_module("torch_tilelang.backend")
    _require_native_ascend(config)
    monkeypatch.setattr(backend, "validate_inputs", lambda inputs: validate_inputs(inputs, require_npu=False))
    monkeypatch.setattr(backend, "build_runner", lambda mod, cfg: build_runner(mod, cfg, kernel_compiler=compiler))
    return make_backend(config)


def _require_native_ascend(config=None):
    backend = importlib.import_module("torch_tilelang.backend")
    try:
        backend._require_ascend_backend(config)
    except UnsupportedGraphError:
        pytest.skip("This backend test needs an Ascend-enabled TileLang build, but does not need an NPU")


@pytest.mark.parametrize(
    ("target", "execution", "target_backend", "execution_backend"),
    [
        (GraphConfig().target, "auto", "ascend", "tvm_ffi"),
        ("ascend", "auto", "ascend", "tvm_ffi"),
        ("ascend", "tvm_ffi", "ascend", "tvm_ffi"),
        ("pto", "auto", "pto", "pto"),
        ("pto", "pto", "pto", "pto"),
        (GraphConfig().target, "pto", "pto", "pto"),
        ('{"kind": "ascend", "keys": ["pto", "ascend"], "mcpu": "dav-3510"}', "auto", "pto", "pto"),
    ],
)
def test_backend_resolves_target_and_execution_together(monkeypatch, target, execution, target_backend, execution_backend):
    from torch_tilelang.runtime import _resolve_graph_context

    config = GraphConfig(target=target, execution_backend=execution, tile_size=256)
    seen = []

    def compile_kernel(func, normalized):
        seen.append(normalized)
        kernel = _cpu_kernel_compiler(func, normalized)
        kernel.backend_context = _resolve_graph_context(normalized)
        return kernel

    backend = _cpu_backend(monkeypatch, config=config, compiler=compile_kernel)
    gm = torch.fx.symbolic_trace(lambda x, y: torch.relu(x + y))
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    run = backend(gm, (x, y))
    torch.testing.assert_close(run(x, y), gm(x, y))
    status = run.graph_report.partitions[0]
    assert status.compiled
    assert status.target_backend == target_backend
    assert status.execution_backend == execution_backend
    assert all(item.execution_backend == execution_backend and item.tile_size == 256 for item in seen)
    for item in seen:
        context = _resolve_graph_context(item)
        assert context.target.kind.name == "ascend"
        assert context.module.name == target_backend
        assert ("pto" in context.target.keys) == (target_backend == "pto")
    # Normalization is local to the graph; user configuration is immutable.
    assert config.target == target
    assert config.execution_backend == execution


def test_graph_execution_defaults_preserve_plain_ascend():
    from torch_tilelang.runtime import _resolve_graph_context

    config = GraphConfig()
    _require_native_ascend(config)
    context = _resolve_graph_context(config)
    assert config.execution_backend == "auto"
    assert context.module.name == "ascend"
    assert context.execution_backend.name == "tvm_ffi"
    assert context.target.attrs["mcpu"] == "dav-3510"


def test_pto_alias_is_normalized_before_constructing_a_target(monkeypatch):
    from tvm.target import Target
    from torch_tilelang.runtime import _resolve_graph_context

    config = GraphConfig(target="pto")
    _require_native_ascend(config)
    original = Target.__init__

    def construct(self, target, *args, **kwargs):
        assert target != "pto", "PTO is a variant alias, not a TVM target kind"
        return original(self, target, *args, **kwargs)

    monkeypatch.setattr(Target, "__init__", construct)
    context = _resolve_graph_context(config)
    assert context.module.name == "pto"


def test_pto_target_rejects_tvm_ffi_pairing_before_graph_compilation():
    _require_native_ascend(GraphConfig(target="pto"))
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    config = GraphConfig(target="pto", execution_backend="tvm_ffi")
    with pytest.raises(ValueError, match="Invalid execution backend.*tvm_ffi"):
        make_backend(config)(gm, (torch.randn(2, 4), torch.randn(2, 4)))


@pytest.mark.parametrize(
    ("target", "execution", "missing_codegen", "other_target"),
    [
        ("ascend", "tvm_ffi", "target.build.tilelang_ascend", "pto"),
        ("pto", "pto", "target.build.tilelang_pto_without_compile", "ascend"),
    ],
)
def test_codegen_availability_checks_only_the_selected_variant(monkeypatch, target, execution, missing_codegen, other_target):
    from tilelang import tvm
    from torch_tilelang.runtime import _resolve_graph_context

    config = GraphConfig(target=target, execution_backend=execution)
    other_config = GraphConfig(target=other_target)
    _require_native_ascend(config)
    _require_native_ascend(other_config)
    original = tvm.get_global_func

    def get_global_func(name, *args, **kwargs):
        return None if name == missing_codegen else original(name, *args, **kwargs)

    monkeypatch.setattr(tvm, "get_global_func", get_global_func)
    with pytest.raises(UnsupportedGraphError, match="requires native codegen"):
        _resolve_graph_context(config)
    assert _resolve_graph_context(other_config).module.name == other_target


@pytest.mark.parametrize(("target", "execution"), [("ascend", "tvm_ffi"), ("pto", "pto")])
def test_native_kernel_compiler_forwards_the_selected_execution_mode(monkeypatch, target, execution):
    import tilelang
    import torch_tilelang.schedules.ascend.schedule as schedule
    from tilelang.cache import _resolve_cache_dispatch
    from torch_tilelang.runtime import _compile_kernel

    config = GraphConfig(target=target, execution_backend=execution)
    _require_native_ascend(config)
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    mod = fx_to_relax(gm, (torch.randn(2, 4), torch.randn(2, 4)))
    func = mod["add"].with_attr("global_symbol", "add")
    monkeypatch.setattr(schedule, "schedule_primfunc", lambda function, _config: function)
    seen = []

    def compile_kernel(function, *, target, execution_backend, out_idx):
        assert out_idx == []
        cache, context, _ = _resolve_cache_dispatch(target, None, execution_backend, False)
        seen.append((function, context, cache))
        return _cpu_kernel_compiler(function, config)

    monkeypatch.setattr(tilelang, "compile", compile_kernel)
    kernel = _compile_kernel(func, config)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    output = torch.empty_like(x)
    kernel(x, y, output)
    torch.testing.assert_close(output, x + y)
    assert seen[0][1].module.name == target
    assert seen[0][1].execution_backend.name == execution


def test_graph_runner_preserves_nested_output_and_old_tensors():
    gm = torch.fx.symbolic_trace(lambda x, y: (torch.relu(x + y), (x * y, x)))
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    runner = build_runner(fx_to_relax(gm, (x, y)), kernel_compiler=_cpu_kernel_compiler)
    first = runner(x, y)
    snapshot = first[0].clone()
    second = runner(x + 3, y)
    torch.testing.assert_close(first, gm(x, y))
    torch.testing.assert_close(second, gm(x + 3, y))
    torch.testing.assert_close(first[0], snapshot)
    assert first[1][1] is x
    assert first[0].data_ptr() != second[0].data_ptr()
    assert runner.kernel_count == 3


def test_graph_runner_counts_calls_and_rejects_shape_mismatch():
    gm = torch.fx.symbolic_trace(lambda x, y: (x + y) + y)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    runner = build_runner(fx_to_relax(gm, (x, y)), kernel_compiler=_cpu_kernel_compiler)
    assert runner.kernel_count == 2
    # Identical legalizations may share a PrimFunc; actual calls still count.
    assert len(runner.kernels) <= runner.kernel_count
    with pytest.raises(ValueError, match="compiled shape"):
        runner(torch.randn(3, 4), torch.randn(3, 4))


def test_only_native_runner_requires_npu_inputs(monkeypatch):
    runtime = importlib.import_module("torch_tilelang.runtime")
    context = SimpleNamespace(
        target="ascend", module=SimpleNamespace(name="ascend"), execution_backend=SimpleNamespace(name="tvm_ffi")
    )
    monkeypatch.setattr(runtime, "_resolve_graph_context", lambda config: context)

    def compile_kernel(func, config):
        kernel = _cpu_kernel_compiler(func, config)
        kernel.backend_context = context
        return kernel

    def fail_if_launched(*args):
        pytest.fail("Native kernel received CPU tensors")

    monkeypatch.setattr(runtime, "_compile_kernel", lambda func, config: fail_if_launched)
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    mod = fx_to_relax(gm, (x, y))
    native = build_runner(mod)
    with pytest.raises(UnsupportedGraphError, match="requires NPU tensors"):
        native(x, y)
    injected = build_runner(mod, kernel_compiler=compile_kernel)
    torch.testing.assert_close(injected(x, y), x + y)


def test_graph_runner_rejects_non_tensor_return_before_compilation():
    builder = relax.BlockBuilder()
    value = relax.Var("value", relax.TensorStructInfo((8,), "float32"))
    with builder.function("main", [value]):
        builder.emit_func_output(relax.ShapeExpr([8]))
    with pytest.raises(UnsupportedGraphError, match="must be tensors"):
        build_runner(builder.get(), kernel_compiler=_cpu_kernel_compiler)


def test_graph_runner_releases_intermediate_after_its_last_consumer():
    first_output = []
    observed_release = []

    def compile_kernel(func, cfg):
        kernel = _cpu_kernel_compiler(func, cfg)
        name = str(func.attrs["global_symbol"])

        def run(*args):
            if name.startswith("add"):
                first_output.append(weakref.ref(args[-1]))
            if name.startswith("multiply"):
                observed_release.append(first_output[0]() is None)
            kernel(*args)

        return run

    gm = torch.fx.symbolic_trace(lambda x, y: torch.relu(x + y) * y)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    runner = build_runner(fx_to_relax(gm, (x, y)), kernel_compiler=compile_kernel)
    torch.testing.assert_close(runner(x, y), gm(x, y))
    assert observed_release == [True]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_frontend_keeps_matmul_and_reduction_output_dtype(dtype):
    x, y = torch.randn(2, 4).to(dtype), torch.randn(4, 8).to(dtype)
    gm = torch.fx.symbolic_trace(lambda a, b: (a @ b, a.sum(-1), a.amax(-1)))
    mod = fx_to_relax(gm, (x, y))
    runner = build_runner(mod, kernel_compiler=_cpu_kernel_compiler)
    actual = runner(x, y)
    torch.testing.assert_close(actual, gm(x, y))
    assert all(output.dtype == dtype for output in actual)
    assert runner.has_reduction


def test_frontend_honors_explicit_reduction_dtype():
    x = torch.randn(2, 4, dtype=torch.float16)
    gm = torch.fx.symbolic_trace(lambda a: a.sum(-1, dtype=torch.float32))
    runner = build_runner(fx_to_relax(gm, (x,)), kernel_compiler=_cpu_kernel_compiler)
    torch.testing.assert_close(runner(x), x.sum(-1, dtype=torch.float32))


@pytest.mark.parametrize(
    "model",
    [lambda x: x.sum(-1, dtype=torch.float16), lambda x: x.softmax(-1, dtype=torch.float16)],
    ids=["sum", "softmax"],
)
def test_explicit_dtype_preserves_input_rounding_on_later_calls(monkeypatch, model):
    backend = _cpu_backend(monkeypatch, config=GraphConfig(strict=True))
    gm = torch.fx.symbolic_trace(model)
    first = torch.zeros(1, 2)
    run = backend(gm, (first,))
    torch.testing.assert_close(run(first), gm(first))
    later = torch.tensor([[65520.0, -65520.0]])
    torch.testing.assert_close(run(later), gm(later), equal_nan=True)
    status, = run.graph_report.partitions
    assert status.compiled_calls == 2 and status.fallback_calls == 0


@pytest.mark.parametrize("dtype,scalar", [(torch.float16, 1.0004), (torch.bfloat16, 1.003)])
def test_scalar_opmath_preserves_rounding_on_later_calls(monkeypatch, dtype, scalar):
    backend = _cpu_backend(monkeypatch, config=GraphConfig(strict=True))
    gm = torch.fx.symbolic_trace(lambda x: x * scalar - x)
    first = torch.zeros(8, dtype=dtype)
    run = backend(gm, (first,))
    torch.testing.assert_close(run(first), gm(first), rtol=0, atol=0)
    later = torch.full_like(first, 1000)
    torch.testing.assert_close(run(later), gm(later), rtol=0, atol=0)
    status, = run.graph_report.partitions
    assert status.compiled_calls == 2 and status.fallback_calls == 0


@pytest.mark.parametrize(
    "dtype,values",
    [(torch.float16, [[65520.0, -65520.0]]), (torch.bfloat16, [[1.004, -1.0]])],
)
def test_mean_dtype_casts_input_before_statistics(dtype, values):
    first = torch.zeros(1, 2)
    gm = torch.fx.symbolic_trace(lambda x: x.mean(-1, dtype=dtype))
    runner = build_runner(fx_to_relax(gm, (first,)), kernel_compiler=_cpu_kernel_compiler)
    for value in (first, torch.tensor(values)):
        # CPU mean(dtype=...) can reduce the original fp32 input directly.
        # Ascend casts the input first, so use that explicit reference here.
        expected = value.to(dtype).float().mean(-1).to(dtype)
        torch.testing.assert_close(runner(value), expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize(
    "model,operation",
    [
        (lambda x: x + 0.0, tirx.Add),
        (lambda x: x * 0.0, tirx.Mul),
        (lambda x: x * -0.0, tirx.Mul),
        (lambda x: x - -0.0, tirx.Sub),
        (lambda x: 0.0 / x, tirx.Div),
    ],
    ids=["add_zero", "mul_zero", "mul_negative_zero", "sub_negative_zero", "zero_div"],
)
def test_zero_scalar_legalization_preserves_ieee_values(model, operation):
    first = torch.ones(8)
    gm = torch.fx.symbolic_trace(model)
    mod = fx_to_relax(gm, (first,))
    operations = []
    for func in mod.functions.values():
        if isinstance(func, tirx.PrimFunc):
            tirx.stmt_functor.post_order_visit(func.body, lambda node: operations.append(node) if isinstance(node, operation) else None)
    # The injected CPU compiler is not a TIR interpreter. Check that the
    # operation actually survived legalization as well as checking execution.
    assert operations
    runner = build_runner(mod, kernel_compiler=_cpu_kernel_compiler)
    later = torch.tensor([float("inf"), -float("inf"), float("nan"), -1.0, 1.0, -0.0, 0.0, 2.0])
    for value in (first, later):
        actual, expected = runner(value), gm(value)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
        zeros = (actual == 0) & (expected == 0)
        assert torch.equal(torch.signbit(actual)[zeros], torch.signbit(expected)[zeros])


@pytest.mark.parametrize(
    "model,dtype",
    [
        (lambda x: x.float(), torch.float32),
        (lambda x: x.half(), torch.float16),
        (lambda x: x.bfloat16(), torch.bfloat16),
        (lambda x: x.to(torch.float32), torch.float32),
        (lambda x: x.type_as(x), torch.float32),
    ],
    ids=["float", "half", "bfloat16", "to", "type_as"],
)
def test_same_dtype_conversion_preserves_input_alias(model, dtype):
    x = torch.ones(2, 4, dtype=dtype)
    gm = torch.fx.symbolic_trace(model)
    runner = build_runner(fx_to_relax(gm, (x,)), kernel_compiler=_cpu_kernel_compiler)
    actual = runner(x)
    assert actual is x
    actual.add_(1)
    torch.testing.assert_close(x, torch.full_like(x, 2))


@pytest.mark.parametrize("method", ["float", "half", "bfloat16"])
def test_dtype_method_layout_change_stays_in_torch(monkeypatch, method):
    backend = _cpu_backend(monkeypatch)
    gm = torch.fx.symbolic_trace(lambda x: getattr(x, method)(memory_format=torch.channels_last))
    x = torch.ones(1, 2, 3, 4)
    run = backend(gm, (x,))
    actual = run(x)
    assert actual.stride() == gm(x).stride()
    assert actual.is_contiguous(memory_format=torch.channels_last)
    assert run.graph_report.partitions == []


def test_frontend_preserves_literal():
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    gm = torch.fx.symbolic_trace(lambda a, b: (a + b) + 5)
    runner = build_runner(fx_to_relax(gm, (x, y)), kernel_compiler=_cpu_kernel_compiler)
    torch.testing.assert_close(runner(x, y), gm(x, y))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_scaled_add_is_rejected_to_preserve_opmath_rounding(dtype):
    gm = torch.fx.symbolic_trace(lambda a, b: torch.add(a, b, alpha=2))
    x, y = torch.randn(2, 4).to(dtype), torch.randn(2, 4).to(dtype)
    with pytest.raises(UnsupportedGraphError, match="alpha=1"):
        fx_to_relax(gm, (x, y))


@pytest.mark.parametrize("mode", ["device", "copy", "layout"])
def test_to_rejects_device_copy_and_layout_changes(mode):
    if mode == "device":
        gm = torch.fx.symbolic_trace(lambda x: x.to(device="cpu"))
    elif mode == "copy":
        gm = torch.fx.symbolic_trace(lambda x: x.to(torch.float32, copy=True))
    else:
        gm = torch.fx.symbolic_trace(lambda x: x.to(memory_format=torch.channels_last))
    with pytest.raises(UnsupportedGraphError, match="device, copy, or layout changes"):
        fx_to_relax(gm, (torch.randn(1, 2, 3, 4),))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_softmax_baseline_has_separate_reductions_and_fp32_statistics(dtype):
    x = torch.randn(2, 16).to(dtype)
    gm = torch.fx.symbolic_trace(lambda a: torch.softmax(a, dim=-1))
    mod = fx_to_relax(gm, (x,))
    runner = build_runner(mod, kernel_compiler=_cpu_kernel_compiler)
    torch.testing.assert_close(runner(x), gm(x))
    assert runner.kernel_count >= 5
    assert runner.output_reductions == (True,)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_arithmetic_keeps_fp32_statistics(dtype):
    def model(x, weight):
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-5)
        return value.to(x.dtype) * weight

    x, weight = torch.randn(2, 16).to(dtype), torch.randn(16).to(dtype)
    gm = torch.fx.symbolic_trace(model)
    runner = build_runner(fx_to_relax(gm, (x, weight)), kernel_compiler=_cpu_kernel_compiler)
    torch.testing.assert_close(runner(x, weight), gm(x, weight))


def test_non_square_pow_remains_a_torch_partition_boundary(monkeypatch):
    backend = _cpu_backend(monkeypatch)
    gm = torch.fx.symbolic_trace(lambda x, y: torch.relu(torch.pow(x + y, 3)))
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    run = backend(gm, (x, y))
    torch.testing.assert_close(run(x, y), gm(x, y))
    assert len(run.graph_report.partitions) == 2


def test_frontend_supports_aten_overloads():
    graph = torch.fx.Graph()
    x, y = graph.placeholder("x"), graph.placeholder("y")
    output = graph.call_function(torch.ops.aten.add.Tensor, (x, y))
    graph.output(output)
    gm = torch.fx.GraphModule({}, graph)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    runner = build_runner(fx_to_relax(gm, (x, y)), kernel_compiler=_cpu_kernel_compiler)
    torch.testing.assert_close(runner(x, y), x + y)


def test_partitioned_backend_keeps_unsupported_torch_operator(monkeypatch):
    backend = _cpu_backend(monkeypatch)
    gm = torch.fx.symbolic_trace(lambda x, y: torch.relu(torch.sin(x + y)))
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    run = backend(gm, (x, y))
    torch.testing.assert_close(run(x, y), gm(x, y))
    assert len(run.graph_report.partitions) == 2
    assert all(part.compiled and part.kernel_count == 1 for part in run.graph_report.partitions)
    torch.testing.assert_close(run(x + 1, y), gm(x + 1, y))
    assert all(part.compiled_calls == 2 for part in run.graph_report.partitions)


def test_partition_reads_updated_module_buffers(monkeypatch):
    backend = _cpu_backend(monkeypatch)

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.ones(2, 4))

        def forward(self, x):
            return torch.relu(x + self.weight)

    gm = torch.fx.symbolic_trace(Model())
    x = torch.randn(2, 4)
    run = backend(gm, (x,))
    torch.testing.assert_close(run(x), gm(x))
    gm.weight.add_(4)
    torch.testing.assert_close(run(x), gm(x))
    assert all(part.compiled_calls == 2 for part in run.graph_report.partitions)


@pytest.mark.parametrize("location", ["activation", "graph"])
@pytest.mark.parametrize("kind", ["pre", "post"])
def test_module_forward_hooks_keep_the_original_graph(monkeypatch, location, kind):
    backend = _cpu_backend(monkeypatch)
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_module("activation", (x,)))
    gm = torch.fx.GraphModule({"activation": torch.nn.ReLU()}, graph)
    module = gm.activation if location == "activation" else gm
    calls = []

    def pre_hook(_module, args):
        calls.append("pre")
        return (args[0] + 1,)

    def post_hook(_module, _args, output):
        calls.append("post")
        return output * 2

    if kind == "pre":
        module.register_forward_pre_hook(pre_hook)
    else:
        module.register_forward_hook(post_hook)
    first = torch.zeros(8)
    run = backend(gm, (first,))
    for value in (first, torch.ones_like(first)):
        expected = torch.relu(value + 1) if kind == "pre" else torch.relu(value) * 2
        torch.testing.assert_close(run(value), expected)
    assert len(calls) == 2
    assert not run.graph_report.partitions
    assert run.graph_report.fallback_reason
    with pytest.raises(UnsupportedGraphError, match="opaque effects"):
        make_backend(GraphConfig(strict=True))(gm, (first,))
    with pytest.raises(UnsupportedGraphError, match="forward hooks"):
        fx_to_relax(gm, (first,))
    assert len(calls) == 2


def test_torch_compile_reuses_the_plan_for_new_tensor_values(monkeypatch):
    backend = _cpu_backend(monkeypatch)
    reports = []

    def capture(gm, inputs):
        run = backend(gm, inputs)
        reports.append(run.graph_report)
        return run

    def model(x, y):
        return torch.relu(x + y)

    compiled = torch.compile(model, backend=capture, fullgraph=True)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    torch.testing.assert_close(compiled(x, y), model(x, y))
    x.add_(4)
    torch.testing.assert_close(compiled(x, y), model(x, y))
    assert len(reports) == 1
    assert reports[0].partitions[0].compiled_calls == 2


def test_requires_grad_falls_back_but_no_grad_allows_inference(monkeypatch):
    backend = _cpu_backend(monkeypatch)
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    x, y = torch.randn(2, 4, requires_grad=True), torch.randn(2, 4)
    run = backend(gm, (x, y))
    actual = run(x, y)
    assert actual.requires_grad
    assert run.graph_report.partitions[0].fallback_calls == 1
    with torch.no_grad():
        torch.testing.assert_close(run(x, y), gm(x, y))
    assert run.graph_report.partitions[0].compiled_calls == 1


def test_compile_failure_reports_eager_fallback(monkeypatch):
    def reject(_func, _config):
        raise UnsupportedGraphError("unsupported schedule")

    backend = _cpu_backend(monkeypatch, compiler=reject)
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    run = backend(gm, (x, y))
    torch.testing.assert_close(run(x, y), gm(x, y))
    status = run.graph_report.partitions[0]
    assert not status.compiled and status.kernel_count == 0
    assert status.fallback_calls == 1
    assert "unsupported schedule" in status.fallback_reason


@pytest.mark.parametrize("target", ["ascend", "pto"])
def test_fusion_compilation_failure_preserves_a_diagnostic(monkeypatch, target):
    import torch_tilelang.fusion as fusion

    def reject(_mod):
        raise UnsupportedGraphError("unsupported fused layout")

    monkeypatch.setattr(fusion, "make_fusion_candidate", reject)
    backend = _cpu_backend(monkeypatch, config=GraphConfig(target=target, fusion="profile"))
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    run = backend(gm, (x, y))
    torch.testing.assert_close(run(x, y), gm(x, y))
    status = run.graph_report.partitions[0]
    assert status.compiled
    assert not status.fusion_decision.accepted
    assert "candidate compilation failed" in status.fusion_decision.reason
    assert "unsupported fused layout" in status.fusion_decision.reason
    assert status.target_backend == target
    assert status.execution_backend == ("pto" if target == "pto" else "tvm_ffi")


def test_runtime_failure_is_not_retried_in_torch(monkeypatch):
    def fail(_func, _config):
        def kernel(*_args):
            raise RuntimeError("device execution failed")

        return kernel

    backend = _cpu_backend(monkeypatch, compiler=fail)
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    run = backend(gm, (x, y))
    with pytest.raises(RuntimeError, match="device execution failed"):
        run(x, y)
    assert run.graph_report.partitions[0].fallback_calls == 0


def test_mutation_keeps_the_entire_original_graph():
    _require_native_ascend()

    def model(x):
        x.add_(1)
        return x + x

    gm = torch.fx.symbolic_trace(model)
    x = torch.randn(2, 4)
    run = make_backend()(gm, (x,))
    assert run.graph_report.fallback_reason
    assert run.graph_report.partitions == []
    expected_input = x.clone()
    expected = gm(expected_input)
    actual = run(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, expected_input)


def test_strict_backend_propagates_compilation_failure(monkeypatch):
    def reject(_func, _config):
        raise UnsupportedGraphError("unsupported schedule")

    backend = _cpu_backend(monkeypatch, config=GraphConfig(strict=True), compiler=reject)
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    with pytest.raises(UnsupportedGraphError, match="unsupported schedule"):
        backend(gm, (x, y))(x, y)


def test_strict_backend_requires_an_ascend_partition():
    _require_native_ascend()
    gm = torch.fx.symbolic_trace(lambda x: torch.sin(x))
    with pytest.raises(UnsupportedGraphError, match="no supported Ascend partitions"):
        make_backend(GraphConfig(strict=True))(gm, (torch.ones(8),))


@pytest.mark.parametrize(
    "options",
    [
        {"fusion": "always"},
        {"tile_size": 0},
        {"benchmark_repeats": 2},
        {"min_speedup": float("nan")},
        {"benchmark_backend": "wall"},
        {"execution_backend": "cython"},
        {"execution_backend": "nvrtc"},
    ],
)
def test_invalid_graph_options_fail_early(options):
    with pytest.raises(ValueError):
        GraphConfig(**options)


def test_non_npu_input_uses_explicit_fallback():
    _require_native_ascend()
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    run = make_backend()(gm, (x, y))
    torch.testing.assert_close(run(x, y), gm(x, y))
    assert run.graph_report.partitions[0].fallback_reason
    assert not run.graph_report.partitions[0].compiled


def test_missing_ascend_backend_has_clear_diagnostic(monkeypatch):
    from tvm.target import Target

    monkeypatch.setattr(Target, "list_kinds", staticmethod(lambda: []))
    gm = torch.fx.symbolic_trace(lambda x, y: x + y)
    inputs = (torch.randn(2, 4), torch.randn(2, 4))
    with pytest.raises(UnsupportedGraphError, match="rebuild with USE_ASCEND=ON"):
        make_backend()(gm, inputs)


@pytest.mark.parametrize(
    ("target", "extra_args", "kwargs"),
    [
        (operator.iadd, (1,), {}),
        (torch.nn.functional.relu, (True,), {}),
        (torch.nn.functional.relu, (), {"inplace": True}),
        (torch.nn.functional.batch_norm, (None, None), {"training": True}),
    ],
)
def test_effects_gate_rejects_inplace_and_opaque_torch_calls(target, extra_args, kwargs):
    from torch_tilelang.backend import _has_unsafe_effects

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    result = graph.call_function(target, (x, *extra_args), kwargs)
    graph.output(result)
    assert _has_unsafe_effects(torch.fx.GraphModule({}, graph))


def test_effects_gate_rejects_unknown_methods_randomness_and_overridden_modules():
    from torch_tilelang.backend import _has_unsafe_effects

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_method("retain_grad", (x,)))
    assert _has_unsafe_effects(torch.fx.GraphModule({}, graph))

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_function(torch.ops.aten.rand_like.default, (x,)))
    assert _has_unsafe_effects(torch.fx.GraphModule({}, graph))

    class DifferentReLU(torch.nn.ReLU):
        def forward(self, x):
            return x * 3

    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    graph.output(graph.call_module("activation", (x,)))
    assert _has_unsafe_effects(torch.fx.GraphModule({"activation": DifferentReLU()}, graph))


def test_execution_does_not_read_graph_metadata_after_construction():
    gm = torch.fx.symbolic_trace(lambda x, y: (torch.relu(x + y), (x * y, x)))
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    runner = build_runner(fx_to_relax(gm, (x, y)), kernel_compiler=_cpu_kernel_compiler)

    class UnavailableIR:
        def __getattr__(self, name):
            raise AssertionError(f"Execution attempted to read IR attribute {name}")

    runner.main = runner.mod = UnavailableIR()
    for value in (x, x + 2):
        actual = runner(value, y)
        torch.testing.assert_close(actual, gm(value, y))
        assert actual[1][1] is value


def test_unused_call_output_is_released_and_duplicate_inputs_keep_identity():
    x = torch.randn(2, 4)
    original = fx_to_relax(torch.fx.symbolic_trace(lambda a, b: a + b), (x, x))
    add = next(func for func in original.functions.values() if isinstance(func, tirx.PrimFunc))
    builder = relax.BlockBuilder()
    gv = builder.add_func(add, "add")
    info = relax.TensorStructInfo((2, 4), "float32")
    a, b = relax.Var("a", info), relax.Var("b", info)
    with builder.function("main", [a, b]):
        builder.emit(relax.call_tir(gv, (a, b), out_sinfo=info))
        result = builder.emit(relax.call_tir(gv, (a, a), out_sinfo=info))
        builder.emit_func_output(result)
    refs = []

    def compile_kernel(func, config):
        kernel = _cpu_kernel_compiler(func, config)

        def run(*args):
            if refs:
                assert refs[0]() is None
                assert args[0] is args[1]
            refs.append(weakref.ref(args[-1]))
            kernel(*args)
        return run

    runner = build_runner(builder.get(), kernel_compiler=compile_kernel)
    actual = runner(x, x + 1)
    torch.testing.assert_close(actual, x + x)
    assert len(refs) == 2


def test_multi_output_kernel_projection_and_nested_aliases():
    buffers = [tirx.decl_buffer((2, 4), "float32", name=name) for name in ("input", "first", "second")]
    pair = tirx.PrimFunc([b.data for b in buffers], tirx.Evaluate(0), buffer_map={b.data: b for b in buffers})
    builder = relax.BlockBuilder()
    gv = builder.add_func(pair.with_attr("global_symbol", "pair"), "pair")
    info = relax.TensorStructInfo((2, 4), "float32")
    xvar = relax.Var("x", info)
    with builder.function("main", [xvar]):
        pair_value = builder.emit(relax.call_tir(gv, (xvar,), out_sinfo=[info, info]))
        first = builder.emit(relax.TupleGetItem(pair_value, 0))
        second = builder.emit(relax.TupleGetItem(pair_value, 1))
        builder.emit_func_output(relax.Tuple([first, relax.Tuple([second, xvar, first])]))

    def compile_pair(func, config):
        def run(x, a, b):
            a.copy_(x + 1)
            b.copy_(x * 2)
        return run

    x = torch.randn(2, 4)
    runner = build_runner(builder.get(), kernel_compiler=compile_pair)
    first = runner(x)
    second = runner(x + 3)
    torch.testing.assert_close(first, (x + 1, (x * 2, x, x + 1)))
    torch.testing.assert_close(second, (x + 4, ((x + 3) * 2, x + 3, x + 4)))
    assert first[0] is first[1][2] and first[1][1] is x
    assert first[0].data_ptr() != second[0].data_ptr()
    assert first[1][0].data_ptr() != second[1][0].data_ptr()


def test_repeated_constant_is_loaded_once_per_request(monkeypatch):
    import numpy as np

    info = relax.TensorStructInfo((2, 4), "float32")
    xvar = relax.Var("x", info)
    constant = relax.const(np.arange(8, dtype="float32").reshape(2, 4))
    builder = relax.BlockBuilder()
    with builder.function("main", [xvar]):
        builder.emit_func_output(relax.Tuple([constant, relax.Tuple([constant, xvar])]))
    copies = []
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    class ConstantTransfer:
        def to(self, *, device):
            copies.append(device)
            return source.to(device=device)

    # Count logical constant transfers, not Torch's internal redispatch of
    # Tensor.to under the device context.
    monkeypatch.setattr(torch, "from_dlpack", lambda value: ConstantTransfer())
    runner = build_runner(builder.get(), kernel_compiler=lambda *args: pytest.fail("No kernel expected"))
    x = torch.randn(2, 4)
    for _ in range(2):
        before = len(copies)
        actual = runner(x)
        assert len(copies) == before + 1
        torch.testing.assert_close(actual[0], source)
        assert actual[0] is actual[1][0] and actual[1][1] is x


def test_overlapping_requests_have_independent_execution_slots():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2)

    def compile_kernel(func, config):
        kernel = _cpu_kernel_compiler(func, config)
        name = str(func.attrs["global_symbol"])

        def run(*args):
            if name.startswith("add"):
                barrier.wait(timeout=10)
            kernel(*args)
        return run

    model = torch.fx.symbolic_trace(lambda x, y: torch.relu(x + y) * y)
    x, y = torch.randn(2, 4), torch.randn(2, 4)
    runner = build_runner(fx_to_relax(model, (x, y)), kernel_compiler=compile_kernel)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(runner, x, y)
        second = pool.submit(runner, x + 3, y + 1)
        torch.testing.assert_close(first.result(), model(x, y))
        torch.testing.assert_close(second.result(), model(x + 3, y + 1))


def test_failed_signature_preserves_cached_runner_report(monkeypatch):
    from unittest.mock import Mock

    compiler = Mock(wraps=_cpu_kernel_compiler)
    backend = _cpu_backend(monkeypatch, config=GraphConfig(strict=True), compiler=compiler)
    model = torch.fx.symbolic_trace(lambda x: x + 1)
    x = torch.randn(4)
    run = backend(model, (x,))
    torch.testing.assert_close(run(x), model(x))
    status, = run.graph_report.partitions
    with pytest.raises(UnsupportedGraphError, match="float16, bfloat16, and float32"):
        run(x.double())

    assert status.compiled and status.kernel_count == 1
    assert status.compiled_calls == 1 and status.fallback_calls == 0
    torch.testing.assert_close(run(x), model(x))
    assert status.compiled_calls == 2 and status.fallback_calls == 0
    compiler.assert_called_once()


def test_overlapping_signatures_count_the_selected_runners(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event
    from torch_tilelang.backend import PartitionReport, _CompiledPartition

    _cpu_backend(monkeypatch)
    model = torch.fx.symbolic_trace(lambda x: x + 1)
    status = PartitionReport("add")
    partition = _CompiledPartition(model, GraphConfig(), status)
    compile_lock = partition._compile_lock
    first_unlocked, resume = Event(), Event()

    class PauseAfterFirstUnlock:
        def __enter__(self):
            compile_lock.acquire()

        def __exit__(self, *exc):
            compile_lock.release()
            if not first_unlocked.is_set():
                first_unlocked.set()
                assert resume.wait(timeout=10)

    partition._compile_lock = PauseAfterFirstUnlock()
    x = torch.randn(4)
    with ThreadPoolExecutor(max_workers=1) as pool:
        compiled = pool.submit(partition, x)
        try:
            assert first_unlocked.wait(timeout=10)
            torch.testing.assert_close(partition(x.double()), model(x.double()))
        finally:
            resume.set()
        torch.testing.assert_close(compiled.result(), model(x))
    assert status.compiled_calls == 1 and status.fallback_calls == 1
