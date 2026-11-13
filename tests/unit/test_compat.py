"""Compiler and profiler adapters preserve the graph execution contract."""

from types import SimpleNamespace

import pytest
import torch

from torch_tilelang import GraphConfig
from torch_tilelang.compat import profiler
from torch_tilelang.compat import tilelang as graph_tilelang
from torch_tilelang.errors import MeasurementUnavailableError


def test_typed_profile_unavailability_is_translated(monkeypatch):
    errors = pytest.importorskip("tilelang.profiler.errors")

    def unavailable(*args, **kwargs):
        raise errors.ProfileUnavailableError("no kernel samples")

    monkeypatch.setattr(profiler, "do_bench", unavailable)
    with pytest.raises(MeasurementUnavailableError, match="no kernel samples"):
        profiler.measure(lambda: None, (), torch.device("cpu"), "wall")


def test_profiler_keeps_the_original_execution_error(monkeypatch):
    failure = RuntimeError("device execution failed")

    def broken(*args, **kwargs):
        raise failure

    monkeypatch.setattr(profiler, "do_bench", broken)
    with pytest.raises(RuntimeError) as result:
        profiler.measure(lambda: None, (), torch.device("cpu"), "wall")
    assert result.value is failure


def test_msprof_without_typed_errors_is_unavailable(monkeypatch):
    monkeypatch.setattr(profiler, "ProfileUnavailableError", None)
    with pytest.raises(MeasurementUnavailableError, match="typed profile errors are unsupported"):
        profiler.measure(lambda: None, (), torch.device("cpu"), "msprof")


@pytest.mark.parametrize(
    "variant,requested,automatic,expected,calls",
    [
        ("ascend", "auto", "tvm_ffi", "tvm_ffi", ["auto"]),
        ("pto", "auto", "tvm_ffi", "pto", ["auto", "pto"]),
        ("pto", "auto", "pto", "pto", ["auto"]),
        ("pto", "pto", "tvm_ffi", "pto", ["pto"]),
    ],
)
def test_graph_adapter_policy_is_independent_of_tilelang_auto_preference(monkeypatch, variant, requested, automatic, expected, calls):
    import tilelang.backend

    target = '{"kind": "ascend", "keys": ["pto", "ascend"], "mcpu": "dav-3510"}' if variant == "pto" else GraphConfig().target
    config = GraphConfig(target=target, execution_backend=requested)
    seen = []

    def resolve(target, *, execution_backend):
        assert target == config.target
        seen.append(execution_backend)
        return SimpleNamespace(
            module=SimpleNamespace(name=variant), target=SimpleNamespace(kind=SimpleNamespace(name="ascend")),
            execution_backend=SimpleNamespace(name=automatic if execution_backend == "auto" else execution_backend),
        )

    monkeypatch.setattr(graph_tilelang.Target, "list_kinds", staticmethod(lambda: ["ascend"]))
    monkeypatch.setattr(graph_tilelang.tvm, "get_global_func", lambda *args, **kwargs: object())
    monkeypatch.setattr(tilelang.backend, "create_backend_context", resolve)
    assert graph_tilelang.resolve_graph_context(config).execution_backend.name == expected
    assert seen == calls
    assert config.execution_backend == requested


def test_graph_rejects_explicit_pto_tvm_ffi_even_when_tilelang_allows_it(monkeypatch):
    import tilelang.backend

    context = SimpleNamespace(module=SimpleNamespace(name="pto"), execution_backend=SimpleNamespace(name="tvm_ffi"))
    monkeypatch.setattr(graph_tilelang.Target, "list_kinds", staticmethod(lambda: ["ascend"]))
    monkeypatch.setattr(graph_tilelang.tvm, "get_global_func", lambda *args, **kwargs: object())
    monkeypatch.setattr(tilelang.backend, "create_backend_context", lambda *args, **kwargs: context)
    with pytest.raises(ValueError, match="Invalid execution backend.*tvm_ffi"):
        graph_tilelang.resolve_graph_context(GraphConfig(target="pto", execution_backend="tvm_ffi"))


def test_graph_does_not_fall_back_to_tvm_ffi_when_pto_adapter_is_unavailable(monkeypatch):
    import tilelang.backend

    failure = ValueError("PTO adapter dependencies unavailable")

    def resolve(target, *, execution_backend):
        if execution_backend == "pto":
            raise failure
        return SimpleNamespace(module=SimpleNamespace(name="pto"), execution_backend=SimpleNamespace(name="tvm_ffi"))

    monkeypatch.setattr(graph_tilelang.Target, "list_kinds", staticmethod(lambda: ["ascend"]))
    monkeypatch.setattr(graph_tilelang.tvm, "get_global_func", lambda *args, **kwargs: object())
    monkeypatch.setattr(tilelang.backend, "create_backend_context", resolve)
    with pytest.raises(ValueError) as result:
        graph_tilelang.resolve_graph_context(GraphConfig(target="pto"))
    assert result.value is failure
