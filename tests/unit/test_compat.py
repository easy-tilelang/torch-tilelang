"""Only typed missing measurements are converted by the profiler boundary."""

import pytest
import torch

from torch_tilelang.compat import profiler
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
