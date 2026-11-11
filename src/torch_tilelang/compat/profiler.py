"""Keep missing profiling evidence distinct from device execution failures."""

from tilelang.profiler import do_bench

from ..errors import MeasurementUnavailableError

try:
    from tilelang.profiler.errors import ProfileUnavailableError
except ModuleNotFoundError as error:
    if error.name != "tilelang.profiler.errors":
        raise
    ProfileUnavailableError = None

_UNAVAILABLE_ERRORS = (ProfileUnavailableError,) if ProfileUnavailableError is not None else ()


def measure(runner, inputs, device, backend: str) -> float:
    """Use TileLang timing and translate only its typed unavailable result."""
    if backend == "msprof" and ProfileUnavailableError is None:
        raise MeasurementUnavailableError(
            "msprof is unavailable with this TileLang build: typed profile errors are unsupported; use event timing"
        )
    try:
        return do_bench(
            lambda: runner(*inputs),
            device=device,
            backend=backend,
            warmup=10,
            rep=30,
            early_stop_baseline=None,
            return_mode="median",
        )
    except _UNAVAILABLE_ERRORS as error:
        raise MeasurementUnavailableError(str(error)) from error
