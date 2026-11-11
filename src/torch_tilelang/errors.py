"""Errors owned by the external graph compiler."""


class MeasurementUnavailableError(RuntimeError):
    """Profiling produced no usable measurements; device failures are separate."""
