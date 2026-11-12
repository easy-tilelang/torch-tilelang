"""Optional pytest settings for the standalone Ascend functional examples."""

import pytest
import torch


def pytest_addoption(parser):
    group = parser.getgroup("Ascend functional examples")
    group.addoption("--device", default="npu:0", help="NPU device for the examples (default: npu:0)")
    group.addoption(
        "--dtype",
        default="all",
        choices=("all", "float16", "bfloat16", "float32"),
        help="Tensor dtype, or all dtypes supported by each example (default: all)",
    )
    group.addoption("--seed", type=int, default=0, help="Random seed for the examples (default: 0)")


def pytest_generate_tests(metafunc):
    if "dtype" not in metafunc.fixturenames:
        return
    supported = metafunc.module.SUPPORTED_DTYPES
    requested = metafunc.config.getoption("--dtype")
    names = supported if requested == "all" else (requested,)
    values = []
    for name in names:
        marks = () if name in supported else pytest.mark.skip(reason=f"{metafunc.module.__name__} supports {', '.join(supported)}")
        values.append(pytest.param(getattr(torch, name), id=name, marks=marks))
    metafunc.parametrize("dtype", values)


@pytest.fixture
def device(pytestconfig):
    import tilelang.testing

    if not tilelang.testing.ascend_backend_compiled():
        pytest.skip("These examples require an Ascend-enabled TileLang build")
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("These examples require an available Ascend NPU")
    selected = torch.device(pytestconfig.getoption("--device"))
    if selected.type != "npu":
        raise pytest.UsageError("Ascend functional examples require --device npu:<index>")
    with torch.npu.device(selected):
        yield selected


@pytest.fixture
def seed(pytestconfig):
    return pytestconfig.getoption("--seed")
