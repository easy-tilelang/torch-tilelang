"""Plan contiguous pure FX regions without moving any call across a native operation."""

from __future__ import annotations

from dataclasses import dataclass
import operator

import torch
from torch import fx
from torch.fx.passes.utils.fuser_utils import erase_nodes, fuse_as_graphmodule, insert_subgm

from .frontend import UnsupportedGraphError


_CAST_METHODS = {"to", "type_as", "float", "half", "bfloat16"}
_BINARY_ARITHMETIC = {"add", "sub", "subtract", "mul", "multiply", "div", "divide", "truediv", "true_divide"}


@dataclass(frozen=True)
class NativeNodeReport:
    """A planned native boundary, independent of any later compilation fallback."""

    name: str
    target: str
    reason: str


def target_name(node: fx.Node) -> str:
    target = node.target
    if isinstance(target, (str, torch._ops.OpOverload, torch._ops.OpOverloadPacket)):
        return str(target)
    return f"{target.__module__}.{target.__qualname__}" if hasattr(target, "__qualname__") else str(target)


def _metadata(node, placeholders):
    return node.meta.get("example_value", node.meta.get("val", placeholders.get(node)))


def _native_reason(node, submodules, placeholders, support, validate_inputs):
    target = node.target
    if node.op == "call_function":
        # Operator names alone cannot establish the semantics of a custom
        # overload: even a supposedly functional vendor schema may write a cache.
        name = getattr(target, "__name__", "").split(".", 1)[0]
        if isinstance(target, torch._ops.OpOverload):
            schema = target._schema
            if schema.is_mutable or any(ret.alias_info is not None for ret in schema.returns):
                return "native operator with mutation or aliasing"
            if target.namespace != "aten":
                return "native external operator"
            if target._overloadname not in {"default", "Tensor", "Scalar", "Tensor_Scalar", "dim", "dim_IntList", "int"}:
                return "native ATen overload outside the TileLang capability set"
        elif not any(target is getattr(module, name, None) for module in (torch, operator, torch.nn.functional)):
            return "native external or opaque callable"
    elif node.op == "call_module":
        module = submodules[target]
        name = "softmax" if type(module) is torch.nn.Softmax else "relu"
    else:
        name = str(target)
        if name.endswith("_") and not name.startswith("__"):
            return "native in-place method"

    if node.kwargs.get("out") is not None or node.kwargs.get("inplace", False):
        return "native operation with an output buffer or in-place write"
    if name == "relu":
        if ((len(node.args) > 1 and node.args[1] is not False)
                or (node.op == "call_module" and getattr(submodules[target], "inplace", False))):
            return "native in-place activation"
    if not support.is_node_supported(submodules, node):
        return "native operator outside the TileLang capability set"
    if not node.args or (name in _BINARY_ARITHMETIC | {"matmul", "mm"} and len(node.args) < 2):
        return "TileLang requires tensor operands passed positionally"
    if node.kwargs.get("rounding_mode") is not None:
        return "native division rounding mode"
    if not node.users:
        return "unused result retained in original order"

    inputs = [_metadata(arg, placeholders) for arg in node.all_input_nodes]
    output = _metadata(node, placeholders)
    if not inputs or not all(isinstance(value, torch.Tensor) for value in (*inputs, output)):
        return "missing static tensor signature or non-tensor operand"
    if any(type(dim) is not int for value in (*inputs, output) for dim in value.shape):
        return "symbolic tensor dimensions are outside the static capability set"
    try:
        validate_inputs([*inputs, output])
    except UnsupportedGraphError as error:
        return str(error)

    if name in _CAST_METHODS and inputs[0].dtype == output.dtype:
        return "same-dtype conversion may return the input"
    if name in _BINARY_ARITHMETIC and len(inputs) == 2 and output.dtype != torch.promote_types(inputs[0].dtype, inputs[1].dtype):
        return "native scalar-tensor dtype promotion differs from TileLang tensor promotion"
    if name in {"mm", "matmul"}:
        if len(inputs) != 2 or any(x.ndim != 2 for x in inputs):
            return "GEMM requires two matrices"
        lhs, rhs = inputs
        if lhs.dtype not in (torch.float16, torch.bfloat16) or lhs.dtype != rhs.dtype or any(
            size % 16 for size in (*lhs.shape, *rhs.shape)
        ):
            return "GEMM requires matching fp16/bf16 matrices and M/N/K divisible by 16"
    if name in {"sum", "mean", "amax", "softmax"}:
        x = inputs[0]
        if name == "softmax" and len(node.args) > 2:
            return "native softmax with additional positional arguments"
        if node.op == "call_module":
            dim = submodules[target].dim
        else:
            dim = node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else None)
        if isinstance(dim, (tuple, list)) and len(dim) == 1:
            dim = dim[0]
        if dim is None and x.ndim == 1 and name != "softmax":
            dim = 0
        if x.ndim == 0 or type(dim) is not int or dim not in (-1, x.ndim - 1) or x.shape[-1] > 16384:
            return "only final-axis reductions of up to 16384 elements are scheduled"

    # The runner allocates fresh dense outputs. Even contiguous singleton axes
    # may have observable noncanonical strides (e.g. from an input view).
    stride = 1
    for extent, actual in zip(reversed(output.shape), reversed(output.stride())):
        if actual != stride:
            return "output stride differs from TileLang's contiguous allocation"
        stride *= extent
    return None


def partition_in_order(gm, example_inputs, support, validate_inputs, report):
    placeholders = dict(zip((n for n in gm.graph.nodes if n.op == "placeholder"), example_inputs))
    submodules = dict(gm.named_modules())
    regions = []
    region = []
    for node in gm.graph.nodes:
        reason = None
        if node.op in ("call_function", "call_method", "call_module"):
            reason = _native_reason(node, submodules, placeholders, support, validate_inputs)
            if reason is None:
                region.append(node)
                continue
        # A get_attr may be a fresh read of live state. Never combine across it.
        if region:
            # An opaque operator's fake dtype may differ from its runtime
            # dtype. A cast-only region could then be an alias with no kernel.
            if all(n.op == "call_method" and n.target in _CAST_METHODS for n in region):
                for cast in region:
                    report.native_nodes.append(NativeNodeReport(cast.name, target_name(cast), "standalone dtype conversions stay native"))
            else:
                regions.append(region)
            region = []
        if reason is not None:
            report.native_nodes.append(NativeNodeReport(node.name, target_name(node), reason))

    planned = {}
    index = 0
    for nodes in regions:
        while hasattr(gm, f"tilelang_partition_{index}"):
            index += 1
        name = f"tilelang_partition_{index}"
        index += 1
        planned[name] = (tuple(n.name for n in nodes), tuple(target_name(n) for n in nodes))
        # This is a consecutive slice of the original graph, so inserting at
        # its last output needs no topological reorder. Native calls stay
        # exactly where they were, including writes with no users.
        fused, inputs, outputs = fuse_as_graphmodule(gm, nodes, name)
        insert_subgm(gm, fused, inputs, outputs)
        erase_nodes(gm, nodes)
    gm.graph.lint()
    gm.recompile()
    return planned
