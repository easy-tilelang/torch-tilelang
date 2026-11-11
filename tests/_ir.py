"""IR inspection helpers independent of TileLang's source-tree tests."""

from tilelang import tvm
from tvm import tirx


def nodes(ir, kind):
    if isinstance(ir, tvm.IRModule):
        ir = ir["main"].body
    elif isinstance(ir, tirx.PrimFunc):
        ir = ir.body
    found = []
    tirx.stmt_functor.post_order_visit(ir, lambda node: found.append(node) if isinstance(node, kind) else None)
    return found


def calls(ir, name):
    return [node for node in nodes(ir, tirx.Call) if isinstance(node.op, tvm.ir.Op) and node.op.name == name]
