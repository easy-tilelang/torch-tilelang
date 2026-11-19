# torch-tilelang 安装手册

> 文档性质：安装与部署说明（开发者版）
> 适用对象：需要在 Ascend NPU 环境使用 `torch.compile(backend="tilelang")` 的开发者
> 支持后端：AscendC（TVM-FFI，默认）、PTO

本文说明如何在本仓库中安装 `torch-tilelang`、验证 Dynamo 后端注册，以及排查常见安装问题。

---

## 1. 前置条件

| 依赖 | 要求 | 说明 |
| --- | --- | --- |
| Python | ≥ 3.10 | 见 `pyproject.toml` 的 `requires-python` |
| setuptools | ≥ 77 | 见 `pyproject.toml` 的 `build-system.requires` |
| torch | 与 `torch_npu` 匹配 | 本包只声明依赖，不锁定已验证版本范围 |
| torch_npu | 与 torch 匹配 | NPU 执行必需 |
| TileLang | 开启 `USE_ASCEND=ON` 的构建 | 必须使用其自带的 TVM/Relax |
| CANN | 与设备/驱动匹配 | AscendC 需要 `bisheng`；PTO 另需 `PTODSL`、`ptoas` |

注意：

- 本包是 **纯 Python 包**，不包含 kernel 编译器；真正的编译、缓存与启动由已安装的 Ascend 版 TileLang 提供。
- `pyproject.toml` 声明 `dependencies = ["torch", "tilelang"]`，安装时应使用 `--no-deps`，避免 pip 从 PyPI 拉取并覆盖本地带 Ascend 支持的 TileLang。

---

## 2. 环境自检

在安装前先确认解释器与依赖：

```bash
python -V
python -m pip -V
python -m pip list | grep -Ei 'torch|tilelang|pytest'
```

确认 TileLang 具备 Ascend 能力：

```bash
python - <<'PY'
import tilelang
from tilelang import tvm
print("tilelang:", tilelang.__version__, tilelang.__file__)
print("Target kinds:", tvm.target.Target.list_kinds())
print("TargetIsAscend:", tvm.get_global_func("tl.TargetIsAscend", allow_missing=True) is not None)
print("ascend codegen:", tvm.get_global_func("target.build.tilelang_ascend", allow_missing=True) is not None)
print("pto codegen:", tvm.get_global_func("target.build.tilelang_pto_without_compile", allow_missing=True) is not None)
PY
```

确认工具链：

```bash
which bisheng
which ptoas      # 仅 PTO 后端需要
python -c "import ptodsl"   # 仅 PTO 后端需要
```

若 `TargetIsAscend` 为 `False` 或缺少对应 codegen，需要重新构建/安装带 `USE_ASCEND=ON` 的 TileLang。

---

## 3. 安装方式

### 3.1 推荐：editable 安装

```bash
cd /path/to/torch-tilelang
python -m pip install --no-deps -e .
```

说明：

- `-e`（editable）：源码修改后无需重装即可生效；但 **entry point 变更（如后端名）仍需重装并新开进程**。
- `--no-deps`：跳过 `torch`、`tilelang` 依赖解析，确保使用当前环境中已验证的 Ascend 版 TileLang。
- 若在离线环境或构建隔离下载失败，可追加 `--no-build-isolation`（要求当前环境 `setuptools ≥ 77`）。

### 3.2 非 editable 安装

```bash
python -m pip install --no-deps .
```

### 3.3 构建并安装 wheel

```bash
python -m pip wheel . --no-deps -w dist
python -m pip install --no-deps dist/*.whl
```

wheel 安装到独立环境后，可使用仓库脚本做入口冒烟检查（需从源码目录之外运行）：

```bash
python /path/to/torch-tilelang/scripts/check_installed.py
```

> `scripts/check_installed.py` 会断言 `torch_tilelang` 位于 `sys.prefix` 下，因此 **不适用于 editable 安装**。

### 3.4 开发/测试依赖（可选）

```bash
python -m pip install pytest ruff
```

注意：`pip install --no-deps -e ".[test]"` 中的 extras 同样会被 `--no-deps` 跳过，建议单独安装。

---

## 4. 安装验证

### 4.1 检查发行版与 entry point

安装后必须 **新开一个 Python 进程**（Dynamo 会缓存后端表）：

```bash
python - <<'PY'
import importlib.metadata as md
from torch._dynamo.backends.registry import lookup_backend

print("dist:", md.version("torch-tilelang"))
print("entry points:", [(e.name, e.value) for e in md.entry_points(group="torch_dynamo_backends")])
print("backend:", lookup_backend("tilelang"))
print("alias:", lookup_backend("torch_tilelang_ascend"))
PY
```

预期：

- `entry points` 中包含 `tilelang` 与 `torch_tilelang_ascend`，值均为 `torch_tilelang.backend:tilelang_backend`；
- `lookup_backend(...)` 返回 `torch_tilelang.backend` 中的函数，而不是抛出 `InvalidBackend`。

### 4.2 无需 NPU 的冒烟测试

```bash
# 单元测试：FX 转换、Relax 合法化、Torch 张量执行
python -m pytest tests/unit -q -ra

# Ascend 源码生成（不执行设备 kernel）
python -m pytest tests/ascend -q -ra -m "not ascend and not pto"

# PTO 源码生成（可选）
python -m pytest tests/ascend -q -ra -m "pto and not ascend"
```

### 4.3 NPU 端到端验证

```bash
# 默认 AscendC（排除 PTO 用例）
python examples/ascend/test_torch_compile.py -m "not pto"

# 包含 PTO
python examples/ascend/test_torch_compile.py

# 指定设备/dtype/seed
python examples/ascend/test_torch_compile.py --device npu:1 --dtype bfloat16 --seed 0
```

更完整的验收（回归、示例、soak、benchmark）见 `scripts/verify_npu.py`（`--help` 查看各 stage）。

---

## 5. 后端选择

| 目标 | 配置 |
| --- | --- |
| AscendC（默认） | 不设置，或 `options={"target": "ascendc"}` |
| PTO | `options={"target": "pto"}` 或 `TILELANG_DEFAULT_TARGET=pto` |

- 显式 `target` 优先于 `TILELANG_DEFAULT_TARGET`；
- 默认 target 为 `{"kind": "ascend", "mcpu": "dav-3510"}`；
- AscendC 使用 codegen `target.build.tilelang_ascend` + `tvm_ffi` 适配；
- PTO 使用 codegen `target.build.tilelang_pto_without_compile` + `pto` 适配。

```python
import torch

compiled = torch.compile(model, backend="tilelang", fullgraph=True, options={"target": "pto"})
```

---

## 6. 常见问题

### 6.1 `InvalidBackend: Invalid backend: 'tilelang'`

**现象**：

```text
torch._dynamo.exc.InvalidBackend: Invalid backend: 'tilelang'
```

**根因**：`torch-tilelang` 未安装到当前解释器，`torch_dynamo_backends` entry point 缺失，`lookup_backend("tilelang")` 无法解析。

**为什么在仓库根目录运行也不行**：

- 包采用 **src 布局**（`src/torch_tilelang/`），仓库根不在可导入 `torch_tilelang` 的 `sys.path` 位置；
- Dynamo 通过 **已安装发行版的 entry point 元数据** 发现后端，而不是扫描当前目录；
- 因此“源码就在当前目录”不等于“后端已注册”。

**诊断命令**：

```bash
python - <<'PY'
import importlib.util, importlib.metadata as md
print("find_spec:", importlib.util.find_spec("torch_tilelang"))
print("entry points:", list(md.entry_points(group="torch_dynamo_backends")))
PY
python -m pip show torch-tilelang
```

若 `find_spec` 为 `None`、entry point 为空、`pip show` 未找到，即可确认是未安装。

**解决**：

```bash
cd /path/to/torch-tilelang
python -m pip install --no-deps -e .
# 新开进程后重试
```

### 6.2 安装后仍报 `InvalidBackend`

按顺序排查：

1. **解释器不一致**：确认 `python -m pip` 与 `python -m pytest` 使用的是同一个 Python（可用 `which python`、`sys.executable` 核对）；
2. **进程未刷新**：Dynamo 会缓存后端表，必须新开进程；
3. **editable 切换/升级后未重装**：切换 checkout 或修改 `pyproject.toml` 的 entry point 后需重新安装；
4. **构建隔离失败**：离线环境追加 `--no-build-isolation`。

### 6.3 依赖被 PyPI 覆盖

如果安装时未加 `--no-deps`，pip 可能安装 PyPI 上的 `tilelang`，覆盖本地 Ascend 构建，导致后续报“Ascend 不可用”或 codegen 缺失。重新安装本地 TileLang，并始终使用：

```bash
python -m pip install --no-deps -e .
```

### 6.4 `Ascend graph compilation is unavailable ...`

说明当前 TileLang 构建未开启 Ascend（`USE_ASCEND=OFF` 或缺少 `tl.TargetIsAscend`）。需重新构建并安装带 Ascend 支持的 TileLang。

### 6.5 `... requires native codegen ...`

说明缺少对应 target 的 codegen：

- AscendC：`target.build.tilelang_ascend`
- PTO：`target.build.tilelang_pto_without_compile`

需使用支持该后端并开启 `USE_ASCEND=ON` 的 TileLang 构建。

### 6.6 PTO 相关失败

PTO 执行额外要求：

- `PTODSL` 可导入；
- `ptoas` 可用；
- 匹配的 Bisheng 编译器。

缓存复用相关问题见 `tests/ascend/test_pto_cache.py`。

---

## 7. 卸载

```bash
python -m pip uninstall torch-tilelang
```

editable 安装卸载后会移除 entry point；如需保留源码，仅需重新执行安装命令即可恢复。

---

## 附录：关键环境变量

| 变量 | 作用 |
| --- | --- |
| `TILELANG_DEFAULT_TARGET` | 默认编译 target（`ascendc` / `pto` / JSON target）；显式 `options["target"]` 优先 |
| `TILELANG_CACHE_DIR` | TileLang 编译缓存目录 |
| `TILELANG_DISABLE_CACHE` | 是否禁用缓存（`1` 禁用） |
| `ASCEND_RT_VISIBLE_DEVICES` | 可见 NPU 设备列表 |
| `ASCEND_HOME_PATH` | CANN 安装路径 |
| `ASCEND_TOOLKIT_HOME` | CANN 工具链路径 |

---

*最后更新：2026-11-19。对应依赖声明见 `pyproject.toml`，后端解析见 `src/torch_tilelang/compat/tilelang.py`。*
