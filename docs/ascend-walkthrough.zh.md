# torch-tilelang 技术说明书

## —— 以 `examples/ascend/test_torch_compile.py` 为例的端到端业务流程

> 文档性质：技术说明书（内部/开发者版）
> 适用代码版本：`main_clear` 分支（提交 `cda1251`）
> 示例目标：解释一个 `torch.compile` 图如何被捕获、分区、编译为 Ascend kernel 并在 NPU 上执行
> 对应示例：`examples/ascend/test_torch_compile.py`

---

## 1. 文档目的与范围

本文以最小可运行示例 `examples/ascend/test_torch_compile.py` 为线索，完整说明 `torch-tilelang` 后端从 **PyTorch Dynamo 捕获** 到 **TileLang Ascend kernel 执行** 的业务流程，包括：

- 各阶段输入/输出、关键模块与函数入口；
- 配置解析与 AscendC/PTO 双后端选择；
- 图的合法性判定、分区、编译、数值校验与执行；
- 本例实际产生的分区数、kernel 数、线程/核配置；
- 运行前置条件、常见失败点与定位路径。

不在本文范围：TileLang 编译器内部 lowering、CANN 驱动细节、PTO 工具链实现。

---

## 2. 系统定位与架构

### 2.1 定位

`torch-tilelang` 是一个 **PyTorch `torch.compile` 第三方后端**：

- 把 Dynamo 捕获的 FX 图转换为 TVM Relax/TIR；
- 对纯、静态、连续的 Ascend 推理图套用 Ascend 专用调度（elementwise / reduction / GEMM）；
- 复用 TileLang 现有 AscendC（TVM-FFI）或 PTO（PTO adapter）编译、缓存与启动能力；
- **中间张量与输出由 Torch 分配和持有**，不经过 TVM VM 的存储分配器。

### 2.2 分层

```
┌──────────────────────────────────────────────┐
│ 用户层   torch.compile(backend="tilelang")     │
├──────────────────────────────────────────────┤
│ 注册层   torch_dynamo_backends entry point     │
├──────────────────────────────────────────────┤
│ 后端层   backend.py（安全判定/分区/降级）        │
│ 前端层   frontend.py（FX → Relax → TIR）        │
│ 调度层   schedules/ascend/*（Ascend 调度）      │
│ 运行时   runtime.py（执行计划/内存生命周期）      │
├──────────────────────────────────────────────┤
│ 适配层   compat/tilelang.py, compat/profiler.py │
├──────────────────────────────────────────────┤
│ 依赖层   TileLang / TVM Relax / AscendC / PTO   │
└──────────────────────────────────────────────┘
```

### 2.3 组件职责

| 组件 | 文件 | 职责 |
| --- | --- | --- |
| 后端注册 | `pyproject.toml:21-23` | 注册 `tilelang`、`torch_tilelang_ascend` 两个后端名 |
| 后端入口 | `backend.py` | 配置解析、安全判定、能力分区、分区编译、降级、报告 |
| 前端 | `frontend.py` | FX→Relax 转换、数值语义修正、输入契约校验 |
| 调度 | `schedules/ascend/` | 调度族选择、SIMT/VF、归约、GEMM+epilogue |
| 运行时 | `runtime.py` | IR 校验、kernel 编译、执行计划、Torch 张量生命周期 |
| 融合 | `fusion.py` | FuseTIR 候选、数值校验、实测融合选择 |
| 兼容层 | `compat/tilelang.py` | target/execution 组合解析、编译与缓存 |
| 计时 | `compat/profiler.py` | `do_bench` 封装、区分测量不可用与设备错误 |
| 配置 | `config.py` | 不可变 `GraphConfig` 与校验 |

---

## 3. 示例用例说明

```python
# examples/ascend/test_torch_compile.py:13-14
def model(x, bias):
    return torch.relu(x + bias) * bias
```

| 项 | 值 |
| --- | --- |
| 输入 `x` | `[7, 65]`，NPU，dtype ∈ {float16, bfloat16, float32} |
| 输入 `bias` | `[65]`，NPU，与 `x` 同 dtype |
| 编译调用 | `torch.compile(model, backend="tilelang", fullgraph=True, options=...)` |
| 选项 default | `{"strict": True}` |
| 选项 pto | `{"strict": True, "target": "pto"}`，带 `pytest.mark.pto` |
| 用例数量 | 2 选项 × 3 dtype = 6 |
| 断言 | `torch.testing.assert_close(compiled(x, bias), model(x, bias))` |

`fullgraph=True` 表示禁止 graph break；`strict=True` 表示任何不支持情况直接报错，不静默降级。

---

## 4. 端到端流程

### 4.1 阶段总览

| 阶段 | 触发点 | 输入 | 输出 | 关键文件 |
| --- | --- | --- | --- | --- |
| 1 | pytest fixture | 命令行/标记 | device/dtype/seed | `examples/ascend/conftest.py` |
| 2 | `torch.compile` | backend 名 + options | `GraphConfig` + backend context | `backend.py:293`, `compat/tilelang.py:14` |
| 3 | Dynamo 捕获 | FX GraphModule | 已分区 GraphModule | `backend.py:304-334` |
| 4 | 首次 `compiled(...)` | args | 编译好的 runner | `backend.py:224`, `frontend.py:259`, `runtime.py:70` |
| 5 | `runner(*args)` | 张量输入 | Torch 张量输出 | `runtime.py:230` |
| 6 | `assert_close` | 编译结果 + eager 结果 | 通过/失败 | 测试体 |

### 4.2 阶段 1：测试入口与 fixture

`examples/ascend/conftest.py`：

1. **参数注入**（`:7-16`）：`--device`（默认 `npu:0`）、`--dtype`（默认 `all`）、`--seed`（默认 0）。
2. **dtype 参数化**（`:19-29`）：读取模块级 `SUPPORTED_DTYPES`，对 `dtype` fixture 参数化；不支持的 dtype 标记为 skip。
3. **device fixture**（`:32-45`）：
   - 校验 TileLang 构建带 Ascend 后端；
   - `importorskip("torch_npu")` 且 `torch.npu.is_available()`；
   - 要求 `--device` 必须是 `npu:<index>`；
   - `with torch.npu.device(selected): yield`，保证整个用例在目标卡上下文执行。
4. **seed fixture**（`:48-50`）：直接返回命令行 seed。

测试体（`:21-26`）设置随机种子、创建输入、编译并断言。

### 4.3 阶段 2：后端解析与配置冻结

#### 4.3.1 后端解析

`torch.compile(backend="tilelang")` 通过 entry point 查到：

```toml
[project.entry-points."torch_dynamo_backends"]
tilelang = "torch_tilelang.backend:tilelang_backend"
torch_tilelang_ascend = "torch_tilelang.backend:tilelang_backend"
```

`tilelang_backend = make_backend()`（`backend.py:339`），是一个可调用后端函数。

PyTorch 将 `options=` 以关键字参数传入（`_TorchCompileWrapper.__call__`），对应 `backend.py:295-301`：

```python
def backend(gm, example_inputs, **kwargs):
    options = kwargs.pop("options", None) or {}
    local_config = GraphConfig(**options) if config is None else replace(config, **options)
    context = _resolve_graph_context(local_config)
    local_config = _resolved_config(local_config, context)
```

#### 4.3.2 配置对象

`GraphConfig`（`config.py:13`）字段：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `target` | `TILELANG_DEFAULT_TARGET` 或 `{"kind":"ascend","mcpu":"dav-3510"}` | 编译 target |
| `execution_backend` | `auto` | `auto/tvm_ffi/pto` |
| `tile_size` | 1024 | 每个核处理的元素块 |
| `num_cores` | 32 | 最大逻辑核数 |
| `threads` | 128 | 每核 SIMT 线程数（1..1024） |
| `vector_width` | 0 | 0 自动 / 1 / 4 |
| `fusion` | `off` | `off/profile` |
| `min_speedup` | 0.05 | 融合接受阈值 |
| `max_regression` | 0.02 | 稳定性/端到端回退阈值 |
| `benchmark_repeats` | 3 | ≥3 |
| `benchmark_backend` | `event` | `event/msprof` |
| `strict` | False | 不支持即报错 |

#### 4.3.3 backend context 解析

`compat/tilelang.py:14 resolve_graph_context` 依次：

1. 识别 target kind（JSON 或字符串），仅允许 `ascend/asc/ascendc/pto/auto`；
2. `"ascendc"` 字面量映射到默认 JSON target，且要求 execution 不是 `pto`；
3. 检查构建支持 Ascend（`Target.list_kinds()`、`tl.TargetIsAscend`）；
4. `create_backend_context(target, execution_backend=...)` 得到 `(module.name, execution_backend.name)`；
5. PTO 场景修正 `auto` 为 `pto`；
6. 校验 target kind 为 `ascend` 且对应 codegen 存在：
   - `("ascend","tvm_ffi") → target.build.tilelang_ascend`
   - `("pto","pto") → target.build.tilelang_pto_without_compile`
7. `resolved_config`（`:50`）把 `target` 与 `execution_backend` 冻结进配置。

**本例结果**：default → `ascend + tvm_ffi`；pto → `pto + pto`。

### 4.4 阶段 3：图捕获、安全判定与分区

#### 分区是什么意思？

**分区就是把计算图中的算子分成若干组，让各组交给合适的执行方式处理。** 本仓库把支持的算子组成 TileLang 子图，不支持但确认无副作用的算子可以留在 Torch 中执行，从而让同一张图混合使用两种执行方式。

例如，下面的模型在支持的 NPU 输入上运行：

```python
def model(x, bias, scale):
    a = x + bias
    b = torch.relu(a)
    c = torch.sin(b)
    return c * scale
```

加法、ReLU 和乘法在当前支持范围内；`sin` 被视为无副作用的 Torch 算子，但没有纳入 TileLang 编译范围。因此会形成以下执行流程：

```text
x、bias
   ↓
[分区 1：add → relu]    TileLang 执行
   ↓
sin                    Torch 执行
   ↓
[分区 2：mul]           TileLang 执行（同时读取 scale）
   ↓
y
```

分区之间传递的仍然是 Torch Tensor，整个模型按图中的依赖执行。这里划分的是计算图中的算子组，并不切分输入张量。若图中含有变异、别名操作或不透明副作用，则先由下述安全判定决定整图降级或报错。

**一个分区不一定对应一个 kernel。** 分区决定哪些算子作为一个子图交给 TileLang 编译；融合决定其中哪些算子能合成更少的 kernel。例如本文的 `add → relu → mul` 属于一个分区，但默认 `fusion="off"` 时生成三个 kernel。即使调用了 `fuse_partitions`，也只是把图节点封装成分区模块，不代表已经完成 kernel 融合。启用 `fusion="profile"` 后，融合候选还需通过调度、数值校验和性能检查，才可能被采用。

#### 代码中的分区流程

上述行为是默认策略 `partition_policy="conservative"`。对含 KV 缓存更新或第三方算子的推理图，可以按下面的方式启用有序分区。

```python
from torch_tilelang import GraphConfig, make_backend

backend = make_backend(GraphConfig(target="ascendc", partition_policy="ordered", strict=True))
decode = torch.compile(model.main_decode, backend=backend, dynamic=False, fullgraph=True)
```

`target="pto"` 使用同一分区策略。有序策略只组合原图中连续且签名受支持的纯计算。视图、原地写入和第三方算子保留在原来的执行位置。即使第三方算子的注册声明没有副作用，它也在原生端执行；首次校验与融合试跑只会执行选中的纯分区。只有类型转换的区域也保留在原生端，避免供应商算子的运行时 dtype 与捕获阶段推断不同而将恒等转换误计为内核。模型带有 forward hook 时继续使用保守保护。

在自定义 Dynamo 回调中保留 `runner = backend(gm, example_inputs)`，可通过 `runner.graph_report` 检查实际覆盖。`native_nodes` 记录计划留在原生端的节点和原因，`partitions` 记录每个 TileLang 分区的 `node_names`、`targets`、内核数及调用计数。设备同步完成后，比较预热前后或请求前后的 `compiled_calls` 和 `fallback_calls`。严格模式要求至少有一个可编译分区，并传播所选分区的构建或校验错误；允许计划中的原生调用。

`backend.py:304-334`：

1. **副作用判定** `_has_unsafe_effects`（`backend.py:106`）：
   - 前向 hook；
   - `call_method` 中的变异方法、别名方法（`view/reshape/...`）、未知方法；
   - `call_module` 中非 `ReLU/Softmax` 或 inplace 模块；
   - `call_function` 的 mutable schema、返回别名、随机性 tag、`out=`、inplace、未知 Python 函数等。
   - 命中：`strict=True` 抛 `UnsupportedGraphError`，否则 eager fallback。

2. **输入类型**：要求全 Tensor，否则 fallback/报错。

3. **能力分区**：
   - `deepcopy(gm.graph)` 复制图结构，复用用户模块与参数；
   - `CapabilityBasedPartitioner(_SupportedOperators(), allows_single_node_partition=True)`（`backend.py:315`）；
   - `_SupportedOperators`（`backend.py:159`）按算子名/模块类型/`alpha`/`memory_format`/`pow` 指数等判定；
   - `fuse_partitions(prefix="tilelang_partition_")` 生成分区模块。

4. **替换与报告**：为每个 `call_module` 分区建立 `PartitionReport` 并用 `_CompiledPartition` 替换子模块（`backend.py:323-328`），返回 `run(*args)` 并挂 `graph_report`。

**本例结果**：`add → relu → mul` 三节点全部受支持且相邻，形成 **1 个分区**。

### 4.5 阶段 4：首次调用的编译流程

`compiled(x, bias)` 进入 `_CompiledPartition.forward`（`backend.py:261`）：

1. 计算签名 `_signature`（`backend.py:202`）：grad 状态 + 每个参数 `(shape, dtype, device, stride, storage_offset, requires_grad)`。
2. 加锁比较签名；首次 → `_prepare`；后续相同签名直接复用 runner（避免重复编译）。

`_prepare`（`backend.py:224`）分四步。

#### 4.5.1 输入契约校验

`frontend.py:37 validate_inputs`：

- 非空且全为 Tensor；
- device 为 `npu`（图转换阶段允许 CPU，用 `require_npu=False`）；
- 同一 device；
- dtype ∈ {float16, bfloat16, float32}；
- 不需要梯度（`requires_grad and grad_enabled` 拒绝）；
- contiguous 且 `storage_offset == 0`；
- 每一维为正的静态整数。

#### 4.5.2 FX → Relax

`frontend.py:259 fx_to_relax`：

1. 再次校验输入（`require_npu=False`），拒绝 forward hook；
2. 构造 `_AscendFXImporter`，重写 `create_convert_map`（`frontend.py:58`）：
   - `matmul/mm`：同 dtype，fp32 累加后回退输出 dtype；
   - `sum/mean/amax/softmax`：fp32 归约与数值稳定 softmax；
   - `add/sub/mul/div`：含标量 opmath 与 dtype 提升；
   - `pow`：仅支持常数指数 2；
   - `to/float/half/bfloat16/type_as`：仅 dtype 转换，不做布局/设备变更。
3. 对 Dynamo 产生的 `OpOverload` 建立 `custom_map`，按基础名映射转换器；
4. `importer.from_fx(...)` 生成未调度的 `relax.call_tir` IR；
5. `LegalizeOps` 对 add/sub/mul/div 使用 `_legalize_binary`（`frontend.py:25`）：**操作数含 0 标量常量时改用 `bb.call_te(topi.xxx)`**，防止 `x*0→0`、`x+0→x` 丢失 NaN/±0；
6. `AnnotateTIROpPattern` + `DeadCodeElimination`。

**本例结果**：生成 **3 个 `call_tir`**（add、relu/max、multiply），对应 `kernel_count = 3`。

#### 4.5.3 调度

`runtime.py:56 _compile_kernel` → `schedules/ascend/schedule.py:14 schedule_primfunc`，按顺序尝试：

```
elementwise → reduction → gemm
```

- **elementwise**（`elementwise.py:430`）适用于本例三个 injective 函数：
  - 校验 buffer 静态/紧凑/零偏移、唯一完整输出 writer；
  - 内联 injective producer；
  - 证明输出坐标所有权、输入读越界安全、标量表达式白名单；
  - 低精度算术提升为 fp32 运算再 cast（`_LegalizeScalarArithmetic`）；
  - `relu` 用带 NaN 语义的 `max` 实现；
  - 决定向量宽度（`(7,65)` 末维 65 不满足 4 对齐 → **vector_width=1**）；
  - 生成 `blockIdx.x`（核）+ `SIMT_VF`（线程域）结构。
- **reduction**（`operators.py:192`）：last-axis sum/max/mean，fp32 累加，max 显式处理 NaN 与全 `-inf`。
- **gemm**（`operators.py:374`）：NN/NT 2-D GEMM，tile 化，可选 bias/relu/cast epilogue（L0C→UB）。

若全部不匹配 → `UnsupportedScheduleError`，进入 fallback 路径。

#### 4.5.4 kernel 编译

`compat/tilelang.py:55`：

```python
tilelang.compile(func, target=config.target, execution_backend=config.execution_backend, out_idx=[])
```

`out_idx=[]` 表示**输出缓冲由调用方提供**（Torch 负责分配）。

#### 4.5.5 执行计划

`runtime.py:70 TorchGraphRunner.__init__`：

1. 解析并冻结 backend context；
2. 校验 main 为 `relax.Function`，参数/返回值均为静态 Tensor；
3. 遍历 body，仅允许 `relax.call_tir`，校验 ABI 为“输入 buffer + 输出 buffer”，收集输出规格与常量；
4. 编译所有 `PrimFunc`，校验各 kernel 解析出的后端与图后端一致；
5. `_build_execution_plan`（`runtime.py:156`）将 IR 编译为槽位化 Python 步骤：
   - 常量步骤：`source.to(device)`；
   - call 步骤：为每个输出 `torch.empty(spec.shape, dtype, device)` 后 `kernel(*inputs, *outputs)`；
   - 计算每个槽位的 last-use，生成释放计划，及时释放中间张量。

#### 4.5.6 数值校验

`fusion.py:142 validate_runner`：

- `no_grad` 下快照 eager 参考输出；
- 运行编译 runner；
- `_assert_outputs_close` 校验输出结构、dtype 容差、NaN、±0 符号位；
- 失败抛 `GraphValidationError`：`strict=True` 抛出，否则回落 eager。

**融合分支**：仅当 `fusion="profile"` 时，额外 `make_fusion_candidate`（`fusion.py:30`，FuseOps+FuseTIR）编译候选，并由 `select_fusion`（`fusion.py:189`）按“正确性 + 可重复设备加速 + 端到端不回退”选择 baseline 或 candidate。本例 `fusion="off"`，**不融合**。

### 4.6 阶段 5：执行

更新 `PartitionReport`（kernel_count、compiled、backend、融合决策）后调用 `runner(*args)`（`runtime.py:230`）：

1. 参数数量、类型、device、shape、dtype、contiguous、offset、梯度校验；
2. `torch.npu.device(device)` 上下文；
3. 对输入 `record_stream`：自定义 FFI 启动绕过 Torch dispatcher，显式记录消费，防止外部输入被提前复用；
4. 按执行计划依次：分配新输出 → 启动 kernel → 释放 last-use 槽位；
5. 返回结果槽位。

**本例结果**：**3 次 kernel 启动**，每次均为 1 核、128 线程、`tile_size=1024`，并用 `lane < 1024 && linear < 455` 做边界保护。

### 4.7 阶段 6：结果断言

`torch.testing.assert_close(compiled(x, bias), model(x, bias))`：

- 左：编译后端结果；
- 右：PyTorch eager 结果；
- 使用 PyTorch 默认容差。

---

## 5. 双后端（AscendC / PTO）

| 维度 | AscendC（default） | PTO |
| --- | --- | --- |
| target | `{"kind":"ascend","mcpu":"dav-3510"}` | `pto` |
| execution | `tvm_ffi` | `pto` |
| codegen | `target.build.tilelang_ascend` | `target.build.tilelang_pto_without_compile` |
| adapter | `TVMFFIKernelAdapter` | `PTOKernelAdapter` |
| 缓存 | TileLang 默认 | 支持磁盘缓存跨进程 reload（`tests/ascend/test_pto_cache.py`） |
| 调度 | 同一套 Ascend 调度 | 同一套，lowering 不同 |

选择优先级：显式 `options["target"]` > `TILELANG_DEFAULT_TARGET` > 默认 AscendC。

---

## 6. 关键设计契约

### 6.1 图与输入

- 仅支持**纯推理**、**静态形状**、**连续且零偏移**的 NPU 张量；
- 仅 float16 / bfloat16 / float32；
- 图内不允许变异、别名写、随机性、`out=`、inplace、未知副作用；
- 不满足时：`strict=True` 报错，否则 eager fallback 并记录原因。

### 6.2 数值语义

- fp16/bf16 归约/算子在 fp32 中计算或累加，再回退原 dtype；
- 保留 NaN 传播与 ±0 符号；
- 0 常量不被代数折叠；
- 低精度逐操作舍入边界在调度合法化中显式保留。

### 6.3 内存与生命周期

- 输出与中间张量由 Torch 分配；
- 执行计划按 last-use 释放槽位；
- 输入通过 `record_stream` 与当前 NPU stream 建立消费关系；
- 支持在另一 stream 上调用并保留旧输出（有对应测试）。

### 6.4 降级策略

| 阶段 | 失败 | 行为 |
| --- | --- | --- |
| 安全判定 | 副作用/别名/随机 | 整图 fallback |
| 分区 | 无受支持分区 | 整图 fallback |
| 编译/调度 | 无可用 schedule | 分区 fallback |
| 数值校验 | 不一致 | 分区 fallback |
| 融合选择 | 候选不正确/不达阈值 | 选 baseline |

`strict=True` 时以上均改为抛出异常。

---

## 7. 本例量化说明

| 指标 | 值 |
| --- | --- |
| 分区数 | 1 |
| Relax `call_tir` 数 | 3（add、relu、mul） |
| `kernel_count` | 3（`fusion="off"`） |
| 元素总数 | 7 × 65 = 455 |
| tiling | `tile_size=1024`，`num_tiles=1` |
| grid | `min(32, 1) = 1` 核 |
| threads | 128 |
| lane 循环 | `ceil(1024/(128×1)) = 8` |
| vector_width | 1（65 不整除 4） |
| 执行启动次数 | 3（每次调用） |

---

## 8. 运行前置条件与命令

### 前置条件

1. 已安装兼容的 `torch` + `torch_npu`；
2. 已安装**带 Ascend 后端**的 TileLang（`USE_ASCEND=ON`）及其 TVM/Relax；
3. 已安装本包（注册 entry point）：

```bash
python -m pip install --no-deps -e .
```

4. 已安装 CANN 与所选后端工具链（AscendC 的 bisheng；PTO 另需 PTODSL/ptoas）。

### 运行

```bash
# 全部用例（含 PTO）
python examples/ascend/test_torch_compile.py

# 仅 AscendC（排除 PTO）
python examples/ascend/test_torch_compile.py -m "not pto"

# 指定设备/dtype/seed
python examples/ascend/test_torch_compile.py --device npu:1 --dtype bfloat16 --seed 0
```

---

## 9. 常见失败与定位

| 现象 | 可能原因 | 定位点 |
| --- | --- | --- |
| `InvalidBackend` | 包未安装/entry point 未注册 | `pyproject.toml:21`、`pip install -e .` |
| `Ascend graph compilation is unavailable...` | TileLang 未开启 Ascend | `compat/tilelang.py:26-27` |
| `... requires native codegen ...` | 缺少对应 codegen | `compat/tilelang.py:43-46` |
| `Ascend graph inputs must ...` | 非 NPU/非连续/非静态/需梯度 | `frontend.py:37` |
| `No Ascend graph schedule applies` | 图形态不被调度支持 | `schedules/ascend/schedule.py:30` |
| `Graph contains mutation...` | 图有副作用/别名 | `backend.py:106` |
| 数值校验失败 | 语义差异/后端错误 | `fusion.py:112` |
| PTO 缓存未命中 | 缓存目录/工具链问题 | `tests/ascend/test_pto_cache.py` |

---

## 附录 A：关键代码索引

| 功能 | 位置 |
| --- | --- |
| 后端注册 | `pyproject.toml:21` |
| 后端工厂/配置解析 | `backend.py:293-303` |
| 副作用判定 | `backend.py:106` |
| 算子能力判定 | `backend.py:159` |
| 分区与替换 | `backend.py:312-328` |
| 分区首次编译 | `backend.py:224-259` |
| 调用签名缓存 | `backend.py:202`, `backend.py:261` |
| 输入契约 | `frontend.py:37` |
| FX→Relax | `frontend.py:259` |
| 二元合法化 | `frontend.py:25` |
| 调度选择 | `schedules/ascend/schedule.py:14` |
| elementwise 调度 | `schedules/ascend/elementwise.py:430` |
| reduction 调度 | `schedules/ascend/operators.py:192` |
| GEMM 调度 | `schedules/ascend/operators.py:374` |
| 运行时构造/编译 | `runtime.py:70` |
| 执行计划 | `runtime.py:156` |
| 执行入口 | `runtime.py:230` |
| 后端 context 解析 | `compat/tilelang.py:14` |
| kernel 编译 | `compat/tilelang.py:55` |
| 数值校验 | `fusion.py:142` |
| 融合选择 | `fusion.py:189` |
| 测试 fixtures | `examples/ascend/conftest.py:7-50` |

---

## 附录 B：术语

| 术语 | 含义 |
| --- | --- |
| Dynamo | PyTorch 图捕获前端 |
| FX GraphModule | Dynamo 产出的可分析图 |
| Partition（分区） | 按算子支持能力划出的子图，作为整体尝试编译；可包含多个 kernel，见第 4.4 节 |
| Relax | TVM 的图级 IR |
| PrimFunc / call_tir | 可调度 TIR 函数及其图级调用 |
| SIMT_VF | Ascend 向量/SIMT 执行域 |
| tvm_ffi / pto | 两种 kernel 执行适配后端 |
| epilogue | GEMM 输出后处理（bias/relu/cast） |
| fallback | 编译失败时回到 PyTorch eager |

---

*最后更新：2026-11-19。对应示例：`examples/ascend/test_torch_compile.py`。若代码行为发生变化，请同步更新本文的代码行号引用。*
