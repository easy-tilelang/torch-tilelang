# torch-tilelang 文档

本目录存放 `torch-tilelang` 的设计与使用文档。

| 文档 | 说明 |
| --- | --- |
| [功能开发现状（HTML · 架构导览 v2）](architecture-v2.zh.html) | 按已实现、有限支持、尚未实现筛选功能，查看有序分区示例、算子边界、源码和验证入口 |
| [项目架构导览（HTML）](architecture.zh.html) | 用浏览器打开，查看项目全景、编译流程和模块依赖，跟读 Ascend 推理示例 |
| [安装手册](install.zh.md) | 环境前置条件、editable/wheel 安装、入口验证与常见问题（含 InvalidBackend 排障） |
| [Ascend 后端编译流程走读](ascend-walkthrough.zh.md) | 以 `examples/ascend/test_torch_compile.py` 为例，说明 FX 捕获、分区、编译、执行全流程 |
