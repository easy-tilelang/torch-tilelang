# Environment records

The tested TileLang/TVM revisions and required build capabilities are documented
in [compatibility](../docs/compatibility.md). Dependency declarations in
`pyproject.toml` identify the packages; they do not imply a verified release
range.

`cpu-wsl-py314.txt` records the CPU verification environment used during this
extraction. It is not an NPU lock. Add device constraints only after recording
Torch, torch_npu, CANN, Bisheng, PTODSL, ptoas and device metadata from a real
NPU verification run.
