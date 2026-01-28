我将对您的代码进行以下优化，重点关注 IO 性能和数据读取效率：

### 1. 优化 HDF5 数据读取 (datasets/tueg.py)
**问题**: 目前代码 `raw = group['eeg'][:]` 会将整个 EEG 记录（可能几分钟长）读入内存，然后再截取前 `input_size` (60秒)。这造成了巨大的 IO 浪费和内存波动。
**方案**: 修改为先检查 HDF5 数据集的形状，然后只读取所需的时间片段。
**代码变更**:
```python
# 伪代码预览
dset = group['eeg']
if dset.shape[0] == 21: # (Channels, Time)
    raw = dset[:, :self.input_size]
else: # (Time, Channels)
    raw = dset[:self.input_size, :].T
```

### 2. 启用 DataLoader 的 Pin Memory (datasets/builder.py)
**问题**: `build_dataloader` 中未启用 `pin_memory=True`。
**方案**: 设置 `pin_memory=True`，这将加速 CPU 到 GPU 的数据传输，这对于高吞吐量的预训练至关重要。

### 3. 修复脚本中的配置问题 (scripts/run_flagship_cross_attn.sh)
**问题**:
1. `dataset.cache_path` 设置在 `OUTPUT_DIR` 下。每次更改实验目录名，都会导致重新索引数据集（耗时）。
2. 脚本注释说是 "Single GPU"，但 `export CUDA_VISIBLE_DEVICES=0,1,2,3` 且代码使用 `DataParallel`，实际上会跑在 4 张卡上。
**方案**:
1. 将 `cache_path` 修改为固定路径（如数据集目录下的 `dataset_index_60s.json`），避免重复索引。
2. 修正 GPU 设置，确保与用户意图一致（如果您想跑 4 卡，我会保持原样但更新注释；如果您想跑单卡，我会修正环境变量）。**假设您希望保持多卡并行以加快预训练**，我会更新注释并优化配置。

### 4. (可选) 内存优化
**问题**: 700MB 的特征文件被加载到字典中，多进程 Worker 可能会导致内存增加。
**方案**: 暂时保持现状，因为 700MB 在现代服务器上通常可接受。如果遇到内存溢出，后续可以改为共享内存方式。

---
**执行步骤**:
1. 修改 `datasets/tueg.py` 优化 `__getitem__`。
2. 修改 `datasets/builder.py` 添加 `pin_memory=True`。
3. 修改 `scripts/run_flagship_cross_attn.sh` 优化 `cache_path` 设置。
