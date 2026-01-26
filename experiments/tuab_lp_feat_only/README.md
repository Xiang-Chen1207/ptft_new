# TUAB 线性探测（仅预测特征）

- 目标：在TUAB异常/正常分类任务上，仅使用预训练模型输出的“预测特征向量”（维度=feature_dim=66）进行线性探测。
- 依赖：`extract_features.py` 已扩展，生成 `train_pred/test_pred`；`run_lp_offline.py` 已支持 `pred` 特征类型与筛选。

## 步骤
1. 从Neuro-KE权重提取特征（含pred）：
   ```bash
   bash extract_pred_features.sh
   ```
2. 仅使用pred做LP：
   ```bash
   bash run_lp_pred_only.sh
   ```

## 产物
- 特征NPZ：`experiments/tuab_lp/features/neuro_ke_features.npz`（含 train_pred/test_pred）
- 结果表：控制台打印各比例的Acc/BAcc，建议重定向保存到 `results_pred_only.txt`

