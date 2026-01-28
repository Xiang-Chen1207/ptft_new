# TUAB 线性探测（Feat-Only 预训练模型）

- 目标：在TUAB异常/正常分类任务上，使用**仅特征预测预训练模型**（sanity_feat_only）提取的特征进行线性探测。
- 对比：与 Neuro-KE（重建+特征预测）模型的表征质量进行对比。

## 两种预训练模型对比

| 模型 | 检查点 | 预训练任务 | 特征提取方式 |
|-----|--------|----------|------------|
| **Neuro-KE** | `output_old/flagship_cross_attn/best.pth` | 重建 + 特征预测 | EEG(GAP) + Feat(CrossAttn) + Pred |
| **Feat-Only** | `output_old/sanity_feat_only/best.pth` | 仅特征预测 | EEG(GAP) + Feat(CrossAttn) + Pred |

## 特征类型说明

| 特征类型 | 维度 | 说明 |
|---------|-----|------|
| `eeg` | 200 | 骨干网络输出的全局平均池化 (GAP) |
| `feat` | 200 | Cross-Attention 查询 Token 输出 |
| `full` | 400 | eeg + feat 拼接 |
| `pred` | 62 | 特征预测头输出的预测特征 |

## 运行步骤

### 1. 提取 Feat-Only 模型特征
```bash
bash extract_feat_only_features.sh
```
输出：`experiments/tuab_lp/features/feat_only_features.npz`

### 2. 运行线性探测（所有特征类型）
```bash
bash run_lp_feat_only.sh
```

### 3. 仅使用预测特征做线性探测（使用 Neuro-KE 特征）
```bash
bash run_lp_pred_only.sh
```

## 与 Neuro-KE 模型对比

运行 `tuab_lp` 目录中的脚本提取 Neuro-KE 特征：
```bash
cd ../tuab_lp
bash extract_pred_features.sh  # 或使用已有的 neuro_ke_features.npz
```

然后对比两个模型的线性探测结果。

## 产物
- 特征NPZ：
  - `experiments/tuab_lp/features/feat_only_features.npz`（Feat-Only模型）
  - `experiments/tuab_lp/features/neuro_ke_features.npz`（Neuro-KE模型）
- 结果表：控制台打印各比例的 Acc/BAcc
