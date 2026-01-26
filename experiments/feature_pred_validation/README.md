# 特征预测编码器有效性验证实验

- 目标：系统评估预训练模型的特征预测头在验证集上的R2、PCC、RMSE，并生成Top-K特征列表与散点图。
- 数据与配置：使用 [pretrain.yaml](file:///vePFS-0x0d/home/cx/ptft/configs/pretrain.yaml) 指向的验证集；可通过命令参数覆盖。

## 两种预训练模型

| 模型 | 检查点 | 预训练任务 |
|-----|--------|----------|
| **Neuro-KE (flagship)** | `output_old/flagship_cross_attn/best.pth` | 重建 + 特征预测 |
| **Feat-Only (sanity)** | `output_old/sanity_feat_only/best.pth` | 仅特征预测 |

## 运行

### 评估 Neuro-KE 模型（重建+特征预测）
```bash
bash run_eval.sh
```

### 评估 Feat-Only 模型（仅特征预测）
```bash
bash run_eval_feat_only.sh
```

### 自定义参数
```bash
CHECKPOINT=/path/to/your_checkpoint.pth \
CONFIG=configs/pretrain.yaml \
OUTPUT=feature_metrics_eval_custom.csv \
BATCH_SIZE=256 \
bash run_eval.sh
```

## 产物
- 指标CSV：
  - `feature_metrics_eval_full.csv`（Neuro-KE模型）
  - `feature_metrics_eval_feat_only.csv`（Feat-Only模型）
- 散点图目录：`feature_scatter_plots/`（每个特征一张）
- Top-K特征打印与保存：见控制台与W&B日志（若开启）
