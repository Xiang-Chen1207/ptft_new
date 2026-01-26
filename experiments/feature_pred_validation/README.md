# 特征预测编码器有效性验证实验

- 目标：系统评估预训练模型的特征预测头在验证集上的R2、PCC、RMSE，并生成Top-K特征列表与散点图。
- 数据与配置：使用 [pretrain.yaml](file:///vePFS-0x0d/home/cx/ptft/configs/pretrain.yaml) 指向的验证集；可通过命令参数覆盖。
- 检查点：默认使用 `output_old/flagship_cross_attn/best.pth`，可替换为你当前最优模型。

## 运行
- 评估并生成CSV与散点图：
  ```bash
  bash run_eval.sh
  ```
- 自定义参数：
  ```bash
  CHECKPOINT=/path/to/your_checkpoint.pth \
  CONFIG=configs/pretrain.yaml \
  OUTPUT=feature_metrics_eval_full.csv \
  BATCH_SIZE=256 \
  bash run_eval.sh
  ```

## 产物
- 指标CSV：`feature_metrics_eval_full.csv`（包含每个特征的R2、PCC、RMSE）
- 散点图目录：`feature_scatter_plots/`（每个特征一张）
- Top-K特征打印与保存：见控制台与W&B日志（若开启）

