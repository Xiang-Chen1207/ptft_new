# TUAB 全量微调实验

- 目标：在TUAB异常/正常分类任务上，对整模型进行端到端微调。
- 评估：Subject-independent Split，主要指标为 Balanced Accuracy (%)

## 三种预训练初始化

| 模型 | 检查点 | 预训练任务 | 说明 |
|-----|--------|----------|------|
| **Recon (Baseline)** | `output_old/baseline_recon/best.pth` | 仅重建 | 标准 MAE 预训练 |
| **Neuro-KE (Flagship)** | `output_old/flagship_cross_attn/best.pth` | 重建 + 特征预测 | 多任务联合预训练 |
| **Feat-Only (Sanity)** | `output_old/sanity_feat_only/best.pth` | 仅特征预测 | 验证特征预测有效性 |

## 运行

### Recon Baseline（仅重建预训练）
```bash
bash run_full_ft_recon.sh
```
输出目录：`output/finetune_recon/`

### Neuro-KE（重建+特征预测联合预训练）
```bash
bash run_full_ft_neuro_ke.sh
```
输出目录：`output/finetune_neuro_ke/`

### Feat-Only（仅特征预测预训练）
```bash
bash run_full_ft_feat_only.sh
```
输出目录：`output/finetune_feat_only/`

## 结果表格（Balanced Accuracy）

| 模型 | 预训练任务 | Test BAcc (%) |
|-----|----------|---------------|
| **Recon (EEG)** | 仅重建 | _待填写_ |
| **Neuro-KE (EEG)** | 重建 + 特征预测 | _待填写_ |
| **Feat-Only (EEG)** | 仅特征预测 | _待填写_ |

> 注：全量微调后，模型使用 EEG backbone 的 GAP 表征 (Dim=200) 进行分类。

## 与线性探测对比

参考 `experiments/tuab_lp/results.md` 中的线性探测结果：

| 模型 | Linear Probe BAcc | Full Fine-Tune BAcc |
|-----|------------------|---------------------|
| Recon (EEG) | 70.02% | _待填写_ |
| Neuro-KE (EEG) | 79.22% | _待填写_ |
| Neuro-KE (Feat) | 78.28% | N/A |

## 说明
- 入口脚本均调用 `main.py` 与 `configs/finetune.yaml`，通过 `--opts` 覆盖预训练权重路径与训练轮数等超参。
- 日志与检查点写入各自的 `output/finetune_*/` 目录。
- 主要指标：`balanced_acc`
- 训练参数：
  - Epochs: 10
  - Learning Rate: 0.001
  - Weight Decay: 0.05
