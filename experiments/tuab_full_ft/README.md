# TUAB 全量微调实验

- 目标：在TUAB异常/正常分类任务上，对整模型进行端到端微调。
- 两种初始化：
  - Neuro-KE（重建+预测特征联合预训练）检查点：`output_old/flagship_cross_attn/best.pth`
  - 仅预测特征预训练检查点：`output_old/sanity_feat_only/best.pth`

## 运行
- 使用Neuro-KE初始化进行全量微调：
  ```bash
  bash run_full_ft_neuro_ke.sh
  ```
- 使用仅预测特征初始化进行全量微调：
  ```bash
  bash run_full_ft_feat_only.sh
  ```

## 说明
- 入口脚本均调用 [main.py](file:///vePFS-0x0d/home/cx/ptft/main.py) 与 [configs/finetune.yaml](file:///vePFS-0x0d/home/cx/ptft/configs/finetune.yaml)，通过 `--opts` 覆盖预训练权重路径与训练轮数等超参。
- 日志与检查点默认写入 `output/finetune/`，指标以 `balanced_acc` 为主。

