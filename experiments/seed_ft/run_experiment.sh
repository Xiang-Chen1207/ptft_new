#!/bin/bash
set -e

# Activate environment
source /vePFS-0x0d/home/cx/miniconda3/bin/activate labram

# Set Project Root
PROJECT_ROOT="/vePFS-0x0d/home/cx/ptft"
cd "$PROJECT_ROOT"

# Set PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# GPUs
export CUDA_VISIBLE_DEVICES=0,1

echo "=== Starting SEED Fine-Tuning Experiment ==="
python main.py --config configs/finetune_seed.yaml

echo "Experiment Completed."
