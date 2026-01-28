#!/bin/bash
# Wrapper for Recon Full Fine-tuning
# Defaults to GPU 0,1,2,3 as per request
GPU=${1:-0,1,2,3}
bash experiments/tuab_full_ft/run_experiment.sh recon $GPU
