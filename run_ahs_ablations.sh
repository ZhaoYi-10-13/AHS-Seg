#!/bin/bash
# AHS-Seg Ablation Study Script
# 消融实验脚本

set -e

# 设置环境
export DETECTRON2_DATASETS=/path/to/datasets

NUM_GPUS=4
BASE_OUTPUT="output/ablations"

# 消融实验配置列表
declare -a CONFIGS=(
    "configs/ablations/vitb_384_ahs_baseline.yaml:baseline"
    "configs/ablations/vitb_384_ahs_hpa_only.yaml:hpa_only"
    "configs/ablations/vitb_384_ahs_sdr_only.yaml:sdr_only"
    "configs/ablations/vitb_384_ahs_afr_only.yaml:afr_only"
    "configs/ablations/vitb_384_ahs_full.yaml:full"
)

echo "=========================================="
echo "AHS-Seg Ablation Study"
echo "=========================================="

for config_pair in "${CONFIGS[@]}"; do
    IFS=':' read -r config name <<< "$config_pair"
    
    echo ""
    echo "Running experiment: ${name}"
    echo "Config: ${config}"
    echo "------------------------------------------"
    
    OUTPUT_DIR="${BASE_OUTPUT}/${name}"
    
    python train_net.py \
        --config-file ${config} \
        --num-gpus ${NUM_GPUS} \
        OUTPUT_DIR ${OUTPUT_DIR} \
        SOLVER.IMS_PER_BATCH 16
    
    echo "Experiment ${name} completed!"
    echo ""
done

echo "=========================================="
echo "All ablation experiments completed!"
echo "=========================================="
