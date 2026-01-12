#!/bin/bash
# AHS-Seg Evaluation Script
# 在多个数据集上评估模型

set -e

# 设置环境
export DETECTRON2_DATASETS=/path/to/datasets

CHECKPOINT="output/ahs_seg_vitb384/model_final.pth"
BASE_CONFIG="configs/vitb_384_ahs.yaml"

# 评估数据集配置
declare -A EVAL_DATASETS=(
    ["ade150"]="datasets/ade150.json"
    ["ade847"]="datasets/ade847.json"
    ["pc59"]="datasets/pc59.json"
    ["pc459"]="datasets/pc459.json"
)

echo "=========================================="
echo "AHS-Seg Cross-Dataset Evaluation"
echo "=========================================="

for dataset in "${!EVAL_DATASETS[@]}"; do
    test_json="${EVAL_DATASETS[$dataset]}"
    
    echo ""
    echo "Evaluating on: ${dataset}"
    echo "Test JSON: ${test_json}"
    echo "------------------------------------------"
    
    python train_net.py \
        --config-file ${BASE_CONFIG} \
        --eval-only \
        MODEL.WEIGHTS ${CHECKPOINT} \
        MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON ${test_json}
    
    echo "Evaluation on ${dataset} completed!"
    echo ""
done

echo "=========================================="
echo "All evaluations completed!"
echo "=========================================="
