#!/bin/bash
# Enhanced CATSeg Evaluation Script
# 评估增强版模型在各数据集上的性能
# Author: ECCV 2026 Submission

export DETECTRON2_DATASETS=../../../../dataset_share_ssd/OV_seg

CONFIG_FILE="configs/vitb_384_enhanced.yaml"
CHECKPOINT=$1
NUM_GPUS=${2:-1}
OUTPUT_DIR=${3:-"output/eval_enhanced"}

if [ -z "$CHECKPOINT" ]; then
    echo "Usage: sh eval_enhanced.sh [CHECKPOINT_PATH] [NUM_GPUS] [OUTPUT_DIR]"
    echo "Example: sh eval_enhanced.sh output/enhanced_catseg/model_final.pth 4"
    exit 1
fi

echo "============================================"
echo "Enhanced CATSeg Evaluation"
echo "Config: $CONFIG_FILE"
echo "Checkpoint: $CHECKPOINT"
echo "GPUs: $NUM_GPUS"
echo "============================================"

# 评估函数
evaluate_dataset() {
    DATASET=$1
    TEST_JSON=$2
    
    echo ""
    echo "Evaluating on: $DATASET"
    echo "----------------------------------------"
    
    python train_net.py \
        --config-file $CONFIG_FILE \
        --num-gpus $NUM_GPUS \
        --eval-only \
        MODEL.WEIGHTS $CHECKPOINT \
        MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON $TEST_JSON \
        DATASETS.TEST "('$DATASET',)" \
        OUTPUT_DIR "$OUTPUT_DIR/$DATASET"
}

# ADE20K-150
evaluate_dataset "ade20k_sem_seg_val" "datasets/ade150.json"

# ADE20K-847
evaluate_dataset "ade20k_full_sem_seg_val" "datasets/ade847.json"

# Pascal Context 59
evaluate_dataset "pascal_context_sem_seg_val" "datasets/pc59.json"

# Pascal Context 459
evaluate_dataset "pascal_context_459_sem_seg_val" "datasets/pc459.json"

# Pascal VOC 2012
evaluate_dataset "voc_sem_seg_val" "datasets/voc20.json"

echo ""
echo "============================================"
echo "Evaluation Complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
