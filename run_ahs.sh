#!/bin/bash
# AHS-Seg Training Script
# 大道至简版本的训练脚本

# 设置环境
export DETECTRON2_DATASETS=/path/to/datasets

# 训练配置
CONFIG="configs/vitb_384_ahs.yaml"
OUTPUT_DIR="output/ahs_seg_vitb384"
NUM_GPUS=4

# 训练命令
python train_net.py \
    --config-file ${CONFIG} \
    --num-gpus ${NUM_GPUS} \
    OUTPUT_DIR ${OUTPUT_DIR} \
    SOLVER.IMS_PER_BATCH 16

echo "Training completed!"
