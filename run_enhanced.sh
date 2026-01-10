#!/bin/bash
# Enhanced CATSeg Training Script
# 使用三个创新点: HA-HEM, MH-FPN, DC-HSP
# Author: ECCV 2026 Submission

# 配置
CONFIG_FILE="configs/vitb_384_enhanced.yaml"
OUTPUT_DIR="output/enhanced_catseg"
NUM_GPUS=4

# 确保输出目录存在
mkdir -p $OUTPUT_DIR

echo "============================================"
echo "Enhanced CATSeg Training"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"
echo "GPUs: $NUM_GPUS"
echo "============================================"
echo ""
echo "创新点:"
echo "  1. HA-HEM: 层级自适应超球面能量调制"
echo "  2. MH-FPN: 多尺度超球面特征金字塔"
echo "  3. DC-HSP: 动态类别感知超球面投影"
echo "============================================"

# 训练命令
python train_net.py \
    --config-file $CONFIG_FILE \
    --num-gpus $NUM_GPUS \
    OUTPUT_DIR $OUTPUT_DIR \
    SOLVER.IMS_PER_BATCH 4 \
    SOLVER.BASE_LR 0.0002 \
    SOLVER.MAX_ITER 80000 \
    TEST.EVAL_PERIOD 5000

echo "Training completed!"
