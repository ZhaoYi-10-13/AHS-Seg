#!/bin/bash
# ==========================================
# ADE20K-150 Metrics 评估脚本
# ==========================================
# 用于评估AHS-Seg模型在ADE20K-150数据集上的性能

set -e

export DETECTRON2_DATASETS=/root/datasets

CONFIG_FILE="${1:-configs/vitb_384_enhanced.yaml}"
NUM_GPUS="${2:-4}"
MODEL_WEIGHTS="${3:-output/ahs_seg_coco_full/model_final.pth}"
OUTPUT_DIR="${4:-output/ahs_seg_coco_full/eval_ade150}"

echo "=========================================="
echo "ADE20K-150 Metrics 评估脚本"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS"
echo "模型权重: $MODEL_WEIGHTS"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查模型权重文件
if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "❌ 错误: 模型权重文件不存在: $MODEL_WEIGHTS"
    exit 1
fi

# 检查数据集是否准备完成
VAL_DIR="/root/datasets/ADEChallengeData2016/images/validation"
if [ ! -d "$VAL_DIR" ] || [ -z "$(ls -A $VAL_DIR 2>/dev/null)" ]; then
    echo "❌ 错误: ADE20K-150数据集未准备完成"
    echo "请运行: python datasets/prepare_ade20k_150.py"
    exit 1
fi

echo "✅ 数据集检查完成"
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "开始评估..."
echo ""

# 运行评估
cd /root/AHS-Seg

/venv/ahs-seg/bin/python train_net.py \
  --config-file "$CONFIG_FILE" \
  --num-gpus "$NUM_GPUS" \
  --eval-only \
  OUTPUT_DIR "$OUTPUT_DIR" \
  MODEL.WEIGHTS "$MODEL_WEIGHTS" \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
  DATASETS.TEST "('ade20k_150_test_sem_seg',)" \
  TEST.SLIDING_WINDOW True \
  MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
  DATALOADER.NUM_WORKERS 4 \
  2>&1 | tee "$OUTPUT_DIR/eval_log.txt"

echo ""
echo "=========================================="
echo "✅ 评估完成！"
echo "=========================================="
echo ""
echo "📊 查看结果:"
echo "  grep copypaste $OUTPUT_DIR/eval_log.txt"
echo ""
echo "🎯 目标指标 (我们的目标):"
echo "  • mIoU: 32.4"
echo ""
echo "📈 H-CLIP baseline:"
echo "  • ViT-B/16: 31.8 mIoU"
echo "  • ViT-L/14: 37.9 mIoU"
echo ""
