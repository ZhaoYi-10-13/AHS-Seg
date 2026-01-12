#!/bin/bash
# ==========================================
# A-847 Metrics 评估脚本
# ==========================================
# 用于评估AHS-Seg模型在ADE20K-847数据集上的性能
# 这是最难的评估数据集（847个类别）

set -e

export DETECTRON2_DATASETS=/root/datasets

CONFIG_FILE="${1:-configs/vitb_384_enhanced.yaml}"
NUM_GPUS="${2:-4}"
MODEL_WEIGHTS="${3:-output/ahs_seg_coco_full/model_final.pth}"
OUTPUT_DIR="${4:-output/ahs_seg_coco_full/eval_ade847}"

echo "=========================================="
echo "A-847 Metrics 评估脚本"
echo "=========================================="
echo "配置文件: $CONFIG_FILE"
echo "GPU数量: $NUM_GPUS"
echo "模型权重: $MODEL_WEIGHTS"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 检查模型权重文件
if [ ! -f "$MODEL_WEIGHTS" ]; then
    echo "❌ 错误: 模型权重文件不存在: $MODEL_WEIGHTS"
    echo ""
    echo "请指定正确的模型权重文件路径，例如:"
    echo "  bash eval_ade847.sh configs/vitb_384_enhanced.yaml 4 output/ahs_seg_coco_full/model_0079999.pth"
    exit 1
fi

# 检查数据集是否准备完成
INDEX_FILE="/root/datasets/ADE20K_2021_17_01/index_ade20k.pkl"
if [ ! -f "$INDEX_FILE" ]; then
    echo "❌ 错误: A-847数据集索引文件不存在"
    echo "请先运行: python datasets/prepare_ade20k_847.py"
    exit 1
fi

VAL_DIR="/root/datasets/ADE20K_2021_17_01/images_detectron2/validation"
if [ ! -d "$VAL_DIR" ] || [ -z "$(ls -A $VAL_DIR 2>/dev/null)" ]; then
    echo "⚠️  警告: A-847数据集未处理完成"
    echo "正在运行数据准备脚本..."
    cd /root/AHS-Seg
    /venv/ahs-seg/bin/python datasets/prepare_ade20k_847.py
    echo ""
fi

echo "开始评估..."
echo ""

# 运行评估
cd /root/AHS-Seg

# 减少NUM_WORKERS以避免资源限制错误
# 评估时不需要太多worker，4个GPU使用4-8个worker即可
/venv/ahs-seg/bin/python train_net.py \
  --config-file "$CONFIG_FILE" \
  --num-gpus "$NUM_GPUS" \
  --eval-only \
  OUTPUT_DIR "$OUTPUT_DIR" \
  MODEL.WEIGHTS "$MODEL_WEIGHTS" \
  MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade847.json" \
  DATASETS.TEST "('ade20k_full_sem_seg_freq_val_all',)" \
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
echo "  cat $OUTPUT_DIR/log.txt | grep copypaste"
echo ""
echo "🎯 目标指标 (H-CLIP baseline):"
echo "  • ViT-B/16: 12.5 mIoU"
echo "  • ViT-L/14: 16.5 mIoU"
echo ""
echo "如果超过这些数值，就是SOTA！🎉"
