#!/bin/bash
# ==========================================
# 多检查点评估脚本
# ==========================================
# 用不同的模型权重测试 A-847 和 A-150 数据集
# 从后往前测试，寻找最佳检查点

set -e

export DETECTRON2_DATASETS=/root/datasets

CONFIG_FILE="configs/vitb_384_enhanced.yaml"
NUM_GPUS=4
BASE_DIR="output/ahs_seg_coco_full"

# 模型检查点列表（从后往前）
CHECKPOINTS=(
    "model_0074999.pth"  # 75K
    "model_0069999.pth"  # 70K
    "model_0064999.pth"  # 65K
    "model_0059999.pth"  # 60K
    "model_0054999.pth"  # 55K
    "model_0049999.pth"  # 50K
    "model_0044999.pth"  # 45K
    "model_0039999.pth"  # 40K
)

echo "=========================================="
echo "🔍 多检查点评估任务"
echo "=========================================="
echo "将测试以下检查点（从75K到40K）:"
for ckpt in "${CHECKPOINTS[@]}"; do
    echo "  • $ckpt"
done
echo ""
echo "每个检查点将在以下数据集上评估:"
echo "  1. ADE20K-847 (A-847) - 最难的数据集"
echo "  2. ADE20K-150 (A-150) - 标准数据集"
echo ""
echo "=========================================="
echo ""

# 结果汇总文件
SUMMARY_FILE="/root/eval_summary.txt"
echo "========================================" > "$SUMMARY_FILE"
echo "AHS-Seg 多检查点评估结果汇总" >> "$SUMMARY_FILE"
echo "评估时间: $(date)" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "" >> "$SUMMARY_FILE"

for checkpoint in "${CHECKPOINTS[@]}"; do
    CKPT_PATH="$BASE_DIR/$checkpoint"
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "⚠️  跳过: $checkpoint (文件不存在)"
        continue
    fi
    
    # 提取迭代数
    ITER=$(echo "$checkpoint" | grep -oP '\d+')
    echo ""
    echo "=========================================="
    echo "📍 检查点: $checkpoint (Iter ${ITER})"
    echo "=========================================="
    echo "" >> "$SUMMARY_FILE"
    echo "-------------------------------------------" >> "$SUMMARY_FILE"
    echo "检查点: $checkpoint (Iter ${ITER})" >> "$SUMMARY_FILE"
    echo "-------------------------------------------" >> "$SUMMARY_FILE"
    
    # ===== 评估 A-847 =====
    echo ""
    echo "🔹 [1/2] 评估 A-847..."
    OUTPUT_DIR_847="$BASE_DIR/eval_ade847_${ITER}"
    mkdir -p "$OUTPUT_DIR_847"
    
    cd /root/AHS-Seg
    /venv/ahs-seg/bin/python train_net.py \
      --config-file "$CONFIG_FILE" \
      --num-gpus "$NUM_GPUS" \
      --eval-only \
      OUTPUT_DIR "$OUTPUT_DIR_847" \
      MODEL.WEIGHTS "$CKPT_PATH" \
      MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade847.json" \
      DATASETS.TEST "('ade20k_full_sem_seg_freq_val_all',)" \
      TEST.SLIDING_WINDOW True \
      MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
      DATALOADER.NUM_WORKERS 4 \
      2>&1 | tee "$OUTPUT_DIR_847/eval_log.txt"
    
    # 提取 A-847 结果
    RESULT_847=$(grep "copypaste:" "$OUTPUT_DIR_847/eval_log.txt" | tail -1 | awk '{print $3}')
    echo "  ✅ A-847 结果: $RESULT_847"
    echo "A-847: $RESULT_847" >> "$SUMMARY_FILE"
    
    # ===== 评估 A-150 =====
    echo ""
    echo "🔹 [2/2] 评估 A-150..."
    OUTPUT_DIR_150="$BASE_DIR/eval_ade150_${ITER}"
    mkdir -p "$OUTPUT_DIR_150"
    
    cd /root/AHS-Seg
    /venv/ahs-seg/bin/python train_net.py \
      --config-file "$CONFIG_FILE" \
      --num-gpus "$NUM_GPUS" \
      --eval-only \
      OUTPUT_DIR "$OUTPUT_DIR_150" \
      MODEL.WEIGHTS "$CKPT_PATH" \
      MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON "datasets/ade150.json" \
      DATASETS.TEST "('ade20k_150_test_sem_seg',)" \
      TEST.SLIDING_WINDOW True \
      MODEL.SEM_SEG_HEAD.POOLING_SIZES "[1,1]" \
      DATALOADER.NUM_WORKERS 4 \
      2>&1 | tee "$OUTPUT_DIR_150/eval_log.txt"
    
    # 提取 A-150 结果
    RESULT_150=$(grep "copypaste:" "$OUTPUT_DIR_150/eval_log.txt" | tail -1 | awk '{print $3}')
    echo "  ✅ A-150 结果: $RESULT_150"
    echo "A-150: $RESULT_150" >> "$SUMMARY_FILE"
    
    echo ""
    echo "✅ $checkpoint 评估完成！"
    echo "  • A-847: $RESULT_847"
    echo "  • A-150: $RESULT_150"
    echo ""
done

echo "" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"
echo "评估完成时间: $(date)" >> "$SUMMARY_FILE"
echo "========================================" >> "$SUMMARY_FILE"

echo ""
echo "=========================================="
echo "🎉 所有评估完成！"
echo "=========================================="
echo ""
echo "📊 查看完整结果:"
echo "  cat $SUMMARY_FILE"
echo ""
echo "🎯 目标对比:"
echo "  • A-847: 12.5 mIoU (H-CLIP baseline: 12.0)"
echo "  • A-150: 32.4 mIoU (H-CLIP baseline: 31.8)"
echo ""
