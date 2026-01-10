#!/bin/bash
# Ablation Studies Script
# 运行所有消融实验
# Author: ECCV 2026 Submission

NUM_GPUS=${1:-4}
BASE_OUTPUT="output/ablations"

echo "============================================"
echo "Running Ablation Studies"
echo "GPUs: $NUM_GPUS"
echo "Output: $BASE_OUTPUT"
echo "============================================"

# 创建输出目录
mkdir -p $BASE_OUTPUT

# 消融实验配置
declare -A ABLATIONS
ABLATIONS["baseline"]="configs/ablations/vitb_384_baseline.yaml"
ABLATIONS["hahem_only"]="configs/ablations/vitb_384_hahem_only.yaml"
ABLATIONS["mhfpn_only"]="configs/ablations/vitb_384_mhfpn_only.yaml"
ABLATIONS["dchsp_only"]="configs/ablations/vitb_384_dchsp_only.yaml"
ABLATIONS["hahem_mhfpn"]="configs/ablations/vitb_384_hahem_mhfpn.yaml"
ABLATIONS["full"]="configs/vitb_384_enhanced.yaml"

# 运行每个消融实验
for name in "${!ABLATIONS[@]}"; do
    config="${ABLATIONS[$name]}"
    output="$BASE_OUTPUT/$name"
    
    echo ""
    echo "============================================"
    echo "Running: $name"
    echo "Config: $config"
    echo "============================================"
    
    python train_net.py \
        --config-file $config \
        --num-gpus $NUM_GPUS \
        OUTPUT_DIR $output \
        SOLVER.IMS_PER_BATCH 4 \
        SOLVER.MAX_ITER 40000 \
        TEST.EVAL_PERIOD 5000
    
    echo "Completed: $name"
done

echo ""
echo "============================================"
echo "All ablation studies completed!"
echo "Results saved in: $BASE_OUTPUT"
echo "============================================"

# 汇总结果
echo ""
echo "Summary of Results:"
echo "==================="
for name in "${!ABLATIONS[@]}"; do
    result_file="$BASE_OUTPUT/$name/inference/sem_seg_predictions.json"
    if [ -f "$result_file" ]; then
        echo "$name: Check $BASE_OUTPUT/$name for detailed results"
    fi
done
