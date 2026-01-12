# Evaluation Results

This document summarizes the evaluation results of the AHS-Seg model on various benchmarks.

## Training Configuration

- **Total Iterations**: 80,000 (80K)
- **Training Duration**: ~11 hours
- **Hardware**: 4x GPU
- **Batch Size**: 16
- **Base Learning Rate**: 0.0002
- **Training Dataset**: COCO-Stuff

## Evaluation Results

### 80K Model (Final)

| Dataset | mIoU | fwIoU | mACC | pACC |
|---------|------|-------|------|------|
| **ADE20K-847** | 11.65 | 50.97 | 22.72 | 61.12 |
| **ADE20K-150** | 31.01 | 59.08 | 49.54 | 70.90 |

### 75K Model Checkpoint

| Dataset | mIoU | fwIoU | mACC | pACC |
|---------|------|-------|------|------|
| **ADE20K-847** | 11.64 | 50.96 | 22.65 | 61.14 |

## Comparison with H-CLIP Baseline

| Method | Backbone | A-847 | A-150 |
|--------|----------|-------|-------|
| H-CLIP (CVPR 2025) | ViT-B/16 | 12.0 | 31.8 |
| CAT-Seg | ViT-B/16 | 12.0 | 31.8 |
| **AHS-Seg (Ours)** | ViT-B/16 | **11.65** | **31.01** |
| **Target (Paper)** | ViT-B/16 | 12.5 | 32.4 |

## Analysis

1. **Training Stability**: The model shows consistent performance between 75K and 80K iterations, indicating stable convergence without overfitting.

2. **Performance vs. Baseline**: 
   - On ADE20K-847: Our model achieves 11.65 mIoU, slightly below the H-CLIP baseline (12.0) by 0.35 points
   - On ADE20K-150: Our model achieves 31.01 mIoU, slightly below the H-CLIP baseline (31.8) by 0.79 points

3. **Gap to Paper Claims**:
   - A-847: Gap of 0.85 mIoU (11.65 vs 12.5)
   - A-150: Gap of 1.39 mIoU (31.01 vs 32.4)

4. **Overall Assessment**: The model successfully trains and achieves competitive results close to the H-CLIP baseline. The slight gap to paper claims may be due to:
   - Hyperparameter tuning
   - Data preprocessing differences
   - Training data variations
   - Implementation details

## Checkpoint Performance

Model checkpoints are saved every 5,000 iterations. All checkpoints are available in the [Vast_Store repository](https://github.com/ZhaoYi-10-13/Vast_Store/tree/models-clean-final/models/ahs_seg_coco_full).

## Reproducibility

To reproduce these results:

```bash
# Training
python train_net.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --num-gpus 4 \
    OUTPUT_DIR output/ahs_seg_coco_full

# Evaluation on ADE20K-847
bash eval_ade847.sh configs/vitb_384_enhanced.yaml 4 \
    output/ahs_seg_coco_full/model_final.pth

# Evaluation on ADE20K-150
bash eval_ade150.sh configs/vitb_384_enhanced.yaml 4 \
    output/ahs_seg_coco_full/model_final.pth

# Multi-checkpoint evaluation
bash eval_multiple_checkpoints.sh
```

## Notes

- All evaluations use sliding window inference with `TEST.SLIDING_WINDOW=True`
- `DATALOADER.NUM_WORKERS=4` for stable evaluation
- Model weights are tracked with Git LFS in the Vast_Store repository
