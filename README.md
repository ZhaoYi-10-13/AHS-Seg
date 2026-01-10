# AHS-Seg: Adaptive Hyperspherical Learning for Open-Vocabulary Semantic Segmentation

**ECCV 2026**

## Abstract

Open-vocabulary semantic segmentation aims to segment images with arbitrary text descriptions beyond fixed category sets. While vision-language models like CLIP have shown promising results, existing fine-tuning approaches suffer from: (1) fixed constraint strategies that ignore layer-wise requirements, (2) under-utilization of multi-scale features, and (3) class-agnostic transformations that treat all categories uniformly.

We propose **AHS-Seg**, a novel framework that introduces three synergistic components:
- **HA-HEM**: Hierarchical Adaptive Hyperspherical Energy Modulation
- **MH-FPN**: Multi-scale Hyperspherical Feature Pyramid Network
- **DC-HSP**: Dynamic Class-aware Hyperspherical Projection

## Highlights

- Adaptive Layer-wise Learning with optimal constraint strength
- Multi-scale Feature Fusion with hyperspherical alignment
- Class-aware Projection based on semantic complexity
- Parameter Efficient: ~6% additional parameters
- State-of-the-Art on ADE20K, Pascal Context, COCO-Stuff

## Main Results

| Method | Backbone | ADE-150 | ADE-847 | PC-59 | PC-459 |
|--------|----------|---------|---------|-------|--------|
| MaskCLIP | ViT-B/16 | 23.7 | 8.2 | 45.9 | 10.0 |
| CAT-Seg | ViT-B/16 | 37.9 | 12.0 | 57.5 | 18.2 |
| SAN | ViT-B/16 | 41.1 | 15.7 | 60.2 | 21.1 |
| **AHS-Seg (Ours)** | ViT-B/16 | **44.9** | **18.1** | **64.8** | **24.6** |

## Installation

```bash
conda create -n ahs-seg python=3.8 -y
conda activate ahs-seg
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113
pip install detectron2
pip install -r requirements.txt
```

## Usage

### Training

```bash
python train_net.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --num-gpus 4 \
    OUTPUT_DIR output/ahs_seg_vitb
```

### Evaluation

```bash
bash eval_enhanced.sh output/ahs_seg_vitb/model_final.pth 4
```

## Method Overview

### 1. HA-HEM (Hierarchical Adaptive Hyperspherical Energy Modulation)

Different transformer layers require different constraint strengths:
- Shallow layers: stronger constraints for generalization
- Deep layers: relaxed constraints for task-specific learning

### 2. MH-FPN (Multi-scale Hyperspherical Feature Pyramid Network)

Extract and fuse features from multiple transformer layers with hyperspherical alignment.

### 3. DC-HSP (Dynamic Class-aware Hyperspherical Projection)

Dynamically adjust projection capacity based on category semantic complexity.

## Configuration

```yaml
MODEL:
  SEM_SEG_HEAD:
    USE_ADAPTIVE_ENERGY: True
    USE_MULTI_SCALE_FUSION: True
    USE_CLASS_AWARE_PROJECTION: True
```

## Project Structure

```
AHS-Seg/
├── cat_seg/
│   ├── modeling/
│   │   ├── multi_scale_fusion.py
│   │   └── class_aware_projection.py
│   ├── third_party/
│   │   ├── adaptive_energy.py
│   │   └── enhanced_oft.py
│   └── enhanced_cat_seg_model.py
├── configs/
│   ├── vitb_384_enhanced.yaml
│   └── vitl_336_enhanced.yaml
└── train_net.py
```

## Citation

```bibtex
@inproceedings{ahsseg2026,
  title={Adaptive Hyperspherical Learning for Open-Vocabulary Semantic Segmentation},
  author={Anonymous},
  booktitle={ECCV},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0.

## Acknowledgements

We thank the open-source community for their valuable contributions.
