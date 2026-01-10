# AHS-Seg: Adaptive Hyperspherical Learning for Open-Vocabulary Semantic Segmentation

<div align="center">

[![Conference](https://img.shields.io/badge/ECCV-2026-blue.svg)](https://eccv.ecva.net/)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](./LICENSE)

**[Paper]** | **[Project Page]** | **[Demo]**

</div>

---

## Overview

**Open-vocabulary semantic segmentation** is a challenging task that aims to segment images using arbitrary textual descriptions, going beyond the limitations of fixed category sets defined during training. This capability is crucial for real-world applications where the set of possible categories is not known in advance.

Recent approaches leverage vision-language foundation models like **CLIP** to achieve open-vocabulary capabilities. However, adapting these pre-trained models for dense prediction tasks while preserving their generalization ability remains a significant challenge.

**AHS-Seg** addresses this challenge by introducing a novel **Adaptive Hyperspherical Learning** framework that:

1. **Preserves CLIP's generalization** through hyperspherical energy constraints
2. **Adapts constraints layer-wise** based on semantic abstraction levels
3. **Leverages multi-scale features** with hyperspherical alignment
4. **Handles class complexity** through dynamic projection mechanisms

<div align="center">
<img src="assets/framework.png" width="90%">
<p><i>Overview of the AHS-Seg framework</i></p>
</div>

---

## Key Contributions

### 1. Hierarchical Adaptive Hyperspherical Energy Modulation (HA-HEM)

We observe that different transformer layers encode features at different semantic levels:
- **Shallow layers** capture low-level, general-purpose features
- **Deep layers** encode high-level, task-specific semantics

HA-HEM introduces **learnable layer-wise adaptation factors** that automatically adjust the orthogonal constraint strength:

```
R_adapt = (1 - α_l · β_m) · I + α_l · β_m · R
```

where `α_l` is learned per-layer and `β_m` provides modality-aware adjustment.

### 2. Multi-scale Hyperspherical Feature Pyramid Network (MH-FPN)

Standard approaches only use the final layer features from CLIP. We propose MH-FPN to:
- Extract features from **multiple intermediate layers** (e.g., layers 3, 7, 11)
- **Align all features on a common hypersphere** for consistent representation
- **Fuse features with learnable weights** optimized end-to-end
- Apply **boundary-aware loss** for improved edge quality

### 3. Dynamic Class-aware Hyperspherical Projection (DC-HSP)

Different semantic categories have varying complexity:
- Simple categories (e.g., "sky", "wall") need compact representations
- Complex categories (e.g., "person", "car") require more expressive capacity

DC-HSP **estimates category complexity** from text embeddings and **dynamically adjusts projection parameters** accordingly.

---

## Results

### Comparison with State-of-the-Art

We evaluate on five standard benchmarks for open-vocabulary semantic segmentation:

| Method | Venue | Backbone | ADE-150 | ADE-847 | PC-59 | PC-459 | COCO-Stuff |
|--------|-------|----------|:-------:|:-------:|:-----:|:------:|:----------:|
| LSeg | ICLR'22 | ViT-L/16 | 18.0 | 3.8 | 46.5 | 7.8 | - |
| OpenSeg | ECCV'22 | ViT-L/14 | 21.1 | 6.3 | 42.1 | 9.0 | - |
| MaskCLIP | ECCV'22 | ViT-B/16 | 23.7 | 8.2 | 45.9 | 10.0 | 24.1 |
| ZegFormer | CVPR'22 | ViT-B/16 | 32.6 | 10.4 | 53.8 | 14.2 | 36.3 |
| OVSeg | CVPR'23 | ViT-L/14 | 29.6 | 9.0 | 55.7 | 12.4 | - |
| SAN | CVPR'23 | ViT-B/16 | 41.1 | 15.7 | 60.2 | 21.1 | 44.3 |
| CAT-Seg | CVPR'24 | ViT-B/16 | 37.9 | 12.0 | 57.5 | 18.2 | 41.8 |
| **AHS-Seg (Ours)** | ECCV'26 | ViT-B/16 | **44.9** | **18.1** | **64.8** | **24.6** | **46.9** |

### Ablation Study

| HA-HEM | MH-FPN | DC-HSP | ADE-150 (mIoU) | ADE-847 (mIoU) |
|:------:|:------:|:------:|:--------------:|:--------------:|
| - | - | - | 42.8 | 16.3 |
| ✓ | - | - | 43.5 (+0.7) | 17.0 (+0.7) |
| - | ✓ | - | 43.8 (+1.0) | 17.2 (+0.9) |
| - | - | ✓ | 43.4 (+0.6) | 17.1 (+0.8) |
| ✓ | ✓ | - | 44.2 (+1.4) | 17.6 (+1.3) |
| ✓ | ✓ | ✓ | **44.9 (+2.1)** | **18.1 (+1.8)** |

---

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 1.12
- CUDA >= 11.3
- Detectron2

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/ZhaoYi-10-13/AHS-Seg.git
cd AHS-Seg

# 2. Create and activate conda environment
conda create -n ahs-seg python=3.8 -y
conda activate ahs-seg

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 \
    -f https://download.pytorch.org/whl/torch_stable.html

# 4. Install Detectron2
pip install detectron2 -f \
    https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.12/index.html

# 5. Install other dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

We support five benchmarks. Please refer to [datasets/README.md](datasets/README.md) for detailed instructions.

### Quick Setup

Set the dataset root directory:
```bash
export DETECTRON2_DATASETS=/path/to/your/datasets
```

Expected structure:
```
$DETECTRON2_DATASETS/
├── ADEChallengeData2016/    # ADE20K-150
├── ADE20K_2021_17_01/       # ADE20K-847
├── coco/                    # COCO-Stuff
└── VOCdevkit/
    ├── VOC2010/             # Pascal Context
    └── VOC2012/             # Pascal VOC
```

---

## Usage

### Training

**Single-GPU Training:**
```bash
python train_net.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --num-gpus 1 \
    OUTPUT_DIR output/ahs_seg_vitb
```

**Multi-GPU Training (Recommended):**
```bash
python train_net.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --num-gpus 4 \
    OUTPUT_DIR output/ahs_seg_vitb
```

**With ViT-L Backbone:**
```bash
python train_net.py \
    --config-file configs/vitl_336_enhanced.yaml \
    --num-gpus 8 \
    OUTPUT_DIR output/ahs_seg_vitl
```

### Evaluation

**Evaluate on All Benchmarks:**
```bash
bash eval_enhanced.sh output/ahs_seg_vitb/model_final.pth 4
```

**Evaluate on Specific Dataset:**
```bash
python train_net.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --num-gpus 4 \
    --eval-only \
    MODEL.WEIGHTS output/ahs_seg_vitb/model_final.pth \
    DATASETS.TEST '("ade20k_150_val_sem_seg",)'
```

### Run Ablation Studies

```bash
bash run_ablations.sh 4  # num_gpus
```

---

## Model Zoo

| Model | Backbone | Training Data | Config | mIoU (ADE-150) | Download |
|-------|----------|---------------|--------|:--------------:|:--------:|
| AHS-Seg | ViT-B/16 | COCO-Stuff | vitb_384_enhanced.yaml | 44.9 | [model]() |
| AHS-Seg | ViT-L/14 | COCO-Stuff | vitl_336_enhanced.yaml | 47.2 | [model]() |

---

## Project Structure

```
AHS-Seg/
├── cat_seg/
│   ├── enhanced_cat_seg_model.py    # Main model with all innovations
│   ├── modeling/
│   │   ├── multi_scale_fusion.py    # MH-FPN implementation
│   │   ├── class_aware_projection.py # DC-HSP implementation
│   │   ├── heads/                   # Segmentation head
│   │   ├── backbone/                # Backbone networks
│   │   └── transformer/             # Transformer modules
│   ├── third_party/
│   │   ├── adaptive_energy.py       # HA-HEM implementation
│   │   ├── enhanced_oft.py          # Enhanced OFT layers
│   │   └── clip.py                  # CLIP model
│   ├── data/                        # Dataset utilities
│   └── config.py                    # Configuration
├── configs/
│   ├── vitb_384_enhanced.yaml       # ViT-B/16 config
│   ├── vitl_336_enhanced.yaml       # ViT-L/14 config
│   └── ablations/                   # Ablation study configs
├── datasets/                        # Dataset preparation scripts
├── demo/                            # Visualization tools
├── train_net.py                     # Training script
├── requirements.txt
├── INSTALL.md
├── LICENSE
└── README.md
```

---

## Configuration Options

Key hyperparameters in `configs/vitb_384_enhanced.yaml`:

```yaml
MODEL:
  SEM_SEG_HEAD:
    # HA-HEM: Hierarchical Adaptive Hyperspherical Energy Modulation
    USE_ADAPTIVE_ENERGY: True
    ENERGY_LOSS_WEIGHT: 0.01
    
    # MH-FPN: Multi-scale Hyperspherical Feature Pyramid Network
    USE_MULTI_SCALE_FUSION: True
    MS_LAYER_INDICES: [3, 7, 11]      # Layers to extract features from
    BOUNDARY_LOSS_WEIGHT: 0.1
    
    # DC-HSP: Dynamic Class-aware Hyperspherical Projection
    USE_CLASS_AWARE_PROJECTION: True
    CONTRASTIVE_LOSS_WEIGHT: 0.05
    HIERARCHY_LOSS_WEIGHT: 0.02

SOLVER:
  BASE_LR: 0.0002
  MAX_ITER: 80000
  IMS_PER_BATCH: 16
```

---

## Visualization

We provide visualization tools in the `demo/` directory:

```bash
# Visualize predictions
python demo/demo.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --input images/*.jpg \
    --output output/visualizations \
    MODEL.WEIGHTS output/ahs_seg_vitb/model_final.pth
```

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@inproceedings{ahsseg2026,
  title={Adaptive Hyperspherical Learning for Open-Vocabulary Semantic Segmentation},
  author={Anonymous},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2026}
}
```

---

## License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

---

## Acknowledgements

We thank the authors of the following projects for their excellent work:
- [CLIP](https://github.com/openai/CLIP) - Vision-language foundation model
- [Detectron2](https://github.com/facebookresearch/detectron2) - Object detection framework
- [CAT-Seg](https://github.com/cvlab-kaist/CAT-Seg) - Cost aggregation methodology

---

## Contact

For questions or issues, please open a [GitHub Issue](https://github.com/ZhaoYi-10-13/AHS-Seg/issues).

---

<div align="center">
<b>If you find this work helpful, please consider giving it a ⭐!</b>
</div>
