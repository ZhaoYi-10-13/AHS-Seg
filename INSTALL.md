# Installation Guide

## Prerequisites

- Linux or macOS
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.3+ (for GPU training)

## Step-by-Step Installation

### 1. Create Conda Environment

```bash
conda create -n ahs-seg python=3.8 -y
conda activate ahs-seg
```

### 2. Install PyTorch

For CUDA 11.3:
```bash
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
```

For CUDA 11.7:
```bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. Install Detectron2

```bash
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.12/index.html
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Verify Installation

```bash
python -c "import torch; print(torch.cuda.is_available())"
python -c "import detectron2; print(detectron2.__version__)"
```

## Dataset Preparation

### ADE20K

```bash
cd datasets
python prepare_ade20k_150.py
python prepare_ade20k_full.py
```

### Pascal Context

```bash
python prepare_pascal_context_59.py
python prepare_pascal_context_459.py
```

### COCO-Stuff

```bash
python prepare_coco_stuff.py
```

## Common Issues

### CUDA Out of Memory

Reduce batch size in config:
```yaml
SOLVER:
  IMS_PER_BATCH: 2
```

### Detectron2 Installation Fails

Try building from source:
```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
pip install -e .
```

## Verification

Run a quick test:
```bash
python train_net.py \
    --config-file configs/vitb_384_enhanced.yaml \
    --num-gpus 1 \
    --eval-only \
    MODEL.WEIGHTS ""
```

If no errors occur, installation is successful.
