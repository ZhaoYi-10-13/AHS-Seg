# AHS-Seg: Adaptive Hyperspherical Semantic Segmentation

## 大道至简的创新设计

本项目提出了三个核心创新点，以 **"大道至简"** 的设计理念，实现参数高效、计算高效且迁移友好的开放词汇语义分割。

---

## 🎯 设计原则

| 原则 | 目标 | 实现 |
|------|------|------|
| **参数高效** | 新增参数 < 2M | 通过低秩分解和参数共享 |
| **计算高效** | 推理时间增加 < 5% | 轻量级模块设计 |
| **迁移友好** | 跨数据集泛化更强 | 自蒸馏 + 原型对齐 |

---

## 🚀 核心创新点

### 1. HPA: Hyperspherical Prototype Alignment (超球面原型对齐)

**核心思想**: 在超球面上维护类别原型，通过动量更新保持稳定性。

**创新性**:
- 原型在超球面上均匀分布，自然形成判别性特征空间
- 动量更新减少训练噪声，提升稳定性
- 可学习的类别特定温度，适应不同类别的复杂度

**公式**:
```
P_c^{t+1} = m * P_c^t + (1-m) * mean(f_i | y_i = c)
L_proto = -log(exp(f·P_y / τ_y) / Σ exp(f·P_c / τ_c))
```

**参数量**: ~0.4M (对于512维, 171类)

---

### 2. SDR: Self-Distillation Regularizer (自蒸馏正则化)

**核心思想**: 使用冻结的CLIP作为教师，防止fine-tuning过程中过度遗忘。

**创新性**:
- 不是简单的KL散度，而是在超球面上的测地距离
- 动态权重：训练初期权重大(保持泛化)，后期权重小(允许适应)
- 计算简单，零额外参数

**公式**:
```
L_distill = w(t) * arccos(f_student · f_teacher)
w(t) = w_final + 0.5 * (w_init - w_final) * (1 + cos(π * t / T))
```

**参数量**: 0 (只使用现有模型)

---

### 3. AFR: Adaptive Feature Rectification (自适应特征矫正)

**核心思想**: 学习识别并矫正对迁移有害的特征。

**创新性**:
- 不确定性估计：高不确定性特征降权
- 通道+空间双重门控
- 轻量级设计，推理时开销极小

**公式**:
```
gate = σ(channel_attn) * σ(spatial_attn) * certainty
f_rectified = f + α * (gate * f - f)
```

**参数量**: ~50K

---

## 📊 与H-CLIP的对比分析

| 方面 | H-CLIP | AHS-Seg (Ours) |
|------|--------|----------------|
| **参数效率** | OFT微调 | OFT + 轻量创新模块 |
| **迁移学习** | 无特殊设计 | 自蒸馏防遗忘 |
| **类别建模** | 文本嵌入 | 文本 + 视觉原型 |
| **特征空间** | 标准余弦 | 超球面测地距离 |
| **泛化策略** | 隐式 | 显式原型对齐 |

---

## 📈 预期效果

基于设计原理，预期在以下方面优于H-CLIP:

1. **跨数据集迁移**: 自蒸馏 + 原型对齐显著减少domain shift
2. **少见类识别**: 原型记忆帮助学习少见类的稳定表示
3. **边界精度**: 特征矫正减少噪声特征的干扰

---

## 🔧 使用方法

### 训练
```bash
# 完整版本
bash run_ahs.sh

# 消融实验
bash run_ahs_ablations.sh
```

### 评估
```bash
bash eval_ahs.sh
```

### 配置选项
```yaml
MODEL:
  SEM_SEG_HEAD:
    # HPA: 原型对齐
    USE_PROTOTYPE_ALIGNMENT: True
    PROTOTYPE_LOSS_WEIGHT: 0.1
    
    # SDR: 自蒸馏
    USE_SELF_DISTILLATION: True
    DISTILL_LOSS_WEIGHT: 0.1
    
    # AFR: 特征矫正
    USE_FEATURE_RECTIFICATION: True
    RECTIFY_LOSS_WEIGHT: 0.05
```

---

## 📁 文件结构

```
AHS-Seg/
├── cat_seg/
│   ├── ahs_seg_model.py                    # 主模型 (大道至简版)
│   └── modeling/
│       └── hyperspherical_innovations.py   # 核心创新模块
├── configs/
│   ├── vitb_384_ahs.yaml                   # 主配置
│   └── ablations/
│       ├── vitb_384_ahs_baseline.yaml      # 无创新点
│       ├── vitb_384_ahs_hpa_only.yaml      # 仅HPA
│       ├── vitb_384_ahs_sdr_only.yaml      # 仅SDR
│       ├── vitb_384_ahs_afr_only.yaml      # 仅AFR
│       └── vitb_384_ahs_full.yaml          # 完整版
├── run_ahs.sh                              # 训练脚本
├── run_ahs_ablations.sh                    # 消融实验脚本
└── eval_ahs.sh                             # 评估脚本
```

---

## 💡 设计哲学

> "大道至简" - 最有效的改进往往是最简洁的。

我们的设计遵循以下哲学:

1. **不堆砌模块**: 每个模块都有明确的目的和理论支撑
2. **参数克制**: 宁可少用参数也不过度设计
3. **可解释性**: 每个loss都有直观的物理意义
4. **即插即用**: 模块可以独立使用或组合使用

---

## 📖 引用

如果本工作对您有帮助，请引用:

```bibtex
@article{ahs-seg-2026,
  title={AHS-Seg: Adaptive Hyperspherical Semantic Segmentation},
  author={Your Name},
  journal={arXiv preprint},
  year={2026}
}
```
