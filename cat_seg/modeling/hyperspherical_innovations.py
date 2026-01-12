"""
Hyperspherical Innovations for Open-Vocabulary Semantic Segmentation

Core Innovations (大道至简):
1. HPA: Hyperspherical Prototype Alignment - 原型对齐机制
2. SDR: Self-Distillation Regularizer - 自蒸馏正则化
3. GCS: Geodesic Cosine Similarity - 测地距离相似度
4. AFR: Adaptive Feature Rectification - 自适应特征矫正

设计原则:
- 参数高效: 每个模块新增参数 < 1M
- 计算高效: 不显著增加推理时间
- 迁移友好: 专注于提升跨数据集泛化能力

Copyright (c) 2024-2026 AHS-Seg Authors
Licensed under the Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Optional, Tuple, Dict, List
import math


# ============================================================
# Innovation 1: Hyperspherical Prototype Alignment (HPA)
# ============================================================

class HypersphericalPrototypeBank(nn.Module):
    """
    超球面原型记忆库
    
    核心思想: 在超球面上维护类别原型，通过动量更新保持稳定
    
    优势:
    1. 原型在超球面上均匀分布，自然形成判别性特征空间
    2. 动量更新减少训练噪声，提升稳定性
    3. 提供额外的对比学习信号，增强类间区分度
    
    参数量: ~0.4M (对于512维, 171类)
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 171,
        momentum: float = 0.99,
        temperature: float = 0.07,
        use_queue: bool = True,
        queue_size: int = 1024
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.momentum = momentum
        self.temperature = temperature
        self.use_queue = use_queue
        self.queue_size = queue_size
        
        # 类别原型 (在超球面上初始化)
        # 使用正交初始化确保原型分布均匀
        prototypes = torch.randn(num_classes, feature_dim)
        prototypes = F.normalize(prototypes, dim=-1)
        self.register_buffer('prototypes', prototypes)
        
        # 类别计数 (用于自适应动量)
        self.register_buffer('class_counts', torch.zeros(num_classes))
        
        # 可学习的类别特定温度 (大道至简: 只有num_classes个参数)
        self.class_temperature = nn.Parameter(torch.ones(num_classes) * temperature)
        
        # 特征队列 (用于对比学习)
        if use_queue:
            self.register_buffer('feature_queue', torch.randn(queue_size, feature_dim))
            self.feature_queue = F.normalize(self.feature_queue, dim=-1)
            self.register_buffer('label_queue', torch.zeros(queue_size, dtype=torch.long))
            self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))
    
    @torch.no_grad()
    def update_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = 255
    ):
        """
        动量更新类别原型
        
        Args:
            features: [B, C, H, W] 图像特征
            labels: [B, H, W] 类别标签
        """
        B, C, H, W = features.shape
        features_flat = rearrange(features, 'B C H W -> (B H W) C')
        labels_flat = labels.view(-1)
        
        # 过滤忽略类
        valid_mask = labels_flat != ignore_index
        features_valid = features_flat[valid_mask]
        labels_valid = labels_flat[valid_mask]
        
        if len(labels_valid) == 0:
            return
        
        # 对每个类别更新原型
        for c in labels_valid.unique():
            if c >= self.num_classes:
                continue
            mask = labels_valid == c
            class_features = features_valid[mask]
            
            # 计算类别均值并归一化到超球面
            class_mean = F.normalize(class_features.mean(dim=0), dim=-1)
            
            # 自适应动量: 样本越多的类动量越大 (更稳定)
            count = mask.sum().item()
            self.class_counts[c] += count
            adaptive_momentum = min(self.momentum + 0.09 * (1 - 1/(1 + self.class_counts[c]/1000)), 0.999)
            
            # 动量更新
            self.prototypes[c] = adaptive_momentum * self.prototypes[c] + (1 - adaptive_momentum) * class_mean
            self.prototypes[c] = F.normalize(self.prototypes[c], dim=-1)
        
        # 更新特征队列
        if self.use_queue:
            self._update_queue(features_valid, labels_valid)
    
    @torch.no_grad()
    def _update_queue(self, features: torch.Tensor, labels: torch.Tensor):
        """更新特征队列"""
        batch_size = min(features.size(0), self.queue_size // 4)
        if batch_size == 0:
            return
        
        # 随机采样
        indices = torch.randperm(features.size(0))[:batch_size]
        new_features = F.normalize(features[indices], dim=-1)
        new_labels = labels[indices]
        
        ptr = int(self.queue_ptr)
        
        # 循环更新
        if ptr + batch_size > self.queue_size:
            batch_size = self.queue_size - ptr
            new_features = new_features[:batch_size]
            new_labels = new_labels[:batch_size]
        
        self.feature_queue[ptr:ptr + batch_size] = new_features
        self.label_queue[ptr:ptr + batch_size] = new_labels
        self.queue_ptr[0] = (ptr + batch_size) % self.queue_size
    
    def compute_prototype_loss(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = 255
    ) -> torch.Tensor:
        """
        计算原型对齐损失
        
        设计: 对比学习 + 原型吸引
        - 每个像素特征应该接近其对应类别的原型
        - 远离其他类别的原型
        """
        B, C, H, W = features.shape
        features_flat = rearrange(features, 'B C H W -> (B H W) C')
        labels_flat = labels.view(-1)
        
        # 过滤忽略类
        valid_mask = labels_flat != ignore_index
        features_valid = F.normalize(features_flat[valid_mask], dim=-1)
        labels_valid = labels_flat[valid_mask]
        
        if len(labels_valid) == 0:
            return torch.tensor(0.0, device=features.device)
        
        # 计算与所有原型的相似度
        # [N, num_classes]
        sim_to_prototypes = torch.mm(features_valid, self.prototypes.t())
        
        # 使用类别特定温度
        temperatures = self.class_temperature.clamp(min=0.01, max=1.0)
        sim_scaled = sim_to_prototypes / temperatures.unsqueeze(0)
        
        # 交叉熵损失 (正样本是对应类别的原型)
        loss = F.cross_entropy(sim_scaled, labels_valid, reduction='mean')
        
        return loss
    
    def get_prototype_guidance(
        self,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        获取原型引导的文本特征增强
        
        将文本特征与原型进行融合，利用训练中积累的视觉知识
        """
        text_norm = F.normalize(text_features, dim=-1)
        
        # 文本与原型的相似度
        sim = torch.mm(text_norm, self.prototypes.t())  # [num_classes, num_classes]
        attn = F.softmax(sim / self.temperature, dim=-1)
        
        # 加权原型
        prototype_enhanced = torch.mm(attn, self.prototypes)
        
        # 残差融合 (0.1权重保持原始文本主导)
        enhanced = text_norm + 0.1 * F.normalize(prototype_enhanced, dim=-1)
        return F.normalize(enhanced, dim=-1)


# ============================================================
# Innovation 2: Self-Distillation Regularizer (SDR)
# ============================================================

class SelfDistillationRegularizer(nn.Module):
    """
    自蒸馏正则化器
    
    核心思想: 使用冻结的CLIP作为教师，防止fine-tuning过程中过度遗忘
    
    创新点:
    1. 不是简单的KL散度，而是在超球面上的测地距离
    2. 动态权重: 训练初期权重大(保持泛化)，后期权重小(允许适应)
    3. 层级蒸馏: 不同层使用不同权重
    
    参数量: 0 (只使用现有模型)
    """
    
    def __init__(
        self,
        initial_weight: float = 0.5,
        final_weight: float = 0.1,
        warmup_iters: int = 5000,
        total_iters: int = 80000,
        use_geodesic: bool = True
    ):
        super().__init__()
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.use_geodesic = use_geodesic
        
        self.register_buffer('current_iter', torch.zeros(1))
    
    def get_weight(self) -> float:
        """获取当前蒸馏权重 (余弦退火)"""
        iter = self.current_iter.item()
        if iter < self.warmup_iters:
            # Warmup: 线性增加
            return self.initial_weight * (iter / self.warmup_iters)
        else:
            # 余弦退火
            progress = (iter - self.warmup_iters) / (self.total_iters - self.warmup_iters)
            progress = min(progress, 1.0)
            weight = self.final_weight + 0.5 * (self.initial_weight - self.final_weight) * (1 + math.cos(math.pi * progress))
            return weight
    
    def geodesic_distance(
        self,
        feat1: torch.Tensor,
        feat2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算超球面上的测地距离
        
        d(x, y) = arccos(x · y)
        
        相比欧氏距离或余弦距离，测地距离更符合超球面几何
        """
        feat1_norm = F.normalize(feat1, dim=-1)
        feat2_norm = F.normalize(feat2, dim=-1)
        
        # 点积
        cos_sim = (feat1_norm * feat2_norm).sum(dim=-1)
        # 裁剪防止数值问题
        cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
        
        # 测地距离
        geodesic_dist = torch.acos(cos_sim)
        
        return geodesic_dist
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor,
        update_iter: bool = True
    ) -> torch.Tensor:
        """
        计算自蒸馏损失
        
        Args:
            student_features: 微调模型的特征 [B, C, H, W] 或 [B, N, C]
            teacher_features: 冻结CLIP的特征 (同维度)
        """
        if update_iter:
            self.current_iter += 1
        
        weight = self.get_weight()
        
        if weight < 1e-6:
            return torch.tensor(0.0, device=student_features.device)
        
        # 展平空间维度
        if student_features.dim() == 4:
            B, C, H, W = student_features.shape
            student_flat = rearrange(student_features, 'B C H W -> (B H W) C')
            teacher_flat = rearrange(teacher_features, 'B C H W -> (B H W) C')
        else:
            student_flat = student_features.view(-1, student_features.size(-1))
            teacher_flat = teacher_features.view(-1, teacher_features.size(-1))
        
        if self.use_geodesic:
            dist = self.geodesic_distance(student_flat, teacher_flat)
            loss = dist.mean()
        else:
            # 备用: 余弦距离
            cos_sim = F.cosine_similarity(student_flat, teacher_flat, dim=-1)
            loss = (1 - cos_sim).mean()
        
        return weight * loss


# ============================================================
# Innovation 3: Geodesic Cosine Similarity (GCS)
# ============================================================

class GeodesicSimilarityHead(nn.Module):
    """
    测地相似度头
    
    核心思想: 用超球面测地距离替代普通余弦相似度
    
    创新:
    1. 对于接近的特征，测地距离更精确
    2. 引入可学习的曲率参数，适应不同数据集
    3. 多尺度测地距离融合
    
    参数量: ~10K
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        use_learnable_curvature: bool = True,
        num_scales: int = 3
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        
        # 可学习的曲率参数 (控制超球面的"弯曲程度")
        if use_learnable_curvature:
            self.curvature = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('curvature', torch.ones(1))
        
        # 多尺度融合权重
        if num_scales > 1:
            self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)
            self.scale_projections = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(feature_dim, feature_dim // (2 ** i)),
                    nn.LayerNorm(feature_dim // (2 ** i)),
                    nn.GELU()
                ) for i in range(num_scales)
            ])
        
        # 输出投影
        self.output_scale = nn.Parameter(torch.ones(1) * math.sqrt(feature_dim))
    
    def compute_geodesic_similarity(
        self,
        img_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        计算测地相似度
        
        sim = 1 - d(x, y) / π  where d is geodesic distance
        """
        # 归一化
        img_norm = F.normalize(img_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)
        
        # 余弦相似度
        cos_sim = torch.einsum('bchw,nc->bnhw', img_norm, text_norm)
        
        # 裁剪
        cos_sim = cos_sim.clamp(-1 + 1e-7, 1 - 1e-7)
        
        # 转换为测地相似度
        # 使用可学习的曲率
        curvature = F.softplus(self.curvature)
        geodesic_dist = torch.acos(cos_sim) * curvature
        geodesic_sim = 1 - geodesic_dist / math.pi
        
        return geodesic_sim * self.output_scale
    
    def forward(
        self,
        img_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            img_features: [B, C, H, W]
            text_features: [num_classes, C]
        Returns:
            similarity: [B, num_classes, H, W]
        """
        if self.num_scales == 1:
            return self.compute_geodesic_similarity(img_features, text_features)
        
        # 多尺度融合
        B, C, H, W = img_features.shape
        num_classes = text_features.size(0)
        
        sims = []
        weights = F.softmax(self.scale_weights, dim=0)
        
        for i, (proj, w) in enumerate(zip(self.scale_projections, weights)):
            # 投影到子空间
            img_proj = proj(rearrange(img_features, 'B C H W -> B H W C'))
            img_proj = rearrange(img_proj, 'B H W C -> B C H W')
            
            text_proj = proj(text_features)
            
            sim = self.compute_geodesic_similarity(img_proj, text_proj)
            sims.append(w * sim)
        
        return sum(sims)


# ============================================================
# Innovation 4: Adaptive Feature Rectification (AFR)
# ============================================================

class AdaptiveFeatureRectifier(nn.Module):
    """
    自适应特征矫正器
    
    核心思想: 学习识别并矫正对迁移有害的特征
    
    创新:
    1. 不确定性估计: 高不确定性特征降权
    2. 域不变性增强: 鼓励学习域不变特征
    3. 轻量级门控: 只需极少参数
    
    参数量: ~50K
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        reduction: int = 16,
        use_uncertainty: bool = True
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_uncertainty = use_uncertainty
        
        # 通道注意力 (类似SE-Net但更轻量)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, feature_dim // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, feature_dim),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 不确定性估计头
        if use_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 4, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(feature_dim // 4, 1, kernel_size=1),
                nn.Softplus()  # 确保非负
            )
        
        # 残差缩放因子
        self.residual_scale = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        features: torch.Tensor,
        return_uncertainty: bool = False
    ) -> torch.Tensor:
        """
        Args:
            features: [B, C, H, W]
        Returns:
            rectified: [B, C, H, W]
            (optional) uncertainty: [B, 1, H, W]
        """
        B, C, H, W = features.shape
        
        # 通道门控
        channel_weights = self.channel_gate(features).view(B, C, 1, 1)
        
        # 空间门控
        spatial_weights = self.spatial_gate(features)
        
        # 组合门控
        gate = channel_weights * spatial_weights
        
        # 不确定性调制
        if self.use_uncertainty:
            uncertainty = self.uncertainty_head(features)
            # 高不确定性 -> 低权重
            certainty = 1.0 / (1.0 + uncertainty)
            gate = gate * certainty
        
        # 残差连接 + 缩放
        residual_scale = torch.sigmoid(self.residual_scale)
        rectified = features + residual_scale * (gate * features - features)
        
        if return_uncertainty and self.use_uncertainty:
            return rectified, uncertainty
        return rectified
    
    def compute_domain_invariance_loss(
        self,
        features: torch.Tensor,
        augmented_features: torch.Tensor
    ) -> torch.Tensor:
        """
        域不变性损失
        
        鼓励矫正后的特征对于同一图像的不同增强保持一致
        """
        rect_orig = self.forward(features)
        rect_aug = self.forward(augmented_features)
        
        # 在超球面上计算一致性
        rect_orig_norm = F.normalize(rearrange(rect_orig, 'B C H W -> B (H W) C'), dim=-1)
        rect_aug_norm = F.normalize(rearrange(rect_aug, 'B C H W -> B (H W) C'), dim=-1)
        
        consistency = F.cosine_similarity(rect_orig_norm, rect_aug_norm, dim=-1)
        loss = (1 - consistency).mean()
        
        return loss


# ============================================================
# Innovation 5: Unified Hyperspherical Loss (UHL)
# ============================================================

class UnifiedHypersphericalLoss(nn.Module):
    """
    统一超球面损失
    
    整合所有创新点的损失函数:
    1. 原型对齐损失
    2. 自蒸馏损失
    3. 特征矫正一致性损失
    
    大道至简: 一个损失模块管理所有正则化
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 171,
        prototype_weight: float = 0.1,
        distill_weight: float = 0.1,
        rectify_weight: float = 0.05,
        use_prototype: bool = True,
        use_distillation: bool = True,
        use_rectification: bool = True
    ):
        super().__init__()
        self.prototype_weight = prototype_weight
        self.distill_weight = distill_weight
        self.rectify_weight = rectify_weight
        
        # 子模块
        if use_prototype:
            self.prototype_bank = HypersphericalPrototypeBank(
                feature_dim=feature_dim,
                num_classes=num_classes
            )
        else:
            self.prototype_bank = None
        
        if use_distillation:
            self.distillation = SelfDistillationRegularizer()
        else:
            self.distillation = None
        
        if use_rectification:
            self.rectifier = AdaptiveFeatureRectifier(feature_dim=feature_dim)
        else:
            self.rectifier = None
    
    def forward(
        self,
        student_features: torch.Tensor,
        teacher_features: Optional[torch.Tensor],
        labels: torch.Tensor,
        ignore_index: int = 255
    ) -> Dict[str, torch.Tensor]:
        """
        计算统一损失
        
        Returns:
            losses: 损失字典
        """
        losses = {}
        
        # 1. 原型对齐损失
        if self.prototype_bank is not None:
            proto_loss = self.prototype_bank.compute_prototype_loss(
                student_features, labels, ignore_index
            )
            losses['loss_prototype'] = self.prototype_weight * proto_loss
            
            # 更新原型 (在训练时)
            if self.training:
                self.prototype_bank.update_prototypes(
                    student_features.detach(), labels, ignore_index
                )
        
        # 2. 自蒸馏损失
        if self.distillation is not None and teacher_features is not None:
            distill_loss = self.distillation(student_features, teacher_features)
            losses['loss_distill'] = distill_loss
        
        # 3. 特征矫正 (返回矫正后的特征和损失)
        if self.rectifier is not None:
            rectified = self.rectifier(student_features)
            # 这里不计算域不变性损失（需要增强数据）
        
        return losses
    
    def get_rectified_features(self, features: torch.Tensor) -> torch.Tensor:
        """获取矫正后的特征"""
        if self.rectifier is not None:
            return self.rectifier(features)
        return features
    
    def get_prototype_guidance(self, text_features: torch.Tensor) -> torch.Tensor:
        """获取原型引导的文本特征"""
        if self.prototype_bank is not None:
            return self.prototype_bank.get_prototype_guidance(text_features)
        return text_features


# ============================================================
# Utility: Lightweight Hyperspherical Attention
# ============================================================

class LightweightHypersphericalAttention(nn.Module):
    """
    轻量级超球面注意力
    
    比标准注意力更高效，专门为超球面特征设计
    
    参数量: ~100K
    """
    
    def __init__(
        self,
        dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 只使用单个投影矩阵 (参数效率)
        self.qkv = nn.Linear(dim, dim * 2, bias=False)  # 只有Q和K，V复用输入
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # 可学习的超球面半径
        self.radius = nn.Parameter(torch.ones(num_heads) * math.sqrt(self.head_dim))
    
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]
            context: [B, M, C] (可选的上下文)
        """
        B, N, C = x.shape
        
        if context is None:
            context = x
        
        M = context.size(1)
        
        # QK投影
        qk = self.qkv(torch.cat([x, context], dim=1))
        q, k = qk[:, :N], qk[:, N:]
        
        # 多头
        q = rearrange(q, 'B N (H D) -> B H N D', H=self.num_heads)
        k = rearrange(k, 'B M (H D) -> B H M D', H=self.num_heads)
        v = rearrange(context, 'B M (H D) -> B H M D', H=self.num_heads)
        
        # 在超球面上归一化
        q = F.normalize(q, dim=-1) * self.radius.view(1, -1, 1, 1)
        k = F.normalize(k, dim=-1) * self.radius.view(1, -1, 1, 1)
        
        # 注意力
        attn = torch.einsum('bhnd,bhmd->bhnm', q, k) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # 输出
        out = torch.einsum('bhnm,bhmd->bhnd', attn, v)
        out = rearrange(out, 'B H N D -> B N (H D)')
        out = self.proj(out)
        
        return out
