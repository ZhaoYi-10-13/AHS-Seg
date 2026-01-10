"""
Multi-scale Hyperspherical Feature Pyramid Network (MH-FPN)

Core Innovation: Multi-scale feature fusion with hyperspherical alignment
- Extracts features from multiple CLIP layers
- Hyperspherical energy alignment for feature space consistency
- Boundary-aware loss for improved segmentation quality

Copyright (c) 2024-2026 AHS-Seg Authors
Licensed under the Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Tuple, Optional


class HypersphericalAlignment(nn.Module):
    """
    hyperspherical能量对齐模块
    
    将不同层的特征projection到相同半径的hyperspherical上，
    保持特征空间的一致性和可比性
    """
    
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        learnable_radius: bool = True
    ):
        """
        Args:
            in_dim: Input特征dimension
            out_dim: Output特征dimension
            learnable_radius: 半径是否可学习
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # dimensionprojection
        if in_dim != out_dim:
            self.proj = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU()
            )
        else:
            self.proj = nn.Identity()
        
        # 可学习的hyperspherical半径
        if learnable_radius:
            self.radius = nn.Parameter(torch.ones(1) * (out_dim ** 0.5))
        else:
            self.register_buffer('radius', torch.tensor(out_dim ** 0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input特征
               - 3D: [B, N, C] (序列形式)
               - 4D: [B, C, H, W] (图像形式)
        Returns:
            hyperspherical对齐后的特征
        """
        is_4d = x.dim() == 4
        
        if is_4d:
            B, C, H, W = x.shape
            x = rearrange(x, 'B C H W -> B (H W) C')
        
        # dimensionprojection
        x = self.proj(x)
        
        # L2归一化
        x_normalized = F.normalize(x, p=2, dim=-1)
        
        # 缩放到目标半径
        x_sphere = x_normalized * self.radius
        
        if is_4d:
            x_sphere = rearrange(x_sphere, 'B (H W) C -> B C H W', H=H, W=W)
        
        return x_sphere


class FeaturePyramidLevel(nn.Module):
    """
    feature pyramid的单层处理模块
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_scale: int = 1  # 相对于最高分辨率的缩放倍数
    ):
        super().__init__()
        self.spatial_scale = spatial_scale
        
        # hyperspherical对齐
        self.hs_align = HypersphericalAlignment(in_channels, out_channels)
        
        # 空间处理（如果需要上采样）
        if spatial_scale > 1:
            self.upsample = nn.Sequential(
                nn.ConvTranspose2d(
                    out_channels, out_channels,
                    kernel_size=spatial_scale * 2,
                    stride=spatial_scale,
                    padding=spatial_scale // 2
                ),
                nn.GroupNorm(out_channels // 16, out_channels),
                nn.GELU()
            )
        else:
            self.upsample = nn.Identity()
        
        # 特征增强
        self.enhance = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels),
            nn.Conv2d(out_channels, out_channels, 1),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.GELU()
        )
    
    def forward(
        self,
        x: torch.Tensor,
        target_size: Tuple[int, int] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input特征 [B, N, C] 或 [B, C, H, W]
            target_size: 目标空间尺寸 (H, W)
        """
        # hyperspherical对齐
        x = self.hs_align(x)
        
        # 确保是4D
        if x.dim() == 3:
            B, N, C = x.shape
            H = W = int(N ** 0.5)
            x = rearrange(x, 'B (H W) C -> B C H W', H=H, W=W)
        
        # 上采样
        x = self.upsample(x)
        
        # 调整到目标尺寸
        if target_size is not None and x.shape[-2:] != target_size:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        # 特征增强
        x = self.enhance(x)
        
        return x


class MultiScaleHypersphericalFPN(nn.Module):
    """
    multi-scalehypersphericalfeature pyramid网络 (MH-FPN)
    
    核心功能:
    1. 从CLIP不同层提取特征
    2. 对每层特征进行hyperspherical对齐
    3. 使用可学习权重进行fusion
    4. Outputmulti-scalefusion后的特征
    """
    
    def __init__(
        self,
        in_channels_list: List[int] = [768, 768, 768],
        out_channels: int = 512,
        feature_size: int = 24,  # CLIP特征图尺寸
        fusion_type: str = 'weighted'  # 'weighted', 'attention', 'concat'
    ):
        """
        Args:
            in_channels_list: 各层Input通道数
            out_channels: Output通道数
            feature_size: 特征图空间尺寸
            fusion_type: fusion类型
        """
        super().__init__()
        self.num_levels = len(in_channels_list)
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.fusion_type = fusion_type
        
        # 各层处理模块
        self.level_processors = nn.ModuleList([
            FeaturePyramidLevel(
                in_channels=in_ch,
                out_channels=out_channels,
                spatial_scale=1  # CLIP各层分辨率相同
            )
            for in_ch in in_channels_list
        ])
        
        # fusion权重
        if fusion_type == 'weighted':
            self.fusion_weights = nn.Parameter(
                torch.ones(self.num_levels) / self.num_levels
            )
        elif fusion_type == 'attention':
            self.fusion_attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(out_channels * self.num_levels, self.num_levels),
                nn.Softmax(dim=-1)
            )
        elif fusion_type == 'concat':
            self.fusion_proj = nn.Sequential(
                nn.Conv2d(out_channels * self.num_levels, out_channels, 1),
                nn.GroupNorm(out_channels // 16, out_channels),
                nn.GELU()
            )
        
        # Outputprojection
        self.output_proj = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(out_channels // 16, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1)
        )
        
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: 各层特征列表
                          每个元素为 [B, N, C] 或 [B, C, H, W]
        Returns:
            fusion后的特征 [B, C, H, W]
        """
        assert len(features_list) == self.num_levels, \
            f"Expected {self.num_levels} features, got {len(features_list)}"
        
        # 处理各层特征
        processed_features = []
        target_size = (self.feature_size, self.feature_size)
        
        for feat, processor in zip(features_list, self.level_processors):
            processed = processor(feat, target_size)
            processed_features.append(processed)
        
        # fusion
        if self.fusion_type == 'weighted':
            weights = F.softmax(self.fusion_weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, processed_features))
            
        elif self.fusion_type == 'attention':
            # 拼接用于compute注意力
            stacked = torch.stack(processed_features, dim=1)  # [B, L, C, H, W]
            B, L, C, H, W = stacked.shape
            # compute注意力权重
            concat_for_attn = stacked.view(B, L * C, H, W)
            attn_weights = self.fusion_attention(concat_for_attn)  # [B, L]
            # 加权求和
            fused = torch.einsum('bl,blchw->bchw', attn_weights, stacked)
            
        elif self.fusion_type == 'concat':
            concat = torch.cat(processed_features, dim=1)
            fused = self.fusion_proj(concat)
        
        # Outputprojection
        output = self.output_proj(fused)
        
        return output
    
    def get_fusion_weights(self) -> torch.Tensor:
        """get当前fusion权重（用于分析）"""
        if self.fusion_type == 'weighted':
            return F.softmax(self.fusion_weights, dim=0).detach()
        else:
            return None


class BoundaryAwareLoss(nn.Module):
    """
    boundary感知loss
    
    使用Sobel算子检测boundary，鼓励模型在boundary区域有更好的预测
    """
    
    def __init__(self, edge_weight: float = 1.0):
        super().__init__()
        self.edge_weight = edge_weight
        
        # Sobel滤波器
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32)
        
        sobel_y = torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ], dtype=torch.float32)
        
        self.register_buffer('sobel_x', sobel_x.view(1, 1, 3, 3))
        self.register_buffer('sobel_y', sobel_y.view(1, 1, 3, 3))
    
    def compute_edges(self, x: torch.Tensor) -> torch.Tensor:
        """
        computeboundary图
        
        Args:
            x: Input [B, C, H, W] 或 [B, H, W]
        Returns:
            boundary图 [B, C, H, W] 或 [B, 1, H, W]
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        B, C, H, W = x.shape
        
        # 对每个通道computeboundary
        x_reshaped = x.view(B * C, 1, H, W)
        
        # Sobel边缘检测
        edge_x = F.conv2d(x_reshaped.float(), self.sobel_x, padding=1)
        edge_y = F.conv2d(x_reshaped.float(), self.sobel_y, padding=1)
        
        # boundary强度
        edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-8)
        
        edge_magnitude = edge_magnitude.view(B, C, H, W)
        
        return edge_magnitude
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测分割 [B, C, H, W] (logits)
            target: 目标分割 [B, H, W] (类别索引) 或 [B, C, H, W] (one-hot)
        Returns:
            boundaryloss
        """
        # 处理target格式
        if target.dim() == 3:
            # 将类别索引转换为one-hot（简化版，只computeboundary）
            target_float = target.float().unsqueeze(1)
        else:
            target_float = target.float()
        
        # 对预测应用sigmoid
        pred_prob = torch.sigmoid(pred)
        
        # computeboundary
        pred_edges = self.compute_edges(pred_prob)
        target_edges = self.compute_edges(target_float)
        
        # 归一化boundary图到[0, 1]
        pred_edges_norm = pred_edges / (pred_edges.max() + 1e-8)
        target_edges_norm = target_edges / (target_edges.max() + 1e-8)
        
        # boundary区域的BCEloss
        boundary_loss = F.binary_cross_entropy(
            pred_edges_norm,
            target_edges_norm.expand_as(pred_edges_norm),
            reduction='mean'
        )
        
        return self.edge_weight * boundary_loss


class MultiScaleBoundaryLoss(nn.Module):
    """
    multi-scaleboundaryloss
    
    在多个尺度上computeboundaryloss，更好地处理不同大小的物体
    """
    
    def __init__(
        self,
        scales: List[float] = [1.0, 0.5, 0.25],
        weights: List[float] = [1.0, 0.5, 0.25]
    ):
        super().__init__()
        self.scales = scales
        self.weights = weights
        self.boundary_loss = BoundaryAwareLoss()
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            pred: 预测分割 [B, C, H, W]
            target: 目标分割 [B, H, W] 或 [B, C, H, W]
        """
        total_loss = 0.0
        
        for scale, weight in zip(self.scales, self.weights):
            if scale != 1.0:
                # 下采样
                H, W = pred.shape[-2:]
                new_size = (int(H * scale), int(W * scale))
                
                pred_scaled = F.interpolate(
                    pred, size=new_size, mode='bilinear', align_corners=False
                )
                
                if target.dim() == 3:
                    target_scaled = F.interpolate(
                        target.float().unsqueeze(1),
                        size=new_size,
                        mode='nearest'
                    ).squeeze(1).long()
                else:
                    target_scaled = F.interpolate(
                        target, size=new_size, mode='bilinear', align_corners=False
                    )
            else:
                pred_scaled = pred
                target_scaled = target
            
            scale_loss = self.boundary_loss(pred_scaled, target_scaled)
            total_loss = total_loss + weight * scale_loss
        
        return total_loss / sum(self.weights)


class FeatureExtractorHook:
    """
    特征提取钩子
    用于从CLIP模型中提取多层特征
    """
    
    def __init__(self):
        self.features = {}
        self.hooks = []
    
    def get_hook(self, name: str):
        """创建钩子函数"""
        def hook(module, input, output):
            self.features[name] = output
        return hook
    
    def register(self, model: nn.Module, layer_names: List[str]):
        """
        注册钩子到指定层
        
        Args:
            model: 模型
            layer_names: 层名称列表（如 ['layer.3', 'layer.7', 'layer.11']）
        """
        for name, module in model.named_modules():
            if name in layer_names:
                hook = module.register_forward_hook(self.get_hook(name))
                self.hooks.append(hook)
    
    def get_features(self, names: List[str]) -> List[torch.Tensor]:
        """get指定层的特征"""
        return [self.features[name] for name in names if name in self.features]
    
    def clear(self):
        """清空特征缓存"""
        self.features = {}
    
    def remove_hooks(self):
        """移除所有钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
