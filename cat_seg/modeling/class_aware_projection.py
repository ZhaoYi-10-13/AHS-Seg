"""
Dynamic Class-aware Hyperspherical Projection (DC-HSP)

Core Innovation: Class-aware adaptive projection
- Dynamically adjusts projection based on semantic complexity
- Contrastive learning for enhanced discriminability
- Hierarchical structure modeling to reduce confusion

Copyright (c) 2024-2026 AHS-Seg Authors
Licensed under the Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import List, Tuple, Optional, Dict


class ClassComplexityEstimator(nn.Module):
    """
    classcomplexityevaluation器
    
    基于class的文本嵌入evaluation其语义complexity
    复杂class（如"person"）获得更大的projection空间
    简单class（如"sky"）使用更紧凑的表示
    """
    
    def __init__(
        self,
        text_dim: int = 512,
        hidden_dim: int = 256,
        num_complexity_levels: int = 5
    ):
        """
        Args:
            text_dim: 文本嵌入dimension
            hidden_dim: 隐藏层dimension
            num_complexity_levels: complexity等级数（软划分）
        """
        super().__init__()
        self.num_levels = num_complexity_levels
        
        # complexityevaluation网络
        self.complexity_net = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output [0, 1]
        )
        
        # 可选：多尺度complexity（不同粒度的complexityevaluation）
        self.multi_scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(text_dim, hidden_dim // (2 ** i)),
                nn.GELU(),
                nn.Linear(hidden_dim // (2 ** i), 1),
                nn.Sigmoid()
            )
            for i in range(3)  # 3个尺度
        ])
    
    def forward(
        self,
        text_embeddings: torch.Tensor,
        return_multi_scale: bool = False
    ) -> torch.Tensor:
        """
        Args:
            text_embeddings: [num_classes, text_dim] 或 [B, num_classes, text_dim]
            return_multi_scale: 是否返回多尺度complexity
            
        Returns:
            complexity: [num_classes, 1] 或 [B, num_classes, 1]，范围[0, 1]
        """
        # 主complexity
        complexity = self.complexity_net(text_embeddings)
        
        if return_multi_scale:
            multi_complexities = [net(text_embeddings) for net in self.multi_scale]
            return complexity, multi_complexities
        
        return complexity


class DynamicProjection(nn.Module):
    """
    dynamicprojection模块
    
    根据classcomplexitydynamic调整projection参数
    P_i = P_base + c_i * ΔP
    """
    
    def __init__(
        self,
        in_dim: int = 512,
        out_dim: int = 512,
        rank: int = 64  # 低秩分解的秩
    ):
        """
        Args:
            in_dim: Inputdimension
            out_dim: Outputdimension
            rank: dynamic偏移的低秩分解秩（控制参数量）
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.rank = rank
        
        # 基础projection矩阵
        self.base_proj = nn.Linear(in_dim, out_dim, bias=False)
        
        # dynamic偏移（低秩分解）
        # ΔP = A @ B, where A: [in_dim, rank], B: [rank, out_dim]
        self.delta_A = nn.Linear(in_dim, rank, bias=False)
        self.delta_B = nn.Linear(rank, out_dim, bias=False)
        
        # complexity调制因子
        self.complexity_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        # 初始化：dynamic偏移初始为0
        nn.init.zeros_(self.delta_B.weight)
    
    def forward(
        self,
        features: torch.Tensor,
        complexity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            features: Input特征 [B, N, C] 或 [num_classes, C]
            complexity: complexity分数 [num_classes, 1] 或 None
            
        Returns:
            projection后的特征
        """
        # 基础projection
        base_output = self.base_proj(features)
        
        if complexity is not None:
            # dynamic偏移
            delta_output = self.delta_B(self.delta_A(features))
            
            # complexity调制
            # 确保complexitydimension匹配
            if complexity.dim() == 2 and features.dim() == 2:
                # [num_classes, 1] * [num_classes, out_dim]
                modulation = complexity * self.complexity_scale
            elif complexity.dim() == 2 and features.dim() == 3:
                # 需要广播
                modulation = complexity.unsqueeze(0) * self.complexity_scale
            else:
                modulation = complexity * self.complexity_scale
            
            output = base_output + modulation * delta_output
        else:
            output = base_output
        
        return output


class DynamicClassAwareProjection(nn.Module):
    """
    dynamicclassawarehypersphericalprojection (DC-HSP)
    
    核心功能:
    1. evaluation每个class的语义complexity
    2. 根据complexitydynamic调整projection参数
    3. 在hyperspherical上进行normalization
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        proj_dim: int = 512,
        rank: int = 64,
        temperature: float = 0.07
    ):
        """
        Args:
            feature_dim: Input特征dimension
            proj_dim: projectionOutputdimension
            rank: dynamicprojection的低秩秩
            temperature: 相似度compute的温度系数
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.proj_dim = proj_dim
        self.temperature = temperature
        
        # classcomplexityevaluation器
        self.complexity_estimator = ClassComplexityEstimator(
            text_dim=feature_dim,
            hidden_dim=256
        )
        
        # 图像特征projection
        self.image_proj = DynamicProjection(
            in_dim=feature_dim,
            out_dim=proj_dim,
            rank=rank
        )
        
        # 文本特征projection
        self.text_proj = DynamicProjection(
            in_dim=feature_dim,
            out_dim=proj_dim,
            rank=rank
        )
        
        # 可学习的hyperspherical半径
        self.radius = nn.Parameter(torch.ones(1) * (proj_dim ** 0.5))
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        return_complexity: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_features: 图像特征 [B, N, C] 或 [B, C, H, W]
            text_features: 文本特征 [num_classes, C] 或 [B, num_classes, C]
            return_complexity: 是否返回complexity分数
            
        Returns:
            proj_image: projection后的图像特征
            proj_text: projection后的文本特征
            (optional) complexity: classcomplexity
        """
        # 处理图像特征dimension
        is_4d = image_features.dim() == 4
        if is_4d:
            B, C, H, W = image_features.shape
            image_features = rearrange(image_features, 'B C H W -> B (H W) C')
        
        # evaluationclasscomplexity
        complexity = self.complexity_estimator(text_features)
        
        # dynamicprojection
        proj_image = self.image_proj(image_features, complexity=None)  # 图像不需要complexity调制
        proj_text = self.text_proj(text_features, complexity=complexity)
        
        # hypersphericalnormalization
        proj_image = F.normalize(proj_image, dim=-1) * self.radius
        proj_text = F.normalize(proj_text, dim=-1) * self.radius
        
        # 恢复空间dimension
        if is_4d:
            proj_image = rearrange(proj_image, 'B (H W) C -> B C H W', H=H, W=W)
        
        if return_complexity:
            return proj_image, proj_text, complexity
        
        return proj_image, proj_text


class ClassContrastiveLoss(nn.Module):
    """
    classcontrastive学习loss
    
    鼓励同类像素特征相似，不同类像素特征分离
    使用原型contrastive而非逐像素contrastive，提高效率
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        num_hard_negatives: int = 5
    ):
        """
        Args:
            temperature: 温度系数
            num_hard_negatives: 困难负样本数量
        """
        super().__init__()
        self.temperature = temperature
        self.num_hard_negatives = num_hard_negatives
    
    def compute_class_prototypes(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        num_classes: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        compute每个class的原型（平均特征）
        
        Args:
            features: [B, C, H, W] 或 [B, N, C]
            labels: [B, H, W] class索引
            num_classes: class数量
            
        Returns:
            prototypes: [num_classes, C]
            valid_mask: [num_classes] 哪些class有效
        """
        if features.dim() == 4:
            B, C, H, W = features.shape
            features = rearrange(features, 'B C H W -> B (H W) C')
            labels = labels.view(B, -1)  # [B, H*W]
        
        B, N, C = features.shape
        device = features.device
        
        prototypes = torch.zeros(num_classes, C, device=device)
        counts = torch.zeros(num_classes, device=device)
        
        for c in range(num_classes):
            mask = (labels == c)  # [B, N]
            if mask.sum() > 0:
                class_features = features[mask]  # [num_pixels, C]
                prototypes[c] = class_features.mean(dim=0)
                counts[c] = mask.sum()
        
        valid_mask = counts > 0
        
        # normalization
        prototypes = F.normalize(prototypes, dim=-1)
        
        return prototypes, valid_mask
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        text_embeddings: torch.Tensor,
        ignore_index: int = 255
    ) -> torch.Tensor:
        """
        Args:
            features: 图像特征 [B, C, H, W]
            labels: 标签 [B, H, W]
            text_embeddings: 文本嵌入 [num_classes, C]
            ignore_index: 忽略的标签值
            
        Returns:
            contrastiveloss
        """
        num_classes = text_embeddings.shape[0]
        
        # 创建有效掩码
        valid_labels = labels.clone()
        valid_labels[labels == ignore_index] = 0  # 临时替换
        
        # computeclass原型
        prototypes, valid_mask = self.compute_class_prototypes(
            features, valid_labels, num_classes
        )
        
        # 文本嵌入normalization
        text_normalized = F.normalize(text_embeddings, dim=-1)
        
        # compute原型与文本的相似度矩阵
        # [num_classes, num_classes]
        similarity = torch.mm(prototypes, text_normalized.t()) / self.temperature
        
        # 对角线是正样本（原型应该与对应的文本嵌入相似）
        # 非对角线是负样本
        labels_one_hot = torch.eye(num_classes, device=features.device)
        
        # 只考虑有效class
        if valid_mask.sum() > 1:  # 至少2个有效class
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            
            # 选择有效的行和列
            valid_sim = similarity[valid_indices][:, valid_indices]
            valid_labels_oh = labels_one_hot[valid_indices][:, valid_indices]
            
            # contrastiveloss
            loss = F.cross_entropy(
                valid_sim,
                valid_labels_oh,
                reduction='mean'
            )
        else:
            loss = torch.tensor(0.0, device=features.device)
        
        return loss


class HierarchyStructureLoss(nn.Module):
    """
    classhierarchy结构loss
    
    鼓励语义相关的class在嵌入空间中接近，
    但保持足够的判别距离
    """
    
    def __init__(
        self,
        parent_margin: float = 0.3,
        sibling_margin: float = 0.5
    ):
        """
        Args:
            parent_margin: 父子class的目标距离
            sibling_margin: 兄弟class的目标距离
        """
        super().__init__()
        self.parent_margin = parent_margin
        self.sibling_margin = sibling_margin
    
    def forward(
        self,
        class_embeddings: torch.Tensor,
        hierarchy_relations: Optional[Dict[str, List[Tuple[int, int]]]] = None
    ) -> torch.Tensor:
        """
        Args:
            class_embeddings: [num_classes, C]
            hierarchy_relations: 字典，包含:
                - 'parent_child': [(parent_idx, child_idx), ...]
                - 'siblings': [(idx1, idx2), ...]
                
        Returns:
            hierarchy结构loss
        """
        if hierarchy_relations is None or len(hierarchy_relations) == 0:
            return torch.tensor(0.0, device=class_embeddings.device)
        
        # normalization
        embeddings_normalized = F.normalize(class_embeddings, dim=-1)
        
        total_loss = 0.0
        num_pairs = 0
        
        # 父子关系loss
        if 'parent_child' in hierarchy_relations:
            for parent_idx, child_idx in hierarchy_relations['parent_child']:
                if parent_idx < len(embeddings_normalized) and child_idx < len(embeddings_normalized):
                    parent_emb = embeddings_normalized[parent_idx]
                    child_emb = embeddings_normalized[child_idx]
                    
                    # 余弦距离
                    dist = 1 - F.cosine_similarity(
                        parent_emb.unsqueeze(0),
                        child_emb.unsqueeze(0)
                    )
                    
                    # 鼓励距离接近margin
                    loss = (dist - self.parent_margin).abs()
                    total_loss = total_loss + loss
                    num_pairs += 1
        
        # 兄弟关系loss
        if 'siblings' in hierarchy_relations:
            for idx1, idx2 in hierarchy_relations['siblings']:
                if idx1 < len(embeddings_normalized) and idx2 < len(embeddings_normalized):
                    emb1 = embeddings_normalized[idx1]
                    emb2 = embeddings_normalized[idx2]
                    
                    dist = 1 - F.cosine_similarity(
                        emb1.unsqueeze(0),
                        emb2.unsqueeze(0)
                    )
                    
                    # 兄弟class应该有适中的距离
                    loss = (dist - self.sibling_margin).abs()
                    total_loss = total_loss + loss
                    num_pairs += 1
        
        if num_pairs > 0:
            return total_loss / num_pairs
        else:
            return torch.tensor(0.0, device=class_embeddings.device)


class ClassAwareLossModule(nn.Module):
    """
    classawareloss模块
    
    整合所有DC-HSP相关的loss函数
    """
    
    def __init__(
        self,
        temperature: float = 0.1,
        contrastive_weight: float = 0.05,
        hierarchy_weight: float = 0.02
    ):
        super().__init__()
        self.contrastive_loss = ClassContrastiveLoss(temperature=temperature)
        self.hierarchy_loss = HierarchyStructureLoss()
        
        self.contrastive_weight = contrastive_weight
        self.hierarchy_weight = hierarchy_weight
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        text_embeddings: torch.Tensor,
        hierarchy_relations: Optional[Dict] = None,
        ignore_index: int = 255
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            features: 图像特征 [B, C, H, W]
            labels: 标签 [B, H, W]
            text_embeddings: 文本嵌入 [num_classes, C]
            hierarchy_relations: classhierarchy关系
            ignore_index: 忽略索引
            
        Returns:
            loss字典
        """
        losses = {}
        
        # contrastiveloss
        loss_contrast = self.contrastive_loss(
            features, labels, text_embeddings, ignore_index
        )
        losses['loss_contrast'] = self.contrastive_weight * loss_contrast
        
        # hierarchy结构loss
        loss_hierarchy = self.hierarchy_loss(
            text_embeddings, hierarchy_relations
        )
        losses['loss_hierarchy'] = self.hierarchy_weight * loss_hierarchy
        
        return losses


# ============================================
# 辅助函数：build常见数据集的classhierarchy关系
# ============================================

def build_ade20k_hierarchy() -> Dict[str, List[Tuple[int, int]]]:
    """
    buildADE20K数据集的classhierarchy关系
    这是一个简化版本，实际使用时可以扩展
    """
    # 示例hierarchy关系（需要根据实际class索引调整）
    hierarchy = {
        'parent_child': [
            # (建筑物, 窗户), (建筑物, 门) 等
            # 这里需要根据ADE20K的实际classID填充
        ],
        'siblings': [
            # (椅子, 桌子), (床, 沙发) 等
        ]
    }
    return hierarchy


def build_coco_stuff_hierarchy() -> Dict[str, List[Tuple[int, int]]]:
    """
    buildCOCO-Stuff数据集的classhierarchy关系
    """
    hierarchy = {
        'parent_child': [],
        'siblings': [
            # 动物class
            # 交通工具class
            # 家具class
        ]
    }
    return hierarchy
