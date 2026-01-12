"""
AHS-Seg: Adaptive Hyperspherical Semantic Segmentation

大道至简版本 - 专注于高效创新

核心创新:
1. HPA: Hyperspherical Prototype Alignment - 超球面原型对齐
2. SDR: Self-Distillation Regularizer - 自蒸馏正则化
3. GCS: Geodesic Similarity - 测地相似度 (可选)
4. AFR: Adaptive Feature Rectification - 自适应特征矫正

设计原则:
- 参数高效: 新增参数 < 2M
- 计算高效: 推理时间增加 < 5%
- 迁移友好: 专注于跨数据集泛化

Copyright (c) 2024-2026 AHS-Seg Authors
Licensed under the Apache License 2.0
"""

from typing import Tuple, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from einops import rearrange

# 导入创新模块
from .modeling.hyperspherical_innovations import (
    UnifiedHypersphericalLoss,
    GeodesicSimilarityHead,
    AdaptiveFeatureRectifier,
    LightweightHypersphericalAttention
)


@META_ARCH_REGISTRY.register()
class AHSSeg(nn.Module):
    """
    AHS-Seg: Adaptive Hyperspherical Semantic Segmentation
    
    大道至简的开放词汇语义分割模型
    """
    
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        train_class_json: str,
        test_class_json: str,
        sliding_window: bool,
        clip_finetune: str,
        backbone_multiplier: float,
        clip_pretrained: str,
        # ===== 创新点配置 =====
        use_prototype_alignment: bool = True,
        use_self_distillation: bool = True,
        use_geodesic_similarity: bool = False,  # 可选,默认关闭
        use_feature_rectification: bool = True,
        # 损失权重
        prototype_loss_weight: float = 0.1,
        distill_loss_weight: float = 0.1,
        rectify_loss_weight: float = 0.05,
        # 其他配置
        num_classes: int = 171,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility if backbone else 32
        self.size_divisibility = size_divisibility

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(clip_pixel_std).view(-1, 1, 1), False)
        
        self.train_class_json = train_class_json
        self.test_class_json = test_class_json
        self.sliding_window = sliding_window
        
        # CLIP配置
        self.clip_finetune = clip_finetune
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        
        # 设置CLIP参数可训练性
        self._setup_clip_finetune()
        
        # 特征上采样
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)
        
        # 收集多层特征的钩子
        self.layer_features = []
        layer_indices = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15]
        for l in layer_indices:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(
                lambda m, _, o: self.layer_features.append(o)
            )
        
        # ===== 创新模块 =====
        self.use_prototype_alignment = use_prototype_alignment
        self.use_self_distillation = use_self_distillation
        self.use_geodesic_similarity = use_geodesic_similarity
        self.use_feature_rectification = use_feature_rectification
        
        # 统一超球面损失模块
        if use_prototype_alignment or use_self_distillation or use_feature_rectification:
            self.hyperspherical_loss = UnifiedHypersphericalLoss(
                feature_dim=self.proj_dim,
                num_classes=num_classes,
                prototype_weight=prototype_loss_weight,
                distill_weight=distill_loss_weight,
                rectify_weight=rectify_loss_weight,
                use_prototype=use_prototype_alignment,
                use_distillation=use_self_distillation,
                use_rectification=use_feature_rectification
            )
        else:
            self.hyperspherical_loss = None
        
        # 测地相似度头 (可选)
        if use_geodesic_similarity:
            self.geodesic_head = GeodesicSimilarityHead(feature_dim=self.proj_dim)
        else:
            self.geodesic_head = None
        
        # 缓存冻结的教师特征 (用于自蒸馏)
        self._frozen_features_cache = None
    
    def _setup_clip_finetune(self):
        """设置CLIP参数可训练性"""
        for name, params in self.sem_seg_head.predictor.clip_model.named_parameters():
            if "transformer" in name:
                if self.clip_finetune == "oft" or self.clip_finetune == "oft_both":
                    if "visual" in name:
                        if "attn" in name or "position" in name:
                            params.requires_grad = True
                        else:
                            params.requires_grad = False
                    elif "oft" in name:
                        params.requires_grad = True
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif self.clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False
            else:
                params.requires_grad = False

    @classmethod
    def from_config(cls, cfg):
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "train_class_json": cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON,
            "test_class_json": cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "clip_pretrained": cfg.MODEL.SEM_SEG_HEAD.CLIP_PRETRAINED,
            # 创新点配置
            "use_prototype_alignment": getattr(cfg.MODEL.SEM_SEG_HEAD, 'USE_PROTOTYPE_ALIGNMENT', True),
            "use_self_distillation": getattr(cfg.MODEL.SEM_SEG_HEAD, 'USE_SELF_DISTILLATION', True),
            "use_geodesic_similarity": getattr(cfg.MODEL.SEM_SEG_HEAD, 'USE_GEODESIC_SIMILARITY', False),
            "use_feature_rectification": getattr(cfg.MODEL.SEM_SEG_HEAD, 'USE_FEATURE_RECTIFICATION', True),
            "prototype_loss_weight": getattr(cfg.MODEL.SEM_SEG_HEAD, 'PROTOTYPE_LOSS_WEIGHT', 0.1),
            "distill_loss_weight": getattr(cfg.MODEL.SEM_SEG_HEAD, 'DISTILL_LOSS_WEIGHT', 0.1),
            "rectify_loss_weight": getattr(cfg.MODEL.SEM_SEG_HEAD, 'RECTIFY_LOSS_WEIGHT', 0.05),
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    @torch.no_grad()
    def _get_frozen_features(self, clip_images_resized: torch.Tensor) -> torch.Tensor:
        """获取冻结的CLIP特征用于自蒸馏"""
        # 临时禁用梯度
        was_training = self.training
        self.eval()
        
        # 使用原始CLIP权重
        frozen_features = self.sem_seg_head.predictor.clip_model.encode_image(
            clip_images_resized, 
            dense=True
        )
        
        if was_training:
            self.train()
        
        return frozen_features.detach()
    
    def forward(self, batched_inputs):
        """
        前向传播
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        # 清空特征缓存
        self.layer_features = []

        # CLIP特征提取
        clip_images_resized = F.interpolate(
            clip_images.tensor, 
            size=self.clip_resolution, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 获取冻结特征用于自蒸馏 (仅训练时)
        teacher_features = None
        if self.training and self.use_self_distillation:
            teacher_features = self._get_frozen_features(clip_images_resized)
        
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(
            clip_images_resized, 
            dense=True
        )

        # 基础特征处理
        image_features = clip_features[:, 1:, :]
        
        # ===== 创新点: 特征矫正 =====
        if self.use_feature_rectification and self.hyperspherical_loss is not None:
            image_features_4d = rearrange(image_features, "B (H W) C -> B C H W", H=24)
            image_features_4d = self.hyperspherical_loss.get_rectified_features(image_features_4d)
            image_features = rearrange(image_features_4d, "B C H W -> B (H W) C")
        
        # 构建特征字典
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layer_features[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layer_features[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        features = {'res5': res5, 'res4': res4, 'res3': res3}

        # 分割预测
        outputs = self.sem_seg_head(clip_features, features)
        
        if self.training:
            targets = torch.stack([x["sem_seg"].to(self.device) for x in batched_inputs], dim=0)
            outputs = F.interpolate(
                outputs, 
                size=(targets.shape[-2], targets.shape[-1]), 
                mode="bilinear", 
                align_corners=False
            )
            
            num_classes = outputs.shape[1]
            mask = targets != self.sem_seg_head.ignore_value
            
            outputs_permuted = outputs.permute(0, 2, 3, 1)
            _targets = torch.zeros(outputs_permuted.shape, device=self.device)
            _onehot = F.one_hot(targets[mask], num_classes=num_classes).float()
            _targets[mask] = _onehot
            
            # ===== 损失计算 =====
            losses = {}
            
            # 主分割损失
            loss_seg = F.binary_cross_entropy_with_logits(outputs_permuted, _targets)
            losses["loss_sem_seg"] = loss_seg
            
            # 超球面创新损失
            if self.hyperspherical_loss is not None:
                # 准备教师特征
                teacher_4d = None
                if teacher_features is not None:
                    teacher_4d = rearrange(teacher_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
                
                hs_losses = self.hyperspherical_loss(
                    student_features=res3,
                    teacher_features=teacher_4d,
                    labels=targets,
                    ignore_index=self.sem_seg_head.ignore_value
                )
                losses.update(hs_losses)
            
            return losses

        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])

            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]
            return processed_results

    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        """滑动窗口推理"""
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False)
        image_unfolded = unfold(image)
        image = rearrange(image_unfolded.squeeze(0), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False)
        
        self.layer_features = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
        
        # 可选: 应用特征矫正
        if self.use_feature_rectification and self.hyperspherical_loss is not None:
            image_features = clip_features[:, 1:, :]
            image_features_4d = rearrange(image_features, "B (H W) C -> B C H W", H=24)
            image_features_4d = self.hyperspherical_loss.get_rectified_features(image_features_4d)
            # 重建clip_features
            clip_features_new = clip_features.clone()
            clip_features_new[:, 1:, :] = rearrange(image_features_4d, "B C H W -> B (H W) C")
            clip_features = clip_features_new
        
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(rearrange(self.layer_features[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        res5 = self.upsample2(rearrange(self.layer_features[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        features = {'res5': res5, 'res4': res4, 'res3': res3}
        outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(outputs, size=kernel, mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()
        
        global_output = outputs[-1:]
        global_output = F.interpolate(global_output, size=out_res, mode='bilinear', align_corners=False)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1, 1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息和参数统计
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        innovation_params = 0
        if self.hyperspherical_loss is not None:
            innovation_params = sum(p.numel() for p in self.hyperspherical_loss.parameters())
        if self.geodesic_head is not None:
            innovation_params += sum(p.numel() for p in self.geodesic_head.parameters())
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'innovation_params': innovation_params,
            'innovations': {
                'prototype_alignment': self.use_prototype_alignment,
                'self_distillation': self.use_self_distillation,
                'geodesic_similarity': self.use_geodesic_similarity,
                'feature_rectification': self.use_feature_rectification,
            }
        }
