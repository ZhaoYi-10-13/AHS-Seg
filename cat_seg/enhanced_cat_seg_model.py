"""
Enhanced CATSeg Model with Adaptive Hyperspherical Learning

Integrates three core innovations:
1. HA-HEM: Hierarchical Adaptive Hyperspherical Energy Modulation
2. MH-FPN: Multi-scale Hyperspherical Feature Pyramid Network
3. DC-HSP: Dynamic Class-aware Hyperspherical Projection

Copyright (c) 2024-2026 AHS-Seg Authors
Licensed under the Apache License 2.0
"""

from typing import Tuple, Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList

from einops import rearrange

# Import innovation modules
from .modeling.multi_scale_fusion import (
    MultiScaleHypersphericalFPN,
    BoundaryAwareLoss,
    MultiScaleBoundaryLoss
)
from .modeling.class_aware_projection import (
    DynamicClassAwareProjection,
    ClassAwareLossModule
)
from .third_party.adaptive_energy import energy_tracker


@META_ARCH_REGISTRY.register()
class EnhancedCATSeg(nn.Module):
    """
    Enhanced CATSeg Model with Adaptive Hyperspherical Learning
    
    Integrates three innovations to improve open-vocabulary semantic segmentation.
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
        # Innovation configuration
        use_adaptive_energy: bool = True,
        use_multi_scale_fusion: bool = True,
        use_class_aware_projection: bool = True,
        energy_loss_weight: float = 0.01,
        boundary_loss_weight: float = 0.1,
        contrastive_loss_weight: float = 0.05,
        hierarchy_loss_weight: float = 0.02,
        ms_layer_indices: List[int] = [3, 7, 11],
    ):
        """
        Args:
            use_adaptive_energy: Enable HA-HEM module
            use_multi_scale_fusion: Enable MH-FPN module
            use_class_aware_projection: Enable DC-HSP module
            energy_loss_weight: Weight for energy regularization loss
            boundary_loss_weight: Weight for boundary-aware loss
            contrastive_loss_weight: Weight for contrastive learning loss
            hierarchy_loss_weight: Weight for hierarchy structure loss
            ms_layer_indices: Layer indices for multi-scale fusion
        """
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
        
        # Innovation module configuration
        self.use_adaptive_energy = use_adaptive_energy
        self.use_multi_scale_fusion = use_multi_scale_fusion
        self.use_class_aware_projection = use_class_aware_projection
        
        # Loss weights
        self.energy_loss_weight = energy_loss_weight
        self.boundary_loss_weight = boundary_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight
        self.hierarchy_loss_weight = hierarchy_loss_weight
        
        # CLIP configuration
        self.clip_finetune = clip_finetune
        self.clip_resolution = (384, 384) if clip_pretrained == "ViT-B/16" else (336, 336)
        self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        
        # Setup CLIP parameter trainability
        self._setup_clip_finetune()
        
        # Feature upsampling layers
        self.upsample1 = nn.ConvTranspose2d(self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(self.proj_dim, 128, kernel_size=4, stride=4)
        
        # Layer indices for multi-scale feature extraction
        self.ms_layer_indices = ms_layer_indices
        self.layer_features = []
        
        # Register hooks to collect multi-layer features
        layer_indices = [3, 7] if clip_pretrained == "ViT-B/16" else [7, 15]
        for l in layer_indices:
            self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(
                lambda m, _, o: self.layer_features.append(o)
            )
        
        # ===============================
        # Innovation 2: MH-FPN (Multi-scale Hyperspherical Feature Pyramid)
        # ===============================
        if use_multi_scale_fusion:
            self.ms_fpn = MultiScaleHypersphericalFPN(
                in_channels_list=[self.proj_dim, self.proj_dim, self.proj_dim],
                out_channels=self.proj_dim,
                feature_size=24,  # CLIP ViT-B/16 feature map size
                fusion_type='weighted'
            )
            
            # Multi-scale boundary loss
            self.boundary_loss = MultiScaleBoundaryLoss(
                scales=[1.0, 0.5],
                weights=[1.0, 0.5]
            )
            
            # Additional hooks for multi-scale feature collection
            self.ms_features = []
            for l in ms_layer_indices:
                if l < len(self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks):
                    self.sem_seg_head.predictor.clip_model.visual.transformer.resblocks[l].register_forward_hook(
                        lambda m, _, o, layer=l: self._store_ms_feature(o, layer)
                    )
        
        # ===============================
        # Innovation 3: DC-HSP (Dynamic Class-aware Hyperspherical Projection)
        # ===============================
        if use_class_aware_projection:
            text_dim = 512 if clip_pretrained == "ViT-B/16" else 768
            self.class_projection = DynamicClassAwareProjection(
                feature_dim=self.proj_dim,
                proj_dim=self.proj_dim,
                rank=64,
                temperature=0.07
            )
            
            self.class_loss_module = ClassAwareLossModule(
                temperature=0.1,
                contrastive_weight=contrastive_loss_weight,
                hierarchy_weight=hierarchy_loss_weight
            )
    
    def _store_ms_feature(self, output, layer_idx):
        """Store multi-scale features for later fusion."""
        self.ms_features.append((layer_idx, output))
    
    def _setup_clip_finetune(self):
        """Setup CLIP parameter trainability based on fine-tuning strategy."""
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
            # Innovation configuration options
            "use_adaptive_energy": getattr(cfg.MODEL.SEM_SEG_HEAD, 'USE_ADAPTIVE_ENERGY', True),
            "use_multi_scale_fusion": getattr(cfg.MODEL.SEM_SEG_HEAD, 'USE_MULTI_SCALE_FUSION', True),
            "use_class_aware_projection": getattr(cfg.MODEL.SEM_SEG_HEAD, 'USE_CLASS_AWARE_PROJECTION', True),
            "energy_loss_weight": getattr(cfg.MODEL.SEM_SEG_HEAD, 'ENERGY_LOSS_WEIGHT', 0.01),
            "boundary_loss_weight": getattr(cfg.MODEL.SEM_SEG_HEAD, 'BOUNDARY_LOSS_WEIGHT', 0.1),
            "contrastive_loss_weight": getattr(cfg.MODEL.SEM_SEG_HEAD, 'CONTRASTIVE_LOSS_WEIGHT', 0.05),
            "hierarchy_loss_weight": getattr(cfg.MODEL.SEM_SEG_HEAD, 'HIERARCHY_LOSS_WEIGHT', 0.02),
            "ms_layer_indices": getattr(cfg.MODEL.SEM_SEG_HEAD, 'MS_LAYER_INDICES', [3, 7, 11]),
        }

    @property
    def device(self):
        return self.pixel_mean.device
    
    def forward(self, batched_inputs):
        """
        Forward pass.
        
        Args:
            batched_inputs: Batch of input dictionaries
        Returns:
            Loss dictionary during training, predictions during inference
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) / self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(clip_images, self.size_divisibility)

        # Clear feature cache
        self.layer_features = []
        if self.use_multi_scale_fusion:
            self.ms_features = []

        # CLIP feature extraction
        clip_images_resized = F.interpolate(
            clip_images.tensor, 
            size=self.clip_resolution, 
            mode='bilinear', 
            align_corners=False
        )
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(
            clip_images_resized, 
            dense=True
        )

        # Base feature processing
        image_features = clip_features[:, 1:, :]
        
        # ===============================
        # Innovation 2: Multi-scale Feature Fusion
        # ===============================
        if self.use_multi_scale_fusion and len(self.ms_features) > 0:
            # Collect multi-scale features
            ms_feature_list = []
            for layer_idx, feat in sorted(self.ms_features, key=lambda x: x[0]):
                # Format conversion: [L, B, C] -> [B, N, C]
                if feat.dim() == 3:
                    feat = feat.permute(1, 0, 2)  # [L, B, C] -> [B, L, C]
                ms_feature_list.append(feat[:, 1:, :])  # Remove CLS token
            
            # Fuse if enough features collected
            if len(ms_feature_list) >= 2:
                # Ensure 3 layers of features
                while len(ms_feature_list) < 3:
                    ms_feature_list.append(image_features)
                
                fused_features = self.ms_fpn(ms_feature_list[:3])
                # Combine fused features with original
                image_features_4d = rearrange(image_features, "B (H W) C -> B C H W", H=24)
                fused_features = 0.5 * image_features_4d + 0.5 * fused_features
                image_features = rearrange(fused_features, "B C H W -> B (H W) C")
        
        # CLIP ViT features for guidance
        res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        res4 = rearrange(self.layer_features[0][1:, :, :], "(H W) B C -> B C H W", H=24)
        res5 = rearrange(self.layer_features[1][1:, :, :], "(H W) B C -> B C H W", H=24)
        res4 = self.upsample1(res4)
        res5 = self.upsample2(res5)
        features = {'res5': res5, 'res4': res4, 'res3': res3}

        # ===============================
        # Innovation 3: Dynamic Class-aware Projection
        # ===============================
        if self.use_class_aware_projection:
            # Get text embeddings
            text_features = self.sem_seg_head.predictor.get_text_embeds(
                self.sem_seg_head.predictor.class_texts if self.training 
                else self.sem_seg_head.predictor.test_class_texts,
                self.sem_seg_head.predictor.prompt_templates,
                self.sem_seg_head.predictor.clip_model
            )
            
            # Dynamic projection
            proj_img, proj_text, complexity = self.class_projection(
                rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24),
                text_features.squeeze(1).mean(1),  # Handle prompt dimension
                return_complexity=True
            )
        
        # Segmentation prediction
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
            
            # ===============================
            # Loss Computation
            # ===============================
            losses = {}
            
            # Main segmentation loss
            loss_seg = F.binary_cross_entropy_with_logits(outputs_permuted, _targets)
            losses["loss_sem_seg"] = loss_seg
            
            # Innovation 1: HA-HEM energy regularization loss
            if self.use_adaptive_energy:
                loss_energy = energy_tracker.compute_total_loss()
                if loss_energy.requires_grad:
                    losses["loss_energy"] = self.energy_loss_weight * loss_energy
                # Clear cache
                energy_tracker.clear_all_caches()
            
            # Innovation 2: Boundary-aware loss
            if self.use_multi_scale_fusion:
                loss_boundary = self.boundary_loss(outputs, targets)
                losses["loss_boundary"] = self.boundary_loss_weight * loss_boundary
            
            # Innovation 3: Class-aware contrastive loss
            if self.use_class_aware_projection:
                class_losses = self.class_loss_module(
                    proj_img,
                    targets,
                    proj_text,
                    hierarchy_relations=None,  # Can add dataset-specific relations
                    ignore_index=self.sem_seg_head.ignore_value
                )
                losses.update(class_losses)
            
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
        """Sliding window inference for high-resolution images."""
        images = [x["image"].to(self.device, dtype=torch.float32) for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False)
        
        self.layer_features = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(clip_images, dense=True)
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
        outputs = fold(outputs.flatten(1).T) / fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
    
    def get_innovation_stats(self) -> Dict:
        """
        Get statistics from innovation modules.
        Used for training monitoring and analysis.
        """
        stats = {}
        
        # HA-HEM statistics
        if self.use_adaptive_energy:
            energy_stats = energy_tracker.get_all_stats()
            if energy_stats:
                stats['ha_hem'] = {
                    'num_modulators': len(energy_stats),
                    'avg_alpha': sum(s['alpha_mean'] for s in energy_stats) / len(energy_stats),
                    'alpha_range': (
                        min(s['alpha_min'] for s in energy_stats),
                        max(s['alpha_max'] for s in energy_stats)
                    )
                }
        
        # MH-FPN statistics
        if self.use_multi_scale_fusion:
            fusion_weights = self.ms_fpn.get_fusion_weights()
            if fusion_weights is not None:
                stats['mh_fpn'] = {
                    'fusion_weights': fusion_weights.tolist()
                }
        
        return stats
