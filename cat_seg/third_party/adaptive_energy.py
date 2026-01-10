"""
Hierarchical Adaptive Hyperspherical Energy Modulation (HA-HEM)

Core Innovation: Layer-wise adaptive constraint learning
- Learns adaptive orthogonal constraint strength for each layer
- Shallow layers: stronger constraints for generalization
- Deep layers: relaxed constraints for task-specific learning
- Modality-aware modulation factors

Copyright (c) 2024-2026 AHS-Seg Authors
Licensed under the Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class AdaptiveEnergyModulator(nn.Module):
    """
    Hierarchical Adaptive Hyperspherical Energy Modulator
    
    Core idea: Different layers require different orthogonal constraint strengths
    - Shallow layers: contain general features, need stronger constraints to preserve CLIP generalization
    - Deep layers: need to learn task-specific knowledge, allow more deviation
    
    Formula: R_adapt = (1 - alpha_l * beta_m) * I + alpha_l * beta_m * R
    where:
    - alpha_l: learnable adaptive factor for layer l
    - beta_m: modality-aware modulation factor (vision/text)
    - R: original orthogonal transformation matrix
    - I: identity matrix
    """
    
    def __init__(
        self, 
        num_layers: int = 12, 
        modality: str = 'vision',
        init_alpha: float = 0.3,
        learnable_beta: bool = True
    ):
        """
        Args:
            num_layers: Number of Transformer layers
            modality: Modality type ('vision' or 'text')
            init_alpha: Initial value for alpha
            learnable_beta: Whether beta is learnable
        """
        super().__init__()
        self.num_layers = num_layers
        self.modality = modality
        
        # Layer-wise adaptive factor alpha_l
        # Use sigmoid activation, initialize such that sigmoid(init_val) ≈ init_alpha
        init_val = torch.log(torch.tensor(init_alpha / (1 - init_alpha + 1e-8)))
        self.alpha_logits = nn.Parameter(torch.ones(num_layers) * init_val)
        
        # Modality-aware factor beta_m
        # Vision modality allows more deviation, text modality keeps stronger constraints
        if learnable_beta:
            beta_init = 1.0 if modality == 'vision' else 0.7
            beta_logit = torch.log(torch.tensor(beta_init / (1 - beta_init + 1e-8)))
            self.beta_logit = nn.Parameter(torch.tensor(beta_logit))
        else:
            beta_init = 1.0 if modality == 'vision' else 0.7
            self.register_buffer('beta_logit', torch.tensor(beta_init))
        
        # Layer weights for energy loss (higher weights for shallow layers)
        # Linear decay from 1.0 (shallow) to 0.3 (deep)
        layer_weights = torch.linspace(1.0, 0.3, num_layers)
        self.register_buffer('layer_weights', layer_weights)
        
        # Cache R matrices for energy loss computation
        self.R_cache = {}
        
    def get_alpha(self, layer_idx: int) -> torch.Tensor:
        """Get adaptive factor for layer l"""
        return torch.sigmoid(self.alpha_logits[layer_idx])
    
    def get_beta(self) -> torch.Tensor:
        """Get modality-aware factor"""
        return torch.sigmoid(self.beta_logit)
    
    def get_modulation(self, layer_idx: int) -> torch.Tensor:
        """Get modulation strength = alpha_l * beta_m"""
        alpha = self.get_alpha(layer_idx)
        beta = self.get_beta()
        return alpha * beta
    
    def modulate(
        self, 
        R: torch.Tensor, 
        layer_idx: int,
        cache_for_loss: bool = True
    ) -> torch.Tensor:
        """
        Apply adaptive modulation to orthogonal transformation matrix
        
        Args:
            R: Orthogonal transformation matrix 
               - 2D: [block_size, block_size]
               - 3D: [num_blocks, block_size, block_size]
            layer_idx: Current layer index (0-indexed)
            cache_for_loss: Whether to cache R for energy loss computation
            
        Returns:
            Modulated transformation matrix R_adapt
        """
        modulation = self.get_modulation(layer_idx)
        
        # Generate identity matrix
        if R.dim() == 2:
            I = torch.eye(R.size(0), device=R.device, dtype=R.dtype)
        elif R.dim() == 3:
            I = torch.eye(R.size(1), device=R.device, dtype=R.dtype)
            I = I.unsqueeze(0).expand(R.size(0), -1, -1)
        else:
            raise ValueError(f"Unsupported R dimension: {R.dim()}")
        
        # Adaptive modulation: R_adapt = (1 - modulation) * I + modulation * R
        # When modulation is close to 0, R_adapt is close to I, maintaining strong orthogonality
        # When modulation is close to 1, R_adapt is close to R, allowing more deviation
        R_adapt = (1 - modulation) * I + modulation * R
        
        # Cache for energy loss computation
        if cache_for_loss:
            self.R_cache[layer_idx] = R_adapt
        
        return R_adapt
    
    def compute_energy_loss(self) -> torch.Tensor:
        """
        Compute adaptive energy loss
        
        Contains two parts:
        1. Orthogonality loss: maintain approximate orthogonality of transformation matrices
        2. Diversity regularization: encourage different constraint strengths across layers
        """
        if len(self.R_cache) == 0:
            return torch.tensor(0.0, device=self.alpha_logits.device)
        
        total_ortho_loss = 0.0
        device = self.alpha_logits.device
        
        for layer_idx, R in self.R_cache.items():
            # Compute Frobenius norm of R^T * R - I
            if R.dim() == 2:
                RtR = torch.mm(R.t(), R)
                I = torch.eye(R.size(0), device=device, dtype=R.dtype)
            else:  # 3D: batch of blocks
                RtR = torch.bmm(R.transpose(-2, -1), R)
                I = torch.eye(R.size(1), device=device, dtype=R.dtype)
                I = I.unsqueeze(0).expand(R.size(0), -1, -1)
            
            # Orthogonality deviation
            ortho_diff = RtR - I
            ortho_loss = torch.norm(ortho_diff, p='fro') ** 2
            
            # Weight by layer importance
            if layer_idx < len(self.layer_weights):
                weight = self.layer_weights[layer_idx]
            else:
                weight = self.layer_weights[-1]
            
            total_ortho_loss = total_ortho_loss + weight * ortho_loss
        
        # Diversity regularization: encourage different alpha values across layers
        alphas = torch.sigmoid(self.alpha_logits)
        alpha_variance = torch.var(alphas)
        diversity_bonus = -0.1 * alpha_variance  # Negative to make it a bonus
        
        # Smoothness regularization: encourage smooth alpha changes between adjacent layers
        if self.num_layers > 1:
            alpha_diff = alphas[1:] - alphas[:-1]
            smoothness_loss = 0.05 * torch.mean(alpha_diff ** 2)
        else:
            smoothness_loss = 0.0
        
        total_loss = total_ortho_loss + diversity_bonus + smoothness_loss
        
        return total_loss
    
    def clear_cache(self):
        """Clear R cache"""
        self.R_cache = {}
    
    def get_alpha_stats(self) -> dict:
        """Get alpha statistics for monitoring and visualization"""
        alphas = torch.sigmoid(self.alpha_logits).detach().cpu()
        return {
            'alpha_mean': alphas.mean().item(),
            'alpha_std': alphas.std().item(),
            'alpha_min': alphas.min().item(),
            'alpha_max': alphas.max().item(),
            'alpha_values': alphas.numpy().tolist(),
            'beta': torch.sigmoid(self.beta_logit).item()
        }


class AdaptiveOFTLayer(nn.Module):
    """
    OFT Layer with Adaptive Energy Modulation
    
    Integrates HA-HEM into OFT transformation
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int = 8,
        num_layers: int = 12,
        modality: str = 'vision',
        use_cayley: bool = True
    ):
        """
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            num_blocks: Number of blocks in block-diagonal matrix
            num_layers: Number of Transformer layers
            modality: Modality type
            use_cayley: Whether to use Cayley parameterization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.use_cayley = use_cayley
        
        assert in_features % num_blocks == 0, \
            f"in_features ({in_features}) must be divisible by num_blocks ({num_blocks})"
        
        self.block_size = in_features // num_blocks
        
        # Adaptive energy modulator
        self.energy_modulator = AdaptiveEnergyModulator(
            num_layers=num_layers,
            modality=modality
        )
        
    def cayley_transform(self, A: torch.Tensor) -> torch.Tensor:
        """
        Cayley transform: convert skew-symmetric matrix to orthogonal matrix
        Q = (I - A)(I + A)^{-1}
        """
        if A.dim() == 2:
            size = A.size(0)
            I = torch.eye(size, device=A.device, dtype=A.dtype)
            # Ensure skew-symmetric
            A_skew = 0.5 * (A - A.t())
            Q = torch.mm(I - A_skew, torch.inverse(I + A_skew))
        else:  # batch
            batch_size, size, _ = A.shape
            I = torch.eye(size, device=A.device, dtype=A.dtype)
            I = I.unsqueeze(0).expand(batch_size, -1, -1)
            # Ensure skew-symmetric
            A_skew = 0.5 * (A - A.transpose(-2, -1))
            Q = torch.bmm(I - A_skew, torch.inverse(I + A_skew))
        return Q
    
    def build_block_diagonal(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Build block-diagonal matrix from block list
        
        Args:
            blocks: [num_blocks, block_size, block_size]
        Returns:
            [in_features, in_features] block-diagonal matrix
        """
        block_list = [blocks[i] for i in range(blocks.size(0))]
        return torch.block_diag(*block_list)
    
    def forward(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        x: torch.Tensor,
        R_blocks: torch.Tensor,
        layer_idx: int = 0
    ) -> torch.Tensor:
        """
        Args:
            weight: Original weight matrix
            bias: Bias
            x: Input features
            R_blocks: Blocks of block-diagonal matrix [num_blocks, block_size, block_size]
            layer_idx: Current layer index
            
        Returns:
            Transformed output
        """
        orig_dtype = x.dtype
        
        # Cayley transform to get orthogonal matrix
        if self.use_cayley:
            R_ortho = self.cayley_transform(R_blocks)
        else:
            R_ortho = R_blocks
        
        # Adaptive energy modulation
        R_adapted = self.energy_modulator.modulate(R_ortho, layer_idx)
        
        # Build block-diagonal matrix
        R_full = self.build_block_diagonal(R_adapted)
        
        # Apply transformation: W' = W @ R
        transformed_weight = torch.mm(weight.to(orig_dtype), R_full.to(orig_dtype))
        
        # Linear transformation
        output = F.linear(x.to(orig_dtype), transformed_weight, bias)
        
        return output
    
    def get_energy_loss(self) -> torch.Tensor:
        """Get energy loss"""
        return self.energy_modulator.compute_energy_loss()
    
    def clear_cache(self):
        """Clear cache"""
        self.energy_modulator.clear_cache()


# ============================================
# Wrapper for replacing original OFT layers
# ============================================

def create_adaptive_oft_qkv(
    in_features: int,
    out_features: int,
    num_blocks: int,
    num_layers: int,
    modality: str
) -> AdaptiveOFTLayer:
    """
    Create adaptive OFT layer for Q, K, V projections
    """
    return AdaptiveOFTLayer(
        in_features=in_features,
        out_features=out_features,
        num_blocks=num_blocks,
        num_layers=num_layers,
        modality=modality,
        use_cayley=True
    )


class GlobalEnergyTracker:
    """
    Global energy loss tracker
    Used to collect energy loss from all adaptive OFT layers
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.modulators = []
        return cls._instance
    
    def register(self, modulator: AdaptiveEnergyModulator):
        """Register an energy modulator"""
        self.modulators.append(modulator)
    
    def compute_total_loss(self) -> torch.Tensor:
        """Compute total energy loss from all modulators"""
        if len(self.modulators) == 0:
            return torch.tensor(0.0)
        
        total = sum(m.compute_energy_loss() for m in self.modulators)
        return total
    
    def clear_all_caches(self):
        """Clear all modulator caches"""
        for m in self.modulators:
            m.clear_cache()
    
    def get_all_stats(self) -> list:
        """Get statistics from all modulators"""
        return [m.get_alpha_stats() for m in self.modulators]
    
    def reset(self):
        """Reset tracker"""
        self.modulators = []


# Global instance
energy_tracker = GlobalEnergyTracker()
