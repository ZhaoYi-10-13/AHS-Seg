"""
Enhanced OFT Module with HA-HEM Integration

Integrates Hierarchical Adaptive Hyperspherical Energy Modulation
into the Orthogonal Fine-Tuning structure for improved performance.

Copyright (c) 2024-2026 AHS-Seg Authors
Licensed under the Apache License 2.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple

from .adaptive_energy import AdaptiveEnergyModulator, energy_tracker


class EnhancedOFTLinearLayer(nn.Module):
    """
    Enhanced OFT Linear Layer
    
    Integrates HA-HEM adaptive energy modulation into OFT
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_blocks: int = 8,
        num_layers: int = 12,
        modality: str = 'vision',
        use_cayley: bool = True,
        use_adaptive: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        self.use_cayley = use_cayley
        self.use_adaptive = use_adaptive
        
        assert in_features % num_blocks == 0
        self.block_size = in_features // num_blocks
        
        # Adaptive energy modulator
        if use_adaptive:
            self.energy_modulator = AdaptiveEnergyModulator(
                num_layers=num_layers,
                modality=modality
            )
            # Register with global tracker
            energy_tracker.register(self.energy_modulator)
    
    def cayley_transform(self, A: torch.Tensor) -> torch.Tensor:
        """Cayley transform"""
        if A.dim() == 2:
            size = A.size(0)
            I = torch.eye(size, device=A.device, dtype=A.dtype)
            A_skew = 0.5 * (A - A.t())
            Q = torch.mm(I - A_skew, torch.inverse(I + A_skew + 1e-6 * I))
        else:
            batch_size, size, _ = A.shape
            I = torch.eye(size, device=A.device, dtype=A.dtype)
            I = I.unsqueeze(0).expand(batch_size, -1, -1)
            A_skew = 0.5 * (A - A.transpose(-2, -1))
            Q = torch.bmm(I - A_skew, torch.inverse(I + A_skew + 1e-6 * I))
        return Q
    
    def build_block_diagonal(self, blocks: torch.Tensor) -> torch.Tensor:
        """Build block-diagonal matrix"""
        if blocks.dim() == 2:
            block_list = [blocks] * self.num_blocks
        else:
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
            weight: Original weight [out_features, in_features]
            bias: Bias
            x: Input [*, in_features]
            R_blocks: Transformation matrix blocks
            layer_idx: Layer index
        """
        orig_dtype = x.dtype
        
        # Cayley transform
        if self.use_cayley:
            R_ortho = self.cayley_transform(R_blocks)
        else:
            R_ortho = R_blocks
        
        # Adaptive modulation
        if self.use_adaptive:
            R_adapted = self.energy_modulator.modulate(R_ortho, layer_idx)
        else:
            R_adapted = R_ortho
        
        # Build complete transformation matrix
        R_full = self.build_block_diagonal(R_adapted)
        
        # Apply transformation
        transformed_weight = torch.mm(weight.to(orig_dtype), R_full.to(orig_dtype))
        
        output = F.linear(x.to(orig_dtype), transformed_weight, bias)
        
        return output


class EnhancedOFTQKVLayer(nn.Module):
    """
    Enhanced QKV Projection OFT Layer
    
    Handles joint transformation of Q, K, V projections
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        num_layers: int = 12,
        modality: str = 'vision',
        use_adaptive: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.use_adaptive = use_adaptive
        
        assert hidden_size % num_blocks == 0
        self.block_size = hidden_size // num_blocks
        
        # Adaptive energy modulator (shared by Q, K, V)
        if use_adaptive:
            self.energy_modulator = AdaptiveEnergyModulator(
                num_layers=num_layers,
                modality=modality
            )
            energy_tracker.register(self.energy_modulator)
    
    def cayley_batch(self, A: torch.Tensor) -> torch.Tensor:
        """Batch Cayley transform"""
        b, r, c = A.shape
        I = torch.eye(r, device=A.device, dtype=A.dtype).unsqueeze(0).expand(b, -1, -1)
        A_skew = 0.5 * (A - A.transpose(-2, -1))
        Q = torch.bmm(I - A_skew, torch.inverse(I + A_skew + 1e-6 * I))
        return Q
    
    def block_diagonal(self, R: torch.Tensor) -> torch.Tensor:
        """Build block-diagonal matrix"""
        if R.dim() == 2:
            blocks = [R] * self.num_blocks
        else:
            blocks = [R[i] for i in range(R.size(0))]
        return torch.block_diag(*blocks)
    
    def forward(
        self,
        attn_weight: torch.Tensor,  # [3*hidden, hidden]
        bias: Optional[torch.Tensor],
        x: torch.Tensor,
        q_R: torch.Tensor,
        k_R: torch.Tensor,
        v_R: torch.Tensor,
        layer_idx: int = 0
    ) -> torch.Tensor:
        """
        Args:
            attn_weight: Joint QKV weights
            bias: Bias
            x: Input
            q_R, k_R, v_R: Transformation matrices for Q, K, V
            layer_idx: Layer index
        """
        orig_dtype = x.dtype
        
        # Cayley transform
        orth_q = self.cayley_batch(q_R) if q_R.dim() == 3 else self.cayley_batch(q_R.unsqueeze(0)).squeeze(0)
        orth_k = self.cayley_batch(k_R) if k_R.dim() == 3 else self.cayley_batch(k_R.unsqueeze(0)).squeeze(0)
        orth_v = self.cayley_batch(v_R) if v_R.dim() == 3 else self.cayley_batch(v_R.unsqueeze(0)).squeeze(0)
        
        # Adaptive modulation
        if self.use_adaptive:
            orth_q = self.energy_modulator.modulate(orth_q, layer_idx)
            orth_k = self.energy_modulator.modulate(orth_k, layer_idx, cache_for_loss=False)
            orth_v = self.energy_modulator.modulate(orth_v, layer_idx, cache_for_loss=False)
        
        # Build block-diagonal matrices
        R_q = self.block_diagonal(orth_q)
        R_k = self.block_diagonal(orth_k)
        R_v = self.block_diagonal(orth_v)
        
        # Separate QKV weights
        q_weight, k_weight, v_weight = attn_weight.chunk(3, dim=0)
        
        # Apply transformation
        filt_q = torch.mm(q_weight.to(orig_dtype), R_q.to(orig_dtype))
        filt_k = torch.mm(k_weight.to(orig_dtype), R_k.to(orig_dtype))
        filt_v = torch.mm(v_weight.to(orig_dtype), R_v.to(orig_dtype))
        
        # Merge
        filt = torch.cat([filt_q, filt_k, filt_v], dim=0)
        
        output = F.linear(x.to(orig_dtype), filt, bias)
        
        return output


class EnhancedOFTOutputLayer(nn.Module):
    """
    Enhanced Output Projection OFT Layer
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_blocks: int = 8,
        num_layers: int = 12,
        modality: str = 'vision',
        use_adaptive: bool = True
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_blocks = num_blocks
        self.use_adaptive = use_adaptive
        
        assert hidden_size % num_blocks == 0
        self.block_size = hidden_size // num_blocks
        
        if use_adaptive:
            self.energy_modulator = AdaptiveEnergyModulator(
                num_layers=num_layers,
                modality=modality
            )
            energy_tracker.register(self.energy_modulator)
    
    def cayley_batch(self, A: torch.Tensor) -> torch.Tensor:
        if A.dim() == 2:
            A = A.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
            
        b, r, c = A.shape
        I = torch.eye(r, device=A.device, dtype=A.dtype).unsqueeze(0).expand(b, -1, -1)
        A_skew = 0.5 * (A - A.transpose(-2, -1))
        Q = torch.bmm(I - A_skew, torch.inverse(I + A_skew + 1e-6 * I))
        
        if squeeze:
            Q = Q.squeeze(0)
        return Q
    
    def block_diagonal(self, R: torch.Tensor) -> torch.Tensor:
        if R.dim() == 2:
            blocks = [R] * self.num_blocks
        else:
            blocks = [R[i] for i in range(R.size(0))]
        return torch.block_diag(*blocks)
    
    def forward(
        self,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        x: torch.Tensor,
        proj_R: torch.Tensor,
        layer_idx: int = 0
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        
        # Cayley transform
        orth_proj = self.cayley_batch(proj_R)
        
        # Adaptive modulation
        if self.use_adaptive:
            orth_proj = self.energy_modulator.modulate(orth_proj, layer_idx)
        
        # Build block-diagonal matrix
        R_full = self.block_diagonal(orth_proj)
        
        # Apply transformation
        filt = torch.mm(R_full.to(orig_dtype), weight.to(orig_dtype))
        
        output = F.linear(x.to(orig_dtype), filt, bias)
        
        return output


# ============================================
# Replacement functions for original OFT layers
# ============================================

def enhanced_oft_forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Enhanced OFT forward pass
    
    Replaces the original oft_forward function
    """
    B, N, C = x.shape
    res_x = x
    orig_dtype = x.dtype
    
    # Get relation transformation
    _, N1, _, _ = self.attn_q_proj_oft_layer_R.shape
    attn_tensor = rearrange(
        self.attn_q_proj_oft_layer_R.cuda().to(orig_dtype),
        'B1 N1 L1 M1 -> B1 (N1 L1) M1'
    )
    attn_tensor = self.attn_q_proj_oft_relation_m_R[int((self.count-4)/4)](attn_tensor)[:, :, :4]
    attn_tensor = rearrange(attn_tensor, 'B1 (N1 L1) M1 -> M1 B1 N1 L1', N1=N1)
    attn_re = self.attn_q_proj_oft_relation_l_R(attn_tensor)[..., self.count-4:self.count]
    q_R, k_R, v_R, proj_R = attn_re[..., 0], attn_re[..., 1], attn_re[..., 2], attn_re[..., 3]
    
    # Use enhanced OFT layer
    layer_idx = (self.count - 4) // 4  # Compute layer index
    
    qkv = self.to_qkv_oft(
        self.attn.in_proj_weight,
        self.attn.in_proj_bias,
        self.ln_1(x),
        q_R, k_R, v_R,
        layer_idx=layer_idx
    )
    
    qkv = qkv.reshape(B, N, 3, self.attn.num_heads, C // self.attn.num_heads).permute(2, 1, 3, 0, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    
    attn = (q @ k.transpose(-2, -1)) * (float(self.attn.head_dim) ** -0.5)
    attn = attn + self.attn_mask.unsqueeze(0).unsqueeze(1).cuda().to(orig_dtype)
    attn = attn.softmax(dim=-1)
    
    oft_out = self.to_out_oft(
        self.attn.out_proj.weight,
        self.attn.out_proj.bias,
        ((attn @ v).transpose(1, 2)).permute(1, 0, 2, 3).reshape(B, N, C),
        proj_R,
        layer_idx=layer_idx
    )
    
    oft_out = self.dp(oft_out)
    final = res_x + oft_out
    final = final + self.mlp(self.ln_2(final))
    
    return final


def enhanced_oft_forward_vision(self, x: torch.Tensor) -> torch.Tensor:
    """
    Enhanced vision OFT forward pass
    """
    orig_dtype = x.dtype
    
    _, N1, _, _ = self.attn_q_proj_oft_layer_R.shape
    attn_tensor = rearrange(
        self.attn_q_proj_oft_layer_R.cuda().to(orig_dtype),
        'B1 N1 L1 M1 -> B1 (N1 L1) M1'
    )
    attn_tensor = self.attn_q_proj_oft_relation_m_R[int((self.count-4)/4)](attn_tensor)[:, :, -6:]
    attn_tensor = rearrange(attn_tensor, 'B1 (N1 L1) M1 -> M1 B1 N1 L1', N1=N1)
    attn_re = self.attn_q_proj_oft_relation_l_R(attn_tensor)[..., self.count-4:self.count]
    q_R, k_R, v_R, proj_R = attn_re[..., 0], attn_re[..., 1], attn_re[..., 2], attn_re[..., 3]
    
    layer_idx = (self.count - 4) // 4
    
    if self.count <= 44:
        B, N, C = x.shape
        res_x = x
        
        qkv = self.to_qkv_oft_vision(
            self.attn.in_proj_weight,
            self.attn.in_proj_bias,
            self.ln_1(x),
            q_R, k_R, v_R,
            layer_idx=layer_idx
        )
        
        qkv = qkv.reshape(B, N, 3, self.n_head, C // self.n_head).permute(2, 1, 3, 0, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * (float(self.attn.head_dim) ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = ((attn @ v).transpose(1, 2)).permute(1, 0, 2, 3).reshape(B, N, C)
        
        oft_out = self.to_out_oft_vision(
            self.attn.out_proj.weight,
            self.attn.out_proj.bias,
            attn,
            proj_R,
            layer_idx=layer_idx
        )
        
        oft_out = self.dp(oft_out)
        final = res_x + oft_out
        final = final + self.mlp(self.ln_2(final))
        return final
    else:
        y = self.to_qkv_oft_vision(
            self.attn.in_proj_weight,
            self.attn.in_proj_bias,
            self.ln_1(x),
            q_R, k_R, v_R,
            layer_idx=layer_idx
        )
        
        L, N, D = y.shape
        y = y.reshape(L, N, 3, D // 3).permute(2, 1, 0, 3).reshape(3 * N, L, D // 3)
        
        y = self.to_out_oft_vision(
            self.attn.out_proj.weight,
            self.attn.out_proj.bias,
            y,
            proj_R,
            layer_idx=layer_idx
        )
        
        q, k, v = y.tensor_split(3, dim=0)
        v = v.transpose(1, 0) + x[:1]
        v = v + self.mlp(self.ln_2(v))
        return v


def set_enhanced_oft(model, dim=8, hidden_size=512, length=12, s=0.1, r=4, count=0, use_adaptive=True):
    """
    Setup enhanced OFT layers for text encoder
    
    Args:
        model: Model
        dim: Block size
        hidden_size: Hidden dimension
        length: Number of layers
        s: Scaling factor
        r: Number of blocks
        count: Counter
        use_adaptive: Whether to use adaptive energy modulation
    """
    from . import model_oft_relation_both_free_half
    
    for _ in model.children():
        if isinstance(_, model_oft_relation_both_free_half.ResidualAttentionBlock):
            count += 4
            
            # Use enhanced OFT layers
            _.to_qkv_oft = EnhancedOFTQKVLayer(
                hidden_size=hidden_size,
                num_blocks=r,
                num_layers=length,
                modality='text',
                use_adaptive=use_adaptive
            )
            _.to_out_oft = EnhancedOFTOutputLayer(
                hidden_size=hidden_size,
                num_blocks=r,
                num_layers=length,
                modality='text',
                use_adaptive=use_adaptive
            )
            
            _.dp = nn.Dropout(_.attn.dropout)
            _.s = s
            _.dim = dim
            _.hidden_size = hidden_size
            _.count = count
            
            # Bind enhanced forward function
            bound_method = enhanced_oft_forward.__get__(_, _.__class__)
            setattr(_, 'forward', bound_method)
            
        elif len(list(_.children())) != 0:
            set_enhanced_oft(_, dim, hidden_size, length, s, r, count, use_adaptive)


def set_enhanced_oft_vision(model, dim=8, hidden_size=512, length=12, s=0.1, r=6, count=0, use_adaptive=True):
    """
    Setup enhanced OFT layers for vision encoder
    """
    from . import model_oft_relation_both_free_half
    
    for _ in model.children():
        if isinstance(_, model_oft_relation_both_free_half.ResidualAttentionBlock):
            count += 4
            print(f'Enhanced OFT Vision Layer: count={count}')
            
            _.dp = nn.Dropout(_.attn.dropout)
            _.s = s
            _.dim = dim
            
            # Use enhanced OFT layers
            if count <= 0:
                _.to_qkv_oft_vision = EnhancedOFTQKVLayer(
                    hidden_size=hidden_size,
                    num_blocks=r,
                    num_layers=length,
                    modality='text',  # First few layers use text modality settings
                    use_adaptive=use_adaptive
                )
                _.to_out_oft_vision = EnhancedOFTOutputLayer(
                    hidden_size=hidden_size,
                    num_blocks=r,
                    num_layers=length,
                    modality='text',
                    use_adaptive=use_adaptive
                )
            else:
                _.to_qkv_oft_vision = EnhancedOFTQKVLayer(
                    hidden_size=hidden_size,
                    num_blocks=r,
                    num_layers=length,
                    modality='vision',
                    use_adaptive=use_adaptive
                )
                _.to_out_oft_vision = EnhancedOFTOutputLayer(
                    hidden_size=hidden_size,
                    num_blocks=r,
                    num_layers=length,
                    modality='vision',
                    use_adaptive=use_adaptive
                )
            
            _.hidden_size = hidden_size
            _.count = count
            
            bound_method = enhanced_oft_forward_vision.__get__(_, _.__class__)
            if count <= 44:
                setattr(_, 'forward', bound_method)
            else:
                setattr(_, 'forward_dense', bound_method)
                
        elif len(list(_.children())) != 0:
            set_enhanced_oft_vision(_, dim, hidden_size, length, s, r, count, use_adaptive)
    
    print(f'Enhanced OFT Vision setup complete: count={count}')
