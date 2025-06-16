import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

class MHAttentionMap(nn.Module):
    """Multi-Head Attention Map for mask generation - Preserves all heads"""
    def __init__(self, query_dim: int, hidden_dim: int, num_heads: int = 8, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        
        # Linear projections
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)
        if bias:
            nn.init.zeros_(self.q_linear.bias)
            nn.init.zeros_(self.k_linear.bias)
            
        self.normalize_fact = (hidden_dim / num_heads) ** -0.5

    def forward(self, q: torch.Tensor, k: torch.Tensor, spatial_shapes: List[Tuple[int, int]], mask: Optional[torch.Tensor] = None):
        """
        Args:
            q: [B, num_queries, query_dim]
            k: [B, L, hidden_dim] (flattened multi-scale features)
        Returns:
            attention_maps: [B, num_heads, num_queries, H, W] (highest resolution)
        """
        B, Q, _ = q.shape
        B, L, D = k.shape
        
        # Linear projections
        q = self.q_linear(q)  # [B, Q, hidden_dim]
        k = self.k_linear(k)  # [B, L, hidden_dim]
        
        # Reshape for multi-head attention
        q = q.view(B, Q, self.num_heads, self.hidden_dim // self.num_heads)
        q = q.permute(0, 2, 1, 3)  # [B, num_heads, Q, head_dim]
        
        k = k.view(B, L, self.num_heads, self.hidden_dim // self.num_heads)
        k = k.permute(0, 2, 3, 1)  # [B, num_heads, head_dim, L]
        
        # Attention energy
        attn_energy = torch.matmul(q, k) * self.normalize_fact  # [B, num_heads, Q, L]
        
        # Apply mask if provided
        if mask is not None:
            attn_energy = attn_energy.masked_fill(~mask.bool().unsqueeze(1), float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_energy, dim=-1)
        attn_weights = self.dropout(attn_weights)  # [B, num_heads, Q, L]
        
        # Reconstruct spatial maps (using highest resolution only)
         H, W = spatial_shapes[0]
        attn_map = attn_weights[..., :H*W]  # First level features
        attn_map = attn_map.view(B, self.num_heads, Q, H, W)
        
        return attn_map  # [B, num_heads, Q, H, W]


class SimpleMaskHead(nn.Module):
    """Improved mask head with FPN-style fusion"""
    def __init__(self,
                 num_heads: int,
                 encoder_dim: int,
                 fpn_dim: int,
                 num_queries: int,
                 hidden_dim: int = 64,
                 mask_out_stride: int = 4):
        """
        Args:
            num_heads: Number of attention heads
            encoder_dim: Channel dimension of encoder features
            fpn_dim: Channel dimension of FPN features
            num_queries: Number of object queries
            hidden_dim: Internal channel dimension
            mask_out_stride: Output stride relative to input image
        """
        super().__init__()
        self.num_queries = num_queries
        self.mask_out_stride = mask_out_stride
        
        # Feature processing
        self.attn_to_feat = nn.Sequential(
            nn.Conv2d(num_heads, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Encoder feature processing
        self.enc_reduce = nn.Conv2d(encoder_dim, hidden_dim, 1)
        
        # FPN integration
        self.fpn_adapter = nn.Conv2d(fpn_dim, hidden_dim, 1)
        
        # Mask refinement
        self.refine_convs = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Mask prediction
        self.mask_predictor = nn.Conv2d(hidden_dim, 1, 1)
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self,
                attention_maps: torch.Tensor,  # [B, num_heads, Q, H, W]
                encoder_features: torch.Tensor, # [B, C_enc, H_enc, W_enc]
                fpn_features: torch.Tensor,     # [B, C_fpn, H_fpn, W_fpn] (higher res)
                target_img_size: Tuple[int, int] = None):
        """
        Args:
            target_img_size: (height, width) of original image
        Returns:
            masks: [B, num_queries, H_out, W_out]
        """
        B, num_heads, Q, H_att, W_att = attention_maps.shape
        _, _, H_enc, W_enc = encoder_features.shape
        _, _, H_fpn, W_fpn = fpn_features.shape
        
        # 1. Process attention maps (all heads -> features)
        # Flatten batch and query dimensions
        attn_flat = attention_maps.permute(0, 2, 1, 3, 4)  # [B, Q, num_heads, H, W]
        attn_flat = attn_flat.reshape(B * Q, num_heads, H_att, W_att)
        attn_feat = self.attn_to_feat(attn_flat)  # [B*Q, hidden_dim, H_att, W_att]
        
        # 2. Process encoder features
        enc_red = self.enc_reduce(encoder_features)  # [B, hidden_dim, H_enc, W_enc]
        
        # Resize to attention map size if needed
        if (H_enc, W_enc) != (H_att, W_att):
            enc_red = F.interpolate(
                enc_red, size=(H_att, W_att), mode='bilinear', align_corners=False
            )
        
        # Duplicate for each query
        enc_red = enc_red.unsqueeze(1)  # [B, 1, hidden_dim, H, W]
        enc_red = enc_red.expand(-1, Q, -1, -1, -1)  # [B, Q, hidden_dim, H, W]
        enc_red = enc_red.reshape(B * Q, -1, H_att, W_att)  # [B*Q, hidden_dim, H, W]
        
        # 3. Combine attention features + encoder features
        combined = torch.cat([attn_feat, enc_red], dim=1)  # [B*Q, 2*hidden_dim, H, W]
        x = self.refine_convs(combined)  # [B*Q, hidden_dim, H, W]
        
        # 4. FPN Fusion (upsample and merge with higher-res feature)
        # Process FPN features
        fpn_proc = self.fpn_adapter(fpn_features)  # [B, hidden_dim, H_fpn, W_fpn]
        
        # Duplicate for each query
        fpn_proc = fpn_proc.unsqueeze(1)  # [B, 1, hidden_dim, H_fpn, W_fpn]
        fpn_proc = fpn_proc.expand(-1, Q, -1, -1, -1)  # [B, Q, hidden_dim, H_fpn, W_fpn]
        fpn_proc = fpn_proc.reshape(B * Q, -1, H_fpn, W_fpn)  # [B*Q, hidden_dim, H_fpn, W_fpn]
        
        # Upsample current features to match FPN resolution
        x_up = F.interpolate(x, size=(H_fpn, W_fpn), mode='bilinear', align_corners=False)
        
        # Feature fusion (element-wise addition)
        x_fused = x_up + fpn_proc
        
        # 5. Final mask prediction
        mask_logits = self.mask_predictor(x_fused)  # [B*Q, 1, H_fpn, W_fpn]
        
        # 6. Resize to target output size
        if target_img_size is not None:
            H_out = target_img_size[0] // self.mask_out_stride
            W_out = target_img_size[1] // self.mask_out_stride
            mask_logits = F.interpolate(
                mask_logits, size=(H_out, W_out), 
                mode='bilinear', align_corners=False
            )
        
        # 7. Reshape to [B, Q, H_out, W_out]
        masks = mask_logits.view(B, Q, H_out, W_out)
        
        return masks