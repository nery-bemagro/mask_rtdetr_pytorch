import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


class MHAttentionMap(nn.Module):
    """Multi-Head Attention Map for generating mask attention from queries"""

    def __init__(self, query_dim: int, hidden_dim: int, num_heads: int = 2, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim, bias=bias)

        nn.init.xavier_uniform_(self.q_linear.weight)
        nn.init.xavier_uniform_(self.k_linear.weight)

        if self.q_linear.bias is not None:
            nn.init.zeros_(self.q_linear.bias)
        if self.k_linear.bias is not None:
            nn.init.zeros_(self.k_linear.bias)

        self.normalize_fact = float(hidden_dim / num_heads) ** -0.5

    def forward(self, q, k, spatial_shapes, mask=None):
        B, Q, _ = q.shape
        B, L, D = k.shape
        
        q = self.q_linear(q)
        k = self.k_linear(k)
        
        q = q.view(B, Q, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(B, L, self.num_heads, self.hidden_dim // self.num_heads).permute(0, 2, 3, 1)
        
        attn = (q @ k) * self.normalize_fact
        
        if mask is not None:
            attn = attn.masked_fill(~mask.bool(), float('-inf'))
            
        attn = F.softmax(attn, dim=-1)
        
        attention_maps = []
        start_idx = 0
        for h, w in spatial_shapes:
            length = h * w
            level_attn = attn[:, :, :, start_idx:start_idx+length]
            attention_maps.append(level_attn.view(B, self.num_heads, Q, h, w))
            start_idx += length
        
        # Use only the first level's attention map (highest resolution)
        attn_map = attention_maps[0]  # Shape: [B, num_heads, Q, H, W]

        # print("attn map shape", attn_map.mean(dim=1).shape)
        
        return attn_map.mean(dim=1)  # Average across heads


class SimpleMaskHead(nn.Module):
    """
    A very simple convolutional mask head for RT-DETR.
    Takes the attention map from MHAttentionMap and refines it.
    Outputs masks at a fixed resolution (e.g., H/4 x W/4 of the input image).
    """
    def __init__(self, hidden_dim: int, num_queries: int, num_convs: int = 2, mask_out_stride: int = 4):
        """
        Args:
            hidden_dim (int): Dimension of the transformer features (used for context, though not directly here).
                              Kept for potential future compatibility or modifications.
            num_queries (int): Number of object queries.
            num_convs (int): Number of convolutional layers.
            mask_out_stride (int): The stride of the output mask relative to the input image size.
                                   e.g., 4 means the mask is H/4 x W/4.
        """
        super().__init__()
        self.num_queries = num_queries
        self.mask_out_stride = mask_out_stride
        self.hidden_dim = hidden_dim  # Not directly used in this simple version
        # Input to this head is the attention map [B, Q, H_attn, W_attn]
        # MHAttentionMap outputs the mean over heads, so input channel is 1 after reshaping
        in_channels = 1
        # Use a fixed intermediate channel size, e.g., 64 or 128
        inter_channels = 64
        convs = []
        # Reshape input [B, Q, H, W] -> [B*Q, 1, H, W]
        # Layer 1
        convs.append(nn.Conv2d(in_channels, inter_channels, kernel_size=3, padding=1, bias=False))
        convs.append(nn.BatchNorm2d(inter_channels))
        convs.append(nn.ReLU())
        # Intermediate layers
        for _ in range(num_convs - 2):
            convs.append(nn.Conv2d(inter_channels, inter_channels, kernel_size=3, padding=1, bias=False))
            convs.append(nn.BatchNorm2d(inter_channels))
            convs.append(nn.ReLU())
        # Define the final convolutional layer separately
        self.final_conv = nn.Conv2d(inter_channels, 1, kernel_size=1)
        # Add the final convolutional layer to the list of convs
        convs.append(self.final_conv)
        self.conv_layers = nn.Sequential(*convs)
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # Calculate the bias for the final convolutional layer
        # Desired initial probability for foreground
        pi = 0.01
        # Calculate the bias
        b = -math.log((1 - pi) / pi)
        # Set the bias for the final_conv
        nn.init.constant_(self.final_conv.bias, b)
    
    def forward(self, attention_map: torch.Tensor, target_img_size: Tuple[int, int]) -> torch.Tensor:
        """
        Args:
            attention_map (torch.Tensor): Output from MHAttentionMap.
                                          Shape: [B, Q, H_attn, W_attn]
            target_img_size (Tuple[int, int]): The original input image size (H, W). Used to determine
                                              the final output mask size based on mask_out_stride.
        Returns:
            torch.Tensor: Predicted masks. Shape: [B, Q, H_out, W_out]
                          where H_out = target_img_size[0] / mask_out_stride
                          and W_out = target_img_size[1] / mask_out_stride
        """
        B, Q, H_attn, W_attn = attention_map.shape
        assert Q == self.num_queries
        # Reshape for convolution: [B, Q, H, W] -> [B*Q, 1, H, W]
        x = attention_map.view(B * Q, 1, H_attn, W_attn)
        # Apply convolutional layers
        x = self.conv_layers(x)  # Shape: [B*Q, 1, H_out, W_out]
        # Reshape back to [B, Q, H_out, W_out]
        masks = x.view(B, Q, *x.shape[-2:])
        return masks
