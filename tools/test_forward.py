import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import torch
import numpy as np
from src.zoo.rtdetr.rtdetr import RTDETR
from src.zoo.rtdetr.rtdetr_decoder import RTDETRTransformer
from src.zoo.rtdetr.hybrid_encoder import HybridEncoder
from src.nn.backbone.presnet import PResNet

# Define the backbone
backbone = PResNet(depth=18)

# Define the encoder
encoder_channels = [64, 128, 256, 512]  # Adjusted to match all four feature maps
encoder_strides = [8, 16, 32, 64]       # Adjusted to match all four feature maps
encoder = HybridEncoder(
    in_channels=encoder_channels,
    feat_strides=encoder_strides,
    hidden_dim=256,
    nhead=8,
    num_encoder_layers=1
)

# Define the decoder
# feat_channels should match the output channels of the encoder (256)
decoder = RTDETRTransformer(
    num_classes=80,  # For COCO
    hidden_dim=256,
    num_queries=300,
    feat_channels=[256, 256, 256, 256],  # Adjusted to match the encoded feature channels
    feat_strides=encoder_strides,
    num_levels=4    # Adjusted to match four feature levels
)

decoder.training = False
# Define the RTDETR model
model = RTDETR(
    backbone=backbone,
    encoder=encoder,
    decoder=decoder
)


# Define a random input tensor
input_tensor = torch.randn(2, 3, 640, 640)  # Batch size 2, 3 color channels, 480x800 image
print(f'Input Tensor Shape: {input_tensor.shape}')

# Pass through the backbone
backbone_features = model.backbone(input_tensor)
print(f'Backbone Features Shape: {[feat.shape for feat in backbone_features]}')

# Pass through the encoder
encoder_features = model.encoder(backbone_features)
print(f'Encoder Features Shape: {[feat.shape for feat in encoder_features]}')

# Pass through the decoder
output = model.decoder(encoder_features, targets=None)
print(f'Output Keys: {output.keys()}')
for key, value in output.items():
    if isinstance(value, torch.Tensor):
        print(f'Output {key} Shape: {value.shape}')
    else:
        print(f'Output {key} Length: {len(value)}')
        for idx, item in enumerate(value):
            print(f'Output {key}[{idx}] Shape: {item.shape if isinstance(item, torch.Tensor) else "Not a tensor"}')
