import torch
import torch.nn as nn
from transformers import SwinModel, SwinConfig

from src.core import register

__all__ = ['Swin']

# Model variant configurations
MODEL_CONFIGS = {
    'tiny': "microsoft/swin-tiny-patch4-window7-224",
    's': "microsoft/swin-small-patch4-window7-224",
    'b': "microsoft/swin-base-patch4-window7-224",
    'l': "microsoft/swin-large-patch4-window7-224",
}

@register
class Swin(nn.Module):
    __inject__ = []  # No dependencies to inject
    __share__ = []   # No shared parameters
    
    def __init__(self, 
                 variant: str = 'b',
                 return_idx: list = [1, 2, 3],
                 freeze_at: int = -1,
                 freeze_norm: bool = False,
                 pretrained: bool = True):
        super(Swin, self).__init__()
        
        # Validate variant
        if variant not in MODEL_CONFIGS:
            raise ValueError(f"Invalid variant '{variant}'. Choose from {list(MODEL_CONFIGS.keys())}")
        
        model_name = MODEL_CONFIGS[variant]
        
        # Load pretrained model or initialize from scratch
        if pretrained:
            self.model = SwinModel.from_pretrained(model_name)
        else:
            config = SwinConfig.from_pretrained(model_name)
            self.model = SwinModel(config)

        self.return_idx = return_idx
        self.variant = variant
        
        # Get patch size for stride calculation
        patch_size = self.model.config.patch_size
        if isinstance(patch_size, tuple):
            patch_size = patch_size[0]
        self.initial_stride = patch_size
        
        # Calculate output channels and strides
        embed_dim = self.model.config.embed_dim
        self.out_channels = []
        self.out_strides = []
        
        # Pre-calculate for all possible indices
        for idx in [1, 2, 3]:
            # Channel calculation
            channel = embed_dim * (2 ** (idx - 1)) if idx > 1 else embed_dim
            self.out_channels.append(channel)
            
            # Stride calculation
            stride = self.initial_stride * (2 ** (idx - 1)) if idx > 1 else self.initial_stride
            self.out_strides.append(stride)
        
        # Filter based on return_idx
        self.out_channels = [self.out_channels[i-1] for i in return_idx]
        self.out_strides = [self.out_strides[i-1] for i in return_idx]

        # Freeze parameters if requested
        if freeze_at >= 0:
            self._freeze_parameters(self.model)
        
        # Freeze norms if requested
        if freeze_norm:
            self._freeze_norm(self.model)

    def _freeze_parameters(self, m: nn.Module):
        for p in m.parameters():
            p.requires_grad = False

    def _freeze_norm(self, m: nn.Module):
        if isinstance(m, nn.LayerNorm):
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        else:
            for _, child in m.named_children():
                self._freeze_norm(child)

    def forward(self, x):
        outputs = self.model(x, output_hidden_states=True)
        
        # Get reshaped feature maps [batch, channels, height, width]
        feature_maps = outputs.reshaped_hidden_states
        
        # Select requested feature levels (indices 1-4 correspond to stages 1-4)
        selected_features = [feature_maps[i] for i in self.return_idx]
        return selected_features
    
def main():
    # Create an instance of the Swin model with a specific variant
    model_variant = 'tiny'  # You can choose from 'tiny', 's', 'b', 'l'
    swin_model = Swin(variant=model_variant, pretrained=True)
    
    # Set the model to evaluation mode (necessary for inference)
    swin_model.eval()
    
    # Create a random input tensor with the appropriate shape
    # The input shape is [batch_size, channels, height, width]
    # For Swin models, the expected input size is generally 224x224
    batch_size = 1
    channels = 3  # RGB image
    height = 640
    width = 640
    input_tensor = torch.randn(batch_size, channels, height, width)
    
    # Perform a forward pass with the random input
    with torch.no_grad():  # Disable gradient computation for inference
        features = swin_model(input_tensor)
    
    # Print the output features
    for i, feature in enumerate(features):
        print(f"Feature map {i+1} shape: {feature.shape}")

if __name__ == "__main__":
    main()