

import torch
import torch.nn as nn
from einops import rearrange
from .motion_module import VanillaTemporalModule


class TemporalWrapper(nn.Module):
    """
    Wraps a temporal transformer module, processing input as [B*T, C, H, W] and applying temporal modeling.
    Internally, reshapes to [B, C, T, H, W] for temporal processing, then back to [B*T, C, H, W].
    """
    def __init__(self, in_channels, num_frames, motion_module_config):
        super().__init__()
        self.num_frames = num_frames
        self.temporal = VanillaTemporalModule(in_channels=in_channels, **motion_module_config)

    def forward(self, x):
        # If input is already [B, C, T, H, W], skip reshape
        if x.dim() == 5:
            return self.temporal(x, temb=None, encoder_hidden_states=None)

        # Otherwise assume input is [B*T, C, H, W]
        B_T, C, H, W = x.shape
        assert B_T % self.num_frames == 0, "Input batch is not divisible by num_frames"
        B = B_T // self.num_frames
        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.num_frames)

        # The temporal module expects [B, C, T, H, W]
        x = self.temporal(x, temb=None, encoder_hidden_states=None)

        # Reshape back to [B*T, C, H, W]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        return x