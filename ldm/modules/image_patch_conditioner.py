# ldm/modules/conditioners/image_patch_conditioner.py
import torch
import torch.nn as nn
from einops import rearrange
try:
    from timm.models.vision_transformer import PatchEmbed
    _HAS_TIMM = True
except Exception:
    _HAS_TIMM = False


class ImagePatchConditioner(nn.Module):
    """
    Convert an image or VAE latent [B,C,H,W] into patch tokens [B, N, D] for UNet cross-attention.
    Additionally, produce a projected context embedding suitable for cross-attention conditioning.
    - If your content is already in latent space, feed it directly here;
    - If it's a pixel image, first encode it into latents with the same VAE for better distribution matching.
    If `return_both=False`, only the context tensor `[B, N, D]` is returned; if `True`, returns `(tokens, context)`.
    """

    def __init__(
        self,
        in_channels: int = 3,        # Input channels: latent usually 3 or 4; pixel images depend on your data
        embed_dim: int = 768,        # Token dimension; should match UNet's context_dim
        patch_size: int = 8,         # Patch size (must evenly divide H and W)
        img_size: int = 64,          # Used only for timm PatchEmbed position check
        use_timm: bool = True,       # Use timm PatchEmbed or custom ConvPatchify
        add_pos: bool = False,       # Optionally add learnable positional embedding
        return_both: bool = False,   # If True, forward returns (tokens, context); else only context
    ):
        super().__init__()
        self.return_both = return_both
        self.embed_dim = embed_dim
        if use_timm and _HAS_TIMM:
            self.pe = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_channels,
                embed_dim=embed_dim
            )
            self.patch_size = patch_size
            self._is_timm = True
        else:
            # Simplified Conv patchify: Conv2d(in->D, kernel_size=stride=patch_size)
            self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
            self.patch_size = patch_size
            self._is_timm = False

        self.pos_embed = None
        if add_pos:
            # Note: simple learnable 1D positional embedding, no complex 2D->1D mapping
            num_patches = (img_size // patch_size) * (img_size // patch_size)
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Projection head to produce context embeddings for UNet cross-attention
        # This projects patch tokens to the context embedding dimension (usually same as embed_dim)
        self.context_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, C, H, W]  (pixel or latent, latent recommended)
        Returns:
            If self.return_both:
                (tokens, context): both of shape [B, N, D]
            else:
                context: [B, N, D]
        """
        if self._is_timm:
            tokens = self.pe(x)        # [B, N, D]
        else:
            z = self.proj(x)           # [B, D, H/ps, W/ps]
            tokens = rearrange(z, 'b d h w -> b (h w) d')  # [B, N, D]

        if self.pos_embed is not None and self.pos_embed.shape[1] == tokens.shape[1]:
            tokens = tokens + self.pos_embed

        # Project tokens to context embedding space for cross-attention
        context = self.context_proj(tokens)  # [B, N, D]

        if self.return_both:
            return tokens, context
        else:
            return context