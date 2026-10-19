import torch
import torch.nn as nn
from einops import rearrange
from typing import Optional, Tuple, Dict, Any, Union
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

class ContentDSALDMWrapper(nn.Module):
    """
    A light-weight wrapper that loads a previously trained *single-frame* DSA Latent Diffusion
    model and uses it to **generate brand-new content frames**.

    This wrapper **does not** read from an input clip (no extraction from videos). It only samples.

    Public APIs:
      - sample_content_frames(...): directly sample [B,C,H,W] in [-1,1]
      - forward(...): alias of sample_content_frames for convenience
    """

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        ldm_config: Optional[Union[Dict[str, Any], str]] = None,
        resize_to: Optional[Tuple[int, int]] = None,   # (H,W) pixel size expected by current VAE
        out_channels: int = 1,
        default_steps: int = 200,
        default_eta: float = 0.0,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.ckpt_path = ckpt_path
        self.ldm_config = ldm_config
        self.resize_to = tuple(resize_to) if resize_to is not None else None
        self.out_channels = int(out_channels)
        self.default_steps = int(default_steps)
        self.default_eta = float(default_eta)
        self._device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.ldm = None        # type: Optional[nn.Module]
        self.sampler = None    # type: Optional[DDIMSampler]

    # --------------- public ---------------
    @torch.no_grad()
    def forward(
        self,
        batch_size: int = 1,
        ddim_steps: Optional[int] = None,
        eta: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
    ) -> torch.Tensor:
        """Alias of sample_content_frames so the wrapper can be used like a module."""
        return self.sample_content_frames(
            batch_size=batch_size,
            ddim_steps=ddim_steps,
            eta=eta,
            height=height,
            width=width,
        )

    @torch.no_grad()
    def sample_content_frames(
        self,
        batch_size: int = 1,
        ddim_steps: Optional[int] = None,
        eta: Optional[float] = None,
    ) -> torch.Tensor:
        self._ensure_ldm_loaded()
        ldm = self.ldm.eval()

        steps = int(ddim_steps if ddim_steps is not None else self.default_steps)
        eta_val = float(eta if eta is not None else self.default_eta)

        with ldm.ema_scope("sampling content"):
            latents, _ = ldm.sample_log(
                cond=None,
                batch_size=batch_size,
                ddim=True,
                ddim_steps=steps,
                eta=eta_val
            )

        x_content = ldm.decode_first_stage(latents)

        return x_content

    # --------------- helpers ---------------
    def _ensure_ldm_loaded(self):
        """Build the single-frame LDM either from a dict config or a YAML path."""
        if getattr(self, "ldm", None) is not None:
            return

        # We require at least a config (dict or yaml path). ckpt is optional.
        if self.ldm_config is None and self.ckpt_path is None:
            raise RuntimeError(
                "ContentDSALDMWrapper: neither `ldm_config` nor `ckpt_path` provided; cannot sample."
            )

        if self.ldm_config is None and self.ckpt_path is not None:
            # Only ckpt with no config is ambiguous for LDM construction.
            # Be explicit to avoid silent misconfigurations.
            raise RuntimeError(
                "ContentDSALDMWrapper: `ldm_config` is required when only a ckpt path is given."
            )

        # Resolve config: support dict or YAML path.
        if isinstance(self.ldm_config, str):
            # Load YAML and accept either a top-level model node or the whole file as the model config.
            cfg = OmegaConf.load(self.ldm_config)
            model_cfg = cfg.get("model", cfg)
        else:
            model_cfg = self.ldm_config

        # Instantiate the single-frame LDM from config
        self.ldm = instantiate_from_config(model_cfg).to(self._device)
        self.ldm.eval()

        # Optionally load weights
        if self.ckpt_path:
            self._safe_load(self.ldm, self.ckpt_path)

    def _safe_load(self, model: nn.Module, ckpt_path: str):
        sd = torch.load(ckpt_path, map_location="cpu")
        state_dict = sd.get("state_dict", sd)
        # drop loss/discriminator keys if present
        drop_prefixes = ("loss.", "discriminator.", "losses.")
        cleaned = {k: v for k, v in state_dict.items() if not any(k.startswith(p) for p in drop_prefixes)}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print(f"[ContentDSALDMWrapper] loaded ckpt={ckpt_path} | missing={len(missing)} unexpected={len(unexpected)}")
        if len(unexpected) > 0:
            print(f"[ContentDSALDMWrapper] unexpected keys (truncated): {unexpected[:8]}")
        print(f"[ContentDSALDMWrapper] device={self._device}")

__all__ = ["ContentDSALDMWrapper"]


class BertPromptContentDSALDMWrapper(ContentDSALDMWrapper):
    """
    Content frame generator using a prompt-conditioned (BERT) single-frame LDM.
    Provides sample_content_frames(batch_size, ...) -> [B,C,H,W] in [-1,1]
    """

    def __init__(
        self,
        ckpt_path=None,
        ldm_config=None,
        prompt_file: Optional[str] = None,   # text prompts, one per line
        default_prompt: str = "DSA image",
        guidance_scale: float = 1.0,         # >1 enables CFG
        default_steps: int = 50,
        default_eta: float = 0.0,
        device: Optional[str] = None,
    ):
        super().__init__(
            ckpt_path=ckpt_path,
            ldm_config=ldm_config,
            resize_to=None,
            out_channels=1,
            default_steps=default_steps,
            default_eta=default_eta,
            device=device,
        )
        self.prompt_file = prompt_file
        self.default_prompt = default_prompt
        self.guidance_scale = float(guidance_scale)

        self._prompts = None
        self._prompt_cursor = 0

    def _load_prompts(self):
        if self._prompts is not None:
            return
        if self.prompt_file is None:
            self._prompts = [self.default_prompt]
            return
        with open(self.prompt_file, "r") as f:
            prompts = [ln.strip() for ln in f if ln.strip()]
        self._prompts = prompts if prompts else [self.default_prompt]

    def _next_prompts(self, batch_size: int) -> list:
        self._load_prompts()
        prompts = []
        for _ in range(batch_size):
            prompts.append(self._prompts[self._prompt_cursor % len(self._prompts)])
            self._prompt_cursor += 1
        return prompts

    @torch.no_grad()
    def sample_content_frames(self, batch_size: int = 1, ddim_steps: Optional[int] = None, eta: Optional[float] = None):
        self._ensure_ldm_loaded()
        ldm = self.ldm.eval()

        steps = int(ddim_steps if ddim_steps is not None else self.default_steps)
        eta_val = float(eta if eta is not None else self.default_eta)

        prompts = self._next_prompts(batch_size)

        # conditioning
        cond = ldm.get_learned_conditioning(prompts)

        # optional CFG (if you want)
        uc = None
        if self.guidance_scale and self.guidance_scale != 1.0:
            uc = ldm.get_learned_conditioning([""] * batch_size)

        sampler = DDIMSampler(ldm)

        # latent shape: (C,H,W) comes from the UNet settings in the single-frame LDM
        # Typical LDM: channels = ldm.model.diffusion_model.in_channels, image_size = ldm.model.diffusion_model.image_size
        c = ldm.model.diffusion_model.in_channels
        h = ldm.model.diffusion_model.image_size
        w = ldm.model.diffusion_model.image_size

        with ldm.ema_scope("sampling content (bert)"):
            z, _ = sampler.sample(
                S=steps,
                batch_size=batch_size,
                shape=(c, h, w),
                conditioning=cond,
                eta=eta_val,
                unconditional_guidance_scale=self.guidance_scale,
                unconditional_conditioning=uc,
                verbose=False,
            )

        x_content = ldm.decode_first_stage(z)
        return x_content