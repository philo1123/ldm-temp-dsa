import torch
import torch.nn as nn
from einops import rearrange

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.openaimodel import MotionDiTBackboneWrapper

class MotionLatentDiffusion(LatentDiffusion):
    """
    Motion prior training in the CMD-style pipeline.

    What this class does:
    - Takes a 5D clip x_seq: [B, C, T, H, W].
    - Uses a *frozen* content diffusion to get a content frame in pixel space: x_content: [B, C, H, W].
    - Encodes x_content with the *same* VAE backbone used by the motion AE
      to obtain z_content (latent).
    - Encodes the video to latent sequence z_seq via VAE.encode_sequence and
      decomposes into motion target z_m using MotionHead with z_content.
    - Trains a diffusion prior in z_m-space, conditioning on x_content via
      the base-class cond_stage_model (from cond_stage_config) via get_learned_conditioning,
      which provides image-based cross-attention tokens for the UNet's cross-attention stack
      (conditioning_key='crossattn').

    Assumptions:
    - self.first_stage_model is an AutoencoderKL_CMHeads (frozen backbone OK).
    - self.content_model exposes:
        infer_content_frame(clip_5d, ddim_steps, eta, mode="extract"|"sample") -> [B,C,H,W]
      Typically you pass mode="extract" during motion prior training so the
      condition is stable and deterministic.
    - Conditioning tokens come from the base class cond_stage_model via get_learned_conditioning.
      The UNet is configured with `conditioning_key='crossattn'`.

    Notes:
    - We override `encode_first_stage` to produce z_m (Tensor) so that
      LatentDiffusion.get_first_stage_encoding() returns it directly.
    - `get_input` prepares both z_m and cross-attn context `c`.
    - If no content diffusion is provided, the VAE pixel-mode path is used to extract x_content during training.
    """

    def __init__(
        self,
        *args,
        content_model_config: dict = None,
        content_mode: str = "extract",      # "extract" (VAE path) or "sample" (diffusion)
        content_ddim_steps: int = 50,
        content_eta: float = 0.0,
        motion_mode: str = "cmd_dit", # "cmd_dit" | "residual_unet" | "flow_unet"
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._last_motion_shape = None
        self._last_keyframe_latent = None
        self.motion_mode = motion_mode
        
        if self.motion_mode == "cmd_dit":
            # DiT path: image cross-attn / keyframe latent
            if self.model.conditioning_key not in (None, "crossattn"):
                print("[WARN] overriding conditioning_key='crossattn' for cmd_dit")
                self.model.conditioning_key = "crossattn"
        else:  # residual_unet or flow_unet
            # UNet path: concat latent z_c
            if self.model.conditioning_key != "concat":
                print("[WARN] overriding conditioning_key='concat' for residual_unet or flow_unet")
                self.model.conditioning_key = "concat"

        # Build content model only if a config is provided; otherwise None (use VAE path)
        self.content_model = None
        if content_model_config is not None:
            try:
                self.content_model = instantiate_from_config(content_model_config)
            except Exception as e:
                raise RuntimeError(f"Failed to build content_model from config: {e}")

        # Content generator (frozen for motion training)
        if self.content_model is not None:
            self.content_model.eval()
            for p in self.content_model.parameters():
                p.requires_grad = False

        self.content_mode = content_mode
        self.content_ddim_steps = int(content_ddim_steps)
        self.content_eta = float(content_eta)

        # Sanity hint: UNet in/out channels should equal motion proj dim
        try:
            proj_dim = getattr(self.first_stage_model.motion_head, "proj_dim", None)
            if proj_dim is not None:
                in_ch = getattr(self.model.diffusion_model, "in_channels", None)
                out_ch = getattr(self.model.diffusion_model, "out_channels", None)
                if in_ch is not None and in_ch != proj_dim:
                    print(f"[WARN] UNet in_channels={in_ch} != motion_proj_dim={proj_dim}.")
                if out_ch is not None and out_ch != proj_dim:
                    print(f"[WARN] UNet out_channels={out_ch} != motion_proj_dim={proj_dim}.")
        except Exception:
            pass

    # -----------------------------
    # Core: build motion latent z_m
    # -----------------------------
    @torch.no_grad()
    def _build_cmd_targets(self, x_seq):
        # CMD/DiT: z_m + z_c
        z_seq, _ = self.first_stage_model.encode_sequence(x_seq, sample_posterior=False)
        z_c, z_m, aux = self.first_stage_model.decompose_content_motion(z_seq, x_seq=x_seq)
        # cache for DiT context
        self._last_keyframe_latent = z_c.detach()
        # cache sampling shape (P,L,HW)
        assert z_m.ndim == 4
        _, P, L, HW = z_m.shape
        self._last_motion_shape = (int(P), int(L), int(HW))
        
        return {"target": z_m.detach(), "z_c": z_c.detach(), "aux": aux}

    @torch.no_grad()
    def _build_residual_targets(self, x_seq):
        # UNet residual: r = z_seq - expand(z_c)
        z_seq, _ = self.first_stage_model.encode_sequence(x_seq, sample_posterior=False) # [B,C,T,H,W]
        z_c, _z_m_unused, aux = self.first_stage_model.decompose_content_motion(z_seq, x_seq=x_seq)
        z_c = z_c.detach()
        z_c_full = z_c.unsqueeze(2).expand_as(z_seq)
        r = (z_seq - z_c_full).detach()  # [B,C,T,H,W]
        B, C, T, H, W = r.shape
        self._last_motion_shape = (int(C), int(H), int(W))
        self._last_bt_hw = (int(B), int(T), int(H), int(W))      # for fold/unfold time later
        
        return {"target": r, "z_c": z_c, "aux": aux}

    @torch.no_grad()
    def _build_flow_targets(self, x_seq):
        z_seq, _ = self.first_stage_model.encode_sequence(x_seq, sample_posterior=False)
        z_c, z_m, aux = self.first_stage_model.decompose_content_motion(z_seq, x_seq=x_seq)
        z_c = z_c.detach()
        # z_c_full = z_c.unsqueeze(2).expand_as(z_seq)
        assert z_m.ndim == 5, f"expect z_m [B,P,T,H',W'], got {z_m.shape}"
        B, P, T, Hp, Wp = z_m.shape

        self._last_motion_shape = (int(P), int(Hp), int(Wp))
        self._last_bt_hw = (int(B), int(T), int(Hp), int(Wp))  # for fold/unfold time & space

        return {"target": z_m.detach(), "z_c": z_c.detach(), "aux": aux}

    # ---------------------------------------------------------
    # Override first-stage encode to feed z_m directly to DDPM
    # ---------------------------------------------------------
    @torch.no_grad()
    def encode_first_stage(self, x):
        x = x.to(self.device)
        assert x.ndim == 5, f"expect [B,C,T,H,W], got {x.shape}"
        if self.motion_mode == "cmd_dit":
            out = self._build_cmd_targets(x)
            return out["target"]                  # z_m
        elif self.motion_mode == "flow_unet":
            out = self._build_flow_targets(x)
            return out["target"]               # z_m [B,P,T,H',W']
        else:
            out = self._build_residual_targets(x)
            return out["target"]                  # r

    def get_learned_conditioning(self, c):
        """
        Identity conditioner for our 'concat' route.
        Always return a Tensor suitable for concat, i.e. [BT, C, H, W].
        """
        if torch.is_tensor(c):
            return c

        if isinstance(c, list):
            assert len(c) == 1, f"Expect single-element list, got {len(c)}"
            assert torch.is_tensor(c[0]), f"List must contain a Tensor, got {type(c[0])}"
            return c[0]

    @torch.no_grad()
    def get_input(
        self,
        batch,
        k,
        return_first_stage_outputs=False,
        force_c_encode=False,
        cond_key=None,
        return_original_cond=False,
        bs=None
    ):
        """
        Prepare (z, c) for motion prior training:
          - z := z_m Tensor [B,P,L,HW]
          - c := (unused for DiT path) keep a dummy tensor to satisfy LDM API
        Also cache z_c for DiT backbone.
        """
        x = batch[k]  # expect [B,C,T,H,W]
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)

        if self.motion_mode == "cmd_dit":
            out = self._build_cmd_targets(x)
            z = out["target"]; z_c = out["z_c"]; aux = out["aux"]
            c = None
        if self.motion_mode == "residual_unet":
            out = self._build_residual_targets(x)
            z = out["target"]; z_c = out["z_c"]; aux = out["aux"]
        if self.motion_mode == "flow_unet":
            out = self._build_flow_targets(x)
            z = out["target"]; z_c = out["z_c"]; aux = out["aux"]

        if self.motion_mode in ["flow_unet", "residual_unet"]:
            B, C, T, H, W = z.shape
            _, Cz, Hc, Wc = z_c.shape

            self._last_bt_hw = (int(B), int(T), int(H), int(W))
            # ---- collapse time into batch for 2D UNet ----
            z = rearrange(z, 'b c t h w -> (b t) c h w')      # [B*T, C, H, W]

            # build concat conditioning from a TEMP expanded copy (do NOT modify original z_c)
            z_c_full_cond = z_c.unsqueeze(2).expand(B, Cz, T, Hc, Wc)
            z_c_bt = rearrange(z_c_full_cond, 'b c t h w -> (b t) c h w')  # [B*T, Cz, Hc, Wc]
            if (Hc != H) or (Wc != W):
                z_c_bt = torch.nn.functional.interpolate(z_c_bt, size=(H, W), mode='bilinear', align_corners=False)

            c = z_c_bt
            if force_c_encode or (not self.cond_stage_trainable):
                c = self.get_learned_conditioning(z_c_bt)
            else:
                c = z_c_bt
    
        ret = [z, c]
        if return_first_stage_outputs:
            B, C, T, H, W = x.shape
            inputs_seq = rearrange(x[0], 'c t h w -> t c h w')
            x_content = aux.get('x_content', None) 
            ret.extend([inputs_seq, x_content, z_c])
        if return_original_cond:
            ret.append(None)
        return ret
    
    @torch.no_grad()
    def reconstruct_sequence_from_latent(self, latent, z_c):
        """
        - cmd_dit:      latent = z_m (already in head-specific shape) → VAE.reconstruct_sequence(z_c, z_m)
        - flow_unet:    latent = [B*T,P,H',W'] → fold to [B,P,T,H',W'] → VAE.reconstruct_sequence(z_c, z_m)
        - residual_unet:latent = [B*T,C,H,W]   → fold to [B,C,T,H,W]   → VAE.reconstruct_sequence(z_c, r)
        """
        if self.motion_mode == "cmd_dit":
            z_seq, *_ = self.first_stage_model.reconstruct_sequence(z_c, latent)
            frames = self.first_stage_model.decode_sequence(z_seq)
            return frames

        elif self.motion_mode == "flow_unet":
            assert hasattr(self, "_last_bt_hw"), "Missing shape cache; make sure get_input() ran first."
            B, T, Hp, Wp = self._last_bt_hw
            assert latent.ndim == 4, f"expect [B*T,P,H',W'], got {latent.shape}"
            z_m = rearrange(latent, '(b t) p h w -> b p t h w', b=B, t=T)
            z_seq, *_ = self.first_stage_model.reconstruct_sequence(z_c, z_m)
            frames = self.first_stage_model.decode_sequence(z_seq)
            return frames

        elif self.motion_mode == "residual_unet":
            assert hasattr(self, "_last_bt_hw"), "Missing shape cache; make sure get_input() ran first."
            B, T, H, W = self._last_bt_hw
            assert latent.ndim == 4, f"expect [B*T,C,H,W], got {latent.shape}"
            r = rearrange(latent, '(b t) c h w -> b c t h w', b=B, t=T)
            z_seq, *_ = self.first_stage_model.reconstruct_sequence(z_c, r)
            frames = self.first_stage_model.decode_sequence(z_seq)
            return frames

        else:
            return self.first_stage_model.decode_sequence(z_seq)

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        if self.motion_mode == "cmd_dit":
            if isinstance(getattr(self.model, "diffusion_model", None), MotionDiTBackboneWrapper):
                assert getattr(self, "_last_keyframe_latent", None) is not None, \
                    "keyframe latent missing; ensure get_input() ran and cached it."
                context = {"keyframe": self._last_keyframe_latent}
                out = self.model.diffusion_model(x_noisy, t, context=context)
                return out
        # fallback (e.g., plain UNet)
        return super().apply_model(x_noisy, t, cond, return_ids=return_ids)

    def _cache_motion_shape(self, z_m: torch.Tensor):
        """Cache motion latent shape (P, L, HW) for DDIM sampler treating (L,HW) as spatial."""
        assert z_m.ndim == 4, f"Expected z_m [B,P,L,HW], got {z_m.shape}"
        _, P, L, HW = z_m.shape
        self._last_motion_shape = (int(P), int(L), int(HW))
        
    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, guidance_scale: float = 1.0, eta: float = 0.0, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = self._last_motion_shape
            samples, intermediates = ddim_sampler.sample(
                S=ddim_steps,
                batch_size=batch_size,
                shape=shape,
                conditioning=cond,
                verbose=False,
                eta=eta,
            )
        else:
            samples, intermediates = self.sample(
                cond=cond,
                batch_size=batch_size,
                return_intermediates=True,
                **kwargs
            )
        return samples, intermediates

    @torch.no_grad()
    def log_images(
        self,
        batch,
        N: int = 8,
        ddim_steps: int = 200,
        ddim_eta: float = 1.0,
        **kwargs
    ):
        """
        Visualize: input clip (first sequence), inferred content frame, and a motion-sampled latent.
        """
        log = dict()
        use_ddim = ddim_steps is not None

        z_m, c, inputs_seq, x_content, z_c_latent = self.get_input(
            batch,
            self.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            bs=N
        )

        # Visual references
        log["inputs_seq"] = inputs_seq      # [T,C,H,W] (first clip)
        log["content_frame"] = x_content    # [1,C,H,W]

        rec_seq = self.reconstruct_sequence_from_latent(z_m, z_c_latent)
        log["rec_seq"] = rec_seq[0].permute(1, 0, 2, 3).contiguous()

        # Sample from the motion prior in UNet-space (unconditional)
        with self.ema_scope("Plotting"):
            bs_for_sampling = z_m.shape[0]

            z_samples, _ = self.sample_log(
                cond=c,
                batch_size=bs_for_sampling,
                ddim=use_ddim,
                ddim_steps=ddim_steps,
                eta=ddim_eta
            )

        recon_sample_motion_seq = self.reconstruct_sequence_from_latent(z_samples, z_c_latent)
        log["recon_sample_motion_seq"] = recon_sample_motion_seq[0].permute(1,0,2,3)  # log first sequence [T,C,H,W]
        
        if getattr(self, "content_model", None) is not None:
            B, T, H, W = self._last_bt_hw
            
            steps = int(ddim_steps) if use_ddim else getattr(self.content_model, "default_steps", 50)
            eta_gen = float(ddim_eta) if use_ddim else getattr(self.content_model, "default_eta", 0.0)

            x_content_gen = self.content_model.sample_content_frames(
                batch_size=B,
                ddim_steps=steps,
                eta=eta_gen,
            )

            log["content_frame_gen"] = x_content_gen
            post_gen = self.first_stage_model.encode(x_content_gen)
            z_c_gen  = post_gen.mode()

            self._last_keyframe_latent = z_c_gen
            
            if self.motion_mode in ["flow_unet", "residual_unet"]:
                zc_full = z_c_gen.unsqueeze(2).expand(z_c_gen.shape[0], z_c_gen.shape[1], T, z_c_gen.shape[2], z_c_gen.shape[3])
                c_gen = rearrange(zc_full, 'b c t h w -> (b t) c h w')
                if (c_gen.shape[-2] != H) or (c_gen.shape[-1] != W):
                    c_gen = torch.nn.functional.interpolate(c_gen, size=(H, W), mode='bilinear', align_corners=False)
            else:
                c_gen = None
    
            with self.ema_scope("Plotting"):
                z_samples_gen, _ = self.sample_log(
                    cond=None if self.motion_mode == "cmd_dit" else c_gen,
                    batch_size=B*T,
                    ddim=use_ddim,
                    ddim_steps=steps,
                    eta=eta_gen
                )

            recon_sample_seq_gen = self.reconstruct_sequence_from_latent(z_samples_gen, z_c_gen)
            log["recon_sample_seq_gen"] = recon_sample_seq_gen[0].permute(1, 0, 2, 3)
        return log
