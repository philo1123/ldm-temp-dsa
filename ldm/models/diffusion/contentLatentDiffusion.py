import torch
from einops import rearrange

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.distributions.distributions import DiagonalGaussianDistribution


class ContentLatentDiffusion(LatentDiffusion):
    """
    Latent diffusion for the CMD-style content latent (z_content).

    Key ideas:
    - If the batch provides a video clip [B, C, L, H, W], we derive z_content on-the-fly by:
        encode_sequence(...) -> decompose_content_motion(...) -> z_content
      and then train diffusion in the content-latent space only (unconditional).
    - If the batch provides a single frame [B, C, H, W], we fall back to standard KL-VAE encode.
    - decode_first_stage(z) just calls the first-stage VAE's decode(z), since ContentHead lives
      in the same latent channel space as the VAE (shape [B, D, H, W]).
    """

    @torch.no_grad()
    def encode_first_stage(self, x):
        """
        Returns an 'encoder_posterior-like' object for the base class.
        We want get_first_stage_encoding(...) to accept either a DiagonalGaussianDistribution
        or a Tensor. Here we will return a Tensor so that
            LatentDiffusion.get_first_stage_encoding()
        passes it through directly as z (= z_content).

        Cases:
        - x.ndim == 5: [B, C, L, H, W] video -> compute z_content via AE + heads
        - x.ndim == 4: [B, C, H, W] image -> use the VAE's native encoder (DiagonalGaussianDistribution)
        """
        x = x.to(self.device)

        # Case A: video -> derive z_content
        if x.ndim == 5:
            assert hasattr(self.first_stage_model, "encode_sequence") and \
                   hasattr(self.first_stage_model, "decompose_content_motion"), \
                   "First-stage model must be AutoencoderKL_CMHeads with encode_sequence() and decompose_content_motion()."

            # 1) Encode each frame to latent sequence z_seq: [B, D, L, H, W]
            z_seq, _posterior_list = self.first_stage_model.encode_sequence(x, sample_posterior=True)

            # 2) Decompose to content/motion
            z_content, _z_m, _aux = self.first_stage_model.decompose_content_motion(z_seq, x)

            # Prefer true pixel content when pixel_mode is enabled; otherwise fall back to decode/encode.
            pixel_mode_enabled = bool(getattr(self.first_stage_model, "pixel_mode", False))
            have_xcontent = isinstance(_aux, dict) and ("x_content" in _aux) and (_aux["x_content"] is not None)
            if pixel_mode_enabled and have_xcontent:
                x_content = _aux["x_content"]  # [B, C, H, W]
            else:
                if not pixel_mode_enabled:
                    print("[ContentLD][warn] first_stage_model.pixel_mode=False; using decode(z_content)->encode fallback (weaker alignment).")
                else:
                    print("[ContentLD][warn] pixel_mode=True but x_content missing; using decode(z_content)->encode fallback.")
                # Fall back: decode z_content to pixel. Undo scale_factor before decode.
                sf = getattr(self, "scale_factor", 1.0)
                x_content = self.first_stage_model.decode(z_content / sf)

            encoder = self.first_stage_model.encode

            posterior = encoder(x_content)
            z_aligned = posterior.sample() if hasattr(posterior, "sample") else posterior

            return z_aligned

        # Case B: single image -> standard KL-VAE posterior (keeps LDM’s native path)
        elif x.ndim == 4:
            posterior = self.first_stage_model.encode(x)
            assert isinstance(posterior, DiagonalGaussianDistribution), \
                "First-stage encode(x) for 2D should return DiagonalGaussianDistribution."
            return posterior

        else:
            raise ValueError(f"Unsupported input shape for first-stage encode: {x.shape}")

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        """
        Override to support 5D video input and to prepare logging payload here.

        Behavior:
          - If x is 5D [B, C, L, H, W], compute z_content via first-stage AE+heads.
          - If `return_first_stage_outputs` is True, also decode a content frame and
            prepare a flattened 8-frame sequence for logging (stored in `self._last_log_payload`).
          - If x is 4D [B, C, H, W], fallback to parent implementation.

        Returns (compatible with base expectations):
            out = [z, c]
            + optionally [x, xrec]
            + optionally [xc]
        """
        x = batch[k]
        if bs is not None:
            x = x[:bs]

        # Video case: [B, C, L, H, W] -> compute z_content and optionally prepare logging payload
        if x.ndim == 5:
            x = x.to(self.device)

            # Derive z_content as a Tensor (shape [B, D, H, W]); get_first_stage_encoding handles scale_factor
            z_aligned = self.get_first_stage_encoding(self.encode_first_stage(x)).detach()
            out = [z_aligned, None]

            if return_first_stage_outputs:
                # Decode all content latents back to images for visualization
                B, C, L, H, W = x.shape
                inputs_seq = rearrange(x[0], 'c l h w -> l c h w')

                z_seq_vis, _ = self.first_stage_model.encode_sequence(x[:1], sample_posterior=True)
                zc_vis, _zm_vis, aux_vis = self.first_stage_model.decompose_content_motion(z_seq_vis, x[:1])
                sf = getattr(self, "scale_factor", 1.0)
                if isinstance(aux_vis, dict) and (aux_vis.get("x_content", None) is not None):
                    content_frame = aux_vis["x_content"][0:1]  # [1,C,H,W]
                else:
                    content_frame = self.first_stage_model.decode(zc_vis / sf)[0:1]

                xrec_from_z = self.decode_first_stage(z_aligned[:1])

                out.extend([inputs_seq, content_frame, xrec_from_z])

            if return_original_cond:
                out.append(None)  # unconditional
            return out

        # Image case: fallback to LatentDiffusion’s native path
        return super().get_input(
            batch, k,
            return_first_stage_outputs=return_first_stage_outputs,
            force_c_encode=force_c_encode,
            cond_key=cond_key,
            return_original_cond=return_original_cond,
            bs=bs,
        )

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        """
        Decode a content latent [B, D, H, W] back to an image [B, C, H, W].

        If z came from the image-path (regular 2D encode), we are still fine:
        - Base class passes through here anyway and our first-stage model is a VAE decoder.
        """
        # Content latents were multiplied by self.scale_factor in get_first_stage_encoding().
        # Follow base-class convention: divide back before decode.
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    @torch.no_grad()
    def log_images(self, batch, N: int = 8, ddim_steps: int = 200, ddim_eta=1., **kwargs):
        """
        Custom logger for content-latent training.
        Produces three panels:
          1) inputs_seq: first series' first N frames (flattened to [N, C, H, W])
          2) content_frame: decoded content frame from the same series ([1, C, H, W])
          3) samples_content: N sampled content frames from the diffusion model ([N, C, H, W])

        Note: All heavy lifting (AE encode/decode) is done inside get_input(return_first_stage_outputs=True).
        """        
        use_ddim = ddim_steps is not None
        
        log = dict()
        z, c, inputs_seq, content_frame, xrec_from_z, *_ = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)

        log["inputs_seq"] = inputs_seq
        log["xrec_from_z"]    = xrec_from_z
        log["content_frame"] = content_frame

        # Use base sampler to draw N content frames; this respects EMA etc.
        with self.ema_scope("Plotting"):
            samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                        ddim_steps=ddim_steps,eta=ddim_eta)

        content_frame_samples = self.decode_first_stage(samples)
        log["samples"] = content_frame_samples[0]

        return log
    
    @torch.no_grad()
    def infer_content_frame(self, clip_5d, ddim_steps=50, eta=0.0, mode="extract"):
        """
        From a video clip [B,C,L,H,W], return a content frame in pixel space.
        - mode="extract": use AE heads to extract (GT-like content).
        - mode="sample": use diffusion sampler to sample content latent and decode.
        """
        self.eval()
        x = clip_5d.to(self.device)

        if mode == "extract":
            z_seq, _ = self.first_stage_model.encode_sequence(x, sample_posterior=True)
            zc, _, aux = self.first_stage_model.decompose_content_motion(z_seq, x)
            if isinstance(aux, dict) and aux.get("x_content", None) is not None:
                return aux["x_content"]        # pixel-space content
            sf = getattr(self, "scale_factor", 1.0)
            return self.first_stage_model.decode(zc / sf)

        # mode == "sample"
        z = self.get_first_stage_encoding(self.encode_first_stage(x))
        with self.ema_scope():
            z_samples, _ = self.sample_log(cond=None, batch_size=z.size(0),
                                        ddim=True, ddim_steps=ddim_steps, eta=eta)
        return self.decode_first_stage(z_samples)