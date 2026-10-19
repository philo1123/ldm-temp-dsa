import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf

from ldm.models.autoencoder import AutoencoderKL
from ldm.models.content_motion_heads import ContentHead, ContentHeadPixel, ContentHeadPickFrame,\
    MotionHead, MotionResidualHead, MotionFlowHead
from copy import deepcopy

class PosteriorSeqWrapper:
    """
    Lightweight wrapper to present a sequence of DiagonalGaussianDistribution
    objects (one per frame) as a single object with a .kl() method that returns
    a vector of length (B*L), matching the flattened per-frame reconstruction loss.
    """
    def __init__(self, posterior_list):
        self.posterior_list = posterior_list  # list of length L; each p.kl() returns [B]

    def kl(self):
        # Each p.kl(): [B]. Stack over time -> [B, L], then flatten to [B*L]
        kls = [p.kl() for p in self.posterior_list]  # list of [B]
        kls = torch.stack(kls, dim=1)               # [B, L]
        return kls.reshape(-1)                       # [B*L]


class AutoencoderKL_CMHeads(AutoencoderKL):
    """
    AutoencoderKL with ContentHead and MotionHead for video content-motion decomposition.
    Optionally supports a pixel-mode pathway that aggregates content in pixel space
    and then encodes to latent for fusion with motion in latent space.
    """
    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        motion_proj_dim=16,
        motion_head_params: dict = None,
        freeze_backbone: bool = True,
        log_extras: bool = True,
        pixel_mode: bool = True,
        content_pixel_channels: int = 1,
        content_head_kind: str = "pickframe",
        pickframe_mode: str = 'choice',
        pickframe_index: int = 3,
        pickframe_choices: tuple = (3, 4),
        # motion head kind: "separable_xy" (CMD origin Dit) or "residual" (UNet) or "flow"
        motion_head_kind: str = "separable_xy",
        # params for MotionResidualHead
        residual_use_tanh: bool = False,
        residual_tanh_div: float = 1.0,
        residual_learnable_scale: bool = False,
        **kwargs
    ):
        ckpt_path = kwargs.pop("ckpt_path", None)
        super().__init__(ddconfig, lossconfig, embed_dim, **kwargs)
        self.freeze_backbone = freeze_backbone
        self.pixel_mode = pixel_mode
        self.content_pixel_channels = content_pixel_channels
        # Determine content head kind; keep backward compatibility with `pixel_mode`
        if content_head_kind is None:
            content_head_kind = "pixel" if pixel_mode else "latent"
        assert content_head_kind in ("pixel", "latent", "pickframe"), \
            f"Unsupported content_head_kind: {content_head_kind}"
        self.content_head_kind = content_head_kind
        self.motion_head_kind = motion_head_kind
        
        # Optional per-kind parameter dict, e.g.
        # motion_head_params = {
        #   "flow": {"code_dim": 16, "ds_factor": 2, "use_residual": False, "flow_scale_alpha": 3.0},
        #   "residual": {"use_tanh": True, "tanh_div": 2.0, "learnable_scale": False},
        #   "separable_xy": {"proj_dim": 32}
        # }
        motion_head_params = motion_head_params or {}
        kind_params = (OmegaConf.to_container(motion_head_params, resolve=True) if OmegaConf.is_config(motion_head_params) else (motion_head_params or {})).get(self.motion_head_kind, {}) or {}
        
        if self.motion_head_kind == "residual":
            self.motion_head = MotionResidualHead(
                use_tanh=residual_use_tanh,
                tanh_div=residual_tanh_div,
                learnable_scale=residual_learnable_scale
            )
        elif self.motion_head_kind == "flow":
            self.motion_head = MotionFlowHead(
                c_lat=embed_dim,
                code_dim=kind_params.get("code_dim", 8),
                ds_factor=kind_params.get("ds_factor", 4),
                use_residual=kind_params.get("use_residual", False),
                flow_scale_alpha=kind_params.get("flow_scale_alpha", 2.0),
                zero_init=kind_params.get("zero_init", True),
            )
        else:
            self.motion_head = MotionHead(channels=embed_dim, proj_dim=motion_proj_dim)
        
        # Select content head implementation based on kind
        if self.content_head_kind == "pixel":
            self.content_head = ContentHeadPixel(
                latent_channels=embed_dim,
                out_channels=content_pixel_channels
            )
        elif self.content_head_kind == "pickframe":
            self.content_head = ContentHeadPickFrame(
                fixed_index=pickframe_index,
                choices=pickframe_choices,
                mode=pickframe_mode
            )
        else:  # "latent"
            self.content_head = ContentHead(channels=embed_dim)

        # Optionally freeze VAE backbone
        if self.freeze_backbone:
            self._freeze_vae_backbone()
            if hasattr(self.loss, "disc_factor"):
                self.loss.disc_factor = 0.0
                
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])
            
        self.log_extras = log_extras
        # Convenience flag: whether the chosen content head consumes pixel-space sequence
        self.uses_pixel_content = (self.content_head_kind in ("pixel", "pickframe"))

    def _freeze_vae_backbone(self):
        """
        Freeze all VAE backbone parameters (encoder, decoder, quantizers),
        leaving only content_head and motion_head trainable.
        """
        print("Freezing VAE backbone parameters, leaving only content and motion heads trainable.")
        for name, param in self.named_parameters():
            if name.startswith("content_head") or name.startswith("motion_head"):
                param.requires_grad = True
            else:
                param.requires_grad = False

    def get_input(self, batch, k):
        """
        For grayscale (1 channel), no extra expansion or permutation.
        """
        x = batch[k]
        x = x.float()
        return x
    
    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if x.ndim == 5:
            # [B, C, L, H, W] -> [B*L, C, H, W]
            x_flat = rearrange(x, 'b c l h w -> (b l) c h w')
        else:
            x_flat = x
        log["inputs"] = x_flat
        
        if not only_inputs:
            xrec, aux = self(x)
            if xrec.ndim == 5:
                xrec_flat = rearrange(xrec, 'b c l h w -> (b l) c h w')
            else:
                xrec_flat = xrec
            if x_flat.shape[1] > 3:
                assert x_flat.shape[1] > 3
                x_flat = self.to_rgb(x_flat)
                xrec_flat = self.to_rgb(xrec_flat)
            log["reconstructions"] = xrec_flat
            if self.uses_pixel_content:
                # show the first content frame
                log["content_frame_pixel"] = aux["x_content"][0:1]
        return log

    def _log_extras(self, aux: dict, prefix: str = "train", on_step: bool = True, on_epoch: bool = True):
        """Log auxiliary diagnostics in a single place.
        Expects keys possibly present in aux: {"alpha", "z_content", "h_xt", "h_yt", "z_seq_recon", "posteriors", "z_content_full", "h_xt_full", "h_yt_full", "x_content"}.
        - alpha: [B, L, C, H, W]
        - z_content: [B, C, H, W]
        - h_xt: [B, proj_dim, L, W]
        - h_yt: [B, proj_dim, L, H]
        - z_seq_recon: [B, C, L, H, W]
        - posteriors: PosteriorSeqWrapper with .kl() -> [B*L]
        - z_content_full: [B, C, L, H, W]
        - h_xt_full: [B, C, L, H, W]
        - h_yt_full: [B, C, L, H, W]
        - x_content: [B, C, H, W]
        """
        if aux is None:
            return

        # alpha stats
        alpha = aux.get("alpha", None)
        if alpha is not None:
            eps = 1e-8
            alpha_mean = alpha.mean().detach()
            # entropy over time dimension (dim=1)
            alpha_safe = alpha.clamp_min(eps)
            alpha_entropy = (-(alpha_safe * alpha_safe.log()).sum(dim=1).mean()).detach()
            self.log(f"{prefix}/alpha_mean", alpha_mean, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}/alpha_entropy", alpha_entropy, on_step=on_step, on_epoch=on_epoch)

        # z_content stats
        z_content = aux.get("z_content", None)
        if z_content is not None:
            self.log(f"{prefix}/z_content_mean", z_content.mean().detach(), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}/z_content_std",  z_content.std().detach(),  on_step=on_step, on_epoch=on_epoch)

        # x_content stats (pixel mode)
        x_content = aux.get("x_content", None)
        if x_content is not None:
            self.log(f"{prefix}/x_content_mean", x_content.mean().detach(), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}/x_content_std", x_content.std().detach(), on_step=on_step, on_epoch=on_epoch)

        # motion magnitudes
        h_xt = aux.get("h_xt", None)
        if h_xt is not None:
            self.log(f"{prefix}/h_xt_l1", h_xt.abs().mean().detach(), on_step=on_step, on_epoch=on_epoch)
        h_yt = aux.get("h_yt", None)
        if h_yt is not None:
            self.log(f"{prefix}/h_yt_l1", h_yt.abs().mean().detach(), on_step=on_step, on_epoch=on_epoch)

        # latent recon std (pre-decode)
        z_seq_recon = aux.get("z_seq_recon", None)
        if z_seq_recon is not None:
            self.log(f"{prefix}/z_seq_recon_std", z_seq_recon.std().detach(), on_step=on_step, on_epoch=on_epoch)

        # raw KL for reference
        posteriors = aux.get("posteriors", None)
        if posteriors is not None:
            try:
                kl_raw = posteriors.kl().mean().detach()
                self.log(f"{prefix}/kl_raw", kl_raw, on_step=on_step, on_epoch=on_epoch)
            except Exception:
                pass

        z_content_full = aux.get("z_content_full", None)
        h_xt_full = aux.get("h_xt_full", None)
        h_yt_full = aux.get("h_yt_full", None)
        z_seq = aux.get("z_seq", None)
        z_seq_recon = aux.get("z_seq_recon", None)

        if (z_content_full is not None) and (h_xt_full is not None) and (h_yt_full is not None):
            motion_full = (h_xt_full + h_yt_full)
            zc_l2 = torch.linalg.vector_norm(z_content_full).detach()
            motion_l2 = torch.linalg.vector_norm(motion_full).detach()
            self.log(f"{prefix}/z_content_full_l2", zc_l2, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}/motion_full_l2", motion_l2, on_step=on_step, on_epoch=on_epoch)

        if (z_seq is not None) and (z_seq_recon is not None):
            latent_residual = (z_seq_recon - z_seq)
            residual_l2 = torch.linalg.vector_norm(latent_residual).detach()
            self.log(f"{prefix}/latent_residual_l2", residual_l2, on_step=on_step, on_epoch=on_epoch)
            
        # --- Explicitly log flow and resid diagnostics if present ---
        flow = aux.get("flow", None)
        if flow is not None:
            # flow: [B, 2, T, H, W]
            self.log(f"{prefix}/flow_mean", flow.mean().detach(), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}/flow_std", flow.std().detach(), on_step=on_step, on_epoch=on_epoch)
            # flow magnitude
            flow_mag = torch.linalg.vector_norm(flow, ord=2, dim=1)  # [B, T, H, W]
            self.log(f"{prefix}/flow_mag_mean", flow_mag.mean().detach(), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}/flow_mag_max", flow_mag.max().detach(), on_step=on_step, on_epoch=on_epoch)
            # --- motion code (m) diagnostics (flow path) ---
            for k in [
                "m_mean","m_std","m_abs_mean"
            ]:
                v = aux.get(k, None)
                if v is not None:
                    self.log(f"{prefix}/{k}", v.detach(), on_step=on_step, on_epoch=on_epoch)
        resid = aux.get("resid", None)
        if resid is not None:
            # resid: [B, C, T, H, W]
            self.log(f"{prefix}/resid_mean", resid.mean().detach(), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}/resid_std", resid.std().detach(), on_step=on_step, on_epoch=on_epoch)
            resid_l1 = resid.abs().mean().detach()
            resid_l2 = torch.linalg.vector_norm(resid).detach()
            self.log(f"{prefix}/resid_l1", resid_l1, on_step=on_step, on_epoch=on_epoch)
            self.log(f"{prefix}/resid_l2", resid_l2, on_step=on_step, on_epoch=on_epoch)


    def encode_sequence(self, x_seq, sample_posterior=True):
        """
        Encode a video sequence frame-by-frame.
        Args:
            x_seq: [B, C, L, H, W]
        Returns:
            z_seq: [B, C, L, H, W]
            posterior_list: list of DiagonalGaussianDistribution for each frame
        """
        B, C, L, H, W = x_seq.shape
        z_list = []
        posterior_list = []
        for t in range(L):
            posterior = self.encode(x_seq[:, :, t, :, :])
            z_t = posterior.sample() if sample_posterior else posterior.mode()
            z_list.append(z_t)
            posterior_list.append(posterior)
        # stack directly on channel-first sequence dimension
        z_seq = torch.stack(z_list, dim=2)  # [B, D, L, H, W]
        return z_seq, posterior_list

    def decode_sequence(self, z_seq):
        """
        Decode a sequence of latent codes frame-by-frame.
        Args:
            z_seq: [B, C, L, H, W]
        Returns:
            x_rec: [B, C, L, H, W]
        """
        B, D, L, H, W = z_seq.shape
        x_list = []
        for t in range(L):
            x_t = self.decode(z_seq[:, :, t, :, :])  # [B, C, H, W]
            x_list.append(x_t)
        x_rec = torch.stack(x_list, dim=2)  # [B, C, L, H, W]
        return x_rec

    def decompose_content_motion(self, z_seq, x_seq=None):
        """
        Decompose sequence latent into content and motion components.
        Args:
            z_seq: [B, C, L, H, W]
            x_seq: [B, C, L, H, W], required if content_head_kind is 'pixel' or 'pickframe'
        Returns:
            z_content: [B, C, H, W]
            z_m: [B, proj_dim, L, W+H]
            aux: dict with extras for regularization
        When pixel_mode is True, content is computed in pixel space via content_head_pixel,
        then encoded to latent space by the VAE encoder.
        """
        aux = {}
        if self.uses_pixel_content:
            assert x_seq is not None, "x_seq must be provided when content_head_kind is 'pixel' or 'pickframe'"
            # Pixel-mode or pickframe: compute pixel-space content via the content head, then encode to latent
            x_content = self.content_head(x_seq)  # [B, C, H, W]
            posterior_c = self.encode(x_content)
            z_content = posterior_c.mode()
            aux["x_content"] = x_content
        else:
            # Latent-mode: content head operates directly on latent sequence
            z_content = self.content_head(z_seq)
        # --- residual path ---
        if isinstance(self.motion_head, MotionResidualHead):
            r = self.motion_head.encode(z_seq, z_content)    # [B,C,T,H,W]
            aux["z_m_kind"] = "residual"
            aux["z_m_shape"] = tuple(r.shape)
            return z_content, r, aux
        # --- flow path ---
        elif isinstance(self.motion_head, MotionFlowHead):
            m = self.motion_head.encode(z_seq, z_content)  # [B,P,T,H/s,W/s]
            aux["z_m_kind"] = "flow"
            aux["z_m_shape"] = tuple(m.shape)
            # --- motion code (m) stats: helpful to diagnose diffusion difficulty ---
            with torch.no_grad():
                m_det = m.detach()
                m_abs = m_det.abs()
                aux["m_mean"] = m_det.mean()
                aux["m_std"] = m_det.std()
                aux["m_abs_mean"] = m_abs.mean()
            return z_content, m, aux
        # --- CMD paper path ---
        else:
            h_xt = self.motion_head.pre_project_xt(z_seq)  # [B, proj_dim, L, W]
            h_yt = self.motion_head.pre_project_yt(z_seq)  # [B, proj_dim, L, H]
            z_m = torch.tanh(torch.cat([h_xt, h_yt], dim=-1))  # [B, proj_dim, L, W+H]
            aux["h_xt"] = h_xt
            aux["h_yt"] = h_yt
            aux["z_m_kind"] = "separable_xy"
            return z_content, z_m, aux
    
    def reconstruct_sequence(self, z_content, z_m, aux: dict = None):
        """
        Reconstruct the sequence latent from content and merged motion latent.
        Args:
            z_content: [B, C, H, W]
            z_m: [B, proj_dim, L, W+H]
            aux: optional dict to be populated with diagnostics (e.g., flow/resid)
        Returns:
            (z_seq_recon, z_content_full, h_xt_full, h_yt_full)
        """
        B, C, H, W = z_content.shape

        # residual path: z_seq = z_c + r
        if isinstance(self.motion_head, MotionResidualHead) and z_m.ndim == 5:
            r_hat = self.motion_head.decode(z_m)                 # identity
            T = r_hat.shape[2]
            z_content_full = z_content.unsqueeze(2).expand(B, C, T, H, W)
            z_seq_recon = z_content_full + r_hat
            if aux is not None:
                aux["resid"] = r_hat
            return z_seq_recon, z_content_full
        elif isinstance(self.motion_head, MotionFlowHead):
            # flow-based reconstruction
            out = self.motion_head.decode(z_m, z_content)
            z_seq_recon = out["z_hat"]
            if aux is not None:
                aux["flow"] = out.get("flow", None)
                aux["resid"] = out.get("resid", None)
            z_content_full = z_content.unsqueeze(2).expand_as(z_seq_recon)
            return z_seq_recon, z_content_full
        # z_m: [B, proj_dim, L, W+H]
        B, proj_dim, L, merged = z_m.shape
        W = z_content.shape[-1]
        H = z_content.shape[-2]
        # split merged spatial dimension only (no channel slicing)
        h_xt = z_m[:, :, :, :W]    # [B, proj_dim, L, W]
        h_yt = z_m[:, :, :, W:]    # [B, proj_dim, L, H]
        # project back to content channels
        h_xt_post = self.motion_head.post_project_xt(h_xt)  # [B, C, L, W]
        h_yt_post = self.motion_head.post_project_yt(h_yt)  # [B, C, L, H]
        # expand motion to full spatial dims [B, C, L, H, W]
        h_xt_full = h_xt_post.unsqueeze(-2).expand(-1, -1, -1, H, -1)
        h_yt_full = h_yt_post.unsqueeze(-1).expand(-1, -1, -1, -1, W)
        # expand content along time: [B, C, H, W] -> [B, C, L, H, W]
        z_content_full = z_content.unsqueeze(2).expand(-1, -1, L, -1, -1)
        # sum to reconstruct sequence latent
        z_seq_recon = z_content_full + h_xt_full + h_yt_full
        if aux is not None:
            aux["h_xt_full"] = h_xt_full
            aux["h_yt_full"] = h_yt_full
        return z_seq_recon, z_content_full


    def forward(self, input, sample_posterior=True):
        """
        Args:
            input: [B, C, L, H, W] for video, or [B, C, H, W] for single frame
            sample_posterior: whether to sample or use mode for latent
        Returns:
            For video:
                x_rec: [B, C, L, H, W]
                z_content: [B, D, H, W]
            For single frame:
                dec: [B, C, H, W]
                posterior: DiagonalGaussianDistribution
        """
        if input.ndim == 5:
            z_seq, posterior_list = self.encode_sequence(input, sample_posterior=sample_posterior)
            z_content, z_m, aux = self.decompose_content_motion(z_seq, x_seq=input if self.pixel_mode else None)
            # attach posterior sequence wrapper for KL computation aligned to flattened frames
            aux["posteriors"] = PosteriorSeqWrapper(posterior_list)
            z_seq_recon, z_content_full = self.reconstruct_sequence(z_content, z_m, aux=aux)
            aux["z_seq_recon"] = z_seq_recon
            aux["z_content_full"] = z_content_full
            x_rec = self.decode_sequence(z_seq_recon)
            aux["z_seq"] = z_seq
            return x_rec, aux
        else:
            # Single image mode (original)
            dec, posterior = super().forward(input, sample_posterior=sample_posterior)  # dec:[B,C,H,W]
            aux = {"posteriors": PosteriorSeqWrapper([posterior])}  # list len=1，.kl()->[B*1]
            return dec, aux
        
    def training_step(self, batch, batch_idx, optimizer_idx=0):
        """
        Modified for video autoencoder: no posterior, only reconstructions.
        All tensors are [B, C, L, H, W].
        """
        # get video input [B, C, L, H, W]
        inputs = self.get_input(batch, self.image_key)
        # forward through autoencoder
        x_rec, aux = self(inputs)

        # flatten temporal dimension into batch for 2D losses
        B, C, L, H, W = inputs.shape
        inputs_flat = rearrange(inputs, 'b c l h w -> (b l) c h w')
        rec_flat    = rearrange(x_rec,  'b c l h w -> (b l) c h w')

        # compute reconstruction + perceptual losses on flattened frames
        posteriors = aux["posteriors"]
        aeloss, log_dict_ae = self.loss(
            inputs_flat, rec_flat, posteriors, optimizer_idx, self.global_step,
            last_layer=self.get_last_layer(), split="train"
        )
            
        if self.log_extras:
            self._log_extras(aux, prefix="train", on_step=True, on_epoch=False)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        return aeloss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Validation for video autoencoder: compute reconstruction + regularizers.
        All tensors are [B, C, L, H, W].
        """
        inputs = self.get_input(batch, self.image_key)                  # [B,C,L,H,W]
        x_rec, aux = self(inputs)                                       # forward

        B, C, L, H, W = inputs.shape
        inputs_flat = rearrange(inputs, 'b c l h w -> (b l) c h w')
        rec_flat    = rearrange(x_rec,  'b c l h w -> (b l) c h w')

        # reconstruction loss (try with KL-aware signature first)
        posteriors = aux["posteriors"]
        aeloss, log_dict_ae = self.loss(
            inputs_flat, rec_flat, posteriors, 0, self.global_step,
            last_layer=self.get_last_layer(), split="val"
        )

        # logging
        self.log("val/aeloss", aeloss, on_step=False, on_epoch=True)
        if self.log_extras:
            self._log_extras(aux, prefix="val", on_step=False, on_epoch=True)
        self.log_dict(log_dict_ae, on_step=False, on_epoch=True)
        return aeloss

    def configure_optimizers(self):
        lr = self.learning_rate
        trainable = [p for p in self.parameters() if p.requires_grad]
        print("Trainable params:", sum(p.numel() for p in trainable))
        opt_ae = torch.optim.AdamW(trainable, lr=lr, weight_decay=0)
        return [opt_ae], []
    