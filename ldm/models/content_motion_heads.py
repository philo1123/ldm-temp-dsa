import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class ContentHeadPickFrame(nn.Module):
    """
    Pick a single frame from the pixel-space sequence as the content frame.

    API matches ContentHeadPixel for drop-in replacement:
        forward(z_seq, x_seq) -> x_content

    Args:
        fixed_index (int or None): If not None and mode == 'fixed', always pick this 0-based index.
        choices (tuple[int]): If mode == 'choice', randomly pick from this list during training.
        mode (str): 'fixed' or 'choice'.
    """
    def __init__(self, fixed_index: int = None, choices=(3, 4), mode: str = 'choice'):
        super().__init__()
        assert mode in  ("fixed", "choice", "peak"), "mode must be 'fixed', 'choice', or 'peak'"
        self.fixed_index = fixed_index
        self.choices = tuple(int(i) for i in choices)
        self.mode = mode
        # Store the last selected index for optional logging / debugging
        self.register_buffer("last_index", torch.tensor(-1), persistent=False)
        
    @torch.no_grad()
    def _select_peak_index(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Rule-based peak frame selection for DSA:
        - vessels are dark
        """
        # x_seq: [B, C, T, H, W]
        B, C, T, H, W = x_seq.shape

        # average channels
        x = x_seq.mean(dim=1)  # [B, T, H, W]

        # candidate frames (keep all frames, including first/last)
        cand = list(range(T))
        if T == 0:
            return torch.empty((0,), dtype=torch.long, device=x_seq.device)

        # build ROI per-sample: union over time, exclude near-black background
        roi = (x > 0.03).any(dim=1)  # [B, H, W]
        roi_area = roi.flatten(1).sum(dim=1)
        # fallback: if ROI too small for a sample, use all pixels for that sample
        use_all = roi_area < 1000
        if use_all.any():
            roi = roi.clone()
            roi[use_all] = True

        # scores per sample
        scores = torch.full((B, T), -1e9, device=x_seq.device, dtype=x_seq.dtype)

        # Compute scores per sample (loop over B is simplest and stable; B is typically small)
        for b in range(B):
            roi_b = roi[b]
            for t in cand:
                xt = x[b, t]  # [H, W]

                # Dark-tail score: 1 - 5th percentile (vessels are dark)
                vals = xt[roi_b]
                if vals.numel() < 16:
                    # very degenerate case; skip
                    continue
                p = torch.quantile(vals, 0.05)
                score_dark = 1.0 - p

                # Edge score to avoid washout frames that are globally darker but lack structure
                inv = 1.0 - xt
                gy, gx = torch.gradient(inv, dim=(-2, -1))
                grad = torch.sqrt(gx * gx + gy * gy)
                score_edge = grad[roi_b].mean()

                scores[b, t] = score_dark + 0.5 * score_edge

        idx = torch.argmax(scores, dim=1).to(dtype=torch.long)  # [B]
        return idx

    def _select_index(self, length: int) -> int:
        """
        Decide which frame index to use given sequence length.
        - If mode == 'fixed' and fixed_index is provided, clamp it into [0, length-1].
        - If mode == 'choice', sample from `choices` during training; pick a deterministic
          middle one during eval for reproducibility.
        - If choices are invalid or empty, fall back to the center frame.
        """
        if self.mode == 'fixed' and self.fixed_index is not None:
            return int(max(0, min(length - 1, self.fixed_index)))

        # 'choice' mode
        valid = [i for i in self.choices if 0 <= i < length]
        if not valid:
            # fallback: center frame
            return max(0, min(length - 1, length // 2))

        if self.training:
            # Randomly choose one during training
            idx = torch.randint(low=0, high=len(valid), size=(1,)).item()
            return int(valid[idx])
        # Deterministic choice at eval time (middle of the valid set)
        return valid[len(valid) // 2]

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            z_seq: [B, D, T, H_l, W_l] (not used; kept for API compatibility)
            x_seq: [B, C, T, H, W]
        Returns:
            x_content: [B, C, H, W]
        """
        assert x_seq.ndim == 5, f"Expected [B, C, T, H, W], got {x_seq.shape}"
        B, C, T, H, W = x_seq.shape
        if self.mode == "peak":
            idx = self._select_peak_index(x_seq)  # [B]
            # store last indices for debugging (only meaningful for small B)
            self.last_index = idx.detach()
            # gather per-sample along time dimension
            x_btchw = x_seq.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
            out = x_btchw[torch.arange(B, device=x_seq.device), idx]  # [B, C, H, W]
            return out
        else:
            idx = self._select_index(T)
            self.last_index = torch.tensor(idx, device=x_seq.device)
            return x_seq[:, :, idx, :, :]
class ContentHeadPixel(nn.Module):
    """
    Compute per-pixel, per-frame weights (alpha) in pixel space using the latent sequence.

    Inputs:
      - z_seq: [B, D, T, H_l, W_l]  (latent sequence per frame from the VAE encoder)
      - x    : [B, C, T, H,  W ]    (raw input frames in pixel space)

    Outputs:
      - z_c_pixel: [B, C, H, W]      pixel-space content frame = sum_t alpha_t * x_t
      - alpha     : [B, T, C, H, W]  per-pixel temporal weights (softmax over T)
    """
    def __init__(
        self,
        latent_channels: int,
        out_channels: int,
        hidden: int = None,
        share_alpha_across_channels: bool = True,
    ):
        super().__init__()
        self.share_alpha_across_channels = share_alpha_across_channels
        if hidden is None:
            hidden = 32

        # If sharing alpha across channels, predict a single logit map and broadcast to C.
        k_out = 1 if share_alpha_across_channels else out_channels

        self.logit_head = nn.Sequential(
            nn.Conv2d(latent_channels, hidden, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, k_out, kernel_size=1)
        )

    def forward(self, z_seq: torch.Tensor, x: torch.Tensor):
        """
        z_seq: [B, D, T, H_l, W_l]
        x    : [B, C, T, H,  W]
        """
        B, D, T, H_l, W_l = z_seq.shape
        Bx, C, Tx, H, W   = x.shape
        assert B == Bx and T == Tx, "z_seq and x must agree on batch/time"

        # 1) Latent -> per-frame logits at latent resolution (time folded into batch)
        z_flat = rearrange(z_seq, 'b d t h w -> (b t) d h w')        # [(B*T), D, H_l, W_l]
        logits_lat = self.logit_head(z_flat)                          # [(B*T), k_out, H_l, W_l]

        # 2) Upsample logits to pixel resolution for per-pixel softmax over time
        logits_px = F.interpolate(logits_lat, size=(H, W), mode='bilinear', align_corners=False)
        logits = rearrange(logits_px, '(b t) k h w -> b t k h w', b=B, t=T)  # [B, T, k_out, H, W]

        if self.share_alpha_across_channels:
            # Broadcast the shared temporal weights to C channels
            logits = logits.expand(-1, -1, C, -1, -1)                 # [B, T, C, H, W]

        # 3) Numerically stable softmax along time (T)
        logits = logits - logits.max(dim=1, keepdim=True)[0]
        alpha  = F.softmax(logits, dim=1)                             # [B, T, C, H, W]

        # 4) Aggregate pixels over time using alpha
        x_btchw   = rearrange(x, 'b c t h w -> b t c h w')            # [B, T, C, H, W]
        z_c_pixel = (alpha * x_btchw).sum(dim=1)                      # [B, C, H, W]

        return z_c_pixel

class ContentHead(nn.Module):
    """
    ContentHead module to extract content latent from a sequence of frames.
    Input:
        z: Tensor of shape [B, C, L, H, W]
    Output:
        z_content: Tensor of shape [B, C, H, W]
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # 1x1 Conv2d to output scalar weight per pixel per frame
        self.weight_conv = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, z):
        B, C, L, H, W = z.shape
        # flatten batch and time into batch dimension
        z_flat = rearrange(z, 'b c l h w -> (b l) c h w')
        scores_flat = self.weight_conv(z_flat)   # [(B*L),C,H,W]
        scores = rearrange(scores_flat, '(b l) c h w -> b l c h w', b=B, l=L)
        alpha = F.softmax(scores, dim=1)   # over L
        z_t = z.permute(0,2,1,3,4)               # [B,L,C,H,W]
        z_content = (alpha * z_t).sum(dim=1)     # [B,C,H,W]
        return z_content

class MotionHead(nn.Module):
    """
    MotionHead for content-motion decomposition with lightweight normalization.
    Provides explicit pre/post projection functions for x (width) and y (height) motion.
    - pre_project_xt/yt: extract motion features (project sequence latents to motion dim)
    - post_project_xt/yt: map motion features back to content latent dim

    Inputs:
        z: Tensor of shape [B, C, L, H, W]
    """
    def __init__(self, channels, proj_dim):
        super().__init__()
        self.channels = channels
        self.proj_dim = proj_dim

        # --- helpers ---
        def make_gn(nc: int) -> nn.GroupNorm:
            # choose up to 32 groups, ensure divisibility; fall back to LayerNorm-like (1 group)
            g = min(32, nc)
            while nc % g != 0 and g > 1:
                g -= 1
            return nn.GroupNorm(g, nc)

        # Pre-projection: GN -> 1x1 Conv -> SiLU -> GN
        self.pre_xt = nn.Sequential(
            make_gn(channels),
            nn.Conv2d(channels, proj_dim, kernel_size=1, bias=True),
            nn.SiLU(),
            make_gn(proj_dim),
        )
        self.pre_yt = nn.Sequential(
            make_gn(channels),
            nn.Conv2d(channels, proj_dim, kernel_size=1, bias=True),
            nn.SiLU(),
            make_gn(proj_dim),
        )

        # Post-projection: GN -> 1x1 Conv (kept linear to not distort reconstruction sum)
        self.post_xt = nn.Sequential(
            make_gn(proj_dim),
            nn.Conv2d(proj_dim, channels, kernel_size=1, bias=True),
        )
        self.post_yt = nn.Sequential(
            make_gn(proj_dim),
            nn.Conv2d(proj_dim, channels, kernel_size=1, bias=True),
        )


    def pre_project_xt(self, z: torch.Tensor) -> torch.Tensor:
        """
        Motion x-feature: average over height dimension, then normalized 1x1 projection.
        Input:
            z: [B, C, L, H, W]
        Output:
            h_xt: [B, proj_dim, L, W]
        """
        # average over height -> [B, C, L, W]
        R_w = z.mean(dim=-2)
        # treat (L, W) as (H_in, W_in) for Conv2d
        return self.pre_xt(R_w)

    def pre_project_yt(self, z: torch.Tensor) -> torch.Tensor:
        """
        Motion y-feature: average over width dimension, then normalized 1x1 projection.
        Input:
            z: [B, C, L, H, W]
        Output:
            h_yt: [B, proj_dim, L, H]
        """
        # average over width -> [B, C, L, H]
        R_h = z.mean(dim=-1)
        # treat (L, H) as (H_in, W_in) for Conv2d
        return self.pre_yt(R_h)

    def post_project_xt(self, h_xt: torch.Tensor) -> torch.Tensor:
        """
        Project motion feature x (width) back to content latent dim.
        Input:  h_xt: [B, proj_dim, L, W]
        Output:        [B, channels, L, W]
        """
        return self.post_xt(h_xt)

    def post_project_yt(self, h_yt: torch.Tensor) -> torch.Tensor:
        """
        Project motion feature y (height) back to content latent dim.
        Input:  h_yt: [B, proj_dim, L, H]
        Output:        [B, channels, L, H]
        """
        return self.post_yt(h_yt)

class MotionResidualHead(nn.Module):
    """
    Identity residual head working in latent space.
    It returns r = z_seq - expand(z_content) with optional tanh gating and learnable scale.
    Shapes:
      z_seq:     [B, C, T, H, W]
      z_content: [B, C, H, W]  (content latent per clip)
      r:         [B, C, T, H, W]
    """
    def __init__(self, use_tanh: bool = False, tanh_div: float = 1.0, learnable_scale: bool = False):
        super().__init__()
        self.use_tanh = bool(use_tanh)
        self.tanh_div = float(tanh_div)
        self.gamma = nn.Parameter(torch.tensor(1.0)) if learnable_scale else None

    def encode(self, z_seq: torch.Tensor, z_content: torch.Tensor) -> torch.Tensor:
        # Broadcast content along time
        zc_full = z_content.unsqueeze(2).expand_as(z_seq)  # [B,C,T,H,W]
        r = z_seq - zc_full
        if self.gamma is not None:
            r = self.gamma * r
        if self.use_tanh:
            # optional gentle squashing to stabilize training (keep it off by default)
            r = torch.tanh(r / self.tanh_div)
        return r

    def decode(self, r: torch.Tensor) -> torch.Tensor:
        # Identity — decoder side just returns residual as-is
        return r
    
# =========================
# Flow-like motion head
# =========================
class MotionFlowHead(nn.Module):
    """
    Flow-like motion head that compresses a latent sequence into a compact motion code
    and decodes it into per-frame flow fields to warp the content latent.

    encode(z_seq, z_content) -> motion code m ∈ R[B, P, T, H/s, W/s]
    decode(m, z_content)     -> dict with:
        - z_hat : [B, C, T, H, W]  reconstructed latent sequence
        - flow  : [B, 2, T, H, W]  per-frame (u,v) in pixels
        - resid : [B, C, T, H, W]  optional refinement residual (may be None)

    Notes
    -----
    - We keep the module lightweight on purpose. The intent is to move temporal
      modeling burden into a small 3D decoder that produces smooth flow, while
      the diffusion model operates in the low-dim motion-code space (m).
    - All ops are done in latent space (C,H,W from VAE).
    """

    def __init__(
        self,
        c_lat: int,              # latent channels C
        code_dim: int = 8,       # P, motion code channels
        ds_factor: int = 4,      # s, spatial downsample factor for code (H/s, W/s)
        use_residual: bool = False,
        flow_scale_alpha: float = 2.0,  # controls initial max displacement in pixels after tanh
        zero_init: bool = True,
    ):
        super().__init__()
        assert ds_factor in (1, 2, 4, 8), "ds_factor must be 1/2/4/8"

        self.c_lat = int(c_lat)
        self.code_dim = int(code_dim)
        self.ds = int(ds_factor)
        self.use_residual = bool(use_residual)
        self.flow_scale_alpha = float(flow_scale_alpha)

        # --------- Encoder: z_seq -> motion code m (B,P,T,H/s,W/s) ----------
        # Input is concatenated residual to stabilize: (z_seq - z_c_full)
        enc_in = c_lat
        self.enc = nn.Sequential(
            # depthwise 3D conv to capture local temporal cues cheaply
            nn.Conv3d(enc_in, enc_in, kernel_size=3, padding=1, groups=enc_in),
            nn.SiLU(inplace=True),
            nn.Conv3d(enc_in, code_dim, kernel_size=1),
            nn.SiLU(inplace=True),
        )
        # Spatial downsample to (H/s, W/s)
        self.enc_pool = nn.AvgPool3d(kernel_size=(1, self.ds, self.ds), stride=(1, self.ds, self.ds))

        # --------- Decoder: m -> (flow[, resid]) at full resolution ----------
        dec_hidden = max(2 * code_dim, 16)

        # upsample to full spatial size via simple interp + conv
        self.dec_up1 = nn.Conv3d(code_dim, dec_hidden, kernel_size=3, padding=1)
        self.dec_up2 = nn.Conv3d(dec_hidden, dec_hidden, kernel_size=3, padding=1)
        self.dec_act = nn.SiLU(inplace=True)
        
        self.dec = nn.Sequential(
            self.dec_up1,
            self.dec_act,
            self.dec_up2,
            self.dec_act,
        )

        # heads
        self.flow_head = nn.Conv3d(dec_hidden, 2, kernel_size=3, padding=1)              # (u,v)
        self.res_head  = nn.Conv3d(dec_hidden, c_lat, kernel_size=3, padding=1) if use_residual else None

        if zero_init:
            nn.init.zeros_(self.flow_head.weight); nn.init.zeros_(self.flow_head.bias)
            if self.res_head is not None:
                nn.init.zeros_(self.res_head.weight); nn.init.zeros_(self.res_head.bias)

    @staticmethod
    def _build_content_residual(z_seq: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        # z_seq: [B,C,T,H,W], z_c: [B,C,H,W] -> residual per frame
        return z_seq - z_c.unsqueeze(2).expand_as(z_seq)

    def encode(self, z_seq: torch.Tensor, z_content: torch.Tensor) -> torch.Tensor:
        """
        Compress a latent sequence into a motion code.
        Inputs:
            z_seq     : [B, C, T, H, W]
            z_content : [B, C, H, W]
        Output:
            m         : [B, P, T, H/s, W/s]
        """
        assert z_seq.ndim == 5 and z_content.ndim == 4, "shapes: z_seq[B,C,T,H,W], z_content[B,C,H,W]"
        # use residual to stabilize encoder
        res = self._build_content_residual(z_seq, z_content)  # [B,C,T,H,W]
        m = self.enc(res)                                     # [B,P,T,H,W]
        if self.ds > 1:
            m = self.enc_pool(m)                              # [B,P,T,H/s,W/s]
        return m

    def _upsample_code(self, m: torch.Tensor, H: int, W: int) -> torch.Tensor:
        # m: [B,P,T,h,w] -> upsample spatial to (H,W)
        return F.interpolate(m, size=(m.shape[2], H, W), mode="trilinear", align_corners=False)

    @staticmethod
    def _flow_to_grid(flow_px: torch.Tensor, H: int, W: int, scale: float = 1.0) -> torch.Tensor:
        """
        Convert pixel displacement (u,v) to normalized sampling grid in [-1,1].
        flow_px: [B, 2, H, W]
        return  : [B, H, W, 2]
        """
        B = flow_px.shape[0]
        u = flow_px[:, 0] / ((W - 1) / 2.0) * scale
        v = flow_px[:, 1] / ((H - 1) / 2.0) * scale
        ys = torch.linspace(-1, 1, H, device=flow_px.device, dtype=flow_px.dtype)
        xs = torch.linspace(-1, 1, W, device=flow_px.device, dtype=flow_px.dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H,W]
        grid = torch.stack([grid_x.expand(B, -1, -1) + u, grid_y.expand(B, -1, -1) + v], dim=-1)
        return grid

    def _warp_sequence(self, z_c: torch.Tensor, flow: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Warp content latent into a sequence using per-frame flow.
        z_c  : [B,C,H,W]
        flow : [B,2,T,H,W]  (pixel units)
        alpha: scalar multiply after tanh to limit magnitude
        return: [B,C,T,H,W]
        """
        B, C, H, W = z_c.shape
        T = flow.shape[2]
        z_pred = []
        for t in range(T):
            flow_t = torch.tanh(flow[:, :, t]) * alpha        # stabilize magnitude
            grid_t = self._flow_to_grid(flow_t, H, W, scale=1.0)   # [-1,1] grid
            z_warp_t = F.grid_sample(z_c, grid_t, mode='bilinear', padding_mode='border', align_corners=False)
            z_pred.append(z_warp_t)
        return torch.stack(z_pred, dim=2)  # [B,C,T,H,W]
 
    def decode(self, m: torch.Tensor, z_content: torch.Tensor) -> dict:
        """
        Decode motion code into flow (and optional residual), warp content, and compose the sequence.
        Inputs:
            m         : [B, P, T, h, w]
            z_content : [B, C, H, W]
        Returns:
            dict with keys:
                z_hat : [B,C,T,H,W]
                flow  : [B,2,T,H,W]
                resid : [B,C,T,H,W] or None
        """
        B, P, T, h, w = m.shape
        H, W = z_content.shape[-2:]

        # upsample spatially to latent resolution
        m_up = self._upsample_code(m, H, W)           # [B,P,T,H,W]
        # h1 = self.dec_act(self.dec_up1(m_up))
        # h2 = self.dec_act(self.dec_up2(h1))
        h = self.dec(m_up)

        flow = self.flow_head(h)                     # [B,2,T,H,W]
        z_warp = self._warp_sequence(z_content, flow, alpha=self.flow_scale_alpha)  # [B,C,T,H,W]

        resid = None
        if self.res_head is not None:
            resid = torch.tanh(self.res_head(h))     # keep residual small
            z_hat = z_warp + resid
        else:
            z_hat = z_warp

        return {"z_hat": z_hat, "flow": flow, "resid": resid}