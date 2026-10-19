import argparse, os, sys, glob, datetime, yaml
import csv
import heapq
from typing import Any, Dict, List, Optional, Tuple
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from frame_selector import FrameSelector

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from einops import rearrange

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    # shape: (C, H, W) -> (H, W) if single channel
    if x.shape[0] == 1:
        x = x.squeeze(0)  # (H, W)
        x = (255. * x).numpy().astype(np.uint8)
        img = Image.fromarray(x, mode='L')  # grayscale
    elif x.shape[0] == 3:
        x = x.permute(1, 2, 0).numpy()  # (H, W, C)
        x = (255. * x).astype(np.uint8)
        img = Image.fromarray(x, mode='RGB')  # rgb image
    else:
        raise ValueError(f"Unsupported channel number: {x.shape[0]}")
    return img

# Save a single sequence [C,T,H,W] into a folder as {prefix}_{sample_id:06}_tXX.png
def save_sequence_single(frames_cthw, outdir, prefix, sample_id):
    frames = frames_cthw.detach().cpu()
    C, T, H, W = frames.shape
    for t in range(T):
        img = custom_to_pil(frames[:, t])
        imgpath = os.path.join(outdir, f"{prefix}_{sample_id:06}_t{t:02}.png")
        img.save(imgpath)


def _load_selector_cfg_from_wrapper_yaml(yaml_path: str) -> Optional[Dict[str, Any]]:
    """Load selector_config block from the content wrapper YAML.

    If missing or disabled, returns None.
    """
    cfg = OmegaConf.load(yaml_path)
    sel = cfg.get("selector_config", None)
    if sel is None:
        return None
    # OmegaConf -> plain dict
    sel = OmegaConf.to_container(sel, resolve=True)
    if not isinstance(sel, dict):
        return None
    if not sel.get("enabled", False):
        return None
    return sel


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_selector_scores_csv(rows: List[Dict[str, Any]], out_path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    _ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _select_topk_global(
    selector: "FrameSelector",
    x_content: torch.Tensor,
    remaining_k: int,
    pool: List[Tuple[float, int, torch.Tensor]],
    base_candidate_id: int,
    score_rows: List[Dict[str, Any]],
    reject_dir: Optional[str] = None,
    save_all_candidates: bool = False,
) -> int:
    """Score a batch of content frames and maintain an online global top-k pool.

    Args:
      selector: FrameSelector instance.
      x_content: [B,C,H,W] content frames in [-1,1].
      remaining_k: target top-k size (final kept size).
      pool: min-heap of (score, candidate_id, frame_tensor) with max size remaining_k.
      base_candidate_id: global running id for candidates.
      score_rows: list to append CSV rows.
      reject_dir: where to optionally save rejected candidates.
      save_all_candidates: if True, save non-kept candidates to reject_dir.

    Returns:
      next_candidate_id
    """
    B = int(x_content.shape[0])
    # Score via PIL conversion to match selector expectations
    pil_imgs = [custom_to_pil(x_content[i]) for i in range(B)]
    # FrameSelector exposes `score(pil_imgs)` and returns a numpy array of scores.
    scores = selector.score(pil_imgs)

    for i in range(B):
        cid = base_candidate_id + i
        s = float(scores[i])
        # Decide if enters top-k pool
        enters = False
        if len(pool) < remaining_k:
            heapq.heappush(pool, (s, cid, x_content[i].detach().cpu()))
            enters = True
        else:
            if s > pool[0][0]:
                # Replace worst
                heapq.heapreplace(pool, (s, cid, x_content[i].detach().cpu()))
                enters = True

        score_rows.append({
            "candidate_id": cid,
            "score": s,
            "kept": int(enters),
        })

        if save_all_candidates and reject_dir is not None and not enters:
            try:
                img = pil_imgs[i].convert("RGB")
                img.save(os.path.join(reject_dir, f"candidate_{cid:06}.png"))
            except Exception:
                pass

    return base_candidate_id + B


@torch.no_grad()
def _generate_sequences_from_content(
    model,
    x_content: torch.Tensor,
    T: int,
    steps: int,
    eta_gen: float,
) -> torch.Tensor:
    """Given selected content frames [B,C,H,W], generate sequences [B,C,T,H,W]."""
    device = next(model.parameters()).device
    x_content = x_content.to(device)

    # encode content
    post = model.first_stage_model.encode(x_content)
    z_c = post.mode()  # [B,Cz,h,w]

    # build concat conditioning
    if not hasattr(model, '_last_motion_shape') or model._last_motion_shape is None:
        raise AttributeError("motion model missing _last_motion_shape. Set motion_shape: [P,Hm,Wm] in --content_wrapper YAML.")
    Hm, Wm = int(model._last_motion_shape[1]), int(model._last_motion_shape[2])

    zc_full = z_c.unsqueeze(2).expand(z_c.shape[0], z_c.shape[1], T, z_c.shape[2], z_c.shape[3])
    c = rearrange(zc_full, 'b c t h w -> (b t) c h w')
    if (c.shape[-2] != Hm) or (c.shape[-1] != Wm):
        c = torch.nn.functional.interpolate(c, size=(Hm, Wm), mode='bilinear', align_corners=False)

    bs_for_sampling = int(x_content.shape[0] * T)
    shape = model._last_motion_shape
    model._last_bt_hw = (int(x_content.shape[0]), int(T), int(Hm), int(Wm))

    with model.ema_scope("Plotting"):
        samples, _ = convsample_ddim(
            model,
            steps=steps,
            batch_size=bs_for_sampling,
            shape=shape,
            eta=eta_gen,
            conditioning=c,
        )

    gen_seq = model.reconstruct_sequence_from_latent(samples, z_c)
    return gen_seq


def custom_to_np(x):
    # saves the batch in adm style as in https://github.com/openai/guided-diffusion/blob/main/scripts/image_sample.py
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])
            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])
            else:
                print(f"Unknown format for key {k}. ")
                img = None
        except:
            img = None
        imgs[k] = img
    return imgs


@torch.no_grad()
def convsample(model, shape, return_intermediates=True,
               verbose=True,
               make_prog_row=False):


    if not make_prog_row:
        return model.p_sample_loop(None, shape,
                                   return_intermediates=return_intermediates, verbose=verbose)
    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )


@torch.no_grad()
def convsample_ddim(model, steps, batch_size, shape, eta=1.0, conditioning=None):
    """DDIM sampling in latent space. `shape` is (C,H,W) for 2D UNet motion latents."""
    ddim = DDIMSampler(model)
    samples, intermediates = ddim.sample(
        S=steps,
        batch_size=batch_size,
        shape=shape,
        conditioning=conditioning,
        eta=eta,
        verbose=False,
    )
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, content_model, B, T, custom_steps=None, eta=1.0):
    """Dataset-free sampling:
    - sample B synthetic content frames with content_model
    - encode to z_c
    - build concat conditioning over T frames
    - DDIM sample motion latent for B*T
    - reconstruct to pixel space

    Returns dict with:
      - x_content: [B,C,H,W]
      - gen_seq:  [B,C,T,H,W]
    """
    log = {}

    # Choose steps/eta (prefer CLI values, else wrapper defaults)
    steps = int(custom_steps) if custom_steps is not None else int(getattr(content_model, 'default_steps', 50))
    eta_gen = float(eta) if eta is not None else float(getattr(content_model, 'default_eta', 0.0))

    # 1) sample synthetic content frames in pixel space (expected in [-1,1])
    x_content = content_model.sample_content_frames(batch_size=B, ddim_steps=steps, eta=eta_gen)
    if not torch.is_tensor(x_content):
        raise TypeError(f"sample_content_frames must return a Tensor, got {type(x_content)}")
    x_content = x_content.to(next(model.parameters()).device)
    log['x_content'] = x_content

    # 2) encode with the motion model's first stage
    post = model.first_stage_model.encode(x_content)
    z_c = post.mode()  # [B,Cz,h,w]

    # 3) build concat conditioning for flow/residual: [(B*T), Cz, Hm, Wm]
    # Motion latent spatial comes from model._last_motion_shape
    if not hasattr(model, '_last_motion_shape') or model._last_motion_shape is None:
        raise AttributeError("motion model missing _last_motion_shape. Set motion_shape: [P,Hm,Wm] in --content_wrapper YAML.")
    Hm, Wm = int(model._last_motion_shape[1]), int(model._last_motion_shape[2])

    zc_full = z_c.unsqueeze(2).expand(z_c.shape[0], z_c.shape[1], T, z_c.shape[2], z_c.shape[3])
    c = rearrange(zc_full, 'b c t h w -> (b t) c h w')
    if (c.shape[-2] != Hm) or (c.shape[-1] != Wm):
        c = torch.nn.functional.interpolate(c, size=(Hm, Wm), mode='bilinear', align_corners=False)

    # 4) DDIM sample motion latent for B*T
    bs_for_sampling = int(B * T)
    shape = model._last_motion_shape  # (P, Hm, Wm)

    # Set _last_bt_hw so reconstruct can reshape correctly
    model._last_bt_hw = (B, T, Hm, Wm)

    with model.ema_scope("Plotting"):
        samples, _ = convsample_ddim(
            model,
            steps=steps,
            batch_size=bs_for_sampling,
            shape=shape,
            eta=eta_gen,
            conditioning=c,
        )

    # 5) reconstruct to pixel space [B,C,T,H,W]
    gen_seq = model.reconstruct_sequence_from_latent(samples, z_c)
    log['gen_seq'] = gen_seq
    log['steps'] = steps
    log['eta'] = eta_gen
    return log

def run(
    model,
    content_model,
    logdir,
    batch_size=8,
    custom_steps=None,
    eta=0.0,
    n_samples=100,
    seed=23,
    selector_cfg: Optional[Dict[str, Any]] = None,
):
    print(f"Using DDIM sampling with {custom_steps} sampling steps and eta={eta}")

    # Fixed temporal length for this project
    T = 8

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Choose steps/eta (prefer CLI values, else wrapper defaults)
    steps = int(custom_steps) if custom_steps is not None else int(getattr(content_model, 'default_steps', 50))
    eta_gen = float(eta) if eta is not None else float(getattr(content_model, 'default_eta', 0.0))

    # If selector is enabled, we will:
    #   (1) generate a candidate pool of content frames up to a budget
    #   (2) keep global top-k by score
    #   (3) generate sequences only for selected content
    selector = None
    if selector_cfg is not None:
        # FrameSelector expects an OmegaConf object (attribute-style access), not a plain dict.
        if not OmegaConf.is_config(selector_cfg):
            selector_cfg = OmegaConf.create(selector_cfg)
        selector = FrameSelector(selector_cfg)

    tstart = time.time()

    # Output structure
    img_root = logdir
    selected_dir = os.path.join(img_root, "selected")
    rejected_dir = os.path.join(img_root, "rejected")
    _ensure_dir(selected_dir)
    _ensure_dir(rejected_dir)

    # If no selector: keep existing behavior (generate + save directly)
    if selector is None:
        total_batches = int(np.ceil(n_samples / float(batch_size)))
        print(f"Sampling {n_samples} sequences dataset-free (batch_size={batch_size}, T={T})")
        n_saved = 0

        for _ in trange(total_batches, desc="Sampling Batches (content_model)"):
            B = int(min(batch_size, n_samples - n_saved))
            if B <= 0:
                break

            logs = make_convolutional_sample(
                model,
                content_model=content_model,
                B=B,
                T=T,
                custom_steps=custom_steps,
                eta=eta,
            )

            x_content = logs['x_content']          # [B,C,H,W]
            gen_bcthw = logs['gen_seq']            # [B,C,T,H,W]

            base_id = n_saved
            for b in range(B):
                sample_id = base_id + b
                sample_dir = os.path.join(img_root, f"{sample_id:06}")
                os.makedirs(sample_dir, exist_ok=True)

                # content frame
                img = custom_to_pil(x_content[b])
                img.save(os.path.join(sample_dir, f"sgen_content_{sample_id:06}.png"))

                # generated sequence
                save_sequence_single(gen_bcthw[b], sample_dir, prefix='sgen_seq', sample_id=sample_id)

            n_saved += B
            if n_saved >= n_samples:
                break

        print(f"sampling finished in {(time.time() - tstart) / 60.:.2f} minutes.")
        return

    # Selector-enabled path: global top-k content selection
    # Budgeting
    sel_budget = selector_cfg.get("budget", {}) if isinstance(selector_cfg, dict) else {}
    multiplier = float(sel_budget.get("max_generate_multiplier", 3.0))
    max_candidates = int(np.ceil(float(n_samples) * multiplier))
    early_stop = bool(sel_budget.get("early_stop", True))

    outputs = selector_cfg.get("outputs", {}) if isinstance(selector_cfg, dict) else {}
    save_all_candidates = bool(outputs.get("save_all_candidates", False))
    write_scores_csv = bool(outputs.get("write_scores_csv", True))

    gen_bs = int(selector_cfg.get("selector_batch_size", batch_size))
    gen_bs = max(1, int(gen_bs))

    print(
        f"[Selector] enabled: global top-{n_samples} from up to {max_candidates} candidates "
        f"(gen_bs={gen_bs}, early_stop={early_stop}, save_all_candidates={save_all_candidates})"
    )

    # Online top-k pool (min-heap)
    pool: List[Tuple[float, int, torch.Tensor]] = []
    score_rows: List[Dict[str, Any]] = []
    candidate_id = 0

    # Candidate generation loop
    while candidate_id < max_candidates:
        B = int(min(gen_bs, max_candidates - candidate_id))
        if B <= 0:
            break

        # Sample candidates (pixel space in [-1,1])
        x_cand = content_model.sample_content_frames(batch_size=B, ddim_steps=steps, eta=eta_gen)
        if not torch.is_tensor(x_cand):
            raise TypeError(f"sample_content_frames must return a Tensor, got {type(x_cand)}")

        # Maintain global top-k
        candidate_id = _select_topk_global(
            selector=selector,
            x_content=x_cand,
            remaining_k=int(n_samples),
            pool=pool,
            base_candidate_id=candidate_id,
            score_rows=score_rows,
            reject_dir=rejected_dir,
            save_all_candidates=save_all_candidates,
        )

        # Early stop: if we have filled pool and all new samples are unlikely to change ranking, we can't know.
        # So we only early-stop once we have at least k items and user asked for early_stop AND we reached budget.
        # Practically, keep looping until budget; early_stop remains as a future optimization.
        if early_stop and candidate_id >= max_candidates:
            break

    if not pool:
        raise RuntimeError("Selector enabled but no candidates were generated.")

    # Extract final selected, sorted by score desc
    pool_sorted = sorted(pool, key=lambda x: x[0], reverse=True)
    pool_sorted = pool_sorted[: int(n_samples)]

    # Mark which candidates are in the final selected set (top-k after the full search)
    final_selected_ids = {int(cid) for (_, cid, _) in pool_sorted}
    for r in score_rows:
        cid = int(r.get("candidate_id"))
        r["final_selected"] = 1 if cid in final_selected_ids else 0

    # Save selected content frames (optional convenience)
    for rank, (s, cid, frame_cpu) in enumerate(pool_sorted):
        try:
            img = custom_to_pil(frame_cpu)
            img.save(os.path.join(selected_dir, f"selected_rank{rank:03}_cid{cid:06}_score{s:.4f}.png"))
        except Exception:
            pass

    if write_scores_csv:
        out_csv = os.path.join(img_root, "selector_scores.csv")
        _write_selector_scores_csv(score_rows, out_csv)
        print(f"✅ Saved selector scores: {out_csv}")

    # Now generate sequences only for the selected content frames
    selected_frames = torch.stack([t for (_, _, t) in pool_sorted], dim=0)  # [K,C,H,W] on CPU

    print(f"Generating sequences for selected frames: K={selected_frames.shape[0]} (batch_size={batch_size}, T={T})")

    n_saved = 0
    total_batches = int(np.ceil(selected_frames.shape[0] / float(batch_size)))

    for bi in trange(total_batches, desc="Generating Sequences (motion_model)"):
        start = bi * batch_size
        end = min((bi + 1) * batch_size, selected_frames.shape[0])
        x_sel = selected_frames[start:end]
        gen_seq = _generate_sequences_from_content(model, x_sel, T=T, steps=steps, eta_gen=eta_gen)  # [B,C,T,H,W]

        Bcur = int(x_sel.shape[0])
        base_id = n_saved
        for b in range(Bcur):
            sample_id = base_id + b
            sample_dir = os.path.join(img_root, f"{sample_id:06}")
            os.makedirs(sample_dir, exist_ok=True)

            # content frame (the selected one)
            img = custom_to_pil(x_sel[b])
            img.save(os.path.join(sample_dir, f"sgen_content_{sample_id:06}.png"))

            # generated sequence
            save_sequence_single(gen_seq[b], sample_dir, prefix='sgen_seq', sample_id=sample_id)

        n_saved += Bcur

    print(f"sampling finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        nargs="?",
        help="load from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-n",
        "--n_samples",
        type=int,
        nargs="?",
        help="number of samples to draw",
        default=50000
    )
    parser.add_argument(
        "-e",
        "--eta",
        type=float,
        nargs="?",
        help="eta for ddim sampling (0.0 yields deterministic sampling)",
        default=1.0
    )
    parser.add_argument(
        "-v",
        "--vanilla_sample",
        default=False,
        action='store_true',
        help="vanilla sampling (default option is DDIM sampling)?",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        nargs="?",
        help="extra logdir",
        default="none"
    )
    parser.add_argument(
        "-c",
        "--custom_steps",
        type=int,
        nargs="?",
        help="number of steps for ddim and fastdpm sampling",
        default=50
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        nargs="?",
        help="the bs",
        default=2
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for dataloader order.",
    )
    parser.add_argument(
        "--content_wrapper",
        type=str,
        required=True,
        help="Path to a YAML defining `content_model_config` for ContentDSALDMWrapper (must implement sample_content_frames).",
    )
    return parser


def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu, eval_mode):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


def load_content_wrapper_from_yaml(yaml_path: str):
    """Load a content model wrapper (e.g., ContentDSALDMWrapper) from a YAML.

    Expected YAML keys:
      - content_model_config: instantiate_from_config block (required)
      - motion_shape: [P, Hm, Wm] (required for dataset-free motion sampling)

    Returns:
      (wrapper_model, motion_shape_tuple)
    """
    cfg = OmegaConf.load(yaml_path)

    # motion_shape is required in dataset-free sampling
    motion_shape = cfg.get('motion_shape', None)
    if motion_shape is None:
        raise ValueError("`motion_shape` is missing in content wrapper yaml. Please set motion_shape: [P, Hm, Wm].")
    if len(motion_shape) != 3:
        raise ValueError(f"`motion_shape` must have 3 ints [P,Hm,Wm], got: {motion_shape}")
    motion_shape = tuple(int(x) for x in motion_shape)

    cm_cfg = cfg.get('content_model_config', cfg)
    m = instantiate_from_config(cm_cfg)
    if hasattr(m, 'cuda'):
        m.cuda()
    if hasattr(m, 'eval'):
        m.eval()
    if not hasattr(m, 'sample_content_frames'):
        raise AttributeError(
            "content_model wrapper must implement sample_content_frames(batch_size, ddim_steps, eta)."
        )
    return m, motion_shape


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.insert(0, os.path.abspath(os.getcwd()))
    command = " ".join(sys.argv)

    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    ckpt = None

    if not os.path.exists(opt.resume):
        raise ValueError("Cannot find {}".format(opt.resume))
    if os.path.isfile(opt.resume):
        # paths = opt.resume.split("/")
        try:
            logdir = '/'.join(opt.resume.split('/')[:-1])
            # idx = len(paths)-paths[::-1].index("logs")+1
            print(f'Logdir is {logdir}')
        except ValueError:
            paths = opt.resume.split("/")
            idx = -2  # take a guess: path/to/logdir/checkpoints/model.ckpt
            logdir = "/".join(paths[:idx])
        ckpt = opt.resume
    else:
        assert os.path.isdir(opt.resume), f"{opt.resume} is not a directory"
        logdir = opt.resume.rstrip("/")
        ckpt = os.path.join(logdir, "model.ckpt")

    base_configs = sorted(glob.glob(os.path.join(logdir, "config.yaml")))
    opt.base = base_configs

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    gpu = True
    eval_mode = True

    if opt.logdir != "none":
        locallog = logdir.split(os.sep)[-1]
        if locallog == "": locallog = logdir.split(os.sep)[-2]
        print(f"Switching logdir from '{logdir}' to '{os.path.join(opt.logdir, locallog)}'")
        logdir = os.path.join(opt.logdir, locallog)

    print(config)

    model, global_step = load_model(config, ckpt, gpu, eval_mode)
    content_model, motion_shape = load_content_wrapper_from_yaml(opt.content_wrapper)
    selector_cfg = _load_selector_cfg_from_wrapper_yaml(opt.content_wrapper)
    model._last_motion_shape = motion_shape
    print(f"[INFO] Using motion_shape from content wrapper yaml: {model._last_motion_shape}")

    print(f"global step: {global_step}")
    print(75 * "=")
    print("logging to:")
    logdir = os.path.join(logdir, "samples", f"{global_step:08}", now)
    imglogdir = os.path.join(logdir, "img")
    numpylogdir = os.path.join(logdir, "numpy")

    os.makedirs(imglogdir)
    os.makedirs(numpylogdir)
    print(logdir)
    print(75 * "=")

    # write config out
    sampling_file = os.path.join(logdir, "sampling_config.yaml")
    sampling_conf = dict(vars(opt))
    sampling_conf["content_wrapper"] = opt.content_wrapper
    # Also record the resolved datamodule batch size from the merged config
    try:
        sampling_conf["data.params.batch_size"] = int(config.data.params.batch_size)
    except Exception:
        pass

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)


    run(
        model,
        content_model=content_model,
        logdir=imglogdir,
        eta=opt.eta,
        custom_steps=opt.custom_steps,
        n_samples=opt.n_samples,
        batch_size=opt.batch_size,
        seed=opt.seed,
        selector_cfg=selector_cfg,
    )

    print("done.")
