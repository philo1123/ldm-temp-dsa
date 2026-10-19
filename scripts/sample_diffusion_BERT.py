import re
import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np

import math
from tqdm import trange
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import instantiate_from_config

from frame_selector import FrameSelector, load_selector_cfg, ensure_dir, save_topk_viz, write_scores_csv

rescale = lambda x: (x + 1.) / 2.

def sanitize_filename(name):
    name = name.strip().replace(" ", "_")
    name = re.sub(r'[^\w\-_.]', '', name)
    return name

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

def _selector_save_decision(
    *,
    img: Image.Image,
    score: float,
    threshold: float,
    allow_keep: bool,
    selected_dir: str,
    rejected_dir: Optional[str],
    sel_idx: int,
    rej_idx: int,
    save_all: bool,
) -> Tuple[str, str, int, int]:
    """Save image according to threshold decision. Returns (decision, out_path, sel_idx, rej_idx)."""
    decision = "reject"
    out_path = ""

    if score >= threshold and allow_keep:
        decision = "keep"
        out_path = os.path.join(selected_dir, f"sample_{sel_idx:06}.png")
        img.save(out_path)
        sel_idx += 1
    else:
        if save_all and rejected_dir is not None:
            out_path = os.path.join(rejected_dir, f"sample_{rej_idx:06}.png")
            img.save(out_path)
            rej_idx += 1

    return decision, out_path, sel_idx, rej_idx


def _selector_fallback_fill(
    *,
    need: int,
    pool: List[Tuple[float, Image.Image]],
    selected_dir: str,
    sel_idx: int,
    rows: List[Dict[str, Any]],
    do_csv: bool,
    category: str,
) -> Tuple[int, int]:
    """Fill remaining slots by taking top scored images from pool. Returns (filled, new_sel_idx)."""
    if need <= 0:
        return 0, sel_idx

    pool_sorted = sorted(pool, key=lambda x: x[0], reverse=True)
    filled = 0
    for score, img in pool_sorted[:need]:
        out_path = os.path.join(selected_dir, f"sample_{sel_idx:06}.png")
        img.save(out_path)
        sel_idx += 1
        filled += 1
        if do_csv:
            rows.append({
                "category": category,
                "score": float(score),
                "decision": "keep_fallback",
                "path": out_path,
            })

    return filled, sel_idx


def _selector_loop(
    *,
    model,
    selector: FrameSelector,
    make_batch_fn,
    batch_size: int,
    target_kept: int,
    budget: int,
    threshold: float,
    fallback: bool,
    save_all: bool,
    do_csv: bool,
    category: str,
    root_dir: str,
    viz_k: int,
) -> None:
    """Generic sampling+selection loop.

    make_batch_fn(cur_bs) -> dict logs, where logs["sample"] is a tensor batch in image space.
    Saves results under root_dir/(selected|rejected) and writes scores/viz if configured.
    """
    selected_dir = os.path.join(root_dir, "selected")
    rejected_dir = os.path.join(root_dir, "rejected")

    ensure_dir(selected_dir)
    if save_all:
        ensure_dir(rejected_dir)
    else:
        rejected_dir = None

    kept = 0
    generated = 0
    sel_idx = 0
    rej_idx = 0
    pool: List[Tuple[float, Image.Image]] = []
    all_scored: List[Tuple[float, Image.Image]] = []
    rows: List[Dict[str, Any]] = []

    while kept < target_kept and generated < budget:
        cur_bs = min(batch_size, budget - generated)
        logs = make_batch_fn(cur_bs)

        pil_imgs = [custom_to_pil(logs["sample"][j]) for j in range(cur_bs)]
        scores = selector.score(pil_imgs)

        for img, score in zip(pil_imgs, scores.tolist()):
            all_scored.append((float(score), img))
            decision, out_path, sel_idx, rej_idx = _selector_save_decision(
                img=img,
                score=float(score),
                threshold=threshold,
                allow_keep=(kept < target_kept),
                selected_dir=selected_dir,
                rejected_dir=rejected_dir,
                sel_idx=sel_idx,
                rej_idx=rej_idx,
                save_all=save_all,
            )

            if decision == "keep":
                kept += 1
            else:
                pool.append((float(score), img))

            if do_csv:
                rows.append({
                    "category": category,
                    "score": float(score),
                    "decision": decision,
                    "path": out_path,
                })

        generated += cur_bs

        # Early stop once we have enough kept.
        if getattr(selector.cfg.budget, "early_stop", False) and kept >= target_kept:
            break

    if kept < target_kept and fallback:
        need = target_kept - kept
        print(f"[SELECTOR] {category}: fallback_topk fill {need}")
        filled, sel_idx = _selector_fallback_fill(
            need=need,
            pool=pool,
            selected_dir=selected_dir,
            sel_idx=sel_idx,
            rows=rows,
            do_csv=do_csv,
            category=category,
        )
        kept += filled

    if do_csv:
        write_scores_csv(rows, os.path.join(root_dir, "scores.csv"))
    if viz_k > 0:
        save_topk_viz(all_scored, os.path.join(root_dir, f"top{viz_k}.png"), k=viz_k)

    print(f"[SELECTOR] {category}: generated={generated} kept={kept} budget={budget}")

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
def convsample_ddim(model, steps, shape, eta=1.0, cond=None):
    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, eta=eta, verbose=False)
    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, cond=None, vanilla=False, custom_steps=None, eta=1.0):
    log = dict()
    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]
    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model, steps=custom_steps, shape=shape, eta=eta, cond=cond)
        t1 = time.time()
    x_sample = model.decode_first_stage(sample)
    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')

    selector_cfg = load_selector_cfg(getattr(opt, "selector_cfg", None))
    selector: Optional[FrameSelector] = None
    if selector_cfg is not None and selector_cfg.enabled:
        print(f"[SELECTOR] enabled=True  bundle={selector_cfg.bundle_path}")
        selector = FrameSelector(selector_cfg)

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1

    # Decide mode:
    # - Conditional: model has cond_stage_model AND prompt_file is provided
    # - Unconditional: otherwise
    is_conditional = (model.cond_stage_model is not None) and (opt.prompt_file is not None)

    if selector is not None:
        # End-to-end sampling + selection
        threshold = float(selector_cfg.strategy.score_threshold)
        fallback = bool(selector_cfg.strategy.fallback_topk)
        save_all = bool(selector_cfg.outputs.save_all_candidates)
        do_csv = bool(selector_cfg.outputs.write_scores_csv)
        viz_k = int(selector_cfg.outputs.viz_topk)

        if not is_conditional:
            target_kept = int(n_samples)
            budget = int(math.ceil(selector_cfg.budget.max_generate_multiplier * target_kept))
            print(f"[SELECTOR] mode=unconditional  target_kept={target_kept}  budget={budget}  thr={threshold:.3f}")

            def _make_uncond_batch(cur_bs: int):
                return make_convolutional_sample(
                    model,
                    batch_size=cur_bs,
                    cond=None,
                    vanilla=vanilla,
                    custom_steps=custom_steps,
                    eta=eta,
                )

            _selector_loop(
                model=model,
                selector=selector,
                make_batch_fn=_make_uncond_batch,
                batch_size=batch_size,
                target_kept=target_kept,
                budget=budget,
                threshold=threshold,
                fallback=fallback,
                save_all=save_all,
                do_csv=do_csv,
                category="uncond",
                root_dir=logdir,
                viz_k=viz_k,
            )
            return

        else:
            # conditional: per-prompt selection
            if not os.path.isfile(opt.prompt_file):
                raise ValueError(f"Prompt file {opt.prompt_file} not found.")
            with open(opt.prompt_file, "r") as f:
                raw_prompts = [line.strip() for line in f if line.strip()]
            if len(raw_prompts) == 0:
                raise ValueError(f"Prompt file {opt.prompt_file} is empty.")

            target_kept = int(n_samples)
            budget = int(math.ceil(selector_cfg.budget.max_generate_multiplier * target_kept))
            print(f"[SELECTOR] mode=conditional(per-prompt)  target_kept_per_prompt={target_kept}  budget_per_prompt={budget}  thr={threshold:.3f}")

            for prompt in raw_prompts:
                match = re.search(
                    r"This is an (\w+) DSA scan taken in (\w+) view, with a primary angle of (-?\d+) degrees and a secondary angle of (-?\d+) degrees",
                    prompt,
                )
                if match:
                    group, plane, angle, secondary_angle = match.groups()
                    subfolder_name = f"{group}_{plane}_{angle}_{secondary_angle}"
                else:
                    subfolder_name = sanitize_filename(prompt)

                prompt_root = os.path.join(logdir, subfolder_name)

                def _make_cond_batch(cur_bs: int, _prompt: str = prompt):
                    cond = model.get_learned_conditioning([_prompt] * cur_bs)
                    return make_convolutional_sample(
                        model,
                        batch_size=cur_bs,
                        cond=cond,
                        vanilla=vanilla,
                        custom_steps=custom_steps,
                        eta=eta,
                    )

                _selector_loop(
                    model=model,
                    selector=selector,
                    make_batch_fn=_make_cond_batch,
                    batch_size=batch_size,
                    target_kept=target_kept,
                    budget=budget,
                    threshold=threshold,
                    fallback=fallback,
                    save_all=save_all,
                    do_csv=do_csv,
                    category=subfolder_name,
                    root_dir=prompt_root,
                    viz_k=viz_k,
                )

            return

    if not is_conditional:
        if model.cond_stage_model is not None and opt.prompt_file is None:
            print("[INFO] cond_stage_model is present but --prompt_file not provided; running unconditional sampling.")
        else:
            print("[INFO] Running unconditional sampling.")

        for i in trange(max(1, (n_samples + batch_size - 1) // batch_size), desc="Sampling Batches (unconditional)"):
            cur_bs = min(batch_size, n_samples - (i * batch_size))
            if cur_bs <= 0:
                break
            logs = make_convolutional_sample(
                model,
                batch_size=cur_bs,
                cond=None,
                vanilla=vanilla,
                custom_steps=custom_steps,
                eta=eta,
            )
            for j in range(cur_bs):
                img = custom_to_pil(logs["sample"][j])
                imgpath = os.path.join(logdir, f"sample_{i * batch_size + j:06}.png")
                img.save(imgpath)
            n_saved += cur_bs
            if n_saved >= n_samples:
                print(f"Finish after generating {n_saved} samples")
                break

    else:
        if not os.path.isfile(opt.prompt_file):
            raise ValueError(f"Prompt file {opt.prompt_file} not found.")
        with open(opt.prompt_file, "r") as f:
            raw_prompts = [line.strip() for line in f if line.strip()]
        if len(raw_prompts) == 0:
            raise ValueError(f"Prompt file {opt.prompt_file} is empty.")

        total_n_samples = n_samples * len(raw_prompts)
        all_prompts: list[str] = []
        for prompt in raw_prompts:
            all_prompts.extend([prompt] * n_samples)

        print(f"Running conditional sampling for {len(all_prompts)} samples using {len(raw_prompts)} prompts")

        for i in trange(max(1, (len(all_prompts) + batch_size - 1) // batch_size), desc="Sampling Batches (conditional)"):
            batch_prompts = all_prompts[i * batch_size:(i + 1) * batch_size]
            if len(batch_prompts) == 0:
                break
            cond = model.get_learned_conditioning(batch_prompts)
            logs = make_convolutional_sample(
                model,
                batch_size=len(batch_prompts),
                cond=cond,
                vanilla=vanilla,
                custom_steps=custom_steps,
                eta=eta,
            )
            for j, prompt in enumerate(batch_prompts):
                match = re.search(
                    r"This is an (\w+) DSA scan taken in (\w+) view, with a primary angle of (-?\d+) degrees and a secondary angle of (-?\d+) degrees",
                    prompt,
                )
                if match:
                    group, plane, angle, secondary_angle = match.groups()
                    subfolder_name = f"{group}_{plane}_{angle}_{secondary_angle}"
                else:
                    subfolder_name = sanitize_filename(prompt)
                subfolder = os.path.join(logdir, subfolder_name)
                os.makedirs(subfolder, exist_ok=True)
                img = custom_to_pil(logs["sample"][j])
                imgpath = os.path.join(subfolder, f"sample_{i * batch_size + j:06}.png")
                img.save(imgpath)
            n_saved += len(batch_prompts)
            if n_saved >= total_n_samples:
                print(f"Finish after generating {n_saved} samples")
                break

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


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
        default=10
    )
    parser.add_argument(
        "--prompt_file",
        type=str,
        default=None,
        help="Optional: path to file containing text prompts (one per line). If omitted, run unconditional sampling."
    )
    parser.add_argument(
        "--selector_cfg",
        type=str,
        default=None,
        help="Optional: YAML config for frame selection (threshold/topk + budget). If omitted, sampling behaves as before.",
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


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    sys.path.append(os.getcwd())
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
    if getattr(model, 'conditioning_key', None) != 'crossattn':
        print("Warning: this model may not support text conditioning as expected.")
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
    sampling_conf = vars(opt)

    with open(sampling_file, 'w') as f:
        yaml.dump(sampling_conf, f, default_flow_style=False)
    print(sampling_conf)


    run(model, imglogdir, eta=opt.eta,
        vanilla=opt.vanilla_sample,  n_samples=opt.n_samples, custom_steps=opt.custom_steps,
        batch_size=opt.batch_size, nplog=numpylogdir)

    print("done.")
