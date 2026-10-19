import argparse, os, sys, glob, datetime, yaml
import torch
import time
import numpy as np
from tqdm import trange

from omegaconf import OmegaConf
from PIL import Image

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config

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

# Save a single sequence [C,T,H,W] with a fixed sample_id
def save_sequence_single(frames_cthw, outdir, prefix, sample_id):
    """frames_cthw: Tensor [C,T,H,W] in [-1,1]. Saves frames with a fixed sample id."""
    frames = frames_cthw.detach().cpu()
    C, T, H, W = frames.shape
    for t in range(T):
        img = custom_to_pil(frames[:, t])
        imgpath = os.path.join(outdir, f"{prefix}_{sample_id:06}_t{t:02}.png")
        img.save(imgpath)


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
def make_convolutional_sample(model, batch, bs, vanilla=False, custom_steps=None, eta=1.0):
    """Sample motion latent conditioned on content frame from the provided batch.

    Returns a dict with:
      - inputs_seq: [T,C,H,W] (first clip)
      - content_frame: [1,C,H,W] (extracted content)
      - rec_seq: reconstructed sequence from GT motion latent
      - recon_sample_motion_seq: reconstructed sequence from sampled motion latent
    """
    log = dict()

    # Use exactly the same pipeline as training
    z_m, c, inputs_seq, x_content, z_c_latent = model.get_input(
        batch,
        model.first_stage_key,
        return_first_stage_outputs=True,
        force_c_encode=True,
        bs=bs,
    )

    log["inputs_seq"] = inputs_seq
    log["content_frame"] = x_content

    # Reconstruct the input sequence from the GT motion latent (sanity)
    rec_seq = model.reconstruct_sequence_from_latent(z_m, z_c_latent)
    log["rec_seq"] = rec_seq[0] # [B,C,T,H,W] # .permute(1, 0, 2, 3).contiguous()  # [T,C,H,W]

    # Build conditioning dict for concat models
    conditioning = None
    if getattr(model.model, 'conditioning_key', None) == 'concat':
        assert c is not None and torch.is_tensor(c), f"Flow/residual requires concat conditioning Tensor, got {type(c)}"
        conditioning = c

    # DDIM sampling in motion latent space
    with model.ema_scope("Plotting"):
        t0 = time.time()
        bs_for_sampling = int(z_m.shape[0])  # flow/residual already folded to [B*T,...]
        shape = model._last_motion_shape     # (C,H',W') for 2D UNet motion latents
        samples, _ = convsample_ddim(
            model,
            steps=custom_steps,
            batch_size=bs_for_sampling,
            shape=shape,
            eta=eta,
            conditioning=conditioning,
        )
        t1 = time.time()

    # Reconstruct sampled sequence
    recon_sample_motion_seq = model.reconstruct_sequence_from_latent(samples, z_c_latent)
    log["recon_sample_motion_seq"] =  recon_sample_motion_seq  # [B,C,T,H,W] # .permute(1, 0, 2, 3).contiguous()  # [T,C,H,W]

    log["time"] = t1 - t0
    log["throughput"] = bs_for_sampling / max(1e-6, (t1 - t0))
    print(f"Throughput for this batch (BT): {log['throughput']}")
    return log

def run(model, data_config, logdir, batch_size=2, split='train', vanilla=False, custom_steps=None, eta=None,
        n_samples=100, max_batches=None, seed=23, nplog=None):
    if vanilla:
        print(f"Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.")
    else:
        print(f"Using DDIM sampling with {custom_steps} sampling steps and eta={eta}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate datamodule exactly like training, but force batch_size to follow CLI --batch_size
    if hasattr(data_config, 'params') and ('batch_size' in data_config.params):
        data_config.params.batch_size = int(batch_size)
    dm = instantiate_from_config(data_config)
    dm.prepare_data()
    dm.setup()
    if split == 'train':
        loader = dm.train_dataloader()
    elif split == 'validation':
        loader = dm.val_dataloader()
    else:
        raise ValueError(f"Unknown split: {split}")

    # Use the real loader batch size for counting batches
    real_bs = getattr(loader, 'batch_size', batch_size)
    print(f"Sampling {n_samples} sequences from split='{split}' (loader bs={real_bs})")

    # Flow/residual sampling internally uses BT; we rely on model.get_input() to fold.
    if max_batches is None:
        total_batches = int(np.ceil(n_samples / float(real_bs)))
    else:
        total_batches = int(max_batches)

    try:
        print("data_config.params.batch_size =", data_config.params.batch_size)
    except Exception:
        pass

    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir, '*.png')))

    # We interpret `batch_size` as number of sequences B (same as dataloader bs).
    # Flow/residual sampling internally uses BT; we rely on model.get_input() to fold.

    it = iter(loader)
    for _ in trange(total_batches, desc=f"Sampling Batches ({split})"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        logs = make_convolutional_sample(
            model,
            batch=batch,
            bs=real_bs,
            vanilla=vanilla,
            custom_steps=custom_steps,
            eta=eta,
        )

        # Save triplets per sequence: real_content frame, real_seq (GT), cgen_seq (generated)
        if 'recon_sample_motion_seq' in logs:
            gen_bcthw = logs['recon_sample_motion_seq']  # [B,C,T,H,W]
            Bgen = gen_bcthw.shape[0]

            # Content frames from model.get_input (typically [B,C,H,W])
            cf = logs.get('content_frame', None)

            # Original dataset sequence from the dataloader batch (dataset returns [C,T,H,W], collate -> [B,C,T,H,W])
            gt = batch.get(model.first_stage_key, None)

            base_id = n_saved
            for b in range(Bgen):
                sample_id = base_id + b
                sample_dir = os.path.join(logdir, f"{sample_id:06}")
                os.makedirs(sample_dir, exist_ok=True)

                # 1) real_content frame
                if isinstance(cf, torch.Tensor) and cf.ndim == 4:
                    img = custom_to_pil(cf[b])
                    img.save(os.path.join(sample_dir, f"real_content_{sample_id:06}.png"))

                # 2) real_seq (ground-truth/original sequence)
                if isinstance(gt, torch.Tensor) and gt.ndim == 5:
                    save_sequence_single(gt[b], sample_dir, prefix='real_seq', sample_id=sample_id)
                # 3) cgen_seq (generated sequence)
                save_sequence_single(gen_bcthw[b], sample_dir, prefix='cgen_seq', sample_id=sample_id)

            n_saved += Bgen

        if n_saved >= n_samples:
            break

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
        "--split",
        type=str,
        default= "train",
        choices=["train", "validation"],
        help="Which dataloader split to draw batches from.",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum number of batches to iterate (defaults to n_samples//batch_size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for dataloader order.",
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
        data_config=config.data,
        logdir=imglogdir,
        eta=opt.eta,
        vanilla=opt.vanilla_sample,
        n_samples=opt.n_samples,
        custom_steps=opt.custom_steps,
        batch_size=opt.batch_size,
        split=opt.split,
        max_batches=opt.max_batches,
        seed=opt.seed,
        nplog=numpylogdir,
    )

    print("done.")
