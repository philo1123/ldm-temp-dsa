# Temporal DSA Latent Diffusion

This repository contains code for generating temporal DSA sequences using a latent diffusion model.

---

## Environment Setup (uv)

This project uses **uv** for reproducible environment management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

### 2. Create the environment

From the repository root:

```bash
uv sync
```


This will create a local virtual environment in `./.venv/` and install all required dependencies.

Activate it:

```bash
source .venv/bin/activate
```

### 3. Install taming-transformers (required)

Due to packaging limitations of `taming-transformers`, it must be cloned manually.

From the repository root:

```bash
git clone https://github.com/CompVis/taming-transformers.git src/taming-transformers
```

After cloning, create a `.pth` file so Python can resolve the `taming` module:

```bash
python - << 'EOF'
import site, os
sp = site.getsitepackages()[0]
pth = os.path.join(sp, "taming_transformers_repo.pth")
with open(pth, "w") as f:
    f.write(os.path.abspath("src/taming-transformers") + "\n")
print("Wrote:", pth)
EOF
```

(Optional) You can verify the installation with:

```bash
python -c "import taming; import taming.modules.vqvae.quantize as q; print('taming OK')"
```



---

## Download Models

Run the following command:

```bash
bash download_models.sh
```

## Generate Temporal DSA Samples

We also provide a script for sampling synthetic temporal DSA sequences. Start it via

```bash
python scripts/sample_diffusion_temp_sampleContent.py \
  -r models/ldm/umm_dsa_f4_flow-ds2-residual/best.ckpt \
  -l <logdir> \
  -n 2 --batch_size <batch_size> -c <\#ddim steps> -e <\#eta>  \
  --content_wrapper scripts/sampling_config/content_wrapper_8-32-32-BERT-selector.yaml

```

For example:

```bash
python scripts/sample_diffusion_temp_sampleContent.py \
  -r models/ldm/umm_dsa_f4_flow-ds2-residual/best.ckpt \
  -l outputs/flow-temp-sampleContent-BERT \
  -n 4 --batch_size 4 -c 200 -e 0.0 \
  --content_wrapper scripts/sampling_config/content_wrapper_8-32-32-BERT-selector.yaml
```

Generated results will be saved under:

```
outputs/flow-temp-sampleContent-BERT/
```

---

## Notes

- Python 3.10 is required (handled automatically by `uv`).
- A CUDA-capable GPU is recommended for sampling.
