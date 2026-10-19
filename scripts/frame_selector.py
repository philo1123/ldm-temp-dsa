import os
import math
import csv
import yaml

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from scripts.umm_phase_CNN.dino_classifier import DinoClassifier

@dataclass
class SelectorOutputs:
    save_all_candidates: bool = False
    write_scores_csv: bool = True
    viz_topk: int = 50


@dataclass
class SelectorStrategy:
    mode: str = "threshold_then_topk"  # threshold | topk | threshold_then_topk
    score_threshold: float = 0.8
    fallback_topk: bool = True


@dataclass
class SelectorBudget:
    max_generate_multiplier: float = 3.0
    early_stop: bool = True


@dataclass
class SelectorConfig:
    enabled: bool = True
    bundle_path: str = ""
    device: str = "cuda"
    selector_batch_size: int = 128
    strategy: SelectorStrategy = field(default_factory=SelectorStrategy)
    budget: SelectorBudget = field(default_factory=SelectorBudget)
    outputs: SelectorOutputs = field(default_factory=SelectorOutputs)


def _deep_get(d: Dict[str, Any], path: List[str], default: Any = None) -> Any:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_selector_cfg(path: str) -> Optional[SelectorConfig]:
    if path is None:
        return None
    if not os.path.isfile(path):
        raise ValueError(f"Selector config not found: {path}")
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    raw_cfg = raw.get("selector_config", raw)

    cfg = SelectorConfig(
        enabled=bool(raw_cfg.get("enabled", True)),
        bundle_path=str(raw_cfg.get("bundle_path", "")),
        device=str(raw_cfg.get("device", "cuda")),
        selector_batch_size=int(raw_cfg.get("selector_batch_size", 128)),
        strategy=SelectorStrategy(
            mode=str(_deep_get(raw_cfg, ["strategy", "mode"], "threshold_then_topk")),
            score_threshold=float(_deep_get(raw_cfg, ["strategy", "score_threshold"], 0.8)),
            fallback_topk=bool(_deep_get(raw_cfg, ["strategy", "fallback_topk"], True)),
        ),
        budget=SelectorBudget(
            max_generate_multiplier=float(_deep_get(raw_cfg, ["budget", "max_generate_multiplier"], 3.0)),
            early_stop=bool(_deep_get(raw_cfg, ["budget", "early_stop"], True)),
        ),
        outputs=SelectorOutputs(
            save_all_candidates=bool(_deep_get(raw_cfg, ["outputs", "save_all_candidates"], False)),
            write_scores_csv=bool(_deep_get(raw_cfg, ["outputs", "write_scores_csv"], True)),
            viz_topk=int(_deep_get(raw_cfg, ["outputs", "viz_topk"], 50)),
        ),
    )

    if cfg.enabled and not cfg.bundle_path:
        raise ValueError("selector_config.enabled is true but bundle_path is empty")
    return cfg


# --- Helper utilities for filesystem, visualization, and CSV ---
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def draw_title_below(img: Image.Image, title: str, pad: int = 6) -> Image.Image:
    w, h = img.size
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text_h = 14
    canvas = Image.new("RGB", (w, h + text_h + pad), (0, 0, 0))
    canvas.paste(img.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((2, h + 2), title, fill=(255, 255, 255), font=font)
    return canvas


def save_topk_viz(pairs: List[Tuple[float, Image.Image]], out_path: str, k: int = 50) -> None:
    if k <= 0 or len(pairs) == 0:
        return
    pairs = sorted(pairs, key=lambda x: x[0], reverse=True)[:k]

    base = pairs[0][1]
    if base.mode != "RGB":
        base = base.convert("RGB")
    tile_w, _ = base.size

    cols = 10
    rows = int(math.ceil(len(pairs) / cols))

    sample_tile = draw_title_below(base, "0.000")
    tile_h2 = sample_tile.size[1]

    grid = Image.new("RGB", (cols * tile_w, rows * tile_h2), (0, 0, 0))
    for idx, (score, img) in enumerate(pairs):
        r = idx // cols
        c = idx % cols
        tile = draw_title_below(img, f"{score:.3f}")
        grid.paste(tile, (c * tile_w, r * tile_h2))

    ensure_dir(os.path.dirname(out_path))
    grid.save(out_path)


def write_scores_csv(rows: List[Dict[str, Any]], out_csv: str) -> None:
    if len(rows) == 0:
        return
    ensure_dir(os.path.dirname(out_csv))
    keys = list(rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

class FrameSelector:
    """Scores PIL images using the canonical DINO + head pipeline.

    This wraps `DinoClassifier.from_bundle_path(...)` so selector scoring is bitwise
    consistent with `train_dino_classifier.py` / `dino_classifier.py`.

    Public API is intentionally stable:
      - `score(pil_imgs: List[PIL.Image]) -> np.ndarray`

    Notes:
    - `bundle_path` must point to `head.joblib` produced by `train_dino_classifier.py`.
    - All preprocessing (pooling, l2norm, PCA) is loaded from the bundle.
    """

    def __init__(self, cfg: SelectorConfig):
        self.cfg = cfg

        # Use the same device semantics as elsewhere.
        device = cfg.device
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"

        self.rater = DinoClassifier.from_bundle_path(
            cfg.bundle_path,
            device=device,
            batch_size=int(cfg.selector_batch_size),
        )

    def score(self, pil_imgs: List[Image.Image]) -> np.ndarray:
        if len(pil_imgs) == 0:
            return np.zeros((0,), dtype=np.float32)
        scores = self.rater.score_pil(pil_imgs)
        return np.asarray(scores, dtype=np.float32)


# Exported symbols for import *
__all__ = [
    "SelectorConfig",
    "SelectorBudget",
    "SelectorStrategy",
    "SelectorOutputs",
    "load_selector_cfg",
    "FrameSelector",
    "ensure_dir",
    "save_topk_viz",
    "write_scores_csv",
]