"""Reusable DINO backbone + classifier head wrapper.

This module provides a single source of truth for the end-to-end inference pipeline:
image -> DINO embeddings -> (optional L2 norm) -> (optional PCA) -> head.predict_proba -> score.

The training / CV / visualization CLI lives in `train_dino_classifier.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image

import timm
from timm.data import create_transform, resolve_data_config

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from tabpfn import TabPFNClassifier


def load_image_rgb(path: str) -> Image.Image:
    """Load image as grayscale then repeat channels to RGB (matches training script)."""
    img = Image.open(path).convert("L")
    return Image.merge("RGB", (img, img, img))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalization."""
    nrm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(nrm, eps)


def predict_scores(clf: Any, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities (column 1)."""
    proba = clf.predict_proba(X)
    if isinstance(proba, list):
        proba = np.asarray(proba)
    return np.asarray(proba)[:, 1]

def fit_pca(X: np.ndarray, pca_dim: int, seed: int):
    if pca_dim <= 0:
        return None
    

    pca = PCA(n_components=int(pca_dim), random_state=seed, svd_solver="randomized")
    pca.fit(X)
    return pca


def apply_pca(pca, X: np.ndarray) -> np.ndarray:
    if pca is None:
        return X
    return pca.transform(X).astype(np.float32)

# ------------------------------ head factories ------------------------------


def build_head(head: str, *, seed: int = 0, C: float = 1.0) -> Any:
    """Construct an untrained classifier head.

    Args:
        head: 'logreg' or 'tabpfn'
        seed: random seed / random_state
        C: inverse regularization strength for LogisticRegression

    Returns:
        An untrained classifier with a `fit` method.
    """
    head = str(head).lower().strip()

    if head == "logreg":
        return LogisticRegression(
            random_state=int(seed),
            max_iter=2000,
            C=float(C),
            class_weight="balanced",
            solver="lbfgs",
        )

    if head == "tabpfn":
        # Recent TabPFN versions use `random_state`.
        return TabPFNClassifier(random_state=int(seed))

    raise ValueError(f"Unknown head: {head}")


def fit_head(head: str, X: np.ndarray, y: np.ndarray, *, seed: int = 0, C: float = 1.0) -> Any:
    """Construct and fit a classifier head."""
    clf = build_head(head, seed=seed, C=C)
    clf.fit(X, y)

    return clf

@dataclass
class DinoClassifierBundle:
    """Serializable bundle for inference.

    NOTE: this is a convenience typed wrapper. We still save/load as a plain dict
    via joblib in the training script.
    """

    clf: Any
    head: str
    model: str
    pool: str
    l2norm: bool
    pca: Any = None
    pca_dim: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DinoClassifierBundle":
        return cls(
            clf=d["clf"],
            head=str(d.get("head", "logreg")),
            model=str(d.get("model", "timm/vit_base_patch16_dinov3.lvd1689m")),
            pool=str(d.get("pool", "cls")),
            l2norm=bool(d.get("l2norm", False)),
            pca=d.get("pca", None),
            pca_dim=int(d.get("pca_dim", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clf": self.clf,
            "head": self.head,
            "model": self.model,
            "pool": self.pool,
            "l2norm": self.l2norm,
            "pca": self.pca,
            "pca_dim": int(self.pca_dim),
        }


class DinoClassifier:
    """Frozen DINO backbone + a small classifier head (logreg / tabpfn).

    This class is intentionally inference-focused and lightweight. It does NOT
    own training logic, metrics, CV, or visualization.

    Key guarantees:
    - Uses timm backbone with `global_pool=""` and `forward_features`.
    - Uses the same grayscale->RGB conversion as training.
    - Applies pre-processing in a fixed order: L2norm -> PCA -> head.
    """

    def __init__(
        self,
        *,
        bundle: Union[DinoClassifierBundle, Dict[str, Any]],
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 32,
    ) -> None:
        if isinstance(bundle, dict):
            self.bundle = DinoClassifierBundle.from_dict(bundle)
        else:
            self.bundle = bundle

        if isinstance(device, str):
            self.device = torch.device(device if (torch.cuda.is_available() and device.startswith("cuda")) else "cpu")
        else:
            self.device = device

        self.batch_size = int(batch_size)

        # Backbone is loaded lazily on first use.
        self._model = None
        self._transform = None

        # Basic sanity checks
        if not hasattr(self.bundle.clf, "predict_proba"):
            raise ValueError("Classifier head must implement predict_proba(X)")
        if self.bundle.pool not in ("cls", "mean"):
            raise ValueError(f"Unsupported pool: {self.bundle.pool}")

    # -------------------------- backbone / transforms --------------------------

    @staticmethod
    def build_timm_backbone(model_name: str, device: torch.device):
        """Create a timm backbone and matching transform (same as training)."""
        timm_version = getattr(timm, "__version__", "unknown")

        try_names = [model_name]
        if not model_name.startswith("hf_hub:"):
            try_names.append(f"hf_hub:{model_name}")

        last_err = None
        model = None
        used = None
        for name in try_names:
            try:
                model = timm.create_model(name, pretrained=True, num_classes=0, global_pool="")
                used = name
                break
            except Exception as e:
                last_err = e

        if model is None:
            msg = (
                f"Failed to load model via timm. Tried {try_names}. Last error: {last_err}\n"
                f"Your timm version is: {timm_version}.\n"
                "DINOv3 backbones are supported by timm starting from v1.0.20. "
                "Please upgrade timm (e.g., `pip install -U timm>=1.0.20`)."
            )
            raise RuntimeError(msg)

        model.eval().to(device)

        cfg = resolve_data_config({}, model=model)
        transform = create_transform(**cfg, is_training=False)

        return model, transform, used, cfg

    def _ensure_backbone(self) -> None:
        if self._model is not None and self._transform is not None:
            return

        model, transform, used, cfg = self.build_timm_backbone(self.bundle.model, self.device)
        self._model = model
        self._transform = transform

        # Keep a minimal debug print (safe for both scripts)
        print(f"✅ Loaded timm model: {used}")
        print(f"   Data config: {cfg}")

    # ------------------------------ embeddings -------------------------------

    @torch.no_grad()
    def embed_pil(self, pil_images: Sequence[Image.Image]) -> np.ndarray:
        """Extract DINO embeddings for a list of PIL images."""
        self._ensure_backbone()
        assert self._model is not None
        assert self._transform is not None

        n = len(pil_images)
        feats: List[np.ndarray] = []

        for start in range(0, n, self.batch_size):
            batch_imgs = pil_images[start : start + self.batch_size]

            # Match training: convert to L then repeat to RGB
            tensors = []
            for img in batch_imgs:
                img_l = img.convert("L")
                img_rgb = Image.merge("RGB", (img_l, img_l, img_l))
                tensors.append(self._transform(img_rgb))

            x = torch.stack(tensors, dim=0).to(self.device)

            out = self._model.forward_features(x)
            if isinstance(out, (tuple, list)):
                out = out[0]

            if out.ndim == 3:
                # [B, T, C]
                if self.bundle.pool == "cls":
                    emb = out[:, 0, :]
                else:
                    emb = out.mean(dim=1)
            elif out.ndim == 2:
                emb = out
            else:
                raise RuntimeError(f"Unexpected output shape: {tuple(out.shape)}")

            feats.append(emb.float().cpu().numpy())

        return np.concatenate(feats, axis=0).astype(np.float32)

    @torch.no_grad()
    def embed_paths(self, image_paths: Sequence[str]) -> np.ndarray:
        """Extract DINO embeddings for a list of image paths."""
        pil_images = [load_image_rgb(p) for p in image_paths]
        return self.embed_pil(pil_images)

    # ------------------------------ scoring ----------------------------------

    def _preprocess_features(self, X: np.ndarray) -> np.ndarray:
        if self.bundle.l2norm:
            X = l2_normalize(X)
        if self.bundle.pca is not None:
            X = self.bundle.pca.transform(X).astype(np.float32)
        return X

    def score_embeddings(self, X: np.ndarray) -> np.ndarray:
        """Score already-extracted embeddings."""
        Xh = self._preprocess_features(X)
        return predict_scores(self.bundle.clf, Xh)

    def score_pil(self, pil_images: Sequence[Image.Image]) -> np.ndarray:
        X = self.embed_pil(pil_images)
        return self.score_embeddings(X)

    def score_paths(self, image_paths: Sequence[str]) -> np.ndarray:
        X = self.embed_paths(image_paths)
        return self.score_embeddings(X)

    # ------------------------------ serialization ----------------------------

    @classmethod
    def from_bundle_path(
        cls,
        bundle_path: str,
        *,
        device: Union[str, torch.device] = "cuda",
        batch_size: int = 32,
        joblib_load_fn=None,
    ) -> "DinoClassifier":
        """Load a joblib bundle produced by the training script."""
        if joblib_load_fn is None:
            from joblib import load as _load

            joblib_load_fn = _load
        bundle = joblib_load_fn(bundle_path)
        return cls(bundle=bundle, device=device, batch_size=batch_size)