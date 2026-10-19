from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# Use the reusable DINO classifier helpers
from scripts.umm_phase_CNN.dino_classifier import DinoClassifier, \
    fit_pca, apply_pca, l2_normalize, predict_scores, fit_head

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from joblib import dump, load

import math
import matplotlib.pyplot as plt


def scan_png_dir(png_dir: str) -> pd.DataFrame:
    rows = []
    for root, _, files in os.walk(png_dir):
        for fn in files:
            if fn.lower().endswith('.png'):
                p = os.path.normpath(os.path.join(root, fn))
                rows.append({
                    'abs_path': p,
                    'image_path': p,
                    'category': os.path.basename(os.path.dirname(p)),
                    'filename': fn,
                })
    if not rows:
        raise FileNotFoundError(f"No PNG files found under: {png_dir}")
    return pd.DataFrame(rows).sort_values('abs_path').reset_index(drop=True)


def resolve_paths(df: pd.DataFrame, csv_path: str, png_root: str | None) -> pd.DataFrame:
    out = df.copy()

    if "abs_path" in out.columns:
        out["abs_path"] = out["abs_path"].astype(str).apply(os.path.normpath)
    else:
        if "image_path" not in out.columns:
            raise ValueError(f"CSV must contain an `image_path` column. Got: {set(out.columns)}")

        def _resolve(p: str) -> str:
            p = str(p)
            if os.path.isabs(p):
                return os.path.normpath(p)
            if png_root:
                return os.path.normpath(os.path.join(png_root, p))
            return os.path.normpath(os.path.join(os.path.dirname(csv_path), p))

        out["abs_path"] = out["image_path"].apply(_resolve)

    if "category" not in out.columns:
        out["category"] = out["abs_path"].apply(lambda x: os.path.basename(os.path.dirname(x)))
    if "filename" not in out.columns:
        out["filename"] = out["abs_path"].apply(lambda x: os.path.basename(x))

    missing = out[~out["abs_path"].apply(os.path.exists)]
    if len(missing) > 0:
        examples = missing["abs_path"].head(10).tolist()
        raise FileNotFoundError(f"Some image paths do not exist. Examples: {examples}")

    return out


def map_label(df: pd.DataFrame) -> pd.DataFrame:
    if "decision" not in df.columns:
        raise ValueError("CSV must contain a `decision` column with values keep/reject for training.")

    y = (df["decision"].astype(str).str.lower().str.strip() == "keep").astype(int)
    out = df.copy()
    out["y"] = y
    return out


def precision_at_k(y: np.ndarray, p: np.ndarray, k: int) -> float:
    if k <= 0 or len(y) == 0:
        return 0.0
    k = min(k, len(y))
    idx = np.argsort(-p)[:k]
    return float(y[idx].mean())


# --- Evaluation helpers ---

def parse_report_ks(report_ks: str) -> List[int]:
    ks: List[int] = []
    for s in str(report_ks).split(','):
        s = s.strip()
        if not s:
            continue
        try:
            ks.append(int(s))
        except ValueError:
            continue
    ks = [k for k in ks if k > 0]
    return sorted(set(ks))


def mean_std(vals: List[float]) -> Tuple[float, float]:
    if not vals:
        return 0.0, 0.0
    v = np.asarray(vals, dtype=np.float32)
    return float(v.mean()), float(v.std(ddof=0))


def eval_one_split(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    *,
    head: str,
    seed: int,
    C: float,
    pca_dim: int,
) -> Tuple[np.ndarray, float, float, float, dict]:
    """Train head on (Xtr,ytr) and evaluate on (Xva,yva). Returns (p_va, roc, ap, p@topk, p@k_map)."""
    pca = fit_pca(Xtr, pca_dim, seed)
    Xtr_h = apply_pca(pca, Xtr)
    Xva_h = apply_pca(pca, Xva)

    clf = fit_head(head, Xtr_h, ytr, seed=seed, C=C)
    p_va = predict_scores(clf, Xva_h)

    roc = float(roc_auc_score(yva, p_va))
    ap = float(average_precision_score(yva, p_va))
    return p_va, roc, ap, clf, {"pca": pca}


def report_holdout_metrics(
    yva: np.ndarray,
    p_va: np.ndarray,
    *,
    topk: int,
    ks: List[int],
) -> None:
    roc = roc_auc_score(yva, p_va)
    ap = average_precision_score(yva, p_va)
    p_at_topk = precision_at_k(yva, p_va, topk)

    print(f"VAL ROC-AUC: {roc:.4f}")
    print(f"VAL PR-AUC : {ap:.4f}")
    print(f"VAL Precision@Top{min(topk, len(yva))}: {p_at_topk:.4f}")

    if ks:
        msg_parts = []
        for k in ks:
            kk = min(k, len(yva))
            msg_parts.append(f"P@{kk}={precision_at_k(yva, p_va, kk):.4f}")
        print("VAL " + ", ".join(msg_parts))


def run_holdout_eval(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    *,
    eval_split: float,
    seed: int,
    head: str,
    C: float,
    pca_dim: int,
    topk: int,
    ks: List[int],
    out_dir: str,
    viz_cols: int,
    val_viz_topk: int,
    val_viz_bottomk: int,
) -> None:
    idx_all = np.arange(len(y))
    idx_tr, idx_va, Xtr, Xva, ytr, yva = train_test_split(
        idx_all, X, y, test_size=float(eval_split), random_state=seed, stratify=y
    )

    # Train/eval with PCA fit only on train split
    pca = fit_pca(Xtr, pca_dim, seed)
    Xtr_h = apply_pca(pca, Xtr)
    Xva_h = apply_pca(pca, Xva)

    clf = fit_head(head, Xtr_h, ytr, seed=seed, C=C)
    p_va = predict_scores(clf, Xva_h)

    report_holdout_metrics(yva, p_va, topk=topk, ks=ks)

    # Optional visualizations
    df_va = df.iloc[idx_va].copy().reset_index(drop=True)
    df_va["score"] = p_va
    df_va = df_va.sort_values("score", ascending=False).reset_index(drop=True)

    if val_viz_topk and val_viz_topk > 0:
        save_topk_grid(
            df_va,
            val_viz_topk,
            os.path.join(out_dir, f"val_top{val_viz_topk}.png"),
            cols=viz_cols,
        )
    if val_viz_bottomk and val_viz_bottomk > 0:
        df_va_bottom = df_va.sort_values("score", ascending=True).reset_index(drop=True)
        save_topk_grid(
            df_va_bottom,
            val_viz_bottomk,
            os.path.join(out_dir, f"val_bottom{val_viz_bottomk}.png"),
            cols=viz_cols,
        )


def run_cv_eval(
    X: np.ndarray,
    y: np.ndarray,
    *,
    cv_folds: int,
    cv_repeats: int,
    seed: int,
    head: str,
    C: float,
    pca_dim: int,
    topk: int,
    ks: List[int],
    val_viz_topk: int,
    val_viz_bottomk: int,
) -> None:
    if val_viz_topk or val_viz_bottomk:
        print("ℹ️  CV mode: per-fold visualizations are skipped (set --cv_folds 0 to use holdout viz).")

    rocs: List[float] = []
    aps: List[float] = []
    p_at_topk: List[float] = []
    p_at_ks = {k: [] for k in ks}

    repeats = int(max(1, cv_repeats))
    for r in range(repeats):
        cv_seed = int(seed) + r
        skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=cv_seed)

        for tr_idx, va_idx in skf.split(X, y):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]

            pca = fit_pca(Xtr, pca_dim, cv_seed)
            Xtr_h = apply_pca(pca, Xtr)
            Xva_h = apply_pca(pca, Xva)

            clf_fold = fit_head(head, Xtr_h, ytr, seed=cv_seed, C=C)
            p_va = predict_scores(clf_fold, Xva_h)

            rocs.append(float(roc_auc_score(yva, p_va)))
            aps.append(float(average_precision_score(yva, p_va)))
            p_at_topk.append(float(precision_at_k(yva, p_va, topk)))
            for k in ks:
                kk = min(k, len(yva))
                p_at_ks[k].append(float(precision_at_k(yva, p_va, kk)))

    m_roc, s_roc = mean_std(rocs)
    m_ap, s_ap = mean_std(aps)
    m_topk, s_topk = mean_std(p_at_topk)

    print(f"CV({cv_folds}-fold x {repeats} repeats) ROC-AUC: {m_roc:.4f} ± {s_roc:.4f}")
    print(f"CV({cv_folds}-fold x {repeats} repeats) PR-AUC : {m_ap:.4f} ± {s_ap:.4f}")
    print(f"CV({cv_folds}-fold x {repeats} repeats) Precision@Top{topk}: {m_topk:.4f} ± {s_topk:.4f}")

    if ks:
        parts = []
        for k in ks:
            mk, sk = mean_std(p_at_ks[k])
            parts.append(f"P@{k}={mk:.4f} ± {sk:.4f}")
        print("CV " + ", ".join(parts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "infer"], default="train")

    parser.add_argument("--csv", type=str, default=None, help="(Alias for --train_csv) Labeled CSV with image_path, decision.")
    parser.add_argument("--train_csv", type=str, default=None, help="Labeled CSV for training (image_path, decision).")
    parser.add_argument("--infer_csv", type=str, default=None, help="Unlabeled CSV for inference (image_path; category optional).")
    parser.add_argument("--png_dir", type=str, default=None, help="Unlabeled directory to recursively scan for PNGs (inference).")

    parser.add_argument("--png_root", type=str, default=None, help="Root to resolve relative image_path in CSV.")

    parser.add_argument("--model", type=str, default="timm/vit_base_patch16_dinov3.lvd1689m")
    parser.add_argument("--pool", choices=["cls", "mean"], default="cls", help="Token pooling for ViT outputs.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--l2norm", action="store_true", help="If set, L2-normalize embeddings before LogReg.")
    parser.add_argument("--C", type=float, default=1.0, help="Inverse regularization strength for LogisticRegression.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--topk", type=int, default=100, help="Report precision@topk (ranking quality).")

    parser.add_argument(
        "--val_viz_topk",
        type=int,
        default=0,
        help="If >0 (train mode with eval_split), save a grid of top-K validation samples ranked by score.",
    )
    parser.add_argument(
        "--val_viz_bottomk",
        type=int,
        default=0,
        help="If >0 (train mode with eval_split), save a grid of bottom-K validation samples (lowest scores).",
    )
    parser.add_argument(
        "--report_ks",
        type=str,
        default="10,20,40",
        help="Comma-separated K values to report Precision@K on validation (e.g., '10,20,40').",
    )

    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="predictions.csv")

    parser.add_argument("--viz_topk", type=int, default=0, help="If >0, save a grid image of top-K predictions (by score).")
    parser.add_argument("--viz_cols", type=int, default=10)

    parser.add_argument(
        "--model_out",
        type=str,
        default="head.joblib",
        help="Filename to save the trained classifier bundle (train mode).",
    )
    parser.add_argument(
        "--model_in",
        type=str,
        default=None,
        help="Path to a saved classifier bundle (joblib) for inference.",
    )
    parser.add_argument("--eval_split", type=float, default=0.2, help="Validation split ratio in train mode. Set 0 to skip.")
    parser.add_argument(
        "--cv_folds",
        type=int,
        default=0,
        help="If >1, run Stratified K-Fold cross-validation in train mode (replaces --eval_split eval).",
    )
    parser.add_argument(
        "--cv_repeats",
        type=int,
        default=1,
        help="Number of repeats for CV (uses seed+repeat as shuffle seed).",
    )

    # --- New arguments for head and PCA ---
    parser.add_argument(
        "--head",
        choices=["logreg", "tabpfn"],
        default="logreg",
        help="Classifier head to train/use on top of DINO embeddings.",
    )
    parser.add_argument(
        "--pca_dim",
        type=int,
        default=0,
        help="If >0, apply PCA to embeddings before the head. Recommended for TabPFN with 768-d embeddings (e.g., 128 or 256).",
    )

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.out_dir, exist_ok=True)

    # Resolve train_csv alias
    train_csv = args.train_csv or args.csv

    # Backbone/transform creation is centralized in DinoClassifier
    model, transform, used_name, cfg = DinoClassifier.build_timm_backbone(args.model, device)
    print(f"✅ Loaded timm model: {used_name}")
    print(f"   Data config: {cfg}")

    if args.mode == "train":
        if not train_csv:
            raise ValueError("Train mode requires --train_csv (or --csv alias).")

        df = pd.read_csv(train_csv)
        df = resolve_paths(df, train_csv, args.png_root)
        df = map_label(df)

        # Use the same embedding extraction as DinoClassifier
        rater_tmp = DinoClassifier(
            bundle={
                "clf": fit_head(
                        "logreg",
                        np.zeros((2, 1), dtype=np.float32),
                        np.array([0, 1], dtype=int),
                        seed=int(args.seed),
                        C=float(args.C),
                    ),
                "head": str(args.head),
                "model": str(args.model),
                "pool": str(args.pool),
                "l2norm": bool(args.l2norm),
                "pca": None,
                "pca_dim": int(args.pca_dim),
            },
            device=device,
            batch_size=int(args.batch_size),
        )
        # Reuse already-loaded backbone to avoid re-loading weights
        rater_tmp._model = model
        rater_tmp._transform = transform
        X = rater_tmp.embed_paths(df["abs_path"].tolist())
        if args.l2norm:
            X = l2_normalize(X)

        y = df["y"].to_numpy(dtype=int)

        # Optional evaluation: CV (stable) or single holdout
        ks = parse_report_ks(args.report_ks)

        if args.cv_folds and args.cv_folds > 1:
            run_cv_eval(
                X,
                y,
                cv_folds=int(args.cv_folds),
                cv_repeats=int(args.cv_repeats),
                seed=int(args.seed),
                head=str(args.head),
                C=float(args.C),
                pca_dim=int(args.pca_dim),
                topk=int(args.topk),
                ks=ks,
                val_viz_topk=int(args.val_viz_topk),
                val_viz_bottomk=int(args.val_viz_bottomk),
            )
        elif args.eval_split and args.eval_split > 0:
            run_holdout_eval(
                df,
                X,
                y,
                eval_split=float(args.eval_split),
                seed=int(args.seed),
                head=str(args.head),
                C=float(args.C),
                pca_dim=int(args.pca_dim),
                topk=int(args.topk),
                ks=ks,
                out_dir=str(args.out_dir),
                viz_cols=int(args.viz_cols),
                val_viz_topk=int(args.val_viz_topk),
                val_viz_bottomk=int(args.val_viz_bottomk),
            )

        # Fit final model (use all data) for export + inference usage
        pca_full = fit_pca(X, args.pca_dim, args.seed)
        X_h = apply_pca(pca_full, X)
        clf = fit_head(args.head, X_h, y, seed=int(args.seed), C=float(args.C))

        # Save model bundle
        model_path = os.path.join(args.out_dir, args.model_out)
        dump(
            {
                "clf": clf,
                "head": args.head,
                "l2norm": bool(args.l2norm),
                "pool": args.pool,
                "model": args.model,
                "pca_dim": int(args.pca_dim),
                "pca": pca_full,
            },
            model_path,
        )
        print(f"✅ Saved {args.head} head: {model_path}")

        # In-sample scores for ranking export
        p = predict_scores(clf, X_h)
        out = df.copy()
        out["score"] = p
        out = out.sort_values("score", ascending=False).reset_index(drop=True)

    else:
        # Inference mode
        if not args.model_in:
            raise ValueError("Infer mode requires --model_in pointing to a saved classifier bundle (joblib).")

        if args.png_dir:
            df = scan_png_dir(args.png_dir)
            df = resolve_paths(df, args.png_dir, png_root=None)
        elif args.infer_csv:
            df = pd.read_csv(args.infer_csv)
            df = resolve_paths(df, args.infer_csv, args.png_root)
            # If the inference CSV is labeled (has `decision`), map it so we can report metrics.
            if "decision" in df.columns:
                df = map_label(df)
        else:
            raise ValueError("Infer mode requires --png_dir or --infer_csv.")

        bundle = load(args.model_in)
        clf = bundle["clf"]
        need_l2 = bool(bundle.get("l2norm", False))
        pca = bundle.get("pca", None)

        rater = DinoClassifier(bundle=bundle, device=device, batch_size=int(args.batch_size))
        # Reuse already-loaded backbone to avoid re-loading weights
        rater._model = model
        rater._transform = transform
        X = rater.embed_paths(df["abs_path"].tolist())
        if need_l2:
            X = l2_normalize(X)
        X_h = apply_pca(pca, X)

        p = predict_scores(clf, X_h)
        out = df.copy()
        out["score"] = p
        out = out.sort_values("score", ascending=False).reset_index(drop=True)

        # If labels exist, report inference metrics on this labeled hold-out set.
        if "y" in out.columns:
            ks = parse_report_ks(args.report_ks)
            print("\n=== Labeled inference evaluation ===")
            report_holdout_metrics(out["y"].to_numpy(dtype=int), out["score"].to_numpy(dtype=float), topk=int(args.topk), ks=ks)

    out_path = os.path.join(args.out_dir, args.out_csv)
    out.to_csv(out_path, index=False)
    print(f"✅ Saved predictions: {out_path}")

    if args.viz_topk and args.viz_topk > 0:
        save_topk_grid(out, args.viz_topk, os.path.join(args.out_dir, f"top{args.viz_topk}.png"), cols=args.viz_cols)


def save_topk_grid(df_sorted: pd.DataFrame, k: int, out_path: str, cols: int = 10):
    k = min(k, len(df_sorted))
    rows = int(math.ceil(k / cols))

    fig = plt.figure(figsize=(cols * 2.0, rows * 2.4))

    for i in range(k):
        row = df_sorted.iloc[i]
        img = Image.open(row["abs_path"]).convert("L")

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")

        if "y" in df_sorted.columns:
            keep_val = int(row.get("y", -1))
            keep_str = "keep" if keep_val == 1 else "reject"
        else:
            keep_str = "?"
        title = f"{row.get('category','')}\nscore={row['score']:.2f} | label={keep_str}"

        # Put title below image
        ax.set_title(title, fontsize=8, y=-0.2)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"✅ Saved visualization: {out_path}")


if __name__ == "__main__":
    main()
