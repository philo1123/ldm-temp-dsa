import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm
from ummdsa_class_dataset import UMMDSAClassificationDataset
import argparse

import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import random
import sys

# --- minimal stdout tee (print -> terminal + file) ---
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

from sklearn.metrics import confusion_matrix
import math
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from torchvision.models import resnet18
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import numpy as np

def generate_artery_only_csv(inference_csv_path, original_metadata_path, keep_all_frames_in_series=False):
    """
    Generate a new CSV containing only images predicted as artery (label=1),
    and preserve the original metadata format.

    Args:
        inference_csv_path (str): Path to the inference result CSV.
        original_metadata_path (str): Path to the original metadata CSV (for full fields).
        keep_all_frames_in_series (bool): If True, keep all frames in series regardless of predicted label.
    """
    filtered_df = pd.read_csv(inference_csv_path)
    if not keep_all_frames_in_series:
        filtered_df = filtered_df[filtered_df["predicted_label"] == 1]

    # Load original metadata for merging extra columns
    original_df = pd.read_csv(original_metadata_path)
    
    # Auto-detect merge keys based on available columns
    merge_keys = ["file_path", "frame_idx"]
    if "series_id" in filtered_df.columns and "series_id" in original_df.columns:
        merge_keys.append("series_id")

    merged_df = pd.merge(
        filtered_df,
        original_df,
        on=merge_keys,
        how="left"
    )

    output_filtered_csv = inference_csv_path.replace(".csv", "_artery_filtered.csv")
    columns_to_keep = original_df.columns.tolist()
    merged_df[columns_to_keep].to_csv(output_filtered_csv, index=False)
    print(f"✅ Saved filtered artery-only CSV to {output_filtered_csv}")
    
    num_missing = merged_df.isnull().any(axis=1).sum()
    print(f"❗ Merge caused {num_missing} rows with missing metadata.")
    
    print(f"Original predicted label==1: {len(filtered_df)}")
    print(f"After merge and column filtering: {len(merged_df)}")

def visualize_predictions(samples, output_path):
    """
    Visualize a set of prediction samples and save the figure.

    Args:
        samples (pd.DataFrame): A dataframe containing the sampled predictions.
        output_path (str): Path to save the output figure.
    """
    num_samples = len(samples)
    cols = 5
    rows = math.ceil(num_samples / cols)

    plt.figure(figsize=(cols * 3, rows * 3))
    for idx, row in enumerate(tqdm(samples.itertuples(), total=len(samples), desc="Visualizing")):
        dcm = pydicom.dcmread(row.file_path)
        pixel_array = dcm.pixel_array
        if pixel_array.ndim == 3:  # multi-frame
            frame_idx = int(row.frame_idx) - 1
            img = pixel_array[frame_idx]
        elif pixel_array.ndim == 2:  # single-frame
            img = pixel_array
        else:
            raise ValueError(f"Unsupported pixel array shape: {pixel_array.shape}")

        img = img.astype(float)
        img -= img.min()
        img /= (img.max() + 1e-5)

        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img, cmap="gray")
        plt.axis('off')
        pred_label = "artery" if row.predicted_label == 1 else "negative"
        # Optional ground-truth label (if present in the CSV)
        gt_label = "?"
        if hasattr(row, "gt_label") and row.gt_label == row.gt_label:  # not NaN
            gt_label = "artery" if int(row.gt_label) == 1 else "negative"
        elif hasattr(row, "label") and row.label == row.label:
            gt_label = "artery" if int(row.label) == 1 else "negative"
        path_parts = row.file_path.split("/")[-5:]
        path_header = "/".join(path_parts[:4])
        file_name = path_parts[4]
        plt.text(
            0.5, -0.12,
            f"{path_header}\n{file_name}\nFrame {row.frame_idx}\nGT: {gt_label} | Pred: {pred_label} | Prob: {row.prob_artery:.2f}",
            transform=plt.gca().transAxes,
            ha="center",
            va="top",
            fontsize=6,
            fontweight="bold" if pred_label == "artery" else "normal",
            wrap=True
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Saved visualization to {output_path}")

# --- PNG visualization for predictions ---
def visualize_png_predictions(df, output_path, top_k=5):
    """
    Visualize top-K PNG predictions per category (prompt) based on prob_artery.

    Args:
        df (pd.DataFrame): inference_png output dataframe containing columns:
            abs_path, category, prob_artery, doctor_keep
        output_path (str): path to save the figure
        top_k (int): number of top images to show per category
    """
    categories = sorted(df['category'].unique())
    num_cats = len(categories)
    rows = num_cats
    cols = top_k

    plt.figure(figsize=(cols * 3, rows * 3))
    plot_idx = 1

    for r, cat in enumerate(categories):
        sub = df[df['category'] == cat].sort_values('prob_artery', ascending=False).head(top_k)
        for _, row in sub.iterrows():
            img = Image.open(row['abs_path']).convert('L')
            plt.subplot(rows, cols, plot_idx)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
            keep_val = int(row.get('doctor_keep', -1))
            keep_str = "?" if keep_val == -1 else str(keep_val)
            title = f"{cat}\nprob={row['prob_artery']:.2f} | keep={keep_str}"
            plt.text(
                0.5, -0.10,
                title,
                transform=plt.gca().transAxes,
                ha="center",
                va="top",
                fontsize=8,
                wrap=True,
                fontweight="bold" if keep_val == 1 else "normal",
            )
            plot_idx += 1

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"✅ Saved PNG visualization to {output_path}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, 2)
    def forward(self, x):
        return self.model(x)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    correct = 0
    total = 0
    running_loss = 0
    for batch in tqdm(loader, desc="Training"):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    avg_loss = running_loss / len(loader)
    return acc, avg_loss

def evaluate(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    acc = correct / total
    avg_loss = running_loss / len(loader)

    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=[1, 0])
    TP = cm[0, 0]
    FN = cm[0, 1]
    FP = cm[1, 0]
    TN = cm[1, 1]
    artery_precision = TP / (TP + FP + 1e-8)
    artery_recall = TP / (TP + FN + 1e-8)

    print(f"Val Artery Precision: {artery_precision:.4f}")
    print(f"Val Artery Recall: {artery_recall:.4f}")

    return acc, avg_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_path", type=str, required=False, help="Path to the multiframe labeled CSV.")
    parser.add_argument("--save_dir", type=str, default="scripts/umm_phase_CNN/checkpoints", help="Directory to save checkpoints.")
    parser.add_argument("--mode", choices=["train", "inference", "inference_series", "inference_png"], required=False, help="train, inference, inference_series, or inference_png mode")
    parser.add_argument("--inference_output", type=str, default="inference_results.csv", help="Path to save inference results.")
    parser.add_argument("--ckpt_path", type=str, help="Path to load checkpoint for inference.")
    parser.add_argument("--png_csv_path", type=str, default=None, help="Path to doctor review CSV for generated PNGs. Must contain image_path and decision columns.")
    parser.add_argument("--png_root", type=str, default=None, help="Optional root dir to resolve relative image_path in png_csv_path.")
    parser.add_argument("--png_dir", type=str, default=None, help="If set, scan this directory recursively for PNGs and run inference_png without labels.")
    parser.add_argument("--num_samples_to_visualize", type=int, default=0, help="Number of inference samples to visualize. Set 0 to disable.")
    parser.add_argument("--visualize_only_csv", type=str, default=None, help="Path to an existing inference CSV for visualization only.")
    parser.add_argument("--model_type", choices=["simple", "resnet"], default="simple", help="Which model to use: simple or resnet")
    args = parser.parse_args()

    if args.visualize_only_csv is None:
        if args.mode is None:
            raise ValueError("You must specify --mode unless using --visualize_only_csv.")
        if args.mode in ["inference", "inference_series", "inference_png"] and not args.ckpt_path:
            raise ValueError(f"{args.mode} mode requires --ckpt_path to load model.")
        if args.mode == "inference_png" and (not args.png_csv_path and not args.png_dir):
            raise ValueError("inference_png mode requires --png_csv_path (labeled) or --png_dir (unlabeled scan).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    inference_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    def _load_png_tensor(p: str):
        img = Image.open(p).convert("L")
        return inference_transform(img)

    if args.model_type == "simple":
        model = SimpleCNN().to(device)
    else:
        model = ResNetClassifier().to(device)

    if args.visualize_only_csv:
        df = pd.read_csv(args.visualize_only_csv, dtype={"frame_idx": int})
        samples = df.sample(n=min(args.num_samples_to_visualize, len(df)))

        output_img_path = args.visualize_only_csv.replace(".csv", ".png")
        visualize_predictions(samples, output_img_path)
        return

    if args.mode == "train":
        dataset = UMMDSAClassificationDataset(
            metadata_path=args.metadata_path,
            transform=train_transform
        )

        # Split train/val
        val_ratio = 0.2
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=32)

        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        os.makedirs(args.save_dir, exist_ok=True)

        # Save all training prints to txt while still showing them in terminal
        log_f = open(os.path.join(args.save_dir, "train.log"), "a", buffering=1)
        sys.stdout = Tee(sys.stdout, log_f)
        print("=== Training started ===")

        for epoch in range(30):
            print(f"Epoch {epoch+1}/30")
            train_acc, train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_acc, val_loss = evaluate(model, val_loader, criterion, device)

            print(f"Train Acc: {train_acc:.4f}, Loss: {train_loss:.4f}")
            print(f"Val   Acc: {val_acc:.4f}, Loss: {val_loss:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model-test.pth"))
                print("✅ Saved new best model.")

    elif args.mode == "inference":
        model.load_state_dict(torch.load(args.ckpt_path))
        model.eval()
        dataset = UMMDSAClassificationDataset(
            metadata_path=args.metadata_path,
            transform=inference_transform,
            inference_mode=True
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=32)
        results = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference"):
                images = batch["image"].to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                for path, frame_idx, pred, prob in zip(batch['file_path'], batch['frame_idx'], preds.cpu(), probs[:,1].cpu()):
                    results.append({
                        "file_path": path,
                        "frame_idx": int(frame_idx),
                        "predicted_label": int(pred),
                        "prob_artery": float(prob)
                    })
        
        df_out = pd.DataFrame(results)

        # Merge GT label from metadata (so visualization can show it)
        try:
            meta_df = pd.read_csv(args.metadata_path)
            if "label" in meta_df.columns:
                meta_small = meta_df[["file_path", "frame_idx", "label"]].copy()
                meta_small["frame_idx"] = meta_small["frame_idx"].astype(int)
                df_out = df_out.merge(meta_small, on=["file_path", "frame_idx"], how="left")
                df_out = df_out.rename(columns={"label": "gt_label"})
        except Exception as e:
            print(f"⚠️ Could not merge GT labels from metadata: {e}")

        df_out.to_csv(args.inference_output, index=False)
        print(f"✅ Saved inference results to {args.inference_output}")

        # Generate artery-only CSV with original metadata fields
        generate_artery_only_csv(args.inference_output, args.metadata_path)

        if args.num_samples_to_visualize > 0:

            df = pd.read_csv(args.inference_output, dtype={"frame_idx": int})
            samples = df.sample(n=min(args.num_samples_to_visualize, len(df)))

            output_img_path = args.inference_output.replace(".csv", ".png")
            visualize_predictions(samples, output_img_path)

    elif args.mode == "inference_png":
        # Inference on generated PNGs with doctor keep/reject labels or unlabeled scan
        model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
        model.eval()

        # Load inputs: labeled CSV OR unlabeled scan directory
        if args.png_csv_path:
            df = pd.read_csv(args.png_csv_path)
        else:
            scan_root = args.png_dir
            png_paths = []
            for root, _, files in os.walk(scan_root):
                for fn in files:
                    if fn.lower().endswith('.png'):
                        png_paths.append(os.path.join(root, fn))
            if len(png_paths) == 0:
                raise FileNotFoundError(f"No PNG files found under --png_dir: {scan_root}")

            rows = []
            for pth in sorted(png_paths):
                rows.append({
                    "abs_path": os.path.normpath(pth),
                    "category": os.path.basename(os.path.dirname(pth)),
                    "image_path": os.path.normpath(pth),
                })
            df = pd.DataFrame(rows)

        labeled = ("decision" in df.columns)
        if args.png_csv_path and not labeled:
            raise ValueError(f"png_csv_path must contain columns {{'image_path','decision'}} for labeled evaluation, got {set(df.columns)}")

        # Resolve paths (scan mode already provides abs_path)
        if "abs_path" not in df.columns:
            def _resolve_path(p: str) -> str:
                p = str(p)
                if os.path.isabs(p):
                    return os.path.normpath(p)
                if args.png_root:
                    return os.path.normpath(os.path.join(args.png_root, p))
                # fall back to relative to the csv location
                return os.path.normpath(os.path.join(os.path.dirname(args.png_csv_path), p))

            df["abs_path"] = df["image_path"].apply(_resolve_path)
        if (df["abs_path"].apply(lambda x: not os.path.exists(x))).any():
            missing = df[df["abs_path"].apply(lambda x: not os.path.exists(x))]["abs_path"].head(10).tolist()
            raise FileNotFoundError(f"Some image paths do not exist. Examples: {missing}")

        # Map labels (optional)
        if labeled:
            df["doctor_keep"] = (df["decision"].astype(str).str.lower() == "keep").astype(int)
        else:
            df["doctor_keep"] = -1

        # Batch inference (simple loop; 400 images is tiny)
        probs = []
        preds = []
        with torch.no_grad():
            for p in tqdm(df["abs_path"].tolist(), desc="Inference (PNG)"):
                x = _load_png_tensor(p).unsqueeze(0).to(device)  # (1,1,H,W)
                out = model(x)
                prob = torch.softmax(out, dim=1)[0, 1].item()
                pred = int(torch.argmax(out, dim=1)[0].item())
                probs.append(prob)
                preds.append(pred)

        df["prob_artery"] = probs
        df["predicted_label"] = preds

        out_csv = args.inference_output
        df.to_csv(out_csv, index=False)
        print(f"✅ Saved PNG inference results to {out_csv}")

        # Metrics (only if labels are available)
        if labeled:
            y = df["doctor_keep"].values
            p = df["prob_artery"].values

            try:
                roc = roc_auc_score(y, p)
                ap = average_precision_score(y, p)
                print(f"ROC-AUC: {roc:.4f}")
                print(f"PR-AUC : {ap:.4f}")
            except Exception as e:
                print(f"⚠️ Could not compute AUC metrics: {e}")

            # Precision at high recall
            prec, rec, thr = precision_recall_curve(y, p)
            target_recall = 0.95
            ok = np.where(rec >= target_recall)[0]
            if len(ok) > 0:
                best_idx = ok[np.argmax(prec[ok])]
                print(f"Best precision with recall>={target_recall:.2f}: {prec[best_idx]:.4f} (recall={rec[best_idx]:.4f})")
            else:
                print(f"Recall never reaches {target_recall:.2f} on this set.")
        else:
            print("ℹ️ Unlabeled PNG run: skipping ROC/PR metrics.")

        # Best-of-K simulation by category (prompt) (only meaningful when labeled)
        if labeled and "category" in df.columns:
            for K in [1, 3, 5]:
                hits = []
                for g, sub in df.groupby("category"):
                    sub = sub.sort_values("prob_artery", ascending=False)
                    topk = sub.head(K)
                    hits.append(int(topk["doctor_keep"].max() == 1))
                print(f"Best-of-{K} success rate over categories: {float(np.mean(hits)):.4f} ({sum(hits)}/{len(hits)})")
        elif not labeled:
            print("ℹ️ Unlabeled PNG run: skipping best-of-K success metrics (still visualizing top-K).")
        else:
            print("ℹ️ No 'category' column found; skipping best-of-K simulation.")

        # Optional visualization: top-K per category (reuses --num_samples_to_visualize)
        if args.num_samples_to_visualize > 0:
            vis_path = out_csv.replace('.csv', f'_top{args.num_samples_to_visualize}.png')
            os.makedirs(os.path.dirname(vis_path) or '.', exist_ok=True)
            visualize_png_predictions(df, vis_path, top_k=args.num_samples_to_visualize)
            print(f"✅ PNG grid saved at: {vis_path}")

        return

    elif args.mode == "inference_series":
        if not args.ckpt_path:
            raise ValueError("inference_series mode requires --ckpt_path to load model.")

        model.load_state_dict(torch.load(args.ckpt_path))
        model.eval()

        dataset = UMMDSAClassificationDataset(
            metadata_path=args.metadata_path,
            transform=inference_transform,
            inference_mode=True
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=60)

        df_rows = []
        with torch.no_grad():
            for batch in tqdm(loader, desc="Inference (series)"):
                images = batch["image"].to(device)
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)
                for path, frame_idx, series_id, pred, prob in zip(
                    batch['file_path'], batch['frame_idx'], batch['series_id'],
                    preds.cpu(), probs[:, 1].cpu()
                ):
                    df_rows.append({
                        "file_path": path,
                        "frame_idx": int(frame_idx),
                        "series_id": series_id,
                        "predicted_label": int(pred),
                        "prob_artery": float(prob)
                    })

        df = pd.DataFrame(df_rows)

        # Drop series with fewer than 8 frames
        min_frames = 8
        min_artery_frames = 2  # Keep if at least 2/8 frames are predicted as artery
        selected_segments = []
        for sid, group in df.groupby("series_id"):
            group = group.sort_values("frame_idx").reset_index(drop=True)
            if len(group) < min_frames:
                continue

            found = False
            for i in range(len(group) - min_frames + 1):
                window = group.iloc[i : i + min_frames]
                if (window["predicted_label"] == 1).sum() >= min_artery_frames:
                    selected_segments.append(window)
                    found = True
                    break

            if not found:
                continue

        if selected_segments:
            final_df = pd.concat(selected_segments)
            out_path = args.inference_output.replace(".csv", "_series_segment.csv")
            final_df.to_csv(out_path, index=False)
            print(f"✅ Saved series-segment CSV to {out_path}")

            # Generate artery-only CSV and visualize like regular inference
            generate_artery_only_csv(out_path, args.metadata_path, keep_all_frames_in_series=True)

            if args.num_samples_to_visualize > 0:
                df = pd.read_csv(out_path, dtype={"frame_idx": int})
                samples = df.sample(n=min(args.num_samples_to_visualize, len(df)))
                output_img_path = out_path.replace(".csv", ".png")
                visualize_predictions(samples, output_img_path)
        else:
            print("⚠️ No series met the minimum frame requirement.")

if __name__ == "__main__":
    main()