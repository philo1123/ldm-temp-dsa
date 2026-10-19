import os
import pandas as pd
import pydicom
from tqdm import tqdm
from PIL import Image
import random
import numpy as np
import shutil

def get_value(dataset, tag, default=None):
    try:
        return dataset[tag].value
    except Exception:
        return default

def determine_phase_label(frame_idx, rep_frame, num_frames, mask_frame):
    if frame_idx <= mask_frame + 2 or frame_idx >= rep_frame + 5:
        return 0  # negative
    elif (rep_frame - 1) <= frame_idx <= (rep_frame + 1):
        return 1  # artery phase
    else:
        return None

def generate_multiframe_label_csv(input_csv, output_dir, window=2):
    df = pd.read_csv(input_csv)

    # only keep records with is_multiframe == True
    multiframe_df = df[df['is_multiframe']].copy()
    if multiframe_df.empty:
        print("⚠️ No multi-frame DICOMs found.")
        return

    labeled_records = []

    for _, row in tqdm(multiframe_df.iterrows(), total=len(multiframe_df), desc="Labeling multi-frame records"):
        try:
            dcm = pydicom.dcmread(row['file_path'], stop_before_pixels=True)
            rep_frame = get_value(dcm, (0x0028, 0x6010))
            num_frames = get_value(dcm, (0x0028, 0x0008))
            mask_frame = get_value(dcm, (0x0025, 0x1008))
            if mask_frame is None:
                mask_frame = 1  # fallback
            if rep_frame is None or num_frames is None:
                print(f"⚠️ Representative Frame Number or Number of Frames missing in {row['file_path']}")
                continue
            rep_frame = int(rep_frame)
            num_frames = int(num_frames)
            mask_frame = int(mask_frame)
            frame_idx = int(row['frame_idx'])

            label = determine_phase_label(frame_idx, rep_frame, num_frames, mask_frame)

            if label is not None:
                labeled_records.append({**row.to_dict(), 'label': label})

        except Exception as e:
            print(f"⚠️ Failed to label {row['file_path']}: {e}")

    if labeled_records:
        labeled_df = pd.DataFrame(labeled_records)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "multiframe_labeled.csv")
        labeled_df.to_csv(output_path, index=False)
        print(f"✅ Saved multi-frame labeled CSV to: {output_path}")
    else:
        print("⚠️ No labeled multi-frame records found.")

def save_sample_images(labeled_csv, save_dir, num_samples=5):
    df = pd.read_csv(labeled_csv)

    pos_dir = os.path.join(save_dir, "positive")
    neg_dir = os.path.join(save_dir, "negative")

    # Clear previous samples if they exist
    for d in [pos_dir, neg_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]

    # Sample num_samples from each class independently
    pos_samples = pos_df.sample(min(num_samples, len(pos_df)), random_state=42)
    neg_samples = neg_df.sample(min(num_samples, len(neg_df)), random_state=42)

    dsa_dataset_root = "/home/midorai-data/dsa_dataset/"

    for _, row in tqdm(pos_samples.iterrows(), total=len(pos_samples), desc="Saving positive samples"):
        dcm = pydicom.dcmread(row['file_path'])
        frame_idx = int(row['frame_idx']) - 1
        img = dcm.pixel_array[frame_idx]
        img = img.astype(np.float32)
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        img = (img * 255).astype(np.uint8)
        relative_path = os.path.relpath(row['file_path'], start=dsa_dataset_root)
        relative_path = relative_path.replace('/', '__')
        save_filename = f"{relative_path}_frame{frame_idx+1}.png"
        Image.fromarray(img).save(os.path.join(pos_dir, save_filename))

    for _, row in tqdm(neg_samples.iterrows(), total=len(neg_samples), desc="Saving negative samples"):
        dcm = pydicom.dcmread(row['file_path'])
        frame_idx = int(row['frame_idx']) - 1
        img = dcm.pixel_array[frame_idx]
        img = img.astype(np.float32)
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        img = (img * 255).astype(np.uint8)
        relative_path = os.path.relpath(row['file_path'], start=dsa_dataset_root)
        relative_path = relative_path.replace('/', '__')
        save_filename = f"{relative_path}_frame{frame_idx+1}.png"
        Image.fromarray(img).save(os.path.join(neg_dir, save_filename))

    print(f"✅ Saved sample images to {save_dir}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Filter multi-frame DSA dataset and label artery phase.")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input metadata CSV.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the labeled CSV.")
    parser.add_argument("--window", type=int, default=2, help="Frame window size around representative frame to consider as artery phase.")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of positive and negative samples to save if > 0.")
    args = parser.parse_args()

    generate_multiframe_label_csv(args.input_csv, args.output_dir, args.window)

    if args.num_samples > 0:
        save_sample_images(os.path.join(args.output_dir, "multiframe_labeled.csv"), args.output_dir, args.num_samples)