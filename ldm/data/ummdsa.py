import os
import glob
import re
import pydicom
import numpy as np
import pandas as pd
from PIL import Image
import torch
from tqdm import tqdm
from .dsa_base import BaseDSADataset
import sys

def encode_categorical_fields(group, plane_type):
    group_map = {'anterior': 0, 'posterior': 1}
    plane_map = {'single_a': 0, 'single_b': 1, 'biplane_a': 2, 'biplane_b': 3}
    group_id = group_map.get(group.lower(), -1)
    plane_id = plane_map.get(plane_type.lower(), -1)
    return group_id, plane_id

def extract_plane_type(image_type_str):
    """
    Extract standardized plane information from the image_type string.

    Returns:
        One of: 'single_a', 'single_b', 'biplane_a', 'biplane_b', or 'unknown'
    """
    if not isinstance(image_type_str, str):
        return 'unknown'
    lowered = image_type_str.lower()
    if "single plane" in lowered:
        if "single a" in lowered:
            return 'single_a'
        elif "single b" in lowered:
            return 'single_b'
    elif "biplane" in lowered:
        if "biplane a" in lowered:
            return 'biplane_a'
        elif "biplane b" in lowered:
            return 'biplane_b'
    return 'unknown'

class UMMDSADataset(BaseDSADataset):
    """
    A dataset class for UMM DSA DICOM images.
    
    This dataset reads DICOM files from a folder structure organized by year/patient/study/anatomy.
    Each DICOM file can be either a single frame (h,w) or multiple frames (frame,h,w).
    
    Args:
        metadata_path (str): Path to the dataset metadata CSV file containing StudyPath information.
        size (int, optional): Target size for resizing the image (width = height = size). Default is None.
        transform (callable, optional): A transform pipeline to process images. If None, a default pipeline is used.
    """
    _split_cache = {}  # Class-level cache per split

    def __init__(self, metadata_path, size=None, transform=None, split="default"):
        super().__init__(paths=[], size=size, transform=transform, split=split)
        if split in UMMDSADataset._split_cache:
            self.metadata, self.all_images, self.frame_counts = UMMDSADataset._split_cache[split]
            return

        self.metadata = pd.read_csv(metadata_path)
        # Replace NaN angles with 0.0
        self.metadata['primary_angle'] = self.metadata['primary_angle'].fillna(0.0)
        self.metadata['secondary_angle'] = self.metadata['secondary_angle'].fillna(0.0)

        # Copy raw angles before normalization
        self.metadata['primary_angle_raw'] = self.metadata['primary_angle']
        self.metadata['secondary_angle_raw'] = self.metadata['secondary_angle']

        self.metadata['plane_type'] = self.metadata['image_type'].apply(extract_plane_type)

        # Z-score normalize the angles
        self.primary_mean = self.metadata['primary_angle'].mean()
        self.primary_std = self.metadata['primary_angle'].std()
        self.secondary_mean = self.metadata['secondary_angle'].mean()
        self.secondary_std = self.metadata['secondary_angle'].std()

        self.metadata['primary_angle'] = (self.metadata['primary_angle'] - self.primary_mean) / max(self.primary_std, 1e-6)
        self.metadata['secondary_angle'] = (self.metadata['secondary_angle'] - self.secondary_mean) / max(self.secondary_std, 1e-6)

        self.all_images = list(zip(
            self.metadata['file_path'],
            self.metadata['frame_idx'],
            self.metadata['primary_angle'],
            self.metadata['secondary_angle'],
            self.metadata['group'],
            self.metadata['plane_type']
        ))
        # Build a dictionary mapping file_path to total number of frames
        self.frame_counts = (
            self.metadata.groupby('file_path')['frame_idx']
            .max()
            .apply(lambda x: x + 1)  # frame_idx starts from 0
            .to_dict()
        )
        UMMDSADataset._split_cache[split] = (
            self.metadata,
            self.all_images,
            self.frame_counts,
        )

    def load_image(self, file_path, frame_idx=None):
        """
        Load a frame from a DICOM file and convert it to grayscale PIL Image.
        
        Args:
            file_path: Path to the DICOM file
            frame_idx: Index of the frame to load (for multi-frame DICOM files)
                      If None, the first frame is loaded
        
        Returns:
            PIL Image of the specified frame
        """
        ds = pydicom.dcmread(file_path)
        img = ds.pixel_array
        
        # Handle multi-frame DICOM files
        if len(img.shape) == 3 and frame_idx is not None:
            img = img[frame_idx]
        
        # Normalize to 0-255 range
        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min) * 255
        else:
            img = np.zeros_like(img)  # If image is constant, fill with zeros
        img = img.astype(np.uint8)
        return Image.fromarray(img)

    def get_sample_info(self, index):
        """
        Get the file path, frame index, and series ID for the given index.
        For UMM DSA, we use the study ID as the series ID.
        """
        sample = self.all_images[index]
        file_path = sample[0]
        frame_idx = sample[1]
        return {
            "file_path": file_path,
            "frame_idx": frame_idx,
            "series_id": 0,
        }
    
    def get_extra_info(self, file_path, frame_idx=None, series_id=None):
        """
        Get additional information for the given file path and frame index.
        
        Args:
            file_path: Path to the DICOM file
            frame_idx: Index of the frame in the DICOM file
            series_id: Study ID for the series
            
        Returns:
            Dict with norm_time and caption
        """
        # Calculate normalized time based on frame index and total frames
        if frame_idx is None:
            frame_idx = 0
            
        total_frames = self.frame_counts.get(file_path, 1)
        norm_time = frame_idx / max(1, total_frames - 1) if total_frames > 1 else 0.5
        
        # Convert to tensor format
        norm_time_tensor = torch.tensor(norm_time, dtype=torch.float32).unsqueeze(-1).unsqueeze(-1)
        
        row = self.metadata[(self.metadata['file_path'] == file_path) & (self.metadata['frame_idx'] == frame_idx)]
        if not row.empty:
            group = row.iloc[0]['group']
            plane = row.iloc[0]['plane_type']
            angle = row.iloc[0]['primary_angle_raw']
            secondary_angle = row.iloc[0]['secondary_angle_raw']
            caption = f"This is an {group} DSA scan taken in {plane} view, with a primary angle of {angle:.0f} degrees and a secondary angle of {secondary_angle:.0f} degrees."
        else:
            caption = "unknown scan"
        
        return {
            "norm_time": norm_time_tensor,
            "caption": caption
        }

    def __len__(self):
        return len(self.all_images)

class UMMDSATrain(UMMDSADataset):
    """
    UMM DSA training dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(split="train", **kwargs)

class UMMDSAValidation(UMMDSADataset):
    """
    UMM DSA validation dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(split="vali", **kwargs)
