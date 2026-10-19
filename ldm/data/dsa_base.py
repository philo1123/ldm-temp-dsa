import os
import glob
import re
import pydicom
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from PIL import Image
import torchvision.transforms as transforms
import torch
from omegaconf import OmegaConf, DictConfig
from ldm.util import instantiate_from_config

from .transform import create_transform_pipeline

from abc import ABC, abstractmethod
from torch.utils.data import Dataset
import torch
from PIL import Image

class BaseDSADataset(Dataset):
    """
    A base class for DSA (Digital Subtraction Angiography) datasets.
    This class provides common functionality for both DIAS and DSCA datasets.
    
    Args:
        paths (list or str): A list of directories (or a single directory) containing images.
        size (int, optional): Target size for resizing images. Default is None.
        transform (callable, optional): A transform pipeline to process images. If None, a default pipeline is used.
    """
    def __init__(self, paths, size=None, transform=None, split="default"):
        if not isinstance(paths, list):
            paths = list(paths)
        self.paths = paths
        self.split = split
        self.size = size
        
        # Initialize transform pipeline
        if transform is None:
            self.transform = create_transform_pipeline(size)
        elif isinstance(transform, DictConfig):
            self.transform = instantiate_from_config(transform)
        else:
            self.transform = transform

    @abstractmethod
    def load_image(self, file_path, frame_idx=None):
        """
        Load an image from a file. For DICOM files, frame_idx may be used.
        Should return a PIL Image.
        """
        pass

    @abstractmethod
    def get_sample_info(self, index):
        """
        Retrieve sample information for the given index.
        Returns a dict with:
            - file_path: str, path to the image file
            - frame_idx: int, frame index within the file
            - series_id: int, series identifier (used for grouping frames)
        """
        pass

    @abstractmethod
    def get_extra_info(self, file_path, frame_idx=None, series_id=None):
        """
        Compute additional information from the file_path and frame_idx.
        
        Args:
            file_path: Path to the image file
            frame_idx: Frame index within the file
            series_id: Series identifier (used for grouping frames)
            
        Returns:
            Dict with keys:
            - norm_time: tensor of shape [1,1], normalized time value
            - caption: str, used for conditional generation
        """
        pass

    def __getitem__(self, index):
        """
        Standard item getter for all DSA datasets.
        Returns a standardized dictionary with the following keys:
        - image: Tensor of shape [H, W, C]
        - file_path: Path to the source file
        - frame_idx: Frame index within the file (0 for single-frame images)
        - series_id: Series identifier tensor of shape [1,1]
        - split: Dataset split ("train", "vali", etc.)
        - norm_time: Normalized time value tensor of shape [1,1]
        - caption: Textual description of the image used as condition
        """
        try:
            # Get basic sample info
            info = self.get_sample_info(index)
            file_path = info["file_path"]
            frame_idx = info.get("frame_idx", None)
            series_id = info.get("series_id", 0)
            
            # Load and transform the image
            image = self.load_image(file_path, frame_idx)
            if self.transform is not None:
                image_tensor = self.transform(image)
            else:
                image_tensor = torch.tensor(image)
            
            # Convert to [H, W, C] format for downstream network
            image_tensor = image_tensor.permute(1, 2, 0)
            
            # Get additional information (called extra_meta in sample info, merged with computed extra info)
            extra = self.get_extra_info(file_path, frame_idx, series_id)
            extra_meta = info.get("extra_meta", {})
            extra.update(extra_meta)
            
            # Ensure all required fields are present
            if not all(k in extra for k in ["norm_time", "caption"]):
                raise ValueError("get_extra_info must return a dict with keys: norm_time, caption")
            
            # Preserve series_id as string (or empty if not provided)
            series_id_str = str(series_id) if series_id is not None else ""
            
            # Combine all information into a single dictionary with standardized format
            sample = {
                "image": image_tensor,
                "file_path": file_path,
                "frame_idx": frame_idx,
                "series_id": series_id_str,
                "split": self.split,
                "norm_time": extra["norm_time"],
                "caption": extra["caption"],
            }
            return sample
        except Exception as e:
            print(f"❌ Error in __getitem__ at index {index}: {e}")
            raise

    @abstractmethod
    def __len__(self):
        pass
