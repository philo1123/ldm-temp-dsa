from ldm.data.ummdsa import UMMDSADataset
import torch
import pandas as pd

class SafeMetadata(pd.DataFrame):
    def __getitem__(self, key):
        if key in self.columns:
            return super().__getitem__(key).fillna(0.0)
        else:
            print(f"Warning: Column '{key}' not found in metadata. Returning default=0.0.")
            return pd.Series([0.0] * len(self), index=self.index)

class UMMDSAClassificationDataset(UMMDSADataset):
    """
    UMM DSA dataset for classification tasks (artery/negative).
    Assumes metadata CSV contains 'label' column unless in inference mode.
    """
    def __init__(self, metadata_path, size=None, transform=None, split="default", inference_mode=False):
        self.metadata = SafeMetadata(pd.read_csv(metadata_path))
        self.transform = transform
        self.size = size
        self.split = split
        self.inference_mode = inference_mode
        
        self.all_images = list(zip(
            self.metadata['file_path'],
            self.metadata['frame_idx'],
            self.metadata['group'],
        ))
        
        if not self.inference_mode:
            if 'label' not in self.metadata.columns:
                raise ValueError("Classification dataset requires 'label' column in metadata CSV.")
            self.labels = list(self.metadata['label'])

    def safe_metadata_value(self, index, key, default=None):
        try:
            return self.metadata.loc[index, key]
        except KeyError:
            # print(f"Warning: metadata key '{key}' not found at index {index}, using default: {default}")
            return default

    def get_sample_info(self, index):
        sample = super().get_sample_info(index)
        sample['series_id'] = self.safe_metadata_value(index, 'series_id', default=str(index))
        if not self.inference_mode:
            sample['label'] = self.labels[index]
        return sample
    
    def __getitem__(self, index):
        sample = super().__getitem__(index)
        image = sample["image"]
        if image.ndim == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)
        if image.ndim == 2:
            image = image.unsqueeze(0)
        sample["image"] = image
        sample["series_id"] = sample["series_id"]  # Keep as string for downstream use
        if not self.inference_mode:
            sample['label'] = torch.tensor(self.labels[index], dtype=torch.long)
        return sample

    def get_extra_info(self, file_path, frame_idx, series_id):
        return {
            'series_id': series_id,
            'norm_time': 0.0,           # default 0
            'caption': ''               # default empty caption
        }