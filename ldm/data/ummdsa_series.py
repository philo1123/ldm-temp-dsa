

import torch
from collections import defaultdict
from .ummdsa import UMMDSADataset

class UMMDSASeriesDataset(UMMDSADataset):
    def __init__(self, metadata_path, size=None, transform=None, split="default"):
        super().__init__(metadata_path, size=size, transform=transform, split=split)

        # Group metadata by series_id
        self.series_dict = defaultdict(list)
        for _, row in self.metadata.iterrows():
            sid = row['series_id']
            self.series_dict[sid].append(row)

        # Sort frames in each series by frame index
        for sid in self.series_dict:
            self.series_dict[sid] = sorted(self.series_dict[sid], key=lambda r: r['frame_idx'])

        self.series_keys = list(self.series_dict.keys())

    def __len__(self):
        return len(self.series_keys)

    def __getitem__(self, index):
        sid = self.series_keys[index]
        frames = self.series_dict[sid]

        images = []
        for row in frames:
            img = self.load_image(row['file_path'], row['frame_idx'])
            if self.transform:
                img = self.transform(img)
            images.append(img)

        video = torch.stack(images, dim=1)  # [C, T, H, W]

        return {
            "image": video,
            "series_id": sid,
            "num_frames": len(images),
        }
        
class UMMDSASeriesTrain(UMMDSASeriesDataset):
    """
    UMM DSA Series training dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(split="train", **kwargs)

class UMMDSASeriesValidation(UMMDSASeriesDataset):
    """
    UMM DSA Series validation dataset.
    """
    def __init__(self, **kwargs):
        super().__init__(split="vali", **kwargs)
