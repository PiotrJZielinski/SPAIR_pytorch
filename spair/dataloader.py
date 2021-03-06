from typing import Optional, Tuple

import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils import data

class SimpleScatteredMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, subset):
        super().__init__()
        self.dataset = h5py.File(file_path, "r")["train/{}".format(subset)]
        self.episode = None

        # static_img = self.dataset[9, ...]
        # img_size = cfg.INPUT_IMAGE_SHAPE[-1]
        # self.static_img = cv2.resize(static_img, dsize=(img_size,img_size))

    def __getitem__(self, index) -> Tuple[np.ndarray, np.ndarray, int]:
        ret = []

        obs = self.dataset["image"][index, ...]
        # obs = self.static_img
        # obs = np.zeros_like(obs)
        obs = obs[..., None]  # Add channel dimension
        image = np.moveaxis(obs, -1, 0)  # move from (x, y, c) to (c, x, y)

        bbox = self.dataset["bbox"][index, ...]

        digit_count = self.dataset["digit_count"][index, ...]

        return image, bbox, digit_count

    def __len__(self):
        return self.dataset["image"].shape[0]


class MultiScaleMNIST(torch.utils.data.Dataset):

    def __init__(
        self,
        file_path: str,
        subset: str = "train",
    ):
        super().__init__()
        self.file_path = file_path
        self.subset = "train"
        with h5py.File(self.file_path, "r") as file:
            self.dataset_length = len(file[self.subset]["images"])
        self.dataset: Optional[h5py.File] = None

    def __len__(self):
        """Get dataset length."""
        return self.dataset_length

    def __getitem__(self, item) -> Tuple[torch.Tensor, np.ndarray, int]:
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, "r")[self.subset]
        image = self.dataset["images"][item]
        pil_image = Image.fromarray(image).resize((128, 128), Image.BICUBIC)
        image = np.array(pil_image) / 255
        boxes = self.dataset["boxes"][item]  # type: ignore
        labels = self.dataset["labels"][item]  # type: ignore
        mask = np.where(labels != -1)
        return torch.from_numpy(image).float().permute(2, 0, 1), boxes.astype(np.float), len(mask)
