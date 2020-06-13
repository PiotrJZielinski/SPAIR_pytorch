import h5py
import numpy as np
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

    def __getitem__(self, index):
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
