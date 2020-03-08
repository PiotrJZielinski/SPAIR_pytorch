import os

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import extract_archive, iterable_to_str, verify_str_arg
from torchvision.transforms import Compose, ToTensor


class SimpleScatteredMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, subset):
        super().__init__()
        self.dataset = h5py.File(file_path, 'r')['train/{}'.format(subset)]
        self.episode = None

        # static_img = self.dataset[9, ...]
        # img_size = cfg.INPUT_IMAGE_SHAPE[-1]
        # self.static_img = cv2.resize(static_img, dsize=(img_size,img_size))

    def __getitem__(self, index):
        ret = []

        obs = self.dataset['image'][index, ...]
        # obs = self.static_img
        # obs = np.zeros_like(obs)
        obs = obs[..., None]  # Add channel dimension
        image = np.moveaxis(obs, -1, 0)  # move from (x, y, c) to (c, x, y)

        bbox = self.dataset['bbox'][index, ...]

        digit_count = self.dataset['digit_count'][index, ...]

        return image, bbox, digit_count

    def __len__(self):
        return self.dataset['image'].shape[0]


class CityscapesNoTarget(VisionDataset):

    def __init__(self, root, split='train', mode='fine', transform=None, ):
        if transform is None:
            transform = ToTensor()
        else:
            transform = Compose([transform, ToTensor()])
        super(CityscapesNoTarget, self).__init__(root, None, transform,
                                                 None)
        self.mode = 'gtFine' if mode == 'fine' else 'gtCoarse'
        self.images_dir = os.path.join(self.root, 'leftImg8bit', split)
        self.split = split
        self.images = []

        verify_str_arg(mode, "mode", ("fine", "coarse"))
        valid_modes = ("train", "train_extra", "test", "val")
        msg = ("Unknown value '{}' for argument split if mode is '{}'. "
               "Valid values are {{{}}}.")
        msg = msg.format(split, mode, iterable_to_str(valid_modes))
        verify_str_arg(split, "split", valid_modes, msg)

        if not os.path.isdir(self.images_dir):

            if split == 'train_extra':
                image_dir_zip = os.path.join(self.root,
                                             'leftImg8bit{}'.format('_trainextra.zip'))
            else:
                image_dir_zip = os.path.join(self.root, 'leftImg8bit{}'.format(
                    '_trainvaltest.zip'))

            if os.path.isfile(image_dir_zip):
                extract_archive(from_path=image_dir_zip, to_path=self.root)
            else:
                raise RuntimeError(
                    'Dataset not found or incomplete. Please make sure all required folders for the'
                    ' specified "split" and "mode" are inside the "root" directory')

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            for file_name in os.listdir(img_dir):
                self.images.append(os.path.join(img_dir, file_name))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is a tuple of all target types if target_type is a list with more
            than one item. Otherwise target is a json object if target_type="polygon", else the image segmentation.
        """

        image = Image.open(self.images[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.images)

    def extra_repr(self):
        lines = ["Split: {split}", "Mode: {mode}"]
        return '\n'.join(lines).format(**self.__dict__)
