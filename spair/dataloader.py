from collections import defaultdict
from typing import Optional, Tuple, Dict, Any, List, Union
from pathlib import Path

import h5py
import json
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


class CLEVR(torch.utils.data.Dataset):
    datasets = {
        "train": ("images/train", "scenes/CLEVR_train_scenes.json"),
        "test": ("images/val", "scenes/CLEVR_val_scenes.json"),
    }

    def __init__(
        self,
        file_path: str,
        subset: str = "train",
    ):
        super().__init__()
        self.file_path = Path(file_path)
        self.subset = "train"
        self.image_dir, annotations_file = self.datasets[self.subset]
        with self.file_path.joinpath(annotations_file).open("r") as fp:
            self.annotations = json.load(fp)["scenes"]
        self.image_names = [ann["image_filename"] for ann in self.annotations]

    def __len__(self):
        """Get dataset length."""
        return len(self.image_names)

    def extract_bbox_and_label(
            self, scene: Dict[str, Any]
    ) -> Dict[str, List[Union[int, float]]]:
        """Create bbox and label from scene annotation.
        .. note: sourced from
           https://github.com/larchen/clevr-vqa/blob/master/bounding_box.py
        :param scene: scene annotation dict according to CLEVR format
        :return: Dict with list of bbox params: x_min, y_min, x_max, y_max, class
        """
        annotation: Dict[str, List[Union[int, float]]] = defaultdict(list)
        objs = scene["objects"]
        rotation = scene["directions"]["right"]

        for obj in objs:
            x, y, _ = obj["pixel_coords"]
            x1, y1, z1 = obj["3d_coords"]

            cos_theta, sin_theta, _ = rotation

            x1 = x1 * cos_theta + y1 * sin_theta
            y1 = x1 * -sin_theta + y1 * cos_theta

            height_d = height_u = width_l = width_r = 6.9 * z1 * (15 - y1) / 2.0

            if obj["shape"] == "cylinder":
                d = 9.4 + y1
                h = 6.4
                s = z1

                height_u *= (s * (h / d + 1)) / ((s * (h / d + 1)) - (s * (h - s) / d))
                height_d = height_u * (h - s + d) / (h + s + d)

                width_l *= 11 / (10 + y1)
                width_r = width_l

            elif obj["shape"] == "cube":
                height_u *= 1.3 * 10 / (10 + y1)
                height_d = height_u
                width_l = height_u
                width_r = height_u

            class_name = (
                f"{obj['size']} {obj['color']} {obj['material']} {obj['shape']}"
            )
            annotation["class"].append(self.CLASS_LABELS.index(class_name) + 1)
            annotation["x_min"].append(max(0, x - width_l))
            annotation["y_min"].append(max(0, y - height_d))
            annotation["x_max"].append(min(480, x + width_r))
            annotation["y_max"].append(min(320, y + height_u))

        return annotation

    def _get_image(self, item: int) -> torch.Tensor:
        image_file = self.image_names[item]
        image_path = self.file_path.joinpath(self.image_dir).joinpath(image_file)
        image = np.array(PIL.Image.open(image_path).convert("RGB")) / 255
        return torch.tensor(image)

    def _get_annotation(self, item: int) -> Tuple[torch.Tensor, torch.Tensor]:
        scene = self.annotations[item]
        ann = self.extract_bbox_and_label(scene)
        boxes = torch.tensor(
            list(zip(ann["x_min"], ann["y_min"], ann["x_max"], ann["y_max"])),
            dtype=torch.float32,
        )
        labels = torch.tensor(ann["class"], dtype=torch.int64)
        return boxes, labels

    def __getitem__(self, item) -> Tuple[torch.Tensor, np.ndarray, int]:
        image_file = self.image_names[item]
        image_path = self.file_path.joinpath(self.image_dir).joinpath(image_file)
        image = np.array(PIL.Image.open(image_path).convert("RGB").resize((128, 128), Image.BICUBIC)) / 255
        boxes, labels = self._get_annotation(item)
        mask = np.where(labels != -1)
        return torch.from_numpy(image).float().permute(2, 0, 1), boxes.astype(np.float), len(mask)