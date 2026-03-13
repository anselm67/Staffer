import os
import random
from pathlib import Path
from typing import Optional, Union

import torch
from torch import Tensor
from torchvision.io import decode_image
from torchvision.transforms import InterpolationMode, v2
from torchvision.tv_tensors import Mask

from model import Config

CVCMUSCIMA_PATH = Path("/home/anselm/datasets/CvcMuscima-Distortions")


class PadToSize(v2.Transform):

    def __init__(self, config: Config):
        super(PadToSize, self).__init__()
        self.config = config

    def transform(self, tensor: Union[Mask, Tensor]) -> Tensor:
        _, h, w = tensor.shape
        scale = max(
            h / self.config.image_shape[0], w / self.config.image_shape[1])
        if scale > 1.0:
            h, w = int(h / scale), int(w / scale)
            tensor = v2.functional.resize(
                tensor, [h, w],
                interpolation=self.config.interpolation,
                antialias=self.config.antialias,
            )
        pad = torch.full(self.config.image_shape, 0, dtype=torch.float32)
        pad[:h, :w] = tensor
        return pad.to(torch.float32)

    def forward(
        self, image: Tensor, mask: Optional[Union[Mask, Tensor]] = None
    ) -> tuple[Tensor, Optional[Tensor]]:
        assert mask is None or image.shape == mask.shape, "Image an mask should have the same shape."
        # Pads and resizes the image and the mask.
        image = self.transform(image)
        image = (image - image.mean()) / image.std()
        # Thresold the mask to { 0, 1 }.
        if mask is not None:
            mask = self.transform(mask)
            mask = (mask > 0).to(torch.float32)
        return image.unsqueeze(0), mask


class PredictTransform(v2.Transform):

    def __init__(self, config: Config):
        super(PredictTransform, self).__init__()
        self.config = config

    def forward(self, tensor: Tensor) -> Tensor:
        _, h, w = tensor.shape
        tensor = 255 - tensor
        scale = max(
            h / self.config.max_height, w / self.config.max_width)
        if scale > 1.0:
            target_h, target_w = int(h / scale), int(w / scale)
            tensor = v2.functional.resize(
                tensor,
                [target_h, target_w],
                interpolation=self.config.interpolation,
                antialias=self.config.antialias
            )
        return tensor


class StaffDataset(torch.utils.data.Dataset):

    home: Path
    data: list[tuple[Path, Path]]
    config: Config

    train_transform: v2.Compose
    predict_transform: v2.Compose

    @staticmethod
    def create(
        home: Path = CVCMUSCIMA_PATH, config: Config = Config()
    ) -> tuple['StaffDataset', 'StaffDataset']:
        image_directories: list[Path] = list()
        # Collects all image directories.
        for root, names, _ in os.walk(home):
            for name in names:
                path = Path(root) / name
                if name == "image" and path.is_dir():
                    image_directories.append(path)
        # For each image directories, collects (image, staves):
        data: list[tuple[Path, Path]] = list()
        for img_dir in image_directories:
            staff_dir = img_dir.parent / "gt"
            for img_file in img_dir.iterdir():
                data.append((img_file, staff_dir / img_file.name))
        train_len = int((1.0 - config.valid_split) * len(data))
        return StaffDataset(config, data[:train_len]), StaffDataset(config, data[train_len:])

    def __init__(self, config: Config, data: list[tuple[Path, Path]]):
        self.config = config
        self.data = data
        # Inits the transforms.
        self.train_transform = v2.Compose([
            PadToSize(config)
        ])
        self.predict_transform = v2.Compose([
            v2.Grayscale(1),
            PredictTransform(config),
            self.train_transform
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path, mask_path = self.data[idx]
        return self.train_transform(
            decode_image(img_path.as_posix()),
            Mask(decode_image(mask_path.as_posix())),
        )

    def pick_one(self) -> tuple[Tensor, Tensor]:
        return self[random.randint(0, len(self) - 1)]

    def stats(self):
        max_width, max_height = 0, 0
        for img_path, staff_path in self.data:
            img = decode_image(img_path.as_posix())
            h, w = img.shape[-2:]
            max_height, max_width = max(max_height, h), max(max_width, w)

            staff = decode_image(staff_path.as_posix())
            assert staff.shape[-2:] == img.shape[-2:], "Image and mask size mismatch!"
        print(f"{len(self.data)} images: {max_height=}, {max_width=}")
