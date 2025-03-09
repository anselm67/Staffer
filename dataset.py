import os
import random
from pathlib import Path
from typing import Literal, Union

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
        target_h, target_w = h // self.config.divider, w // self.config.divider
        tensor = v2.functional.resize(
            tensor,
            [target_h, target_w],
            interpolation=InterpolationMode.NEAREST)
        pad = torch.full(self.config.image_shape, 0)
        pad[:target_h, :target_w] = tensor
        return pad.to(torch.float32)

    def forward(self, image: Tensor, mask: Union[Mask, Tensor]) -> tuple[Tensor, Tensor]:
        assert image.shape == mask.shape, "Image an mask should have the same shape."
        # Pads and resizes the image and the mask.
        image, mask = self.transform(image), self.transform(mask)
        # Normalizes the image.
        image = (image - image.mean()) / image.std()
        # Thresold the mask to { 0, 1 }.
        # mask = (mask > 0).to(torch.float32)
        return image.unsqueeze(0), mask


class StaffDataset(torch.utils.data.Dataset):

    home: Path
    data: list[tuple[Path, Path]]
    config: Config

    train_transform: v2.Compose

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
        random.shuffle(data)
        split = int(config.valid_split * len(data))
        return StaffDataset(config, data[:split]), StaffDataset(config, data[split:])

    def __init__(self, config: Config, data: list[tuple[Path, Path]]):
        self.config = config
        self.data = data
        # Inits the transforms.
        self.train_transform = v2.Compose([
            PadToSize(config)
        ])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        img_path, mask_path = self.data[idx]
        return self.train_transform(
            decode_image(img_path.as_posix()),
            Mask(decode_image(mask_path.as_posix())),
        )

    def stats(self):
        max_width, max_height = 0, 0
        for img_path, staff_path in self.data:
            img = decode_image(img_path.as_posix())
            h, w = img.shape[:2]
            max_height, max_width = max(max_height, h), max(max_width, w)

            staff = decode_image(staff_path.as_posix())
            assert staff.shape[0] == h and staff.shape[1] == w, "Image and mask size mismatch!"
        print(f"{len(self.data)} images: {max_height=}, {max_width=}")
