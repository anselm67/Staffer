#!/usr/bin/env python3

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
from cv2.typing import MatLike
from einops import rearrange
from numpy.typing import NDArray

CVCMUSCIMA_PATH = Path("/home/anselm/datasets/CvcMuscima-Distortions")
LOG_PATH = Path("untracked/vit_log.json")
MODEL_PATH = Path("untracked/model.pt")

type Source = Literal["data", "valid"]


@dataclass()
class Config:
    image_shape: tuple[int, int]

    # Maximums as obtained with the "stats" command.
    max_width: int = 3993
    max_height: int = 3325

    in_channels: int = 1
    divider: int = 5
    embed_dim: int = 256
    mlp_dim: int = 256

    num_heads: int = 8
    patch_size: int = 16
    dropout: float = 0.1
    num_layers = 4

    batch_size = 8
    valid_split: float = 0.2

    def scale_to_patch(self, value: int) -> int:
        ret = value // self.divider
        return int(round(ret / self.patch_size) * self.patch_size)

    def __init__(self):
        self.image_shape = (
            self.scale_to_patch(self.max_height),
            self.scale_to_patch(self.max_width),
        )
        assert self.patch_size ** 2 == self.embed_dim


def transform(img: MatLike, config: Config) -> torch.Tensor:
    h, w, _ = img.shape
    target_h, target_w = h // config.divider, w // config.divider
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (target_w, target_h))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    pad = torch.full(config.image_shape, 0)
    pad[:target_h, :target_w] = torch.tensor(img)
    return pad.to(torch.float32)


def image_transform(img: MatLike, config: Config) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape
    scale = min(h / config.image_shape[0], w / config.image_shape[1])
    target_h, target_w = int(h / scale), int(w / scale)
    img = cv2.resize(img, (target_w, target_h))
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    pad = torch.full(config.image_shape, 0)
    target_h, target_w = min(target_h, pad.shape[0]), min(
        target_w, pad.shape[1])
    pad[:target_h, :target_w] = torch.tensor(img)[:target_h, :target_w]
    return pad.to(torch.float32)


class Dataset:

    home: Path
    data: list[tuple[Path, Path]]
    valid: list[tuple[Path, Path]]
    config: Config

    def __init__(self, home=CVCMUSCIMA_PATH, config=Config()):
        self.home = home
        self.config = config
        image_directories: list[Path] = list()
        # Collects all image directories.
        for root, names, _ in os.walk(home):
            for name in names:
                path = Path(root) / name
                if name == "image" and path.is_dir():
                    image_directories.append(path)
        # For each image directories, collects (image, staves):
        data = list()
        for img_dir in image_directories:
            staff_dir = img_dir.parent / "gt"
            for img_file in img_dir.iterdir():
                data.append((img_file, staff_dir / img_file.name))
        random.shuffle(data)
        split = int(config.valid_split * len(data))
        self.data, self.valid = data[:split], data[split:]

    def pick_one(self, source: Source) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.data if source == "data" else self.valid
        img_path, staff_path = random.choice(data)
        return (
            transform(cv2.imread(img_path.as_posix()), self.config),
            transform(cv2.imread(staff_path.as_posix()), self.config),
        )

    def batch(self, source: Source) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.data if source == "data" else self.valid
        images = list()
        staves = list()
        for img_path, staff_path in random.sample(data, self.config.batch_size):
            # Reads and transforms the image.
            image = transform(cv2.imread(img_path.as_posix()), self.config)
            image = (image - image.mean()) / image.std()
            images.append(image.unsqueeze(0))

            # Reads and transforms the mask.
            staff = transform(cv2.imread(staff_path.as_posix()), self.config)
            staff = (staff > 0).to(torch.float32)
            staves.append(staff)

        return torch.stack(images), torch.stack(staves)

    def stats(self):
        max_width, max_height = 0, 0
        for img_path, staff_path in self.data:
            img = cv2.imread(img_path.as_posix())
            h, w = img.shape[:2]
            max_height, max_width = max(max_height, h), max(max_width, w)

            staff = cv2.imread(staff_path.as_posix())
            assert staff.shape[0] == h and staff.shape[1] == w, "Image and mask size mismatch!"
        print(f"{len(self.data)} images: {max_height=}, {max_width=}")


class PatchEmbedding(nn.Module):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.num_patch = (
            config.image_shape[0] // config.patch_size,
            config.image_shape[1] // config.patch_size)
        self.proj = nn.Conv2d(config.in_channels, config.embed_dim,
                              kernel_size=config.patch_size, stride=config.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(
            self.num_patch[0] * self.num_patch[1], config.embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x += self.pos_embed
        return x


class TransformerBlock(nn.Module):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.attn = nn.MultiheadAttention(
            config.embed_dim, config.num_heads, dropout=config.dropout, batch_first=True
        )
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, config.mlp_dim),
            nn.GELU(),
            nn.Linear(config.mlp_dim, config.embed_dim),
            nn.Dropout(config.dropout)
        )
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        x = x + self.attn(x_norm, x_norm, x_norm)[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):

    config: Config

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbedding(config)
        self.transformer = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.mask_head = nn.Linear(
            config.embed_dim,
            config.patch_size ** 2
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.mask_head(x)

        Hp = self.config.image_shape[0] // self.config.patch_size
        Wp = self.config.image_shape[1] // self.config.patch_size
        B, _, C = x.shape
        x = x.view(
            B, Hp, Wp,
            self.config.patch_size, self.config.patch_size)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.reshape(
            B, Hp * self.config.patch_size,
            Wp * self.config.patch_size
        )

        return x


@click.command()
@click.argument("path",
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default=CVCMUSCIMA_PATH)
def show(path: Path = CVCMUSCIMA_PATH):
    ds = Dataset(Path(path))
    while True:
        img, staff = tuple(map(lambda x: x.numpy(), ds.pick_one("data")))
        cv2.imshow("music", img)
        cv2.imshow("staff", cv2.bitwise_and(img, staff))

        if cv2.waitKey() == ord('q'):
            break


@click.command()
@click.argument("path",
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default=CVCMUSCIMA_PATH)
def stats(path: Path = CVCMUSCIMA_PATH):
    ds = Dataset(Path(path))
    ds.stats()


@click.command()
def summary():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViT(config).to(device)
    torchsummary.summary(model, input_size=(
        1, config.image_shape[0], config.image_shape[1]))


class Log:

    path: Path

    losses: list[float]

    def __init__(self, path=LOG_PATH):
        self.path = path
        self.load()

    def load(self):
        if self.path.exists():
            with open(self.path, "r") as fp:
                obj = json.load(fp)
            self.losses = obj["losses"]
        else:
            self.losses = list()
            self.losses = list()

    def save(self):
        with open(self.path, "w+") as fp:
            json.dump({
                "losses": self.losses,
            }, fp, indent=4)

    def log(self, loss: float):
        self.losses.append(loss)
        self.save()


@click.command()
@click.argument("path",
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default=CVCMUSCIMA_PATH)
@click.option("epochs", "-e", default=16, help="Number of epochs to train the model.")
@click.option("model_path", "-o", type=click.Path(file_okay=True, dir_okay=False),
              default=MODEL_PATH)
def train(path: Path = CVCMUSCIMA_PATH, epochs: int = 64, model_path: Path = MODEL_PATH):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = Dataset(Path(path))
    log = Log()
    config = Config()

    model = ViT(config).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    bce = nn.BCEWithLogitsLoss()

    batch_per_epoch = len(ds.data) // config.batch_size

    done = 0
    report_every = 20
    report_cycle = 0
    start_time = time.time()

    for e in range(epochs):

        for b in range(batch_per_epoch):

            opt.zero_grad()

            images, staves = ds.batch("data")
            images, staves = images.to(device), staves.to(device)

            yhat = model.forward(images)
            loss = bce(yhat, staves)

            loss.backward()
            opt.step()

            done += config.batch_size
            if done // report_every != report_cycle:
                report_cycle = done // report_every
                now = time.time()
                log.log(loss.item())
                print(
                    f"Epoch {e} {now - start_time:.2f}s: " +
                    f"processed {done} samples, batch {b} of {batch_per_epoch} " +
                    f", loss={loss.item():2.5f}"
                )
                start_time = now

        torch.save({
            "state_dict": model.state_dict(),
            "epoch": e,
            "count": done,
        }, Path(model_path).as_posix())


@click.command()
@click.argument("path",
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default=CVCMUSCIMA_PATH)
@click.option("model_path", "-m", type=click.Path(file_okay=True, dir_okay=False),
              default=MODEL_PATH)
@click.option("image_path", "-i", type=click.Path(file_okay=True, dir_okay=False, exists=True),
              required=False, default=None)
def predict(path: Path, model_path: Path, image_path: Optional[Path] = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = Config()
    obj = torch.load(model_path)
    ds = Dataset(Path(path), config)
    model = ViT(config)
    model.load_state_dict(obj["state_dict"])
    model.to(device)

    if image_path is not None:
        image = cv2.imread(Path(image_path).as_posix())
        image = image_transform(image, config)
        image = (image - image.mean()) / image.std()
        image = image.unsqueeze(0).to(device)

        yhat = model(image.unsqueeze(0))
        staff = (yhat.squeeze(0) > 0.5).to(torch.float32).cpu().numpy()

        cv2.imshow("image", image.squeeze(0).cpu().numpy())
        cv2.imshow("staff", staff)
        cv2.waitKey(0)
    else:
        while True:

            image = ds.pick_one("data")[0].unsqueeze(0).to(device)
            image = (image - image.mean()) / image.std()
            yhat = model(image.unsqueeze(0))

            staff = (yhat.squeeze(0) > 0.5).to(torch.float32).cpu().numpy()

            cv2.imshow("image", image.squeeze(0).cpu().numpy())
            cv2.imshow("predict", staff)

            if cv2.waitKey(0) == ord('q'):
                break


def moving_average(y: NDArray[np.float32], window_size: int = 10) -> NDArray[np.float32]:
    return np.convolve(y, np.ones(window_size) / window_size, mode='valid')


@click.command()
@click.argument("log_path",
                type=click.Path(file_okay=True, dir_okay=False, exists=True),
                default=LOG_PATH)
@click.option("--smooth/--no-smooth", default=True,
              help="Smooth the curves before plotting them.")
def plot(log_path: Path = LOG_PATH, smooth: bool = True):
    log = Log(Path(log_path))
    # State and function to quit the tracking loop.
    quit: bool = False

    def on_key(event):
        nonlocal quit
        quit = (event.key == 'q')

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('key_press_event', on_key)
    loss_plot, = ax.plot([], [], 'r', label='Loss.')
    print("Press 'q' to quit.")

    while not quit:
        log.load()
        if smooth:
            losses = moving_average(np.array(log.losses, dtype=np.float32))
        else:
            losses = log.losses
        loss_plot.set_xdata(range(0, len(losses)))
        loss_plot.set_ydata(losses)
        ax.relim()
        ax.autoscale_view()
        ax.legend()
        fig.canvas.draw_idle()
        plt.pause(1)


@click.group()
def cli():
    pass


cli.add_command(show)
cli.add_command(stats)
cli.add_command(train)
cli.add_command(plot)
cli.add_command(predict)
cli.add_command(summary)

if __name__ == '__main__':
    cli()
