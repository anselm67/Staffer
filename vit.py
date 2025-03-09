#!/usr/bin/env python3

import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

import click
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchsummary
from numpy.typing import NDArray

from dataset import StaffDataset
from model import Config, ViT

CVCMUSCIMA_PATH = Path("/home/anselm/datasets/CvcMuscima-Distortions")
LOG_PATH = Path("untracked/vit_log.json")
MODEL_PATH = Path("untracked/model.pt")


@click.command()
@click.argument("path",
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default=CVCMUSCIMA_PATH)
def show(path: Path = CVCMUSCIMA_PATH):
    ds, _ = StaffDataset.create(Path(path))
    while True:
        img, staff = tuple(map(lambda x: x.numpy(), ds.pick_one()))
        cv2.imshow("music", img)
        cv2.imshow("staff", cv2.bitwise_and(img, staff))

        if cv2.waitKey() == ord('q'):
            break


@click.command()
@click.argument("path",
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default=CVCMUSCIMA_PATH)
def stats(path: Path = CVCMUSCIMA_PATH):
    config = replace(Config(), split=0)
    ds, _ = StaffDataset.create(Path(path), config)
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
    config = Config()

    ds, _ = StaffDataset.create(Path(path), config)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config.batch_size,
        num_workers=4
    )
    log = Log()

    model = ViT(config).to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)

    bce = nn.BCEWithLogitsLoss()

    done = 0
    report_every = 20
    report_cycle = 0
    start_time = time.time()

    for e in range(epochs):

        for images, staves in loader:

            opt.zero_grad()

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
                    f"processed {done} samples, " +
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
    ds, _ = StaffDataset.create(Path(path), config)
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

            image = ds.pick_one()[0].to(device)
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
