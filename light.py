#!/usr/bin/env python3

from pathlib import Path
from typing import Any, Optional

import click
import cv2
import lightning as L
import torch
from torch import Tensor, nn, optim, utils
from torchvision.io import decode_image

from dataset import StaffDataset
from model import Config, ViT


class LitStaffer(L.LightningModule):

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.loss_f = nn.BCEWithLogitsLoss()

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        y, gt = batch
        yhat = self.model(y)
        loss = self.loss_f(yhat, gt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        y, gt = batch
        yhat = self.model(y)
        val_loss = self.loss_f(yhat, gt)
        self.log("val_loss", val_loss)
        return val_loss

    def configure_optimizers(self) -> optim.Adam:
        return optim.Adam(self.parameters(), lr=1e-4)


@click.command()
def show():
    config = Config()
    ds, _ = StaffDataset.create(config=config)
    loader = utils.data.DataLoader(
        ds, num_workers=4, batch_size=config.batch_size)
    for images, masks in loader:
        for idx in range(config.batch_size):
            image, mask = images[idx], masks[idx]

            cv2.imshow("image", image.squeeze(0).cpu().numpy())
            cv2.imshow("mask", mask.cpu().numpy())
            if cv2.waitKey(0) == ord('q'):
                return


@click.command()
def train():
    config = Config()

    train_ds, valid_ds = StaffDataset.create()
    train_loader = utils.data.DataLoader(
        train_ds, num_workers=4, batch_size=config.batch_size
    )
    valid_loader = utils.data.DataLoader(
        valid_ds, num_workers=4, batch_size=config.batch_size
    )

    staffer = LitStaffer(ViT(config))
    trainer = L.Trainer(max_epochs=16)
    trainer.fit(staffer, train_loader, valid_loader)


@click.command()
@click.argument("checkpoint", type=str)
@click.argument("image_path",
                type=click.Path(file_okay=True, dir_okay=False, exists=True),
                required=False)
def predict(checkpoint: str, image_path: Optional[Path] = None):
    config = Config()

    ds, _ = StaffDataset.create()
    loader = utils.data.DataLoader(
        ds, num_workers=4, batch_size=config.batch_size)
    model = ViT(config)
    LitStaffer.load_from_checkpoint(checkpoint, model=model)

    device = next(model.parameters()).device

    for images, _ in loader:

        # Un-Squeezes for a batch_size of 1.
        yhats = model(images.to(device))
        yhats = (yhats > 0.5).to(torch.float32)

        for idx in range(0, config.batch_size):
            image, yhat = images[idx], yhats[idx]

            cv2.imshow("image", image.squeeze(0).cpu().numpy())
            cv2.imshow("staff", yhat.cpu().numpy())
            if cv2.waitKey(0) == ord('q'):
                return


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(predict)
cli.add_command(show)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli()
