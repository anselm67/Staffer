#!/usr/bin/env python3

from dataclasses import asdict, replace
from pathlib import Path
from typing import cast

import click
import cv2
import lightning as L
import torch
from torch import Tensor, nn, optim, utils
from torchvision.io import decode_image

from dataset import StaffDataset
from model import Config, ViT


def accuracy(pred: Tensor, gt: Tensor) -> tuple[float, float]:
    true_positives = (pred * gt).sum()
    false_positives = (pred * (1 - gt)).sum()
    false_negatives = ((1 - pred) * gt).sum()
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    return precision.item(), recall.item()


class LitStaffer(L.LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.save_hyperparameters(asdict(config))
        self.model = ViT(config)
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
        val_precision, val_recall = accuracy((yhat > 0.5).to(torch.float), gt)
        self.log("val_loss", val_loss)
        self.log("val_precision", val_precision)
        self.log("val_recall", val_recall)
        return val_loss

    def predict_step(self, image: Tensor) -> Tensor:
        yhat = self.model(image.unsqueeze(0)).squeeze(0)
        return (yhat > 0.5).to(torch.float32)

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
@click.option("epochs", "-e", type=int, default=32)
def train(epochs: int):
    config = Config()

    train_ds, valid_ds = StaffDataset.create()
    train_loader = utils.data.DataLoader(
        train_ds, num_workers=4, batch_size=config.batch_size, shuffle=True
    )
    valid_loader = utils.data.DataLoader(
        valid_ds, num_workers=4, batch_size=config.batch_size
    )

    staffer = LitStaffer(config)
    trainer = L.Trainer(max_epochs=epochs, limit_val_batches=10)
    trainer.fit(staffer, train_loader, valid_loader)


@click.command()
@click.argument("checkpoint", type=str)
def test(checkpoint: str):
    config = replace(Config(), batch_size=4)

    ds, _ = StaffDataset.create(config=config)
    loader = utils.data.DataLoader(
        ds, num_workers=4, batch_size=config.batch_size)
    model = LitStaffer.load_from_checkpoint(checkpoint, config=config)
    trainer = L.Trainer()

    for images, gts in loader:
        yhats = cast(list[Tensor], trainer.predict(model, [images]))
        for image, yhat, gt in zip(images.unbind(0), yhats, gts):
            precision, recall = accuracy(yhat, gt)
            print(
                f"Precision: {100 * precision:.2f}%, Recall: {100 * recall:.2f}%")
            cv2.imshow("image", image.squeeze(0).cpu().numpy())
            cv2.imshow("staff", yhat.cpu().numpy())
            if cv2.waitKey(0) == ord('q'):
                return


@click.command()
@click.argument("checkpoint", type=str)
@click.option("image_path", "-i", type=click.Path(file_okay=True, dir_okay=False, exists=True),
              required=True)
def predict(checkpoint, image_path: Path):
    config = Config()
    _, ds = StaffDataset.create(config=config)

    model = LitStaffer.load_from_checkpoint(checkpoint, config=config)
    image = decode_image(Path(image_path).as_posix())
    image, _ = ds.predict_transform(image)
    yhat = model.predict_step(image.cuda())
    staff = (yhat > 0.5).to(torch.float32).cpu().numpy()

    cv2.imshow("Image", image.squeeze(0).cpu().numpy())
    cv2.imshow("Staff", staff)
    cv2.waitKey(0)


@click.command()
def stats():
    config = replace(Config(), valid_split=0)
    ds, _ = StaffDataset.create(config=config)
    ds.stats()


@click.group()
def cli():
    pass


cli.add_command(train)
cli.add_command(test)
cli.add_command(show)
cli.add_command(stats)
cli.add_command(predict)

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    cli()
