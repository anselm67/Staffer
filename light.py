import os

import lightning as L
from torch import Tensor, nn, optim, utils

from dataset import StaffDataset
from model import Config, ViT


class LitStaffer(L.LightningModule):

    def __init__(self, config: Config):
        super().__init__()
        self.model = ViT(config)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        y, gt = batch
        yhat = self.model(y)
        loss = nn.functional.mse_loss(yhat, gt)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> optim.Adam:
        return optim.Adam(self.parameters(), lr=1e-4)


config = Config()

train_ds, valid_ds = StaffDataset.create()
train_loader = utils.data.DataLoader(
    train_ds, num_workers=4, batch_size=config.batch_size)

staffer = LitStaffer(config)
trainer = L.Trainer(max_epochs=8)
trainer.fit(model=staffer, train_dataloaders=train_loader)
