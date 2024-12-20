import torch
from lightning import LightningModule, Trainer, seed_everything

# TODO: experiment with SWA
# from lightning.pytorch.callbacks import (
#     StochasticWeightAveraging,
# )
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import WandbLogger
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR

from scaling.augmentations import BasicECGAugmentation, RandomCropOrPad
from scaling.datasets import PhysionetDM
from scaling.models.model_factory import MODELS

seed_everything(42)


class LitModel(LightningModule):
    def __init__(self, model_name, loss_fn, lr_decay_gamma=0.95, **model_kwargs):
        super().__init__()
        self.model: nn.Module = MODELS[model_name](**model_kwargs)
        self.loss_fn = loss_fn
        self.lr_decay_gamma = lr_decay_gamma

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = ExponentialLR(optimizer, gamma=self.lr_decay_gamma)
        return [optimizer], [scheduler]


def train(
    project,
    name,
    meta_file_path,
    fold,
    model_name,
    loss_fn,
    lr_decay_gamma,
    fast_dev_run=False,
    **model_kwargs,
):

    # Prepare model
    model = LitModel(model_name, loss_fn, lr_decay_gamma, **model_kwargs)

    # Prepare data
    transform = RandomCropOrPad(target_length=1024)
    dm = PhysionetDM(
        meta_file_path, fold, train_transform=transform, val_transform=transform
    )

    # Logging
    logger = WandbLogger(project=project, name=name)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Callbacks
    early_stopping = EarlyStopping(monitor="val_loss", patience=5)
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        filename="{model_name}_fold{fold}_epoch{epoch:02d}-loss{val_loss:.2f}",
        save_top_k=1,
    )
    # swa = StochasticWeightAveraging(swa_lrs=1e-2)

    # Train!
    trainer = Trainer(
        max_epochs=100,
        logger=logger,
        log_every_n_steps=50,
        callbacks=[lr_monitor, early_stopping, model_checkpoint],
        gradient_clip_algorithm="norm",
        gradient_clip_val=1,
        fast_dev_run=fast_dev_run,
    )

    trainer.fit(model, dm)


def cross_validate(
    project,
    meta_file_path,
    model_name,
    loss_fn,
    lr_decay_gamma,
    fast_dev_run=False,
    **model_kwargs,
):
    for fold in [0, 1, 2]:
        train(
            project=project,
            name=f"{model_name}_fold{fold}",
            meta_file_path=meta_file_path,
            fold=fold,
            model_name=model_name,
            loss_fn=loss_fn,
            lr_decay_gamma=lr_decay_gamma,
            fast_dev_run=fast_dev_run,
            **model_kwargs,
        )


if __name__ == "__main__":
    path = "/sc-scratch/sc-scratch-gbm-radiomics/ecg/physionet_challenge/training_pt/metadata_v4.csv"
    cross_validate(
        project="scaling",
        meta_file_path=path,
        model_name="resnet",
        loss_fn=nn.CrossEntropyLoss(),
        lr_decay_gamma=0.95,
        in_channels=12,
        num_classes=9,
    )
