from enum import Enum
from typing import Union

from lightning import LightningModule
from torch import nn

from scaling.loss import AsymmetricLoss
from scaling.metrics import scalar_metrics
from scaling.models.model_factory import MODELS


class AvailableLoss(Enum):
    BCE = nn.BCEWithLogitsLoss()
    ASL = AsymmetricLoss()


class LitModel(LightningModule):
    def __init__(
        self,
        model_name: str = "resnet18",
        loss_fn: Union[AvailableLoss, str] = "BCE",
        **model_kwargs
    ):
        """
        Args:
            model_name (str): Name of the model to use.
            loss_fn (AvailableLoss): Loss function to use.
            init_lr (float, optional): Initial learning rate. Defaults to 1e-3.
            weight_decay (float, optional): Weight decay. Defaults to 0.01.
            lr_decay_gamma (float, optional): Learning rate decay factor for ExponentialLR. Defaults to 0.95.
        """
        super().__init__()
        self.loss_fn = (
            AvailableLoss[loss_fn].value if isinstance(loss_fn, str) else loss_fn.value
        )
        self.save_hyperparameters()
        self.model: nn.Module = MODELS[model_name](**model_kwargs)
        self.metrics = scalar_metrics()

    def forward(self, x):
        return self.model(x)

    def step(self, batch):
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self.step(batch)
        self.log("loss/train", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self.step(batch)
        self.log("loss/valid", loss)

        performance = self.metrics(y, y_hat)
        self.log_dict("performance", performance)

        return loss
