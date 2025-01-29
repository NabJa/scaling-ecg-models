from enum import Enum
from typing import Optional, Union

import torch
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
        width: Optional[int] = None,
        depth: Optional[int] = None,
        initial_kernel_size: Optional[int] = None,
        initial_stride: Optional[int] = None,
        initial_padding: Optional[int] = None,
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
        self.width = width
        self.depth = depth
        self.initial_kernel_size = initial_kernel_size
        self.initial_stride = initial_stride
        self.initial_padding = initial_padding

        if width is not None:
            model_kwargs["width"] = width
        if depth is not None:
            model_kwargs["depth"] = depth
        if initial_kernel_size is not None:
            model_kwargs["initial_kernel_size"] = initial_kernel_size
        if initial_stride is not None:
            model_kwargs["initial_stride"] = initial_stride
        if initial_padding is not None:
            model_kwargs["initial_padding"] = initial_padding

        self.save_hyperparameters()
        self.model: nn.Module = MODELS[model_name](**model_kwargs)
        self.metrics = scalar_metrics()

    def forward(self, x):
        return self.model(x)

    def step(self, batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        performance = self.metrics(y_hat, y.int())
        self.log_dict(performance)

        return loss

    def on_validation_end(self):
        self.metrics.reset()
        return super().on_validation_end()
