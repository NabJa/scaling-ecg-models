import math
from collections import OrderedDict
from fractions import Fraction
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning import LightningModule
from sklearn.cluster import KMeans
from torch import Tensor, nn
from torchvision.ops.misc import ConvNormActivation

from scaling.loss import AsymmetricLoss
from scaling.metrics import scalar_metrics


class SqueezeExcitation(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Conv1d(input_channels, squeeze_channels, kernel_size=1)
        self.fc2 = nn.Conv1d(squeeze_channels, input_channels, kernel_size=1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input


class Conv1dNormActivation(ConvNormActivation):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Optional[Union[int, Tuple[int, int], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm1d,
        activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
        dilation: Union[int, Tuple[int, int]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
    ) -> None:

        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            groups,
            norm_layer,
            activation_layer,
            dilation,
            inplace,
            bias,
            torch.nn.Conv1d,
        )


class SimpleStemIN(Conv1dNormActivation):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        kernel_size: int = 3,
    ) -> None:
        padding = (
            kernel_size - 1
        ) // 2  # Together with stride=2 this ensures the output is halved.
        super().__init__(
            width_in,
            width_out,
            kernel_size=kernel_size,
            stride=2,
            padding=padding,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ) -> None:
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        layers["a"] = Conv1dNormActivation(
            width_in,
            w_b,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )
        layers["b"] = Conv1dNormActivation(
            w_b,
            w_b,
            kernel_size=3,
            stride=stride,
            groups=g,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            layers["se"] = SqueezeExcitation(
                input_channels=w_b,
                squeeze_channels=width_se_out,
                activation=activation_layer,
            )

        layers["c"] = Conv1dNormActivation(
            w_b,
            width_out,
            kernel_size=1,
            stride=1,
            norm_layer=norm_layer,
            activation_layer=None,
        )
        super().__init__(layers)


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj = None
        should_proj = (width_in != width_out) or (stride != 1)
        if should_proj:
            self.proj = Conv1dNormActivation(
                width_in,
                width_out,
                kernel_size=1,
                stride=stride,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            norm_layer,
            activation_layer,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation_layer(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is not None:
            x = self.proj(x) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: Callable[..., nn.Module],
        norm_layer: Callable[..., nn.Module],
        activation_layer: Callable[..., nn.Module],
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float] = None,
        stage_index: int = 0,
    ) -> None:
        super().__init__()

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                norm_layer,
                activation_layer,
                group_width,
                bottleneck_multiplier,
                se_ratio,
            )

            self.add_module(f"block{stage_index}-{i}", block)


class BlockParams:
    def __init__(
        self,
        depths: List[int],
        widths: List[int],
        group_widths: List[int],
        bottleneck_multipliers: List[float],
        strides: List[int],
        se_ratio: Optional[float] = None,
    ) -> None:
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.se_ratio = se_ratio

    def __repr__(self) -> str:
        return (
            f"BlockParams(depths={self.depths}, widths={self.widths}, "
            f"group_widths={self.group_widths}, bottleneck_multipliers={self.bottleneck_multipliers}, "
            f"strides={self.strides}, se_ratio={self.se_ratio})"
        )

    @classmethod
    def from_init_params(
        cls,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        input_resolution: Optional[int] = None,
        se_ratio: Optional[float] = None,
        max_stages: int = 7,  # Add a parameter for maximum number of stages
        **kwargs: Any,
    ) -> "BlockParams":
        QUANT = 8
        STRIDE = 2

        if w_a < 0 or w_0 <= 0 or w_m <= 1 or w_0 % 8 != 0:
            raise ValueError("Invalid RegNet settings")

        # Compute continuous widths and block capacities
        widths_cont = torch.arange(depth) * w_a + w_0
        block_capacity = torch.round(torch.log(widths_cont / w_0) / math.log(w_m))

        # Enforce maximum capacity
        if input_resolution is not None:
            max_capacity = math.log2(input_resolution) - 3
            block_capacity = torch.clamp(block_capacity, max=max_capacity)

        # Quantize widths to multiples of QUANT
        block_widths = (
            (torch.round(w_0 * torch.pow(w_m, block_capacity) / QUANT) * QUANT)
            .int()
            .tolist()
        )

        # Determine unique widths and limit to max_stages
        unique_widths = sorted(list(set(block_widths)))
        if len(unique_widths) > max_stages:
            # If there are more unique widths than max_stages, merge the closest widths
            # into max_stages groups

            # Use KMeans clustering to group widths into max_stages clusters
            kmeans = KMeans(n_clusters=max_stages, random_state=0).fit(
                np.array(unique_widths).reshape(-1, 1)
            )
            cluster_centers = sorted(
                [int(center[0]) for center in kmeans.cluster_centers_]
            )

            # Map each width to the nearest cluster center
            block_widths = [
                min(cluster_centers, key=lambda x: abs(x - w)) for w in block_widths
            ]
            unique_widths = cluster_centers

        # Split into stages based on unique widths
        splits = [
            w != wp or r != rp
            for w, wp, r, rp in zip(
                block_widths + [0],
                [0] + block_widths,
                block_widths + [0],
                [0] + block_widths,
            )
        ]
        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = (
            torch.diff(torch.tensor([d for d, t in enumerate(splits) if t]))
            .int()
            .tolist()
        )

        # Adjust compatibility
        strides = [STRIDE] * len(stage_widths)
        bottleneck_multipliers = [bottleneck_multiplier] * len(stage_widths)
        group_widths = [group_width] * len(stage_widths)
        stage_widths, group_widths = cls._adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return cls(
            depths=stage_depths,
            widths=stage_widths,
            group_widths=group_widths,
            bottleneck_multipliers=bottleneck_multipliers,
            strides=strides,
            se_ratio=se_ratio,
        )

    def _get_expanded_params(self):
        return zip(
            self.widths,
            self.strides,
            self.depths,
            self.group_widths,
            self.bottleneck_multipliers,
        )

    @staticmethod
    def _adjust_widths_groups_compatibilty(
        stage_widths: List[int], bottleneck_ratios: List[float], group_widths: List[int]
    ) -> Tuple[List[int], List[int]]:
        # Compute initial bottleneck widths and group widths
        widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
        group_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

        # Adjust widths using LCM of group width and ratio numerator
        ws_bot = []
        for w_bot, g, ratio in zip(widths, group_widths_min, bottleneck_ratios):
            # Convert ratio to simplified fraction
            f = Fraction(ratio).limit_denominator()
            numerator = f.numerator

            # Calculate LCM of group width and ratio numerator
            gcd = math.gcd(g, numerator)
            lcm = (g * numerator) // gcd

            # Find nearest valid width >= original width
            adjusted = ((w_bot + lcm - 1) // lcm) * lcm
            ws_bot.append(adjusted)

        # Recompute stage widths using precise division
        stage_widths = [round(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]

        return stage_widths, group_widths_min


class RegNet(nn.Module):
    def __init__(
        self,
        block_params: BlockParams,
        num_classes: int = 26,
        stem_width: int = 32,
        init_kernel_size=3,
        block_type: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        if block_type is None:
            block_type = ResBottleneckBlock
        if activation is None:
            activation = nn.ReLU

        # Ad hoc stem
        self.stem = SimpleStemIN(
            width_in=12,
            width_out=stem_width,
            kernel_size=init_kernel_size,
            norm_layer=norm_layer,
            activation_layer=activation,
        )

        current_width = stem_width

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(block_params._get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_type,
                        norm_layer,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        block_params.se_ratio,
                        stage_index=i + 1,
                    ),
                )
            )

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_features=current_width, out_features=num_classes)

        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.out_channels
                nn.init.normal_(m.weight, mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.trunk_output(x)

        x = self.avgpool(x)
        x = x.flatten(start_dim=1)
        x = self.fc(x)

        return x


class RegNetModule(LightningModule):
    def __init__(
        self,
        depth,
        w_init,
        w_0,
        w_a,
        w_m,
        group_width,
        bottleneck_multiplier,
        se_ratio,
        init_kernel_size=3,
        input_resolution=1024,
        num_classes=26,
        gamma_neg=4,
        gamma_pos=1,
        init_lr=1e-3,
        lr_decay_gamma=0.98,
        weight_decay=0.0,
        optimizer="sgd",
    ):
        super().__init__()

        self.save_hyperparameters()

        block_params = BlockParams.from_init_params(
            depth=depth,
            w_0=w_0,
            w_a=w_a,
            w_m=w_m,
            group_width=group_width,
            bottleneck_multiplier=bottleneck_multiplier,
            input_resolution=input_resolution,
            se_ratio=se_ratio,
        )
        self.model = RegNet(
            init_kernel_size=init_kernel_size,
            stem_width=w_init,
            block_params=block_params,
            num_classes=num_classes,
        )

        self.loss_fn = AsymmetricLoss(gamma_neg=gamma_neg, gamma_pos=gamma_pos)
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

    def configure_optimizers(self):
        if self.hparams.optimizer.lower() == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.init_lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer.lower() == "sgdm":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.init_lr,
                weight_decay=self.hparams.weight_decay,
                momentum=0.9,
            )
        elif self.hparams.optimizer.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.init_lr,
                weight_decay=self.hparams.weight_decay,
            )
        elif self.hparams.optimizer.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.init_lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(
                f"Invalid optimizer {self.hparams.optimizer}. Use 'sgd', 'adam' or 'adamw'."
            )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=self.hparams.lr_decay_gamma
        )
        return [optimizer], [scheduler]
