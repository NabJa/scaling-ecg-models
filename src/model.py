from functools import partial
from typing import Any, Callable, List, Optional, Sequence

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops.stochastic_depth import StochasticDepth


class CNBlockConfig1D:
    def __init__(
        self,
        input_channels: int,
        out_channels: Optional[int],
        num_layers: int,
    ) -> None:
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "input_channels={input_channels}"
        s += ", out_channels={out_channels}"
        s += ", num_layers={num_layers}"
        s += ")"
        return s.format(**self.__dict__)


class LayerNorm1D(nn.LayerNorm):
    """
    Rearranges input from (N, C, S) to (N, S, C) to apply Layer Normalization over channels (C),
    as nn.LayerNorm normalizes the last dimension. Returns input to original shape (N, C, S) after normalization.
    """

    def forward(self, x: Tensor) -> Tensor:
        x = rearrange(x, "N C S -> N S C")
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = rearrange(x, "N S C -> N C S")
        return x


class CNBlock1D(nn.Module):
    """
    A ConvNeXt residual block for 1D inputs with depthwise convolution, LayerNorm, LayerScale,
    GELU activation, inverted bottlneck, and stochastic depth regularization.
    """

    def __init__(
        self,
        dim,
        layer_scale: float,
        stochastic_depth_prob: float,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        bottleneck_inversion_factor: int = 4,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        inverted_bottleneck_dim = dim * bottleneck_inversion_factor

        self.block = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Rearrange("N S C -> N C S"),
            norm_layer(dim),
            nn.Linear(
                in_features=dim,
                out_features=inverted_bottleneck_dim,
                bias=True,
            ),
            nn.GELU(),
            nn.Linear(
                in_features=inverted_bottleneck_dim,
                out_features=dim,
                bias=True,
            ),
            Rearrange("N C S -> N S C"),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

    def forward(self, input: Tensor) -> Tensor:
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result


class ConvNeXt1D(nn.Module):
    """ConvNeXt1D model for ECG data classification."""

    def __init__(
        self,
        block_setting: List[CNBlockConfig1D],
        stochastic_depth_prob: float = 0.0,
        layer_scale: float = 1e-6,
        channels=12,
        num_classes: int = 5,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Args:
            block_setting: List of CNBlockConfig1D for each stage.
            stochastic_depth_prob: Probability of dropping out a block. The probability is linearly increased.
            layer_scale: Layer scale for LayerScale module.
            channels: Number of input channels.
            num_classes: Number of output classes.
            block: Block module to use. Defaults to CNBlock1D.
            norm_layer: Normalization layer to use. Defaults to LayerNorm1D.
        """
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (
            isinstance(block_setting, Sequence)
            and all([isinstance(s, CNBlockConfig1D) for s in block_setting])
        ):
            raise TypeError("The block_setting should be List[CNBlockConfig1D]")

        if block is None:
            block = CNBlock1D

        if norm_layer is None:
            norm_layer = partial(LayerNorm1D, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            nn.Conv1d(
                channels,
                firstconv_output_channels,
                kernel_size=4,
                stride=4,
                padding=0,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = (
                    stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                )
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        LayerNorm1D(cnf.input_channels, eps=1e-6),
                        nn.Conv1d(
                            cnf.input_channels,
                            cnf.out_channels,
                            kernel_size=2,
                            stride=2,
                        ),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels
            if lastblock.out_channels is not None
            else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels),
            nn.Flatten(1),
            nn.Linear(lastconv_output_channels, num_classes),
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


def convnext1d_tiny(**kwargs: Any) -> ConvNeXt1D:
    return ConvNeXt1D(
        [
            CNBlockConfig1D(input_channels=24, out_channels=48, num_layers=3),
            CNBlockConfig1D(input_channels=48, out_channels=96, num_layers=3),
            CNBlockConfig1D(input_channels=96, out_channels=None, num_layers=3),
        ],
        **kwargs,
    )


def convnext1d_small(**kwargs: Any) -> ConvNeXt1D:
    return ConvNeXt1D(
        [
            CNBlockConfig1D(input_channels=64, out_channels=96, num_layers=3),
            CNBlockConfig1D(input_channels=96, out_channels=128, num_layers=3),
            CNBlockConfig1D(input_channels=128, out_channels=256, num_layers=3),
            CNBlockConfig1D(input_channels=256, out_channels=None, num_layers=3),
        ],
        **kwargs,
    )


def convnext1d_large(**kwargs: Any) -> ConvNeXt1D:
    return ConvNeXt1D(
        [
            CNBlockConfig1D(input_channels=96, out_channels=128, num_layers=3),
            CNBlockConfig1D(input_channels=128, out_channels=256, num_layers=3),
            CNBlockConfig1D(input_channels=256, out_channels=512, num_layers=3),
            CNBlockConfig1D(input_channels=512, out_channels=None, num_layers=3),
        ],
        **kwargs,
    )


def get_convnext(name: str, **kwargs: Any) -> ConvNeXt1D:
    if name == "tiny":
        return convnext1d_tiny(**kwargs)
    elif name == "small":
        return convnext1d_small(**kwargs)
    elif name == "large":
        return convnext1d_large(**kwargs)
    else:
        raise ValueError(f"Unknown ConvNeXt model name: {name}")
