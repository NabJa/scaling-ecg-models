from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Optional, Sequence

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops.stochastic_depth import StochasticDepth


@dataclass
class BlockConfig:
    input_channels: int
    out_channels: Optional[int]
    num_layers: int
    norm_layer: Optional[Callable[..., nn.Module]] = None
    activation: Optional[Callable[..., nn.Module]] = None


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


class ResNetBlock(nn.Module):
    """
    A ResNet residual block for 1D inputs with two Conv1d layers, BatchNorm, and ReLU activation.
    """

    def __init__(
        self,
        input_channels,
        activation=None,
        norm_layer=None,
        layer_scale=None,
        stochastic_depth_prob=None,
    ):
        super().__init__()

        self.layer_scale = layer_scale
        if layer_scale is not None:
            self.layer_scale = nn.Parameter(torch.ones(input_channels, 1) * layer_scale)

        self.stochastic_depth_prob = stochastic_depth_prob
        if stochastic_depth_prob is not None:
            self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        self.activation = activation if activation is not None else nn.ReLU
        self.norm = norm_layer if norm_layer is not None else nn.BatchNorm1d

        self.block = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1),
            self.norm(input_channels),
            self.activation(),
            nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=1),
            self.norm(input_channels),
        )

    def forward(self, input: Tensor) -> Tensor:
        residual = input

        out = self.block(input)

        if self.layer_scale is not None:
            out = self.layer_scale * out

        if self.stochastic_depth_prob is not None:
            out = self.stochastic_depth(out)

        out += residual
        out = self.activation()(out)

        return out


class ConvNextBlock(nn.Module):
    """
    A ConvNeXt residual block for 1D inputs with depthwise convolution, LayerNorm, LayerScale,
    GELU activation, inverted bottlneck, and stochastic depth regularization.
    """

    def __init__(
        self,
        input_channels,
        activation=None,
        norm_layer=None,
        layer_scale=None,
        stochastic_depth_prob=None,
        bottleneck_inversion_factor=4,
    ) -> None:
        super().__init__()

        self.layer_scale = layer_scale
        if layer_scale is not None:
            self.layer_scale = nn.Parameter(torch.ones(input_channels, 1) * layer_scale)

        self.stochastic_depth_prob = stochastic_depth_prob
        if stochastic_depth_prob is not None:
            self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if activation is None:
            activation = nn.GELU

        inverted_bottleneck_dim = input_channels * bottleneck_inversion_factor

        self.block = nn.Sequential(
            nn.Conv1d(
                input_channels,
                input_channels,
                kernel_size=7,
                padding=3,
                groups=input_channels,
                bias=True,
            ),
            Rearrange("N C S -> N S C"),
            norm_layer(input_channels),
            nn.Linear(
                in_features=input_channels,
                out_features=inverted_bottleneck_dim,
                bias=True,
            ),
            activation(),
            nn.Linear(
                in_features=inverted_bottleneck_dim,
                out_features=input_channels,
                bias=True,
            ),
            Rearrange("N S C -> N C S"),
        )

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.layer_scale is not None:
            result = self.layer_scale * result
        if self.stochastic_depth_prob is not None:
            result = self.stochastic_depth(result)
        result += input
        return result


def increase_stochastic_depth_prob(
    stochastic_depth_prob, stage_block_id, total_stage_blocks
):
    """Helper function to linearly increase the stochastic depth probability based on the stage block ID."""
    if stochastic_depth_prob is None:
        return None
    return stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)


class EcgClassifier(nn.Module):
    """Model for ECG classification."""

    def __init__(
        self,
        block_setting: List[BlockConfig],
        channels=12,
        num_classes: int = 5,
        block: Optional[Callable[..., nn.Module]] = None,
        stochastic_depth_prob: Optional[float] = None,
        layer_scale: Optional[float] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        initial_kernel_size: int = 4,
        initial_stride: int = 4,
        activation: Optional[Callable[..., nn.Module]] = None,
        final_norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        """
        Args:
            block_setting: List of BlockConfig instances defining the model architecture.
            channels: Number of input channels.
            num_classes: Number of output classes.
            block: Block class to use for the model.
            stochastic_depth_prob: Probability of applying stochastic depth regularization.
            layer_scale: Scaling factor for the layer output.
            norm_layer: Normalization layer to use in the model.
            initial_kernel_size: Kernel size for the initial convolutional layer.
            initial_stride: Stride for the initial convolutional layer.
            activation: Activation function to use in the model.
            final_norm_layer: Normalization layer to use in the final classifier.
        """
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (
            isinstance(block_setting, Sequence)
            and all([isinstance(s, BlockConfig) for s in block_setting])
        ):
            raise TypeError("The block_setting should be List[CNBlockConfig1D]")

        if block is None:
            block = ConvNextBlock

        if final_norm_layer is None:
            final_norm_layer = LayerNorm1D

        layers: List[nn.Module] = []
        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            nn.Conv1d(
                channels,
                firstconv_output_channels,
                kernel_size=initial_kernel_size,
                stride=initial_stride,
            )
        )

        # Blocks
        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                sd_prob = increase_stochastic_depth_prob(
                    stochastic_depth_prob, stage_block_id, total_stage_blocks
                )
                stage.append(
                    block(
                        cnf.input_channels, activation, norm_layer, layer_scale, sd_prob
                    )
                )
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
            final_norm_layer(lastconv_output_channels),
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


def resnet18(**kwargs):
    """
    This is a basic ResNet-18 model configuration. Arguments are passed to the EcgClassifier class.
    It allows for easy variation of configuration parameters like activation function, normalization layer, etc.
    """
    block_settings = [
        BlockConfig(input_channels=64, out_channels=64, num_layers=2),
        BlockConfig(input_channels=64, out_channels=128, num_layers=2),
        BlockConfig(input_channels=128, out_channels=256, num_layers=2),
        BlockConfig(input_channels=256, out_channels=512, num_layers=2),
    ]
    return EcgClassifier(block_settings, **kwargs)


def get_classifier(depth, width, width_grow_rate=2, **kwargs):
    """
    Get a EcgClassifier with specified depth and width.

    Args:
        depth: Number of layers in the model.
        width: Number of channels in the model.
        **kwargs: Additional keyword arguments for the model.

    Returns:
        A EcgClassifier with the specified depth and width.
    """

    block_settings = []
    next_width = width
    for _ in range(depth - 1):
        next_width *= width_grow_rate
        block_settings.append(
            BlockConfig(input_channels=width, out_channels=next_width, num_layers=2)
        )

    # Set output channels to None for the last block
    block_settings.apend(
        BlockConfig(input_channels=next_width, out_channels=None, num_layers=2)
    )

    return EcgClassifier(block_settings, **kwargs)
