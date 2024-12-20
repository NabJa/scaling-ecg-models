import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def stretch_predictions(
    patch_predictions: torch.Tensor, input_length: int
) -> torch.Tensor:
    """
    Optimally stretch the patch predictions across the input length by strategically placing
    and repeating patches directly in the output tensor using advanced indexing.

    Args:
    - patch_predictions: Tensor of patch predictions, shape [batch_size, num_patches, num_classes]
    - input_length: The length of the input samples.

    Returns:
    - stretched_predictions: Optimally stretched tensor of predictions to input length,
      shape [batch_size, input_length, num_classes]
    """
    batch_size, num_patches, num_classes = patch_predictions.size()
    stretches, remainder = divmod(input_length, num_patches)

    stretched_predictions = torch.zeros(
        batch_size, input_length, num_classes, device=patch_predictions.device
    )

    index = 0
    for i in range(num_patches):
        length = stretches + (1 if i < remainder else 0)
        if length > 0:
            stretched_predictions[:, index : index + length, :] = (
                patch_predictions[:, i, :].unsqueeze(1).expand(-1, length, -1)
            )
        index += length

    return stretched_predictions


def get_conv_config(in_channels, out_channels):
    """simplifies _get_conv_channels_from_config"""
    return [
        (in_channels, 32, 1, 3),
        (32, 64, 1, 5),
        (64, 128, 1, 7),
        (128, 256, 1, 9),
        (256, out_channels, 1, 13),
    ]


class SinusoidalPositionalEncoding(nn.Module):
    """Positional encoding module."""

    def __init__(self, d_model, max_len=5000):
        """ """
        super().__init__()

        pos_enc = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(np.log(10000.0) / d_model)
        )
        pos_enc[:, 0::2] = torch.sin(position * div_term)
        pos_enc[:, 1::2] = torch.cos(position * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_enc", pos_enc)

    def forward(self, input_x):
        """Forward pass."""
        return input_x.add(self.pos_enc[: input_x.size(0), :])


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer block with Layer Normalization."""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        """Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        """
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """

        # Self-attention
        src2 = self.self_attn(
            src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)  # Apply LayerNorm

        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)  # Apply LayerNorm again

        return src


class ConvNext1d(nn.Module):
    """ConvNext1d block"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        expansion_ratio=4,
        layer_scale_init_value=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.expanded_channels = in_channels * expansion_ratio
        self.kernel_size = kernel_size

        # Define the layers
        self.conv1 = nn.Conv1d(
            in_channels, self.expanded_channels, stride=stride, kernel_size=1
        )
        self.norm1 = nn.LayerNorm(self.expanded_channels)
        self.depthwise_conv = nn.Conv1d(
            self.expanded_channels,
            self.expanded_channels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            groups=self.expanded_channels,
            stride=1,
        )
        self.norm2 = nn.LayerNorm(self.expanded_channels)
        self.conv2 = nn.Conv1d(
            self.expanded_channels, out_channels, stride=1, kernel_size=1
        )

        self.gelu = nn.GELU()

        # Define skip connection if in and out channels are not the same
        self.use_skip_connection = in_channels != out_channels or stride != 1
        if self.use_skip_connection:
            self.skip_conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=1, stride=stride
            )

        self.gamma = (
            nn.Parameter(
                layer_scale_init_value * torch.ones((out_channels, 1)),
                requires_grad=True,
            )
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor):
        """Forward pass."""
        # Inverted bottleneck
        identity = x
        x = self.conv1(x)
        x = x.permute(0, 2, 1)  # New line to swap the last two dimensions
        x = self.norm1(x)
        x = x.permute(0, 2, 1)  # Swap the dimensions back to their original order
        x = self.gelu(x)
        # Large kernel size depthwise conv
        x = self.depthwise_conv(x)
        x = x.permute(0, 2, 1)  # New line to swap the last two dimensions
        x = self.norm2(x)
        x = x.permute(0, 2, 1)  # Swap the dimensions back to their original order
        x = self.gelu(x)

        # Output layer
        x = self.conv2(x)

        # Layer scaling
        if self.gamma is not None:
            x.mul_(self.gamma)

        # Skip connection
        if self.use_skip_connection:
            identity = self.skip_conv(identity)

        return x.add(identity)


class ConvPatchEmbedding(nn.Module):
    """
    Implements a convolutional patch embedding layer, specifically designed for ECG data analysis.
    This layer converts input ECG data into patch embeddings suitable for transformer models.
    It includes a sequence of convolutional layers, max-pooling, global average pooling,
    and a linear layer to produce embeddings of a specified dimension.
    Leads are processed separately and concatted after embedding.

    Attributes:
        dim_model: Dimensionality of the input embeddings.
        conv_type: Type of convolution layer to use.
        conv_mode: Mode of convolution, defining the convolutional kernel configuration.
    """

    def __init__(
        self,
        dim_model,
        channels=12,
    ):
        super().__init__()
        self.dim_model = dim_model
        self.half_dim_model = dim_model // 2  # Dimension for each lead
        self.activate_noise_weighting = False  # hard-coded for the experiments

        self.maxpool = nn.MaxPool1d(2, 2)
        # Dynamically create conv layers
        self.conv_layers = nn.ModuleList()
        self.conv_channels = get_conv_config(1, self.half_dim_model)

        for in_channels, out_channels, stride, kernel_size in self.conv_channels:
            layer = ConvNext1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=1,
            )
            self.conv_layers.append(layer)

        self.bn_layers = nn.ModuleList(
            [
                nn.BatchNorm1d(out_channels)
                for _, out_channels, _, _ in self.conv_channels
            ]
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.half_dim_model, self.half_dim_model)
        if self.activate_noise_weighting:
            self.noise_quality_layer = nn.Linear(self.half_dim_model, 1)

    def _apply_conv_to_lead(self, input_tensor):
        """Apply convolutional layers to a single lead"""
        for i_layer, conv in enumerate(self.conv_layers):
            input_tensor = conv(input_tensor)
            if self.conv_channels[i_layer][2] == 1:
                input_tensor = self.maxpool(input_tensor)
        input_tensor = self.global_avg_pool(input_tensor)
        input_tensor = input_tensor.squeeze(-1)
        return self.linear(input_tensor)

    def forward(self, x):
        """
        Forward pass of the model
        Processes an input with shape (batch_size, patch_len, ecg_leads, ecg_samples)
        """
        batch_size, patch_len, _, ecg_samples = x.shape

        # Separate the input for each lead
        lead1_input = x[:, :, 0, :].view(-1, 1, ecg_samples)
        lead2_input = x[:, :, 1, :].view(-1, 1, ecg_samples)

        lead1_output = self._apply_conv_to_lead(lead1_input)
        lead2_output = self._apply_conv_to_lead(lead2_input)

        if self.activate_noise_weighting:
            # Compute noise quality scores
            lead1_quality_scores = torch.sigmoid(
                self.noise_quality_layer(lead1_output)
            )  # Shape: (batch_size, patch_len, 1)
            lead2_quality_scores = torch.sigmoid(
                self.noise_quality_layer(lead2_output)
            )  # Shape: (batch_size, patch_len, 1)
            # Apply weighting based on quality scores
            lead1_output = lead1_output * lead1_quality_scores
            lead2_output = lead2_output * lead2_quality_scores

        # Reshape and concatenate outputs
        lead1_output = lead1_output.view(batch_size, patch_len, self.half_dim_model)
        lead2_output = lead2_output.view(batch_size, patch_len, self.half_dim_model)
        combined_output = torch.cat([lead1_output, lead2_output], dim=-1)

        return combined_output


class ECGVisionTransformer(nn.Module):
    """
    A Vision Transformer model specifically designed for ECG signal analysis.
    This model adopts the transformer architecture for processing ECG data,
     which is represented through multiple leads.
     It utilizes a convolutional patch embedding for time-series data, positional encoding,
    and a standard transformer encoder for sequence modeling.
    The classification head is based on the CLS token.

    Attributes:
        num_classes: Number of output classes.
        num_leads: Number of ECG leads (input channels).
        dim_model: Dimensionality of the input embeddings.
        num_heads: Number of attention heads in the transformer.
        num_encoder_layers: Number of layers in the transformer encoder.
        dim_feedforward: Dimensionality of the feedforward network model.
        patch_size (optional): Size of the patches to be extracted from the input data.
        dropout (optional): Dropout rate. Default: 0.15.
    """

    def __init__(
        self,
        num_classes=26,
        num_leads=12,
        dim_model=256,
        num_heads=4,
        num_encoder_layers=6,
        dim_feedforward=256,
        patch_size=256,
        dropout=0.15,
    ):
        super().__init__()
        self.num_leads = num_leads
        self.patch_size = patch_size
        self.step_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim_model))

        self.time_patch_embedding = ConvPatchEmbedding(dim_model, num_leads)

        self.pos_encoder = SinusoidalPositionalEncoding(dim_model, max_len=1024)

        encoder_layer = TransformerEncoderLayer(
            dim_model,
            num_heads,
            dim_feedforward,
            dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_encoder_layers, enable_nested_tensor=False
        )
        self.layer_norm = nn.LayerNorm(dim_model)
        self.dropout = nn.Dropout(dropout)
        self.class_output_layer = nn.Linear(dim_model, num_classes)

    def forward(self, x):

        # This architecture expects channel-last data
        x = rearrange(x, "b c n -> b n c")

        # Extract patches
        time_patches = x.unfold(1, self.patch_size, self.step_size).contiguous()
        patch_embeddings = self.time_patch_embedding(time_patches)

        # Add class token and positional encoding
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        patch_embeddings = torch.cat((cls_tokens, patch_embeddings), dim=1)
        patch_embeddings = patch_embeddings.permute(1, 0, 2)
        patch_embeddings = self.pos_encoder(patch_embeddings)

        # Transformer encoder
        transformer_output = self.transformer_encoder(patch_embeddings)
        transformer_output = self.dropout(transformer_output)
        transformer_output = transformer_output.permute(1, 0, 2)

        # Classification head
        cls_output = transformer_output[:, 0, :]
        class_predictions = self.class_output_layer(cls_output)

        return class_predictions
