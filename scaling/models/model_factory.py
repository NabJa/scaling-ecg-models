from scaling.models.convnext import CNBlockConfig, ConvNeXt
from scaling.models.getemed import ECGVisionTransformer
from scaling.models.resnet import BasicBlock, Bottleneck, ResNet
from scaling.models.vit import ECGViT


def getemed_small(**kwargs) -> ECGVisionTransformer:
    return ECGVisionTransformer(
        dim_model=128,
        num_heads=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        patch_size=64,
        dropout=0.05,
        **kwargs,
    )


def getemed_base(**kwargs) -> ECGVisionTransformer:
    return ECGVisionTransformer(
        dim_model=256,
        num_heads=4,
        num_encoder_layers=6,
        dim_feedforward=256,
        patch_size=64,
        dropout=0.15,
        **kwargs,
    )


def getemed_large(**kwargs) -> ECGVisionTransformer:
    return ECGVisionTransformer(
        dim_model=512,
        num_heads=16,
        num_encoder_layers=6,
        dim_feedforward=1024,
        patch_size=64,
        dropout=0.15,
        **kwargs,
    )


def convnext_mini(**kwargs) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 2),
        CNBlockConfig(192, 384, 2),
        CNBlockConfig(384, 768, 2),
        CNBlockConfig(768, None, 2),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return ConvNeXt(block_setting, stochastic_depth_prob, **kwargs)


def convnext_tiny(**kwargs) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.1)
    return ConvNeXt(block_setting, stochastic_depth_prob, **kwargs)


def convnext_small(**kwargs) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return ConvNeXt(block_setting, stochastic_depth_prob, **kwargs)


def convnext_base(**kwargs) -> ConvNeXt:
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return ConvNeXt(block_setting, stochastic_depth_prob, **kwargs)


def convnext_large(**kwargs) -> ConvNeXt:

    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return ConvNeXt(block_setting, stochastic_depth_prob, **kwargs)


def resnet18(**kwargs) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnext18(**kwargs) -> ResNet:
    groups = kwargs.pop("groups", 32)
    width_per_group = kwargs.pop("width_per_group", 4)

    return ResNet(
        Bottleneck,
        [2, 2, 2, 2],
        groups=groups,
        width_per_group=width_per_group,
        **kwargs,
    )


def resnext50(**kwargs) -> ResNet:
    groups = kwargs.pop("groups", 32)
    width_per_group = kwargs.pop("width_per_group", 4)
    return ResNet(
        Bottleneck,
        [3, 4, 6, 3],
        groups=groups,
        width_per_group=width_per_group,
        **kwargs,
    )


def resnext101(**kwargs) -> ResNet:
    groups = kwargs.pop("groups", 32)
    width_per_group = kwargs.pop("width_per_group", 8)
    return ResNet(
        Bottleneck,
        [3, 4, 23, 3],
        groups=groups,
        width_per_group=width_per_group,
        **kwargs,
    )


def vit_tiny(**kwargs) -> ECGViT:
    seq_len = kwargs.pop("seq_len", 1024)
    patch_size = kwargs.pop("patch_size", 64)
    return ECGViT(
        seq_len=seq_len,
        patch_size=patch_size,
        dim=256,
        depth=6,
        heads=8,
        mlp_mult=4,
        dim_head=64,
        **kwargs,
    )


def vit_small(**kwargs) -> ECGViT:
    seq_len = kwargs.pop("seq_len", 1024)
    patch_size = kwargs.pop("patch_size", 64)
    return ECGViT(
        seq_len=seq_len,
        patch_size=patch_size,
        dim=512,
        depth=8,
        heads=8,
        mlp_mult=4,
        dim_head=64,
        **kwargs,
    )


def vit_base(**kwargs) -> ECGViT:
    seq_len = kwargs.pop("seq_len", 1024)
    patch_size = kwargs.pop("patch_size", 64)
    return ECGViT(
        seq_len=seq_len,
        patch_size=patch_size,
        dim=768,
        depth=12,
        heads=12,
        mlp_mult=4,
        dim_head=64,
        **kwargs,
    )


def vit_large(**kwargs) -> ECGViT:
    seq_len = kwargs.pop("seq_len", 1024)
    patch_size = kwargs.pop("patch_size", 64)
    return ECGViT(
        seq_len=seq_len,
        patch_size=patch_size,
        dim=1024,
        depth=24,
        heads=16,
        mlp_mult=4,
        dim_head=64,
        **kwargs,
    )


MODELS = {
    "convnext_mini": convnext_mini,
    "convnext_tiny": convnext_tiny,
    "convnext_small": convnext_small,
    "convnext_base": convnext_base,
    "convnext_large": convnext_large,
    "resnet18": resnet18,
    "resnet50": resnet50,
    "resnet101": resnet101,
    "resnext18": resnext18,
    "resnext50": resnext50,
    "resnext101": resnext101,
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "vit_large": vit_large,
    "getemed_small": getemed_small,
    "getemed_base": getemed_base,
    "getemed_large": getemed_large,
}
