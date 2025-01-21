### Using lightning CLI as entry point. See https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html

import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.cli import LightningCLI

from scaling.datasets.physionet import PhysionetDM
from scaling.models.module import LitModel

train_transform_defaults = dict(
    class_path="scaling.augmentations.ECGAugmentation",
    init_args=dict(
        crop_size=1024,
        max_time_warp=0.2,
        scaling=(0.8, 1.2),
        gaussian_noise_std=0.01,
        wandering_max_amplitude=1.0,
        wandering_frequency_range=(0.5, 2.0),
        max_mask_duration=50,
        mask_prob=0.5,
    ),
)


valid_transform_defaults = dict(
    class_path="scaling.augmentations.ECGAugmentation",
    init_args=dict(
        crop_size=1024,
        max_time_warp=None,
        scaling=None,
        gaussian_noise_std=None,
        wandering_max_amplitude=None,
        wandering_frequency_range=None,
        max_mask_duration=None,
        mask_prob=None,
    ),
)

checkpoint_defaults = dict(
    monitor="loss/valid",
    save_top_k=1,
    mode="min",
    filename="model-{epoch:02d}",
)

earlystopping_defaults = dict(
    monitor="loss/valid",
    patience=5,
    mode="min",
)


logger_defaults = dict(
    class_path="lightning.pytorch.loggers.WandbLogger",
    init_args=dict(
        name=None,
        project="PhysioNetScaling",
        log_model=False,
        offline=False,
    ),
)

trainer_defaults = dict(
    deterministic=True,
    check_val_every_n_epoch=2,
    gradient_clip_val=1.0,
    max_epochs=100,
    precision=32,
    callbacks=[
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(**earlystopping_defaults),
        ModelCheckpoint(**checkpoint_defaults),
    ],
    logger=logger_defaults,
)


class PhysionetCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):

        # Add optimization arguments
        parser.add_optimizer_args(torch.optim.AdamW)
        parser.add_lr_scheduler_args(torch.optim.lr_scheduler.ExponentialLR)

        # Set additional defaults
        parser.set_defaults({"lr_scheduler.gamma": 0.98})
        parser.set_defaults({"data.val_transform": valid_transform_defaults})
        parser.set_defaults({"data.train_transform": train_transform_defaults})


def cli_main():
    cli = PhysionetCLI(
        LitModel,
        PhysionetDM,
        seed_everything_default=42,
        trainer_defaults=trainer_defaults,
        save_config_kwargs=dict(overwrite=True),
    )


if __name__ == "__main__":
    cli_main()
