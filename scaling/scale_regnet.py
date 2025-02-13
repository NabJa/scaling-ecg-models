import gc
import math
from functools import partial
from pathlib import Path

import numpy as np
import optuna
import torch
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from optuna.samplers import RandomSampler

import wandb
from scaling.augmentations import ECGAugmentation
from scaling.datasets.physionet import PhysionetDM
from scaling.models.regnet import RegNetModule

seed_everything(42)


def doubling_space(start, end):
    """Generate an array of values that doubles from start to end (inclusive)."""
    assert 0 < start and 0 < end and start < end
    times_to_double = math.log2(end / start) + 1
    return start * 2 ** np.arange(times_to_double)


def suggest_param_grid(trial):

    init_lr = trial.suggest_float("init_lr", 1e-3, 1e-2)
    init_kernel_size = trial.suggest_categorical("init_kernel_size", [3, 9, 15])

    w_init = trial.suggest_int("w_init", 32, 64, step=8)
    w_0 = trial.suggest_int("w_0", 32, 128, step=8)
    w_0 = max(w_0, w_init)

    w_m = trial.suggest_float("w_m", 1.1, 3.0, step=0.1)
    w_a = trial.suggest_int("w_a", 10, 50)
    depth = trial.suggest_int("depth", 4, 20)

    group_width = trial.suggest_categorical(
        "group_width", doubling_space(1, 32).astype(int).tolist()
    )

    bottleneck_multiplier = trial.suggest_categorical(
        "bottleneck_multiplier", doubling_space(0.5, 4.0).tolist()
    )

    se_ratio = trial.suggest_categorical("se_ratio", [0.25, None])

    return {
        "w_init": w_init,
        "init_lr": init_lr,
        "init_kernel_size": init_kernel_size,
        "depth": depth,
        "w_0": w_0,
        "w_a": w_a,
        "w_m": round(w_m, 1),
        "group_width": group_width,
        "bottleneck_multiplier": round(bottleneck_multiplier, 1),
        "se_ratio": se_ratio,
    }


def prepare_dataset(crop_size, batch_size, num_workers):

    meta_path = "/sc-scratch/sc-scratch-gbm-radiomics/ecg/physionet_challenge/training_pt/metadata_v5.csv"
    train_transform = ECGAugmentation(
        crop_size=crop_size,
        max_time_warp=0.2,
        scaling=(0.8, 1.2),
        gaussian_noise_std=0.01,
        wandering_max_amplitude=1.0,
        wandering_frequency_range=(0.5, 2.0),
        max_mask_duration=50,
        mask_prob=0.5,
    )
    val_transform = ECGAugmentation(crop_size=crop_size)
    return PhysionetDM(
        meta_path,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )


def objective(trial, dirpath="./", crop_size=1024, batch_size=32, num_workers=16):

    hyper_params = suggest_param_grid(trial)

    model = RegNetModule(**hyper_params)
    datamodule = prepare_dataset(
        crop_size=crop_size, batch_size=batch_size, num_workers=num_workers
    )

    model_checkpoints = Path(dirpath) / "models" / f"{int(trial.number):03}"
    model_checkpoints.mkdir(parents=True, exist_ok=True)

    trainer = Trainer(
        accelerator="gpu",
        max_epochs=20,
        gradient_clip_val=1.0,
        deterministic=True,
        logger=WandbLogger(
            name=f"Trial-{trial.number:03}",
            project="PhysioNetDesignSpace",
            reinit=True,
            save_dir=dirpath,
        ),
        callbacks=[
            ModelCheckpoint(
                dirpath=model_checkpoints,
                monitor="loss/valid",
                mode="min",
                save_top_k=1,
            )
        ],
    )

    trainer.fit(model, datamodule=datamodule)

    final_loss = trainer.callback_metrics["loss/valid"].item()

    wandb.finish()

    # Free memory
    del model
    del datamodule
    del trainer

    torch.cuda.empty_cache()
    gc.collect()

    return final_loss


if __name__ == "__main__":

    root = "/sc-scratch/sc-scratch-gbm-radiomics/ecg/design_spaces/ds_a"

    study = optuna.create_study(
        study_name="PhysioNetDesignSpace",
        storage=f"sqlite:///{root}/physionet_design_space.db",
        load_if_exists=True,
        sampler=RandomSampler(),
    )

    obj = partial(objective, dirpath=root, batch_size=32, num_workers=16)
    study.optimize(obj, n_trials=120, gc_after_trial=True)
