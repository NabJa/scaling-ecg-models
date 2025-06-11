"""
Physionet dataset is processed into one .pt file for every ecg.
Labels and meta information is stored in a separate .csv file.

Preprocessing steps:
    1. Resample ECG signals (default = 500Hz).
        - Polyphase filtering is used when original sampling rate and target sampling rate are integer multiples of each other
        - otherwise, the FFT method is used for resampling
    2. Data are filtered using a zero-phase method with 3rd order Butterworth
       bandpass filter with frequency band from 1 Hz to 47 Hz.
    3. Z-score normalization using median and std.
"""

import argparse
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from tqdm import tqdm

from scaling.preprocessing.preprocess import (
    butter_filter,
    resample_ecg,
    zscore_normalize,
)

DX_MAP = pd.read_csv(
    "https://raw.githubusercontent.com/physionetchallenges/physionetchallenges.github.io/master/2020/Dx_map.csv"
)


def save_as_csv(data, path) -> None:
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def load_mat(filepath: Path) -> np.ndarray:
    return loadmat(filepath)["val"]


class MetaData:
    """Wrapper class for metadata in .hea files."""

    def __init__(self, path):
        with open(path, mode="r") as file:
            lines = file.readlines()

        self.sample, self.leads, self.sampling_rate, self.length = (
            lines[0].strip().split(" ")
        )

        self.leads = int(self.leads)
        self.sampling_rate = int(self.sampling_rate)
        self.length = int(self.length)

        self.age = lines[13][6:].strip()
        self.sex = lines[14][6:].strip()
        self.dx = [int(x) for x in lines[15][6:].strip().split(",")]

    def to_dict(self):
        return {
            "sample": self.sample,
            "leads": self.leads,
            "age": self.age,
            "sex": self.sex,
            "sampling_rate": self.sampling_rate,
            "length": self.length,
            "dx": self.dx,
        }


def preprocess_file(
    filepath: Path, output_dir: Path, target_sampling_rate: int
) -> dict:
    output_filepath = output_dir / f"{filepath.stem}.pt"

    try:
        metadata = MetaData(filepath.with_suffix(".hea"))
        ecg = load_mat(filepath)
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return {}

    ecg = butter_filter(ecg, metadata.sampling_rate)
    ecg, method = resample_ecg(ecg, metadata.sampling_rate, target_sampling_rate)
    ecg = zscore_normalize(ecg)

    meta = metadata.to_dict()
    meta["method"] = method
    meta["new_length"] = ecg.shape[1]
    meta["filepath"] = str(output_filepath)
    meta["original_filepath"] = str(filepath)

    torch.save(torch.tensor(ecg), output_filepath)
    return meta


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess Physionet dataset.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Path to the raw Physionet dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the processed dataset directory.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=500,
        help="Resampling rate for ECG signals.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = args.input_dir
    output_dir: Path = args.output_dir

    output_ecg_folder = output_dir / "ecgs"
    output_ecg_folder.mkdir(parents=True, exist_ok=True)

    target_sampling_rate = int(args.sampling_rate)

    all_files = list(input_dir.rglob("*.mat"))

    _prprocess = partial(
        preprocess_file,
        output_dir=output_ecg_folder,
        target_sampling_rate=target_sampling_rate,
    )

    with Pool(processes=12) as pool:
        metas = list(tqdm(pool.imap(_prprocess, all_files), total=len(all_files)))

    save_as_csv(metas, output_dir / "metadata.csv")


if __name__ == "__main__":
    main()
