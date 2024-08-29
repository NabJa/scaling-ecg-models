from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, resample, resample_poly, sosfiltfilt


def poly_resample_ecg(
    ecg: np.ndarray, sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    gcd = np.gcd(sample_rate, target_sample_rate)
    up = target_sample_rate // gcd
    down = sample_rate // gcd

    resampled_ecg = resample_poly(ecg, up=up, down=down, axis=1)
    return resampled_ecg


def fft_resample_ecg(
    ecg: np.ndarray, sample_rate: int, target_sample_rate: int
) -> np.ndarray:
    ecg_length_in_s = ecg.shape[1] / sample_rate
    num = np.round(ecg_length_in_s * target_sample_rate)
    actual_sample_rate = num / ecg_length_in_s
    error_in_sample_rate = abs(actual_sample_rate - target_sample_rate)

    assert (
        error_in_sample_rate < 0.5
    ), f"Actual sample rate {actual_sample_rate} is not within 0.5 Hz of target sample rate {target_sample_rate}."

    resampled_ecg = resample(ecg, num=int(num), axis=1)
    return resampled_ecg


def resample_ecg(
    ecg: np.ndarray, sample_rate: int, target_sample_rate: int
) -> Tuple[np.ndarray]:
    """Resample an ECG. We use polymorphic resmpling if the original sampling rate and target sampling
    rate are integer multiples of each other. Otherwise, FFT resampling is used"""

    if sample_rate % target_sample_rate == 0 or target_sample_rate % sample_rate == 0:
        return poly_resample_ecg(ecg, sample_rate, target_sample_rate), "Polyphase"

    return fft_resample_ecg(ecg, sample_rate, target_sample_rate), "FFT"


def butter_filter(
    ecg: np.ndarray,
    sample_rate: float,
    lower_freq: float = 1,
    upper_freq: float = 47,
    order: int = 3,
) -> np.ndarray:
    sos = butter(
        N=order,
        Wn=[lower_freq, upper_freq],
        fs=sample_rate,
        btype="bandpass",
        output="sos",
    )

    return sosfiltfilt(sos, ecg)


def zscore_normalize(ecg: np.ndarray, clip: Optional[int] = None) -> np.ndarray:
    mean = ecg.mean(axis=-1, keepdims=True)
    std = ecg.std(axis=-1, keepdims=True)
    z_ecg = (ecg - mean) / std
    if clip:
        z_ecg = np.clip(z_ecg, -clip, clip)
    return z_ecg
