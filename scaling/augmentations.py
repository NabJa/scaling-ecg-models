import torch
import torch.nn.functional as F


class RandomAugmentation:
    """Parent class for handling randomness in augmentations."""

    def __init__(self, seed=None):
        """
        Args:
            seed (int, optional): Random seed for reproducibility. Defaults to None.
        """
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)

    def random_uniform(self, low, high):
        """Generates a random float between `low` and `high`."""
        return torch.empty(1).uniform_(low, high, generator=self.rng).item()

    def random_int(self, low, high):
        """Generates a random integer between `low` and `high - 1`."""
        return torch.randint(low, high, (1,), generator=self.rng).item()


class RandomCropOrPad(RandomAugmentation):
    """Crops or pads the ECG signal to a target length."""

    def __init__(self, target_length, seed=None):
        super().__init__(seed)
        self.target_length = target_length

    def __call__(self, signal):
        """
        Args:
            signal (torch.Tensor): Input signal of shape (C, N), where C is the number of channels and N is the number of time steps.

        Returns:
            torch.Tensor: Cropped or padded signal of the same shape as input.
        """
        length = signal.shape[1]
        if length < self.target_length:
            # Pad the signal
            padding = self.target_length - length
            left_pad = self.random_int(0, padding + 1)
            right_pad = padding - left_pad
            return F.pad(signal, (left_pad, right_pad))
        else:
            # Crop the signal
            start = self.random_int(0, length - self.target_length + 1)
            return signal[:, start : start + self.target_length]


class RandomMaskChannels(RandomAugmentation):
    """Ronadomly masks a subset of channels in the ECG signal."""

    def __init__(self, mask_prob=0.1, seed=None):
        """
        Args:
            mask_prob (float): Probability of masking a channel.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.mask_prob = mask_prob

    def __call__(self, signal):
        """
        Args:
            signal (torch.Tensor): Input signal of shape (C, N), where C is the number of channels and N is the number of time steps.

        Returns:
            torch.Tensor: Signal with masked channels.
        """
        mask = torch.rand(signal.shape[0], generator=self.rng) > self.mask_prob

        masked = signal * mask.unsqueeze(1)

        if masked.any():
            return masked

        # Ensure at least one channel is not masked
        idx = self.random_int(0, signal.shape[0])
        masked[idx] = signal[idx]
        return masked


class TimeWarping(RandomAugmentation):
    """Randomly stretches or compresses the ECG signal in time."""

    def __init__(self, max_warp=0.2, seed=None):
        """
        Args:
            max_warp (float): Maximum percentage to warp the signal (e.g., 0.2 for ±20%).
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.max_warp = max_warp

    def __call__(self, signal):
        """
        Args:
            signal (torch.Tensor): Input signal of shape (C, N), where C is the number of channels and N is the number of time steps.

        Returns:
            torch.Tensor: Time-warped signal of the same shape as input.
        """
        warp_factor = 1 + self.random_uniform(-self.max_warp, self.max_warp)
        new_length = int(signal.shape[1] * warp_factor)

        # Interpolate to the new length
        signal_resampled = F.interpolate(
            signal.unsqueeze(0),  # Add batch dimension
            size=(new_length,),
            mode="linear",
            align_corners=False,
        ).squeeze(0)

        # Resize back to original length
        if new_length > signal.shape[1]:
            return signal_resampled[:, : signal.shape[1]]
        else:
            padded_signal = torch.zeros_like(signal)
            padded_signal[:, :new_length] = signal_resampled
            return padded_signal


class TimeMasking(RandomAugmentation):
    """Randomly masks a time segment of the ECG signal."""

    def __init__(self, max_mask_duration=50, seed=None):
        """
        Args:
            max_mask_duration (int): Maximum duration (in samples) of the masked segment.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.max_mask_duration = max_mask_duration

    def __call__(self, signal):
        """
        Args:
            signal (torch.Tensor): Input signal of shape (C, N), where C is the number of channels and N is the number of time steps.

        Returns:
            torch.Tensor: Time-masked signal of the same shape as input.
        """
        mask_duration = self.random_int(1, self.max_mask_duration + 1)
        start_idx = self.random_int(0, max(1, signal.shape[1] - mask_duration))
        signal[:, start_idx : start_idx + mask_duration] = 0
        return signal


class AmplitudeScaling(RandomAugmentation):
    """Randomly scales the amplitude of the ECG signal."""

    def __init__(self, min_scale=0.8, max_scale=1.2, seed=None):
        """
        Args:
            min_scale (float): Minimum scaling factor.
            max_scale (float): Maximum scaling factor.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, signal):
        """
        Args:
            signal (torch.Tensor): Input signal of shape (C, N).

        Returns:
            torch.Tensor: Amplitude-scaled signal of the same shape as input.
        """
        scale_factor = self.random_uniform(self.min_scale, self.max_scale)
        return signal * scale_factor


class GaussianNoise(RandomAugmentation):
    """Adds Gaussian noise to the ECG signal."""

    def __init__(self, mean=0.0, std=0.01, seed=None):
        """
        Args:
            mean (float): Mean of the Gaussian noise.
            std (float): Standard deviation of the Gaussian noise.
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.mean = mean
        self.std = std

    def __call__(self, signal):
        """
        Args:
            signal (torch.Tensor): Input signal of shape (C, N).

        Returns:
            torch.Tensor: Signal with added Gaussian noise.
        """
        noise = torch.normal(self.mean, self.std, size=signal.shape, generator=self.rng)
        return signal + noise


class RandomWandering(RandomAugmentation):
    """Adds low-frequency wandering noise to the ECG signal."""

    def __init__(self, max_amplitude=1.0, frequency_range=(0.5, 2.0), seed=None):
        """
        Args:
            max_amplitude (float): Maximum amplitude of the wandering noise.
            frequency_range (tuple): Range of frequencies for the wandering noise (low, high).
            seed (int, optional): Random seed for reproducibility.
        """
        super().__init__(seed)
        self.max_amplitude = max_amplitude
        self.frequency_range = frequency_range

    def __call__(self, signal):
        """
        Args:
            signal (torch.Tensor): Input signal of shape (C, N).

        Returns:
            torch.Tensor: Signal with added wandering noise.
        """
        length = signal.shape[1]
        freq = self.random_uniform(*self.frequency_range)
        t = torch.arange(0, length, dtype=torch.float32)
        wandering_noise = self.max_amplitude * torch.sin(
            2 * torch.pi * freq * t / length
        )
        return signal + wandering_noise.unsqueeze(0)


class Compose:
    """Simply aplies multiple augmentations in sequence."""

    def __init__(self, *augmentations):
        self.augmentations = augmentations

    def __call__(self, signal):
        for augmentation in self.augmentations:
            signal = augmentation(signal)
        return signal


class BasicECGAugmentation:
    def __init__(
        self,
        crop_size=1000,
        max_warp=0.2,
        min_scale=0.5,
        max_scale=2.0,
        noise_std=0.01,
        max_amplitude=1.0,
        frequency_range=(0.5, 2.0),
        max_mask_duration=50,
        mask_prob=0.5,
    ):
        """
        Args:
            crop_size: Crops or pads to this size. Defaults to 1000.
            max_warp: Warps time by this percentage. Defaults to 0.2.
            min_scale: Amplitude scaling. Defaults to 0.5.
            max_scale: Amplitude scaling. Defaults to 2.0.
            noise_std: Gaussian noise std. Defaults to 0.01.
            max_amplitude: Amplitude of random wandering. Defaults to 1.0.
            frequency_range: Frequence range of random wandering. Defaults to (0.5, 2.0).
            max_mask_duration: Max duration of zero masking. Defaults to 50.
            mask_prob: Probability to completely mask a lead (channel). Defaults to 0.5.
        """
        self.transform = Compose(
            RandomCropOrPad(crop_size),
            TimeWarping(max_warp),
            AmplitudeScaling(min_scale, max_scale),
            GaussianNoise(std=noise_std),
            RandomWandering(max_amplitude, frequency_range),
            TimeMasking(max_mask_duration),
            RandomMaskChannels(mask_prob),
        )

    def __call__(self, signal):
        return self.transform(signal)
