from typing import List

import pandas as pd
import torch
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import (
    MultilabelAccuracy,
    MultilabelAUROC,
    MultilabelAveragePrecision,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelRecall,
)
from torchmetrics.functional import calibration_error

from scaling.datasets.physionet import CLASSES, SINUS_RYTHM, WEIGHTS


class MultiLabelCalibrationError(Metric):
    def __init__(
        self,
        num_labels: int,
        n_bins: int = 26,
        norm: str = "l1",
    ):
        super().__init__()
        self.num_labels = num_labels
        self.n_bins = n_bins
        self.norm = norm

        # Separate lists to accumulate predictions and targets for each label
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("targets", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Update metric states with predictions and targets.
        Assumes preds and targets are of shape (batch_size, num_labels).
        """
        if preds.shape[1] != self.num_labels or targets.shape[1] != self.num_labels:
            raise ValueError(
                f"Expected {self.num_labels} labels, but got {preds.shape[1]}."
            )

        self.preds.append(preds.detach().clone())
        self.targets.append(targets.detach().clone())

    def compute(self):
        """
        Compute calibration error for each label and return the results.
        """
        # Concatenate all batches
        preds = torch.cat(self.preds, dim=0)
        targets = torch.cat(self.targets, dim=0)

        labelwise_errors = []

        for i in range(self.num_labels):
            label_preds = preds[:, i]
            label_targets = targets[:, i]

            # Compute calibration error for this label
            ce = calibration_error(
                label_preds,
                label_targets,
                task="binary",
                n_bins=self.n_bins,
                norm=self.norm,
            )
            labelwise_errors.append(ce)

        return torch.mean(torch.tensor(labelwise_errors))


def compute_modified_confusion_matrix(labels, outputs):
    """
    Compute a binary multi-class, multi-label confusion matrix using PyTorch.
    Rows are the labels, and columns are the outputs.
    """

    # Compute normalization factor for each recording
    normalization = torch.clamp(
        (labels.bool() | outputs.bool()).sum(dim=1, keepdim=True).float(), min=1.0
    )

    # Compute contributions to the confusion matrix for each recording
    contributions = (
        labels.unsqueeze(2) * outputs.unsqueeze(1).float()
    ) / normalization.unsqueeze(2)

    # Sum over all recordings to compute the confusion matrix
    A = contributions.sum(dim=0)

    return A


class PhysionetMetric(Metric):
    """
    Custom TorchMetric for computing the Challenge metric.
    """

    full_state_update = False

    def __init__(
        self,
        weights: pd.DataFrame,
        classes: List[set],
        sinus_rhythm: set,
    ):
        super().__init__()

        self.register_buffer("weights", torch.tensor(weights.values, dtype=torch.float))
        self.classes = classes

        if sinus_rhythm not in classes:
            raise ValueError("The sinus rhythm class is not available.")

        self.sinus_rhythm_index = classes.index(sinus_rhythm)

        # Define states for metric accumulation
        self.add_state(
            "observed_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("correct_score", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "inactive_score", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Update the metric states with new data."""
        # Ensure internal buffers are on the same device as outputs
        device = outputs.device
        self.weights = self.weights.to(device)

        # Compute confusion matrices
        A_observed = compute_modified_confusion_matrix(labels, outputs)
        self.observed_score += torch.nansum(self.weights * A_observed)

        A_correct = compute_modified_confusion_matrix(labels, labels)
        self.correct_score += torch.nansum(self.weights * A_correct)

        inactive_outputs = torch.zeros_like(outputs, dtype=torch.bool, device=device)
        inactive_outputs[:, self.sinus_rhythm_index] = 1
        A_inactive = compute_modified_confusion_matrix(labels, inactive_outputs)
        self.inactive_score += torch.nansum(self.weights * A_inactive)

        self.total += labels.size(0)

    def compute(self):
        """Compute the final Challenge metric."""
        if self.correct_score != self.inactive_score:
            normalized_score = (self.observed_score - self.inactive_score) / (
                self.correct_score - self.inactive_score
            )
        else:
            normalized_score = 0.0

        return normalized_score

    def aggregate(self):
        return self.compute()


def scalar_metrics(num_labels=26, physionet_metric=True):
    metrics = {
        "Recall": MultilabelRecall(num_labels=num_labels),
        "Precision": MultilabelPrecision(num_labels=num_labels),
        "AUROC": MultilabelAUROC(num_labels=num_labels),
        "F1": MultilabelF1Score(num_labels=num_labels),
        "AP": MultilabelAveragePrecision(num_labels=num_labels),
        "Accuracy": MultilabelAccuracy(num_labels=num_labels),
        "ECE": MultiLabelCalibrationError(num_labels=num_labels),
    }
    if physionet_metric:
        metrics["PhysioAccuracy"] = PhysionetMetric(
            weights=WEIGHTS, classes=CLASSES, sinus_rhythm=SINUS_RYTHM
        )
    return MetricCollection(metrics)
