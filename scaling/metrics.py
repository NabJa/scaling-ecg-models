from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ptb.utils import CLASS_INDEX
from torchmetrics.classification import (
    BinaryCalibrationError,
    MultilabelAUROC,
    MultilabelF1Score,
    MultilabelPrecision,
    MultilabelPrecisionRecallCurve,
    MultilabelRecall,
    MultilabelROC,
)


def plot_prec_rec_curve(precision, recall):
    for i, (p, r) in enumerate(zip(precision, recall)):
        plt.plot(p, r, label=CLASS_INDEX[i], alpha=0.7, lw=3)
    plt.plot([0, 1], [1, 0], color="black", ls="--")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    plt.grid()
    plt.legend()
    plt.show()


def plot_roc_curve(tprs, fprs):
    for i, (p, r) in enumerate(zip(tprs, fprs)):
        plt.plot(p, r, label=CLASS_INDEX[i], alpha=0.7, lw=3)
    plt.plot([0, 1], [0, 1], color="black", ls="--")
    plt.xlabel("True Positive Rate")
    plt.ylabel("False Positive Rate")
    plt.grid()
    plt.legend()
    plt.show()


class MulltilabelCalibrationError:
    """
    Multilabel Top-label Calibration Error.
    See https://arxiv.org/abs/1909.10155
    """

    def __init__(self, nlabels=5, n_bins=15, norm="l1", **kwargs):
        """
        Args:
            nlabels (int): Number of labels.
            n_bins (int): Number of bins for calibration error.
            norm (str): Norm used to compare empirical and expected probability bins. One of: 'l1', 'l2', 'max'.
            **kwargs: Additional keyword arguments to be passed to BinaryCalibrationError.
        """

        self.nlabels = nlabels
        self.ece_fns = [
            BinaryCalibrationError(n_bins=n_bins, norm=norm, **kwargs)
            for _ in range(nlabels)
        ]

    def __call__(self, pred, y):
        for i, fn in enumerate(self.ece_fns):
            fn(pred[:, i], y[:, i])

    def compute(self):
        return torch.tensor([fn.compute() for fn in self.ece_fns])

    def reset(self):
        for fn in self.ece_fns:
            fn.reset()


class MetricTracker:
    def __init__(self):
        self.precision = MultilabelPrecision(5)
        self.recall = MultilabelRecall(5)
        self.auroc = MultilabelAUROC(5)
        self.prcurve = MultilabelPrecisionRecallCurve(5)
        self.roccurve = MultilabelROC(5)

    def __call__(self, pred, gt):
        pred, gt = pred.detach().cpu(), gt.detach().cpu().int()

        self.precision(pred, gt)
        self.recall(pred, gt)
        self.auroc(pred, gt)
        self.prcurve(pred, gt)
        self.roccurve(pred, gt)

    def aggregate(self):
        precision = self.precision.compute()
        recall = self.recall.compute()
        auroc = self.auroc.compute()
        tprs, fprs, _ = self.roccurve.compute()
        precisions, recalls, _ = self.prcurve.compute()
        return {
            "precision": precision,
            "recall": recall,
            "auroc": auroc,
            "tprs": tprs,
            "fprs": fprs,
            "precisions": precisions,
            "recalls": recalls,
        }

    def reset(self):
        self.precision.reset()
        self.recall.reset()
        self.auroc.reset()
        self.prcurve.reset()
        self.roccurve.reset()


class MultilabelScalarTracker:
    """Tracks scalar metrics for multilabel classification."""

    def __init__(self):
        self.precision = MultilabelPrecision(5, average="none")
        self.recall = MultilabelRecall(5, average="none")
        self.auroc = MultilabelAUROC(5, average="none")
        self.f1score = MultilabelF1Score(5, average="none")
        self.ece = MulltilabelCalibrationError(5)

    def __call__(self, pred, gt):
        pred, gt = pred.detach().cpu(), gt.detach().cpu().int()
        self.precision(pred, gt)
        self.recall(pred, gt)
        self.auroc(pred, gt)
        self.f1score(pred, gt)
        self.ece(pred, gt)

    def aggregate(self):
        return {
            "precision": self.precision.compute(),
            "recall": self.recall.compute(),
            "auroc": self.auroc.compute(),
            "f1score": self.f1score.compute(),
            "ece": self.ece.compute(),
        }

    def reset(self):
        self.precision.reset()
        self.recall.reset()
        self.auroc.reset()
        self.f1score.reset()
        self.ece.reset()
