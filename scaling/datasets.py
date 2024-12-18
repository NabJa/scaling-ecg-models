import numpy as np
import pandas as pd
import torch

DX_SCORED = pd.read_csv(
    "https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/refs/heads/main/dx_mapping_scored.csv"
)
DX_SCORED["IndexU"] = list(DX_SCORED.index)

# Some snomed codes are treated as equivalent in PhysioNet.
EQUIVALENCE_MAP = {
    733534002: 164909002,
    713427006: 59118001,
    284470004: 63593006,
    427172004: 17338001,
}


def remove_dx_equivalence(df):
    """Updates the IndexU column in the DataFrame to remove equivalence between SNOMED codes."""

    df = df.copy()
    df["IndexU"] = np.nan
    for key, value in EQUIVALENCE_MAP.items():
        query = (df.SNOMEDCTCode == key).argmax()
        df.loc[df.SNOMEDCTCode == value, "IndexU"] = float(query)

    _unchanged = df.IndexU.isna()
    df.loc[_unchanged, "IndexU"] = np.arange(_unchanged.sum())
    df.IndexU = df.IndexU.astype(int)
    return df


class PhysioNet:
    def __init__(
        self, meta_file_path, ecg_transform=None, remove_quivalent_labels=True
    ):
        self.meta = pd.read_csv(meta_file_path)

        # Some files are corrupted and cannot be loaded. Drop them.
        self.meta = self.meta.dropna(how="all")

        self.ecg_transform = ecg_transform

        self.dx_map = (
            remove_dx_equivalence(DX_SCORED) if remove_quivalent_labels else DX_SCORED
        )
        self.nlabels = self.dx_map.IndexU.nunique()

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        sample = self.meta.iloc[index]

        label = self.get_label(sample)
        ecg = self.get_ecg(sample)

        return ecg, label

    def get_label(self, sample):
        """Returns one-hot encoded label for the given index."""

        return self.one_hot(*eval(sample.dx))

    def get_ecg(self, sample):
        """Returns the ECG signal for the given index. Apply transformation if specified."""

        ecg: torch.Tensor = torch.load(sample.filepath, weights_only=False)
        if self.ecg_transform is not None:
            ecg = self.ecg_transform(ecg)
        return ecg.float()

    def one_hot(self, *snomed_codes):
        """Returns a one-hot encoded tensor for the given SNOMED codes."""

        indeces = self.dx_map.loc[
            self.dx_map.SNOMEDCTCode.isin(snomed_codes), "IndexU"
        ].values
        one_hot = torch.zeros(self.nlabels)
        if len(indeces) == 0:
            return one_hot
        one_hot[torch.tensor(indeces)] = 1
        return one_hot

    def index_to_label(self, index, label_col="Abbreviation"):
        """Returns the SNOMED code for the given index."""
        labels = self.dx_map.loc[self.dx_map.IndexU == index, label_col].values
        return ", ".join(labels)
