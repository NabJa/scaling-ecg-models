import pandas as pd
import torch

DX_SCORED = pd.read_csv(
    "https://raw.githubusercontent.com/physionetchallenges/evaluation-2021/refs/heads/main/dx_mapping_scored.csv"
)


def one_hot(*snomed_codes):
    one_hot = torch.zeros(len(DX_SCORED))
    indeces = torch.tensor(DX_SCORED[DX_SCORED.SNOMEDCTCode.isin(snomed_codes)].index)
    one_hot[indeces] = 1
    return one_hot


class PhysioNet:
    def __init__(self, meta_file_path, ecg_transform=None):
        self.meta = pd.read_csv(meta_file_path)
        self.ecg_transform = ecg_transform

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, index):
        sample = self.meta.iloc[index]
        label = one_hot(*eval(sample.dx))

        ecg = torch.load(sample.filepath, weights_only=False)
        if self.ecg_transform is not None:
            ecg = self.ecg_transform(ecg)

        return ecg, label, sample
