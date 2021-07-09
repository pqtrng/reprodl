from pathlib import Path

import pandas as pd
import torch
import torchaudio


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(
        self, path: Path = Path("data/ESC-50"), sample_rate: int = 8000, folds=[1]
    ) -> None:
        super().__init__()
        # Load csv
        self.path = path
        # self.csv = pd.read_csv(path / Path("meta/esc50.csv"))
        self.csv = pd.read_csv(path / Path("meta/extra_esc50.csv"))
        self.csv = self.csv[self.csv["fold"].isin(folds)]

        # Transform
        self.resample = torchaudio.transforms.Resample(
            orig_freq=44100, new_freq=sample_rate
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)
        self.db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __getitem__(self, index):
        """Return (xb, yb) pair, after applying all transformations on the audio file

        Args:
            index (int): index of pair
        """

        row = self.csv.iloc[index]
        wav, _ = torchaudio.load(self.path / "audio" / row["filename"])
        label = row["target"]
        xb = self.db(self.melspec(self.resample(wav)))
        return xb, label

    def __len__(self):
        """Get total number of samples in dataset

        Returns:
            int: Number of samples
        """
        return len(self.csv)
