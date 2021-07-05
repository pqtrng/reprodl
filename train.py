import torch, torchaudio
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import functional
from pathlib import Path
import pandas as pd

torch.cuda.is_available()

datapath = Path("data/ESC-50")
datapath.exists()
csv = pd.read_csv(datapath / Path("meta/esc50.csv"))
csv.head()


class ESC50Dataset(torch.utils.data.Dataset):
    def __init__(
        self, path: Path = Path("data/ESC-50"), sample_rate: int = 8000, folds=[1]
    ) -> None:
        super().__init__()
        # Load csv
        self.path = path
        self.csv = pd.read_csv(path / Path("meta/esc50.csv"))
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


train_data = ESC50Dataset(folds=[1, 2, 3])
val_data = ESC50Dataset(folds=[4])
test_data = ESC50Dataset(folds=[5])

train_loader = torch.utils.data.DataLoader(
    dataset=train_data, num_workers=20, batch_size=8, shuffle=True
)
val_loader = torch.utils.data.DataLoader(dataset=val_data, num_workers=20, batch_size=8)
test_loader = torch.utils.data.DataLoader(
    dataset=test_data, num_workers=20, batch_size=8
)


class AudioNet(pl.LightningModule):
    def __init__(self, n_classes=50, base_filters=32) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=base_filters, kernel_size=11, padding=5
        )
        self.bn1 = nn.BatchNorm2d(num_features=base_filters)
        self.conv2 = nn.Conv2d(
            in_channels=base_filters,
            out_channels=base_filters,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(num_features=base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(
            in_channels=base_filters,
            out_channels=base_filters * 2,
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(num_features=base_filters * 2)
        self.conv4 = nn.Conv2d(
            in_channels=base_filters * 2,
            out_channels=base_filters * 4,
            kernel_size=3,
            padding=1,
        )
        self.bn4 = nn.BatchNorm2d(num_features=base_filters * 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=base_filters * 4, out_features=n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool1(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))

        x = self.conv4(x)
        x = F.relu(self.bn4(x))

        x = self.pool2(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.fc1(x[:, :, 0, 0])

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(input=y_hat, dim=1)
        acc = functional.accuracy(y_hat, y)
        self.log(name="val_acc", value=acc, on_epoch=True, prog_bar=True)
        return acc

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = torch.argmax(input=y_hat, dim=1)
        acc = functional.accuracy(y_hat, y)
        self.log(name="test_acc", value=acc)
        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def train():
    audio_net = AudioNet()
    trainer = pl.Trainer(gpus=1, max_epochs=25)
    trainer.fit(
        model=audio_net, train_dataloader=train_loader, val_dataloaders=val_loader
    )
    trainer.test(model=audio_net, test_dataloaders=test_loader)


if __name__ == "__main__":
    pl.seed_everything()
    train()
