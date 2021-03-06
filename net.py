import torch
import torchmetrics
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch import nn
from torch.nn import functional as F

import dvclive


class AudioNet(LightningModule):
    """Neural network to classify audio files.

    Args:
        LightningModule (Object): Base class
    """

    def __init__(self, hparams: DictConfig):
        super().__init__()
        self.metric = torchmetrics.Accuracy()
        self.save_hyperparameters(hparams)
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=self.hparams.base_filters,
            kernel_size=11,
            padding=5,
        )
        self.bn1 = nn.BatchNorm2d(num_features=self.hparams.base_filters)
        self.conv2 = nn.Conv2d(
            in_channels=self.hparams.base_filters,
            out_channels=self.hparams.base_filters,
            kernel_size=3,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(num_features=self.hparams.base_filters)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(
            in_channels=self.hparams.base_filters,
            out_channels=self.hparams.base_filters * 2,
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm2d(num_features=self.hparams.base_filters * 2)
        self.conv4 = nn.Conv2d(
            in_channels=self.hparams.base_filters * 2,
            out_channels=self.hparams.base_filters * 4,
            kernel_size=3,
            padding=1,
        )
        self.bn4 = nn.BatchNorm2d(num_features=self.hparams.base_filters * 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(
            in_features=self.hparams.base_filters * 4,
            out_features=self.hparams.num_classes,
        )

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
        features, target = batch
        prediction = self(features)
        loss = F.cross_entropy(input=prediction, target=target)
        self.log("train_loss", loss, on_step=True)

        # logging metric
        train_loss = loss.data.cpu().numpy().reshape(1)[0].item()
        dvclive.log(name="train_loss", val=train_loss)
        dvclive.next_step()

        return loss

    def validation_step(self, batch, batch_idx):
        features, target = batch
        y_hat = self(features)
        y_hat = torch.argmax(input=y_hat, dim=1)
        acc = self.metric(preds=y_hat, target=target)
        self.log(name="val_acc", value=acc, on_epoch=True, prog_bar=True)

        # logging metric
        val_acc = acc.cpu().numpy().reshape(1)[0].item()
        dvclive.log(name="val_acc", val=val_acc)
        dvclive.next_step()

        return acc

    def test_step(self, batch, batch_idx):
        features, target = batch
        y_hat = self(features)
        y_hat = torch.argmax(input=y_hat, dim=1)
        acc = self.metric(preds=y_hat, target=target)
        self.log(name="test_acc", value=acc)

        # logging metric
        test_acc = acc.cpu().numpy().reshape(1)[0].item()
        dvclive.log(name="test_acc", val=test_acc)
        dvclive.next_step()

        return acc

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.optimizer.lr)
        return optimizer
