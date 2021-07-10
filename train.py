import json
import logging
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import wandb
from omegaconf import DictConfig
from omegaconf import OmegaConf

from data import ESC50Dataset
from net import AudioNet


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="default")
def train(cfg: DictConfig):
    """Train AudioNet on ESC50 dataset.

    Args:
        cfg (DictConfig): Training configuration
    """
    cfg.data.num_workers = os.cpu_count()
    if not torch.cuda.is_available():
        cfg.trainer.gpus = 0
        cfg.trainer.max_epochs = 2

    logger.info(OmegaConf.to_yaml(cfg=cfg))

    data_path = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.path)

    # Load data
    train_loader = torch.utils.data.DataLoader(
        dataset=ESC50Dataset(path=data_path, folds=cfg.data.train_folds),
        num_workers=cfg.data.num_workers,
        batch_size=cfg.data.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=ESC50Dataset(path=data_path, folds=cfg.data.val_folds),
        num_workers=cfg.data.num_workers,
        batch_size=cfg.data.batch_size,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=ESC50Dataset(path=data_path, folds=cfg.data.test_folds),
        num_workers=cfg.data.num_workers,
        batch_size=cfg.data.batch_size,
    )

    pl.seed_everything(cfg.seed)

    wandb.init(project="reprodl")

    audio_net = AudioNet(hparams=cfg.model)

    trainer = pl.Trainer(
        gpus=cfg.trainer.gpus,
        max_epochs=cfg.trainer.max_epochs,
        logger=pl.loggers.WandbLogger(),
    )

    trainer.fit(
        model=audio_net, train_dataloader=train_loader, val_dataloaders=val_loader
    )

    torch.save(trainer.model.state_dict(), "model.pth")

    trainer.test(model=audio_net, test_dataloaders=test_loader)

    # summary dvc
    train_loss = trainer.logged_metrics["train_loss"].data.cpu().numpy().reshape(1)[0]
    valid_accuracy = trainer.logged_metrics["val_acc"].data.cpu().numpy().reshape(1)[0]
    test_accuracy = trainer.logged_metrics["test_acc"].data.cpu().numpy().reshape(1)[0]

    summary_data = {
        "stages": {
            "train": {
                "train_loss": train_loss.astype(float),
                "valid_accuracy": valid_accuracy.astype(float),
                "test_accuracy": test_accuracy.astype(float),
            }
        }
    }

    with open("summary.json", "w") as current_file:
        json.dump(
            summary_data, current_file, ensure_ascii=True, indent=4, sort_keys=True
        )


if __name__ == "__main__":
    train()
