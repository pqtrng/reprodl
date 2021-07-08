import torch
import pytorch_lightning as pl
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
from data import ESC50Dataset
from net import AudioNet


logger = logging.getLogger(__name__)


@hydra.main(config_path="configs", config_name="default")
def train(cfg: DictConfig):

    wandb.init(project="reprodl")

    logger.info(OmegaConf.to_yaml(cfg=cfg))

    data_path = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.path)

    # Load data
    train_loader = torch.utils.data.DataLoader(
        dataset=ESC50Dataset(path=data_path, folds=cfg.data.train_folds),
        num_workers=cfg.data.workers,
        batch_size=cfg.data.batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=ESC50Dataset(path=data_path, folds=cfg.data.val_folds),
        num_workers=cfg.data.workers,
        batch_size=cfg.data.batch_size,
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=ESC50Dataset(path=data_path, folds=cfg.data.test_folds),
        num_workers=cfg.data.workers,
        batch_size=cfg.data.batch_size,
    )

    pl.seed_everything(cfg.seed)

    audio_net = AudioNet(hparams=cfg.model)

    trainer = pl.Trainer(**cfg.trainer, logger=pl.loggers.WandbLogger())

    trainer.fit(
        model=audio_net, train_dataloader=train_loader, val_dataloaders=val_loader
    )

    torch.save(trainer.model.state_dict(), "model.pth")


if __name__ == "__main__":
    train()
