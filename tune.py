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
def tune(cfg: DictConfig):

    config = {
        "sample_rate": cfg.data.sample_rate,
        "batch_size": cfg.data.batch_size,
        "lr": cfg.model.optimizer.lr,
        "base_filters": cfg.model.base_filters,
    }

    wandb.init(project="reprodl", config=config)

    cfg.data.sample_rate = wandb.config.sample_rate
    cfg.data.batch_size = wandb.config.batch_size
    cfg.model.optimizer.lr = wandb.config.lr
    cfg.model.base_filters = wandb.config.base_filters

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


if __name__ == "__main__":
    tune()
