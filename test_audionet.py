import pytorch_lightning as pl
import torch, hydra
from net import AudioNet


def testAudioNet():
    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(
            config_name="default.yaml",
            overrides=[
                "trainer.max_epochs=50",
                #    "trainer.gpus=0"
            ],
        )

        pl.seed_everything(seed=cfg.seed)

        data = torch.utils.data.TensorDataset(
            torch.randn(5, 1, 200, 100), torch.randint(low=0, high=10, size=(5,))
        )

        net = AudioNet(hparams=cfg.model)

        trainer = pl.Trainer(**cfg.trainer, overfit_batches=1)

        trainer.fit(
            model=net,
            train_dataloader=torch.utils.data.DataLoader(
                data,
            ),
        )

        print(trainer.logged_metrics["train_loss"].item())
        assert trainer.logged_metrics["train_loss"].item() <= 0.1
