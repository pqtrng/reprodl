import logging
import os
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def retrieve_data(path, target_dataset, data_size):
    original_dataset_path = path / Path("meta/esc50.csv")
    original_dataset = pd.read_csv(original_dataset_path)

    not_retrieved_index = original_dataset.index.difference(target_dataset.index)

    retrieved_index = np.random.choice(not_retrieved_index, data_size)

    return original_dataset.loc[retrieved_index]


def add_data(retrieved_dataset, target_dataset):
    return target_dataset.append(retrieved_dataset, ignore_index=False, sort=True)


@hydra.main(config_path="configs", config_name="default")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg))
    path = Path(hydra.utils.get_original_cwd()) / Path(cfg.data.path)

    # Retrieve data
    target_dataset_path = path / Path("meta/extra_esc50.csv")
    if os.path.isfile(target_dataset_path):
        target_dataset = pd.read_csv(target_dataset_path)
    else:
        target_dataset = pd.DataFrame()

    retrieved_dataset = retrieve_data(
        path=path, target_dataset=target_dataset, data_size=cfg.data.retrieval_size
    )

    target_dataset = add_data(
        retrieved_dataset=retrieved_dataset, target_dataset=target_dataset
    )

    target_dataset.to_csv(target_dataset_path, index=False)
    print("Saved")


if __name__ == "__main__":
    main()
