from os import PathLike
from pathlib import Path

import pandas as pd
import yaml
import lightning as L
import fire
import pickle as pkl

import torch

from enhance_data import DataEnhancer
from basketball_config import BasketballConfig
from basketball_dataset import BasketballDataset
from time_series_transformer_model import TimeSeriesTransformerModel
from torch.utils.data import DataLoader
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from loguru import logger


class Main:
    def train_model(
        self,
        config_path: PathLike,
        data_path: PathLike,
        model_save_dir: PathLike = "model",
        epochs: int = 10,
    ):
        with open(config_path, "r") as f:
            config = BasketballConfig.model_validate(yaml.safe_load(f))

        mlflow_logger = MLFlowLogger(
            experiment_name="basketball",
            tracking_uri="http://127.0.0.1:5000",
        )

        logger.info("Loading data from {}", data_path)
        data = pd.read_csv(data_path)

        logger.info("Enhancing data")
        data = DataEnhancer(data, config).enhance_data(data)

        logger.info("Creating dataset")
        basketball_dataset = BasketballDataset(
            data, config, num_sequences=100000, sequence_length=30
        )

        train, val = torch.utils.data.random_split(basketball_dataset, [0.8, 0.2])
        train_dataloader, val_dataloader = (
            DataLoader(train, batch_size=32, shuffle=True),
            DataLoader(val, batch_size=32, shuffle=True),
        )
        example_x, example_y = basketball_dataset[0]
        model = TimeSeriesTransformerModel(
            seq_len=example_x.shape[0],
            features_per_step=example_x.shape[1],
            embed_dim=32,
            n_heads=4,
            num_layers=1,
        )
        trainer = L.Trainer(max_epochs=epochs, logger=mlflow_logger)

        logger.info("Training model")
        trainer.fit(
            model,
            train_dataloader,
            val_dataloaders=[val_dataloader],
        )
        trainer.save_checkpoint(Path(model_save_dir) / "model.ckpt")

    def evaluate_model(
        self,
        config_path: PathLike,
        data_path: PathLike,
        model_path: PathLike = "model/model.ckpt",
        column_transformer_path: PathLike = "column_transformer.pkl",
    ):
        with open(config_path, "r") as f:
            config = BasketballConfig.model_validate(yaml.safe_load(f))

        with open(column_transformer_path, "rb") as f:
            column_transformer = pkl.load(f)

        logger.info("Loading data from {}", data_path)
        data = pd.read_csv(data_path)

        logger.info("Enhancing data")
        data = DataEnhancer(data, config).enhance_data(data)

        logger.info("Creating dataset")
        basketball_dataset = BasketballDataset(
            data, config, num_sequences=50000, sequence_length=30, column_transformer=column_transformer
        )

        model = TimeSeriesTransformerModel.load_from_checkpoint(model_path)
        model.eval()

        dataloader = DataLoader(basketball_dataset, batch_size=32, shuffle=False)
        trainer = L.Trainer()

        logger.info("Evaluating model")
        trainer.validate(model, dataloader)


if __name__ == "__main__":
    fire.Fire(Main)
