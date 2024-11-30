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
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('medium')


class Main:
    def train_model(
        self,
        config_path: PathLike,
        data_path: PathLike,
        model_save_dir: PathLike = "model",
        num_sequences: int = 100000,
        sequence_length: int = 60,
        sequence_stride: int = 60,
        test_val_split: float = 0.2,
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
            data, config, num_sequences=num_sequences, sequence_length=sequence_length, sequence_stride=sequence_stride
        )

        train, val = torch.utils.data.random_split(
            basketball_dataset, [1 - test_val_split, test_val_split]
        )
        train_dataloader, val_dataloader = (
            DataLoader(train, batch_size=32, shuffle=True),
            DataLoader(val, batch_size=32, shuffle=True),
        )
        example_x, example_y = basketball_dataset[0]
        model = TimeSeriesTransformerModel(
            seq_len=example_x.shape[0],
            features_per_step=example_x.shape[1],
            embed_dim=32,
            n_heads=2,
            num_layers=2,
        )

        if torch.backends.mps.is_available():
            trainer_args = {"accelerator": "cpu"}
        else:
            trainer_args = {}

        trainer = L.Trainer(max_epochs=epochs, logger=mlflow_logger, callbacks=[EarlyStopping(monitor="val_loss", patience=5)], **trainer_args)

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
            data,
            config,
            num_sequences=1100,
            sequence_length=180,
            sequence_stride=180,
            column_transformer=column_transformer,
        )

        model = TimeSeriesTransformerModel.load_from_checkpoint(model_path)
        model.eval()

        if torch.backends.mps.is_available():
            trainer_args = {"accelerator": "cpu"}
        else:
            trainer_args = {}
        dataloader = DataLoader(basketball_dataset, batch_size=32, shuffle=False)
        trainer = L.Trainer(**trainer_args)

        logger.info("Evaluating model")
        trainer.validate(model, dataloader)


if __name__ == "__main__":
    fire.Fire(Main)
