from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle as pkl

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from basketball_config import BasketballConfig


class BasketballDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        config: BasketballConfig,
        num_sequences=1000,
        sequence_length=20,
        pred_horizon=1,
        column_transformer: Optional[ColumnTransformer] = None,
    ):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.pred_horizon = pred_horizon

        self.config = config

        self.categorical_cols = [
            col
            for col, col_type in self.config.column_types.items()
            if col_type == "categorical"
        ]
        self.numerical_cols = [
            col
            for col, col_type in self.config.column_types.items()
            if col_type == "numeric"
        ]

        if column_transformer is None:
            self.column_transformer = self.get_column_transformer(df)
            with open("column_transformer.pkl", "wb") as f:
                pkl.dump(self.column_transformer, f)
        else:
            self.column_transformer = column_transformer

        self.data = []
        self.labels = []

        self.load_data(df)

    def get_column_transformer(self, df: pd.DataFrame):
        transformers = []

        if self.numerical_cols:
            transformers.append(
                (
                    "num",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                        ("scaler", StandardScaler())
                    ]),
                    self.numerical_cols
                )
            )

        if self.categorical_cols:
            transformers.append(
                ("cat", OneHotEncoder(handle_unknown="ignore",), self.categorical_cols)
            )

        column_transformer = ColumnTransformer(
            transformers=transformers, remainder="drop"
        )

        column_transformer.fit(df)

        return column_transformer


    def load_data(self, df: pd.DataFrame):
        with tqdm(total=self.num_sequences) as pbar:
            for game in self.games_iterator(df):
                feature_cols = game[self.config.column_types.keys()]
                transformed_features = self.transform_df(feature_cols)
                pred_col = game["WinningTeam"]

                for i in range(len(game) - self.sequence_length - self.pred_horizon):
                    if i / len(game) > 0.35:
                        break

                    features = transformed_features[i:i + self.sequence_length]

                    pred_label_raw = pred_col.values[
                        i + self.sequence_length + self.pred_horizon - 1
                    ]

                    pred_label = torch.tensor(pred_label_raw, dtype=torch.float32).reshape(1)

                    self.data.append(features)
                    self.labels.append(pred_label)

                    pbar.update(1)
                    if len(self.data) == self.num_sequences:
                        return

    def transform_df(self, df: pd.DataFrame):
        """
        Transforms the data from the dataframe to a tensor that has the total number of features per time step.

        Args:
            df (pd.DataFrame): The input DataFrame with features to be transformed.

        Returns:
            torch.Tensor: A tensor of shape (sequence_length, num_features), where
                          num_features is the total number of features after transformation.
        """
        transformed = self.column_transformer.transform(df)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        tensor = torch.tensor(transformed, dtype=torch.float32)

        return tensor


    def games_iterator(self, df: pd.DataFrame):
        for game_id, game_df in df.groupby("URL"):
            yield game_df

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
