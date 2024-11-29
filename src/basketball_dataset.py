import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder

from basketball_config import BasketballConfig


class BasketballDataset(Dataset):
    def __init__(self, df: pd.DataFrame, config: BasketballConfig, num_sequences=1000, sequence_length=20, pred_horizon=1):
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.pred_horizon = pred_horizon

        self.config = config
        self.categorical_cols = [col for col, col_type in self.config.column_types.items() if col_type == "categorical"]
        self.one_hot_encoder = self.get_one_hot_encoder(df)

        self.data = []
        self.labels = []

        self.load_data(df)

    def get_one_hot_encoder(self, df: pd.DataFrame):
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(df[self.categorical_cols])
        return enc

    def load_data(self, df: pd.DataFrame):
        with tqdm(total=self.num_sequences) as pbar:
            for game in self.games_iterator(df):
                feature_cols = game[self.config.column_types.keys()]
                pred_col = game["WinningTeam"]
                for i in range(len(game) - self.sequence_length - self.pred_horizon):
                    features = self.transform_df(feature_cols.iloc[i:i + self.sequence_length])
                    pred_label_raw = pred_col.values[i + self.sequence_length + self.pred_horizon - 1]

                    pred_label = torch.tensor([1.0]) if pred_label_raw == "Home" else torch.tensor([0.0])

                    self.data.append(features)
                    self.labels.append(pred_label)

                    pbar.update(1)
                    if len(self.data) == self.num_sequences:
                        return


    def transform_df(self, df: pd.DataFrame):
        """
        Transforms the data from the dataframe to a tensor that is has the total number of features per time step
        """
        transformed_series = []

        categorical_df = df[self.categorical_cols]
        categorical_data = self.one_hot_encoder.transform(categorical_df).toarray()
        transformed_series.append(categorical_data)

        numerical_df = df.drop(self.categorical_cols, axis=1)
        numerical_data = numerical_df.values
        numerical_data = np.nan_to_num(numerical_data)
        transformed_series.append(numerical_data)

        concatenated = np.hstack(transformed_series)

        return torch.tensor(concatenated, dtype=torch.float32)

    def games_iterator(self, df: pd.DataFrame):
        for game_id, game_df in df.groupby("URL"):
            yield game_df

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
