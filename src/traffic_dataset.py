import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from tsf_loader import convert_tsf_to_dataframe


class TrafficDataset(Dataset):
    def __init__(self, path: str, num_sequences=1000, sequence_length=20, pred_horizon=1):
        df = convert_tsf_to_dataframe(path)[0]

        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.pred_horizon = pred_horizon

        self.data = []
        self.labels = []

        self.load_data(df)

    def load_data(self, df: pd.DataFrame):
        while True:
            for series in df["series_value"]:
                for i in range(len(series) - self.sequence_length - self.pred_horizon):
                    self.data.append(torch.tensor(series[i : i + self.sequence_length], dtype=torch.float))

                    pred_value = series[i + self.sequence_length + self.pred_horizon - 1]
                    self.labels.append(torch.tensor([1.0]) if pred_value > series[i + self.sequence_length - 1] else torch.tensor([0.0]))

                    if len(self.data) == self.num_sequences:
                        return


    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

