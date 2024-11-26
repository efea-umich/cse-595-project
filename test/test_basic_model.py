import lightning as L
from torch.utils.data import DataLoader
from sanity_check_model import SanityCheckTSModel
from test_dataset import CorrelatedTimeSeriesDataset

from traffic_dataset import TrafficDataset
from time_series_transformer_model import TimeSeriesTransformerModel

traffic_data_path = "../data/traffic_hourly_dataset.tsf"

def test_basic_model():
    dataset = TrafficDataset(traffic_data_path, num_sequences=50000, sequence_length=20)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SanityCheckTSModel(seq_len=20, features_per_step=1)

    trainer = L.Trainer(max_epochs=3)
    trainer.fit(model, dataloader)


def test_transformer_model():
    dataset = TrafficDataset(traffic_data_path, num_sequences=50000, sequence_length=20)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TimeSeriesTransformerModel(seq_len=20, features_per_step=1, embed_dim=32, n_heads=2)

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, dataloader)
