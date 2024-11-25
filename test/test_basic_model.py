import lightning as L

from sanity_check_model import SanityCheckTSModel
from test_dataset import CorrelatedTimeSeriesDataset

from time_series_transformer_model import TimeSeriesTransformerModel

def test_basic_model():
    dataset = CorrelatedTimeSeriesDataset(sequence_length=20, noise_std=0)
    model = SanityCheckTSModel(seq_len=20)

    trainer = L.Trainer(max_epochs=3)
    trainer.fit(model, dataset)

def test_transformer_model():
    dataset = CorrelatedTimeSeriesDataset(sequence_length=20, noise_std=0)
    model = TimeSeriesTransformerModel(seq_len=20)

    trainer = L.Trainer(max_epochs=3)
    trainer.fit(model, dataset)
