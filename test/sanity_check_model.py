import lightning as L
import torch
import torch.nn as nn


class SanityCheckTSModel(L.LightningModule):
    def __init__(self, seq_len: int, features_per_step: int):
        super().__init__()
        self.seq_len = seq_len
        self.features_per_step = features_per_step
        self.lstm = nn.LSTM(self.features_per_step, 32, batch_first=True)
        self.linear = nn.Linear(32, 1)
        self.layer_norm = nn.LayerNorm(self.seq_len)

    def forward(self, x):
        x = x.reshape(-1, self.seq_len, self.features_per_step)
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
