import lightning as L
import torch
import torch.nn as nn

class SanityCheckTSModel(L.LightningModule):
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(1, 32, batch_first=True)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        x = x.reshape(-1, self.seq_len, 1)
        lstm_out, _ = self.lstm(x)
        return self.linear(lstm_out[:, -1])

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
