import lightning as L
import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding

class TimeSeriesTransformerModel(L.LightningModule):
    def __init__(self, seq_len: int):
        super(TimeSeriesTransformerModel, self).__init__()

        self.seq_len = seq_len
        self.embed_dim = 16
        self.nhead = 1

        self.embedder = nn.Linear(1, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(self.embed_dim, self.nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)

        self.linear = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        x = x.reshape(-1, self.seq_len, 1)
        x = self.embedder(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat[:, -1, :]
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

    def predict(self, x):
        return self(x)
