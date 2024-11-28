import lightning as L
import torch
import torch.nn as nn

from positional_encoding import PositionalEncoding


class TimeSeriesTransformerModel(L.LightningModule):
    def __init__(self, seq_len: int, features_per_step: int, embed_dim: int, n_heads: int):
        super(TimeSeriesTransformerModel, self).__init__()

        self.seq_len = seq_len
        self.features_per_step = features_per_step
        self.embed_dim = embed_dim
        self.n_heads = n_heads

        self.embedder = nn.Linear(self.features_per_step, self.embed_dim)
        self.pos_encoder = PositionalEncoding(self.embed_dim)
        encoder_layers = nn.TransformerEncoderLayer(self.embed_dim, self.n_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=1)

        self.linear = nn.Linear(self.embed_dim, 1)

    def forward(self, x):
        x = x.reshape(-1, self.seq_len, self.features_per_step)
        x = self.embedder(x)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        pred_raw = self.linear(x)
        pred = torch.sigmoid(pred_raw)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat[:, -1, :]
        loss = nn.BCELoss()(y_hat, y)
        accuracy = (y_hat.round() == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
        return [optimizer], [scheduler]

    def predict(self, x):
        return self(x)
