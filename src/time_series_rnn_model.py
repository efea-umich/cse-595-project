import lightning as L
import torch
import torch.nn as nn
import torchmetrics

from positional_encoding import PositionalEncoding


class TimeSeriesRNNModel(L.LightningModule):
    def __init__(
        self,
        seq_len: int,
        features_per_step: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super(TimeSeriesRNNModel, self).__init__()
        self.save_hyperparameters()

        self.seq_len = seq_len
        self.features_per_step = features_per_step
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedder = nn.Linear(self.features_per_step, self.hidden_dim)
        self.rnn = nn.RNN(
            input_size=self.hidden_dim, 
            hidden_size=self.hidden_dim, 
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            nonlinearity="relu"  # You can change it to 'tanh' if preferred
        )
        self.linear = nn.Linear(self.hidden_dim, 1)

    def forward(self, x, padding_mask=None):
        """
        Forward pass with optional padding mask.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, features_per_step)
            padding_mask (Tensor): Optional boolean mask of shape (batch_size, seq_len)
                                   with True for valid positions and False for padding.
        """
        x = x.reshape(-1, self.seq_len, self.features_per_step)
        x = self.embedder(x)

        # RNN expects the input shape (batch_size, seq_len, input_size)
        rnn_out, _ = self.rnn(x)  # we ignore the hidden state here

        # Take the output from the last time step
        rnn_out_last = rnn_out[:, -1, :]

        pred_raw = self.linear(rnn_out_last)
        pred = torch.sigmoid(pred_raw)
        return pred

    def create_padding_mask(self, x):
        """
        Create a padding mask based on input tensor `x`.

        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, features_per_step)

        Returns:
            Tensor: Boolean mask of shape (batch_size, seq_len) where True indicates valid positions.
        """
        return ~torch.any(x != 0, dim=-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        padding_mask = self.create_padding_mask(x)
        y_hat = self(x, padding_mask=padding_mask)

        y_hat = y_hat[:, -1].reshape(y.shape)

        loss = nn.BCELoss()(y_hat, y)
        accuracy = (y_hat.round() == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        padding_mask = self.create_padding_mask(x)
        y_hat = self(x, padding_mask=padding_mask)

        y_hat = y_hat[:, -1].reshape(y.shape)

        loss = nn.BCELoss()(y_hat, y)
        accuracy = (y_hat.round() == y).float().mean()

        self.log("val_loss", loss, on_step=True, on_epoch=True)
        self.log("val_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)
        self.log("auc", torchmetrics.functional.auroc(y_hat, y.int(), task="binary"), on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return [optimizer], [scheduler]

    def predict(self, x):
        padding_mask = self.create_padding_mask(x)
        return self(x, padding_mask=padding_mask)
