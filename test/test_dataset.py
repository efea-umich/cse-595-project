import torch
from torch.utils.data import Dataset, DataLoader


class CorrelatedTimeSeriesDataset(Dataset):
    def __init__(self, num_sequences=1000, sequence_length=20, noise_std=0.1):
        """
        Initializes the dataset.
        Args:
            num_sequences (int): Number of sequences in the dataset.
            sequence_length (int): Length of each sequence.
            noise_std (float): Standard deviation of the noise added to the series.
        """
        self.num_sequences = num_sequences
        self.sequence_length = sequence_length
        self.noise_std = noise_std
        self.data = []
        self.labels = []

        # Generate sequences
        for _ in range(num_sequences):
            sequence, label = self._generate_sequence()
            self.data.append(sequence)
            self.labels.append(label)

        self.data = torch.stack(self.data)
        self.labels = torch.stack(self.labels)

    def _generate_sequence(self):
        """
        Generates a single sequence with a non-trivial correlation.
        Returns:
            torch.Tensor: The generated sequence.
            torch.Tensor: The target for the sequence.
        """
        sequence = torch.zeros(self.sequence_length)
        for t in range(self.sequence_length):
            if t == 0:
                sequence[t] = torch.randn(1)  # Random initial value
            else:
                # Value depends on a weighted sum of the last three timesteps
                w1, w2, w3 = 0.6, 0.3, 0.1  # Example weights
                seq_val = (
                        w1 * sequence[t - 1] +
                        (w2 * sequence[t - 2] if t > 1 else 0) +
                        (w3 * sequence[t - 3] if t > 2 else 0)
                )
                sequence[t] = seq_val + torch.randn(1) * self.noise_std  # Add noise

        # Use a weighted sum of the last three timesteps as the label
        label = (
                        0.5 * sequence[-1] +
                        0.3 * sequence[-2] +
                        0.2 * sequence[-3]
                ) + torch.randn(1) * self.noise_std  # Add noise to label

        return sequence, label

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
