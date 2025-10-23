"""
Single Batch Dataset for Overfitting Sanity Checks

This dataset wraps a single batch of data and returns the same samples
repeatedly, allowing the training loop to overfit to that batch.
This is useful for verifying:
1. Model has sufficient capacity to represent the data
2. Loss function is working correctly
3. Gradients are flowing properly
"""

import torch
from torch.utils.data import Dataset
from typing import Optional


class SingleBatchDataset(Dataset):
    """
    Dataset that returns samples from a single batch repeatedly.

    This is useful for overfitting sanity checks - if the model cannot
    overfit to a single batch, there's a problem with the model architecture,
    loss function, or gradient flow.

    Args:
        batch: Tensor of shape (batch_size, height, width) containing the batch
        num_repeats: How many times to repeat the batch (default: 50)
                     This controls the number of batches per epoch.
                     With num_repeats=50 and batch_size=32:
                     - Dataset has 1,600 "samples"
                     - DataLoader yields 50 batches per epoch
                     - Each batch contains the same 32 samples
    """

    def __init__(self, batch: torch.Tensor, num_repeats: int = 50):
        self.batch = batch
        self.num_repeats = num_repeats
        self.batch_size = batch.size(0)

    def __len__(self) -> int:
        """Return total number of samples (batch_size * num_repeats)."""
        return self.batch_size * self.num_repeats

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return a sample from the batch, cycling through indices."""
        # Map global index to batch index
        batch_idx = idx % self.batch_size
        return self.batch[batch_idx]


def create_single_batch_dataset(
    full_dataset: Dataset,
    batch_size: int = 128,
    num_repeats: int = 625,
) -> SingleBatchDataset:
    """
    Create a SingleBatchDataset from the first batch of a full dataset.

    Args:
        full_dataset: The full dataset to extract a batch from
        batch_size: Size of the batch to extract (default: 128)
        num_repeats: How many times to repeat the batch (default: 50)

    Returns:
        SingleBatchDataset containing the first batch from full_dataset
    """
    # Extract first batch_size samples
    batch_samples = []
    for i in range(min(batch_size, len(full_dataset))):
        batch_samples.append(full_dataset[i])

    # Stack into a single batch tensor
    batch = torch.stack(batch_samples)

    return SingleBatchDataset(batch, num_repeats=num_repeats)
