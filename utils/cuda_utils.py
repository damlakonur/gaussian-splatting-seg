from typing import Union, Callable, Optional

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate


def move_to_device(batch, device: Union[torch.device, str] = "cuda"):
    """Recursively move batch data to the specified device."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)
    elif isinstance(batch, dict):
        new_dict = {}
        for k, v in batch.items():
            new_dict[k] = move_to_device(v, device)
        return new_dict
    elif isinstance(batch, list):
        return [move_to_device(v, device) for v in batch]
    else:
        return batch


class GPUCacheLoader:
    """
    DataLoader that pre-loads all data to GPU for faster access.
    Useful for small datasets that can fit in GPU memory.
    """
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        device: str,
        collate_fn: Callable = default_collate,
        verbose: bool = False,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.collate_fn = collate_fn

        self.data = []
        for i in tqdm(range(len(dataset)), disable=not verbose, desc="Loading data into GPU"):
            self.data.append(
                move_to_device(
                    dataset[i],
                    device,
                )
            )

    def _get_iterator(self):
        indices = np.arange(len(self.data))
        if self.shuffle:
            np.random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            data_batch = [self.data[j] for j in indices[i: i + self.batch_size]]
            if self.collate_fn is not None:
                data_batch = self.collate_fn(data_batch)
            yield data_batch

    def __iter__(self):
        return self._get_iterator()

