import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys


def sbox_layer(x):
    """CHES20 S-box layer."""
    y1 = (x[0] & x[1]) ^ x[2]
    y0 = (x[3] & x[0]) ^ x[1]
    y3 = (y1 & x[3]) ^ x[0]
    y2 = (y0 & y1) ^ x[3]
    return np.stack([y0, y1, y2, y3], axis=1)


class CHES20Dataset(Dataset):
    """PyTorch Dataset for CHES20 dataset."""

    def __init__(self, data_path, split, input_length, data_desync=0):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.data_desync = data_desync

        # Load data
        data = np.load(data_path)
        self.traces = data['traces']
        self.nonces = data['nonces']
        self.umsk_keys = data['umsk_keys']

        # Shift nonces and keys
        shift = 17
        self.nonces = self.nonces >> shift
        self.umsk_keys = self.umsk_keys >> shift
        if len(self.umsk_keys.shape) == 1:
            self.umsk_keys = np.reshape(self.umsk_keys, [1, -1])

        # Compute labels using S-box
        sbox_in = np.bitwise_xor(self.nonces, self.umsk_keys)
        sbox_in = sbox_in.T
        sbox_out = sbox_layer(sbox_in)
        self.labels = (sbox_out & 0x1)
        self.labels = self.labels.astype(np.float32)

        # Truncate traces
        assert (self.input_length + self.data_desync) <= self.traces.shape[1]
        self.traces = self.traces[:, :(self.input_length + self.data_desync)]

        self.num_samples = self.traces.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        trace = self.traces[idx]
        label = self.labels[idx]

        # Apply random shift during training if data_desync > 0
        if self.data_desync > 0 and self.input_length < trace.shape[0]:
            max_shift = min(self.data_desync, trace.shape[0] - self.input_length)
            shift = np.random.randint(0, max_shift + 1)
            trace = trace[shift:shift + self.input_length]
        else:
            trace = trace[:self.input_length]

        trace = torch.from_numpy(trace.astype(np.float32))
        label = torch.from_numpy(label.astype(np.float32))

        return trace, label


def get_dataloader(data_path, split, input_length, batch_size,
                   data_desync=0, shuffle=True, num_workers=4):
    """Create DataLoader for CHES20 dataset.

    Args:
        data_path: Path to npz file
        split: 'train' or 'test'
        input_length: Length of input traces
        batch_size: Batch size
        data_desync: Maximum desynchronization for data augmentation
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    dataset = CHES20Dataset(data_path, split, input_length, data_desync)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train')
    )

    return dataloader


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python data_utils_ches20.py <data_path> <batch_size> <split>")
        sys.exit(1)

    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    split = sys.argv[3]

    dataset = CHES20Dataset(data_path, split, input_length=5000)

    print(f"traces shape    : {dataset.traces.shape}")
    print(f"labels shape    : {dataset.labels.shape}")
    print(f"traces dtype    : {dataset.traces.dtype}")
    print()

    dataloader = get_dataloader(data_path, split, 5000, batch_size, shuffle=True)

    for i, (traces, labels) in enumerate(dataloader):
        if i >= 1:
            break
        print(f"Batch traces shape: {traces.shape}, labels shape: {labels.shape}")
        print(f"Batch traces dtype: {traces.dtype}, labels dtype: {labels.dtype}")
        print(f"Sample trace values: {traces[0, :10]}")
        print(f"Sample labels: {labels[0]}")
