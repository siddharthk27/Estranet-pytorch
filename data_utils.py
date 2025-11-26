import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import os
import sys


class ASCADDataset(Dataset):
    """PyTorch Dataset for ASCAD datasets."""

    def __init__(self, data_path, split, input_length, data_desync=0):
        self.data_path = data_path
        self.split = split
        self.input_length = input_length
        self.data_desync = data_desync

        corpus = h5py.File(data_path, 'r')
        if split == 'train':
            split_key = 'Profiling_traces'
        elif split == 'test':
            split_key = 'Attack_traces'

        # Load traces and labels
        self.traces = corpus[split_key]['traces'][:, :(self.input_length + self.data_desync)]
        self.labels = np.reshape(corpus[split_key]['labels'][()], [-1])
        self.labels = self.labels.astype(np.int64)
        self.num_samples = self.traces.shape[0]

        # Extract metadata
        self.plaintexts = self.get_plaintexts(corpus[split_key]['metadata'])
        self.masks = self.get_masks(corpus[split_key]['metadata'])
        self.keys = self.get_keys(corpus[split_key]['metadata'])

        corpus.close()

    def get_plaintexts(self, metadata):
        plaintexts = []
        for i in range(len(metadata)):
            plaintexts.append(metadata[i]['plaintext'][2])
        return np.array(plaintexts)

    def get_keys(self, metadata):
        keys = []
        for i in range(len(metadata)):
            keys.append(metadata[i]['key'][2])
        return np.array(keys)

    def get_masks(self, metadata):
        masks = []
        for i in range(len(metadata)):
            masks.append(np.array(metadata[i]['masks']))
        masks = np.stack(masks, axis=0)
        return masks

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
        label = torch.tensor(label, dtype=torch.long)

        return trace, label


def get_dataloader(data_path, split, input_length, batch_size,
                   data_desync=0, shuffle=True, num_workers=4):
    """Create DataLoader for ASCAD dataset.

    Args:
        data_path: Path to h5 file
        split: 'train' or 'test'
        input_length: Length of input traces
        batch_size: Batch size
        data_desync: Maximum desynchronization for data augmentation
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    dataset = ASCADDataset(data_path, split, input_length, data_desync)

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
        print("Usage: python data_utils.py <data_path> <batch_size> <split>")
        sys.exit(1)

    data_path = sys.argv[1]
    batch_size = int(sys.argv[2])
    split = sys.argv[3]

    dataset = ASCADDataset(data_path, split, input_length=5000)

    print(f"traces shape    : {dataset.traces.shape}")
    print(f"labels shape    : {dataset.labels.shape}")
    print(f"plaintext shape : {dataset.plaintexts.shape}")
    print(f"keys shape      : {dataset.keys.shape}")
    print(f"traces dtype    : {dataset.traces.dtype}")
    print()

    dataloader = get_dataloader(data_path, split, 5000, batch_size, shuffle=True)

    for i, (traces, labels) in enumerate(dataloader):
        if i >= 1:
            break
        print(f"Batch traces shape: {traces.shape}, labels shape: {labels.shape}")
        print(f"Batch traces dtype: {traces.dtype}, labels dtype: {labels.dtype}")
        print(f"Sample trace values: {traces[0, :10]}")
        print(f"Sample labels: {labels}")
