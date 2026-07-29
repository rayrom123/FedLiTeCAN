import json
import os
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class MemmapArrayDataset(Dataset):
    def __init__(self, x_path: str, y_path: str, indices_path: str = None):
        self.x = np.load(x_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")
        self.indices = np.load(indices_path, mmap_mode="r") if indices_path else None
        self.length = len(self.indices) if self.indices is not None else len(self.y)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        src_idx = int(self.indices[item]) if self.indices is not None else item
        x = torch.from_numpy(np.asarray(self.x[src_idx], dtype=np.float32))
        y = torch.tensor(int(self.y[src_idx]), dtype=torch.long)
        return x, y


def client_dir(memmap_root: str, client_id: int) -> str:
    return os.path.join(memmap_root, "federated_data", f"client_{client_id}")


def global_dir(memmap_root: str) -> str:
    return os.path.join(memmap_root, "global_test")


def has_client_memmap(memmap_root: str, client_id: int) -> bool:
    cdir = client_dir(memmap_root, client_id)
    required = ["x.npy", "y.npy", "train_idx.npy", "val_idx.npy", "test_idx.npy", "meta.json"]
    return all(os.path.exists(os.path.join(cdir, name)) for name in required)


def has_global_memmap(memmap_root: str) -> bool:
    gdir = global_dir(memmap_root)
    return all(os.path.exists(os.path.join(gdir, name)) for name in ["x.npy", "y.npy", "meta.json"])


def load_meta(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_client_memmap(memmap_root: str, client_id: int):
    cdir = client_dir(memmap_root, client_id)
    train_ds = MemmapArrayDataset(
        os.path.join(cdir, "x.npy"),
        os.path.join(cdir, "y.npy"),
        os.path.join(cdir, "train_idx.npy"),
    )
    val_ds = MemmapArrayDataset(
        os.path.join(cdir, "x.npy"),
        os.path.join(cdir, "y.npy"),
        os.path.join(cdir, "val_idx.npy"),
    )
    test_ds = MemmapArrayDataset(
        os.path.join(cdir, "x.npy"),
        os.path.join(cdir, "y.npy"),
        os.path.join(cdir, "test_idx.npy"),
    )
    meta = load_meta(os.path.join(cdir, "meta.json"))
    return train_ds, val_ds, test_ds, meta


def load_global_memmap(memmap_root: str):
    gdir = global_dir(memmap_root)
    ds = MemmapArrayDataset(os.path.join(gdir, "x.npy"), os.path.join(gdir, "y.npy"))
    meta = load_meta(os.path.join(gdir, "meta.json"))
    return ds, meta


def counts_to_alpha(counts, num_classes: int, device):
    counts = np.asarray(counts, dtype=np.float64)
    total = max(float(counts.sum()), 1.0)
    weights = [np.sqrt(total / counts[c]) if c < len(counts) and counts[c] > 0 else 1.0
               for c in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32).to(device)
