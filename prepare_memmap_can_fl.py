"""Prepare disk-backed memmap arrays for CAN FL training on Kaggle.

This converts each .pt file one at a time into .npy arrays under /kaggle/working
so server/client processes can stream mini-batches from disk instead of loading
the full dataset into RAM.
"""
import argparse
import gc
import json
import os
from collections import Counter

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from model_cnn1d import NUM_GLOBAL_CLASSES


DEFAULT_DATA_ROOT = "/kaggle/input/datasets/npngn123/data-can-fl/CAN_label_skew_FL_only_pt"
DEFAULT_OUT = "/kaggle/working/can_fl_memmap"


def load_pt_xy(path: str):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        x, y = blob["x"], blob["y"]
    elif isinstance(blob, (tuple, list)) and len(blob) >= 2:
        x, y = blob[0], blob[1]
    else:
        raise TypeError(f"Unsupported pt payload type in {path}: {type(blob)}")
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    return x.to(torch.float32).numpy(), y.numpy().astype(np.int64)


def save_meta(path: str, **kwargs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(kwargs, f, indent=2, sort_keys=True)


def save_xy(out_dir: str, x: np.ndarray, y: np.ndarray):
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "x.npy"), x.astype(np.float32, copy=False))
    np.save(os.path.join(out_dir, "y.npy"), y.astype(np.int64, copy=False))


def split_indices(y: np.ndarray, seed: int):
    idx = np.arange(len(y), dtype=np.int64)
    try:
        tmp_idx, test_idx, y_tmp, _ = train_test_split(
            idx, y, test_size=0.2, stratify=y, random_state=seed)
        train_idx, val_idx = train_test_split(
            tmp_idx, test_size=0.25, stratify=y_tmp, random_state=seed)
    except ValueError:
        tmp_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=seed)
        train_idx, val_idx = train_test_split(tmp_idx, test_size=0.25, random_state=seed)
    return train_idx.astype(np.int64), val_idx.astype(np.int64), test_idx.astype(np.int64)


def convert_client(data_root: str, out_root: str, client_id: int, seed: int, force: bool):
    src = os.path.join(data_root, "federated_data", f"client_{client_id}.pt")
    out_dir = os.path.join(out_root, "federated_data", f"client_{client_id}")
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path) and not force:
        print(f"[skip] client {client_id}: {out_dir}")
        return
    print(f"[convert] client {client_id}: {src}")
    x, y = load_pt_xy(src)
    train_idx, val_idx, test_idx = split_indices(y, seed)
    save_xy(out_dir, x, y)
    np.save(os.path.join(out_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(out_dir, "val_idx.npy"), val_idx)
    np.save(os.path.join(out_dir, "test_idx.npy"), test_idx)
    train_counts = np.bincount(y[train_idx], minlength=NUM_GLOBAL_CLASSES).astype(int).tolist()
    class_counts = np.bincount(y, minlength=NUM_GLOBAL_CLASSES).astype(int).tolist()
    save_meta(
        meta_path,
        source=src,
        n_samples=int(len(y)),
        x_shape=list(x.shape),
        class_counts=class_counts,
        train_counts=train_counts,
        n_train=int(len(train_idx)),
        n_val=int(len(val_idx)),
        n_test=int(len(test_idx)),
    )
    del x, y, train_idx, val_idx, test_idx
    gc.collect()


def convert_global(data_root: str, out_root: str, force: bool):
    src = os.path.join(data_root, "global_test_data.pt")
    out_dir = os.path.join(out_root, "global_test")
    meta_path = os.path.join(out_dir, "meta.json")
    if os.path.exists(meta_path) and not force:
        print(f"[skip] global test: {out_dir}")
        return
    print(f"[convert] global test: {src}")
    x, y = load_pt_xy(src)
    save_xy(out_dir, x, y)
    class_counts = np.bincount(y, minlength=NUM_GLOBAL_CLASSES).astype(int).tolist()
    save_meta(
        meta_path,
        source=src,
        n_samples=int(len(y)),
        x_shape=list(x.shape),
        class_counts=class_counts,
    )
    del x, y
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUT)
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-global", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    for cid in range(args.num_clients):
        convert_client(args.data_root, args.output_root, cid, args.seed, args.force)
    if not args.skip_global:
        convert_global(args.data_root, args.output_root, args.force)
    print(f"Memmap data ready at: {args.output_root}")


if __name__ == "__main__":
    main()
