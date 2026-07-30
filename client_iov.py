"""Flower client cho CAN federated learning voi CNN1D.

Data train mac dinh:
  /kaggle/input/datasets/npngn123/data-can-fl/CAN_label_skew_FL_only_pt/
    federated_data/client_<client_id>.pt
"""
import argparse
import logging
import os
import time
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from model_cnn1d import CNN1D_IDS, FocalLoss, NUM_GLOBAL_CLASSES, INPUT_LEN
from can_memmap import counts_to_alpha, has_client_memmap, load_client_memmap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "/kaggle/input/datasets/npngn123/data-can-fl/CAN_label_skew_FL_only_pt"


def subsample_capped(x: np.ndarray, y: np.ndarray, max_samples: int, seed=42):
    """Giu cac lop thieu so nhieu nhat co the, cat bot lop da so neu can."""
    if max_samples <= 0 or len(y) <= max_samples:
        return x, y
    rng = np.random.default_rng(seed)
    counts = Counter(y.tolist())
    classes = sorted(counts, key=lambda c: counts[c])
    remaining = max_samples
    keep_idx = []
    for i, c in enumerate(classes):
        quota = remaining // (len(classes) - i)
        idx = np.where(y == c)[0]
        if len(idx) > quota:
            idx = rng.choice(idx, quota, replace=False)
        keep_idx.append(idx)
        remaining -= len(idx)
    keep = np.concatenate(keep_idx)
    rng.shuffle(keep)
    return x[keep], y[keep]


def get_model_parameters(model: torch.nn.Module):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def set_model_parameters(model: torch.nn.Module, parameters: List[np.ndarray]) -> None:
    keys = model.state_dict().keys()
    state = OrderedDict({k: torch.tensor(v) for k, v in zip(keys, parameters)})
    model.load_state_dict(state, strict=True)


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


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_id: int, data_root: str, device: torch.device,
                 max_samples: int, batch_size: int, memmap_root: str = ""):
        self.client_id = client_id
        self.data_root = data_root
        self.memmap_root = memmap_root
        self.device = device
        self.max_samples = max_samples
        self.batch_size = batch_size
        self.data_loaded = False

        self.model = CNN1D_IDS(input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES,
                               dropout=0.15).to(device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.criterion = torch.nn.CrossEntropyLoss()

    def _data_path(self) -> str:
        return f"{self.data_root}/federated_data/client_{self.client_id}.pt"

    def _has_data(self) -> bool:
        if self.memmap_root:
            return has_client_memmap(self.memmap_root, self.client_id)
        return os.path.exists(self._data_path())

    def _ensure_data_loaded(self):
        if self.data_loaded:
            return

        path = self._data_path()
        if self.memmap_root:
            if not has_client_memmap(self.memmap_root, self.client_id):
                raise FileNotFoundError(f"Missing memmap for client {self.client_id} in {self.memmap_root}")
            train_ds, val_ds, test_ds, meta = load_client_memmap(self.memmap_root, self.client_id)
            self.train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
            self.val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
            self.test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
            self.num_train = len(train_ds)
            logger.info(
                f"Client {self.client_id}: streaming memmap from {self.memmap_root}, "
                f"n_train={len(train_ds)} n_val={len(val_ds)} n_test={len(test_ds)}"
            )
            alpha = counts_to_alpha(meta.get("train_counts", []), NUM_GLOBAL_CLASSES, self.device)
            self.criterion = FocalLoss(alpha=alpha, gamma=2.0)
            self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
            self.data_loaded = True
            return

        if not os.path.exists(path):
            raise FileNotFoundError(path)
        x, y = load_pt_xy(path)
        logger.info(
            f"Client {self.client_id}: loaded FL data from {path}, "
            f"x={x.shape}, classes={dict(sorted(Counter(y.tolist()).items()))}"
        )

        x, y = subsample_capped(x, y, self.max_samples)
        logger.info(f"Client {self.client_id}: after subsample n={len(y)}")

        try:
            x_tmp, x_test, y_tmp, y_test = train_test_split(
                x, y, test_size=0.2, stratify=y, random_state=42)
            x_train, x_val, y_train, y_val = train_test_split(
                x_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)
        except ValueError:
            x_tmp, x_test, y_tmp, y_test = train_test_split(
                x, y, test_size=0.2, random_state=42)
            x_train, x_val, y_train, y_val = train_test_split(
                x_tmp, y_tmp, test_size=0.25, random_state=42)

        def loader(xa, ya, shuffle):
            ds = TensorDataset(torch.from_numpy(xa), torch.from_numpy(ya))
            return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle)

        self.train_loader = loader(x_train, y_train, True)
        self.val_loader = loader(x_val, y_val, False)
        self.test_loader = loader(x_test, y_test, False)
        self.num_train = len(y_train)

        cnt = Counter(y_train.tolist())
        total = len(y_train)
        weights = [np.sqrt(total / cnt[c]) if c in cnt else 1.0
                   for c in range(NUM_GLOBAL_CLASSES)]
        alpha = torch.tensor(weights, dtype=torch.float32).to(self.device)
        self.criterion = FocalLoss(alpha=alpha, gamma=2.0)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        self.data_loaded = True

    def get_parameters(self, config) -> List[np.ndarray]:
        return get_model_parameters(self.model)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        set_model_parameters(self.model, parameters)

    def fit(self, parameters, config) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        if not self._has_data():
            logger.info(f"Client {self.client_id}: skip round (missing train file)")
            return self.get_parameters({}), 0, {
                "skipped": True,
                "train_loss": 0.0,
                "train_accuracy": 0.0,
            }

        self._ensure_data_loaded()

        epochs = int(config.get("local_epochs", 1))
        self.model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        for epoch in range(epochs):
            running, correct, total = 0.0, 0, 0
            for xb, yb in self.train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                running += loss.item() * yb.size(0)
                correct += (out.argmax(1) == yb).sum().item()
                total += yb.size(0)

            epoch_loss = running / total if total > 0 else 0.0
            epoch_acc = correct / total if total > 0 else 0.0
            logger.info(
                f"Client {self.client_id} epoch {epoch+1}/{epochs}: "
                f"train_loss={epoch_loss:.4f} train_acc={epoch_acc:.4f}"
            )

        return self.get_parameters({}), self.num_train, {
            "train_loss": float(epoch_loss),
            "train_accuracy": float(epoch_acc),
        }

    def evaluate(self, parameters, config) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        if not self._has_data():
            logger.info(f"Client {self.client_id}: skip eval (missing train file)")
            metrics: Dict = {
                "skipped": True,
                "accuracy": 0.0,
                "balanced_accuracy": 0.0,
                "eval_loss": 0.0,
            }
            for avg in ("micro", "macro", "weighted"):
                metrics[f"{avg}_precision"] = 0.0
                metrics[f"{avg}_recall"] = 0.0
                metrics[f"{avg}_f1"] = 0.0
            return 0.0, 0, metrics

        self._ensure_data_loaded()

        self.model.eval()
        loss_sum, correct, total = 0.0, 0, 0
        preds, targs = [], []
        with torch.no_grad():
            for xb, yb in self.test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.model(xb)
                loss_sum += self.criterion(out, yb).item() * yb.size(0)
                p = out.argmax(1)
                correct += (p == yb).sum().item()
                total += yb.size(0)
                preds.extend(p.cpu().numpy())
                targs.extend(yb.cpu().numpy())

        test_loss = loss_sum / total if total > 0 else 0.0
        acc = correct / total if total > 0 else 0.0
        bal_acc = balanced_accuracy_score(targs, preds) if total > 0 else 0.0

        metrics: Dict = {
            "accuracy": float(acc),
            "balanced_accuracy": float(bal_acc),
            "eval_loss": float(test_loss),
        }
        for avg in ("micro", "macro", "weighted"):
            prec, rec, f1, _ = precision_recall_fscore_support(
                targs, preds, average=avg, zero_division=0)
            metrics[f"{avg}_precision"] = float(prec)
            metrics[f"{avg}_recall"] = float(rec)
            metrics[f"{avg}_f1"] = float(f1)

        logger.info(
            f"Client {self.client_id} eval: eval_loss={test_loss:.4f} "
            f"acc={acc:.4f} micro_f1={metrics['micro_f1']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} weighted_f1={metrics['weighted_f1']:.4f}"
        )
        return float(test_loss), total, metrics


def main():
    parser = argparse.ArgumentParser(description="CNN1D CAN FL Flower client")
    parser.add_argument("--client-id", type=int, required=True, choices=range(10))
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--memmap-root", type=str, default="",
                        help="Thu muc memmap da tao bang prepare_memmap_can_fl.py")
    parser.add_argument("--server-address", type=str, default="127.0.0.1:8081")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Gioi han so mau moi client (0 = dung het)")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--connect-retries", type=int, default=120,
                        help="So lan thu ket noi server truoc khi fail")
    parser.add_argument("--retry-wait", type=int, default=5,
                        help="So giay cho giua cac lan reconnect")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    client = FlowerClient(
        client_id=args.client_id,
        data_root=args.data_root,
        device=device,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        memmap_root=args.memmap_root,
    )

    for attempt in range(args.connect_retries):
        try:
            fl.client.start_numpy_client(server_address=args.server_address, client=client)
            break
        except Exception as e:
            logger.error(f"Connect attempt {attempt+1} failed: {e}")
            if attempt < args.connect_retries - 1:
                time.sleep(args.retry_wait)
            else:
                raise


if __name__ == "__main__":
    main()
