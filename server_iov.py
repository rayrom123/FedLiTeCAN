"""Flower server cho CAN federated incremental learning voi CNN1D + FedAvg.

Mac dinh chay 6 task, moi task 30 communication rounds, moi round 1 local epoch.
Server danh gia tap trung tren global_test_data.pt sau moi round va ghi:
train loss, eval loss, accuracy, micro/macro/weighted precision/recall/F1.

Vi du:
  python server_iov.py --mode train
  python server_iov.py --mode test --checkpoint checkpoints_can_il/round_180.pth
"""
import argparse
import csv
import logging
import os
from collections import Counter, OrderedDict
from typing import Dict, List, Optional

import flwr as fl
import numpy as np
import torch
from flwr.common import Parameters
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, TensorDataset

from model_cnn1d import CNN1D_IDS, FocalLoss, NUM_GLOBAL_CLASSES, INPUT_LEN
from client_iov import subsample_capped

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_DATA_ROOT = "/kaggle/input/datasets/npngn123/data-can-fl-il/CAN_label_skew_final_pt"
DEFAULT_TEST = DEFAULT_DATA_ROOT + "/global_test_data.pt"
CKPT_DIR = "checkpoints_can_il"
CSV_FILE = "metrics_can_il.csv"

METRIC_KEYS = [
    "train_loss", "train_accuracy",
    "eval_loss", "accuracy",
    "micro_precision", "micro_recall", "micro_f1",
    "macro_precision", "macro_recall", "macro_f1",
    "weighted_precision", "weighted_recall", "weighted_f1",
]
CSV_HEADER = ["global_round", "task_id", "task_round"] + METRIC_KEYS


def get_model_parameters(model):
    return [v.cpu().numpy() for _, v in model.state_dict().items()]


def ndarrays_to_state_dict(model, ndarrays):
    keys = model.state_dict().keys()
    return OrderedDict({k: torch.tensor(v) for k, v in zip(keys, ndarrays)})


def append_csv_row(path: str, row: List):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


def load_global_test(test_file: str, max_samples: int, batch_size: int):
    logger.info(f"Loading global test set: {test_file}")
    blob = torch.load(test_file, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        x, y = blob["x"], blob["y"]
    elif isinstance(blob, (tuple, list)) and len(blob) >= 2:
        x, y = blob[0], blob[1]
    else:
        raise TypeError(f"Unsupported global test payload type: {type(blob)}")

    if not torch.is_tensor(x):
        x = torch.tensor(x)
    if not torch.is_tensor(y):
        y = torch.tensor(y)
    x = x.to(torch.float32).numpy()
    y = y.numpy().astype(np.int64)

    logger.info(f"Global test: n={len(y)}, classes={dict(sorted(Counter(y.tolist()).items()))}")
    x, y = subsample_capped(x, y, max_samples)
    logger.info(f"Evaluating each round on n={len(y)} samples")

    loader = DataLoader(
        TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
        batch_size=batch_size,
        shuffle=False,
    )
    return loader, y


def make_criterion(y: np.ndarray, device):
    cnt = Counter(y.tolist())
    total = len(y)
    weights = [np.sqrt(total / cnt[c]) if c in cnt else 1.0
               for c in range(NUM_GLOBAL_CLASSES)]
    return FocalLoss(alpha=torch.tensor(weights, dtype=torch.float32).to(device), gamma=2.0)


def evaluate_on_global_test(model, loader, criterion, device) -> Dict[str, float]:
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    preds, targs = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss_sum += criterion(out, yb).item() * yb.size(0)
            p = out.argmax(1)
            correct += (p == yb).sum().item()
            total += yb.size(0)
            preds.extend(p.cpu().numpy())
            targs.extend(yb.cpu().numpy())

    metrics = {
        "eval_loss": loss_sum / total if total > 0 else 0.0,
        "accuracy": correct / total if total > 0 else 0.0,
    }
    for avg in ("micro", "macro", "weighted"):
        prec, rec, f1, _ = precision_recall_fscore_support(
            targs, preds, average=avg, zero_division=0)
        metrics[f"{avg}_precision"] = float(prec)
        metrics[f"{avg}_recall"] = float(rec)
        metrics[f"{avg}_f1"] = float(f1)
    return metrics


def log_and_save_metrics(global_round: int, task_id: int, task_round: int,
                         eval_metrics: Dict[str, float], train_metrics: Dict[str, float]):
    merged = {
        "train_loss": train_metrics.get("train_loss", float("nan")),
        "train_accuracy": train_metrics.get("train_accuracy", float("nan")),
        **eval_metrics,
    }
    logger.info(
        f"[Task {task_id} Round {task_round} | Global {global_round}] "
        f"train_loss={merged['train_loss']:.4f} eval_loss={merged['eval_loss']:.4f} "
        f"acc={merged['accuracy']:.4f} | "
        f"micro P/R/F1={merged['micro_precision']:.4f}/{merged['micro_recall']:.4f}/{merged['micro_f1']:.4f} | "
        f"macro P/R/F1={merged['macro_precision']:.4f}/{merged['macro_recall']:.4f}/{merged['macro_f1']:.4f} | "
        f"weighted P/R/F1={merged['weighted_precision']:.4f}/{merged['weighted_recall']:.4f}/{merged['weighted_f1']:.4f}"
    )
    append_csv_row(CSV_FILE, [
        global_round,
        task_id,
        task_round,
        *[round(float(merged[k]), 6) for k in METRIC_KEYS],
    ])


class IncrementalFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, template_model, local_epochs=1, start_round=0,
                 num_tasks=5, task_rounds=30, **kwargs):
        super().__init__(**kwargs)
        self.template_model = template_model
        self.local_epochs = local_epochs
        self.start_round = start_round
        self.num_tasks = num_tasks
        self.task_rounds = task_rounds
        self.latest_parameters: Optional[Parameters] = None
        self.latest_fit_metrics: Dict[str, float] = {}

    def task_for_round(self, global_round: int):
        task_id = min(((global_round - 1) // self.task_rounds) + 1, self.num_tasks)
        task_round = ((global_round - 1) % self.task_rounds) + 1
        return task_id, task_round

    def configure_fit(self, server_round, parameters, client_manager):
        global_round = self.start_round + server_round
        task_id, task_round = self.task_for_round(global_round)
        config = {
            "local_epochs": self.local_epochs,
            "server_round": global_round,
            "task_id": task_id,
            "task_round": task_round,
        }
        sample_size, min_num = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        return [(c, fl.common.FitIns(parameters, config)) for c in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        global_round = self.start_round + server_round
        task_id, task_round = self.task_for_round(global_round)
        config = {
            "server_round": global_round,
            "task_id": task_id,
            "task_round": task_round,
        }
        sample_size, min_num = self.num_evaluation_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num)
        return [(c, fl.common.EvaluateIns(parameters, config)) for c in clients]

    def aggregate_fit(self, server_round, results, failures):
        params, metrics = super().aggregate_fit(server_round, results, failures)
        global_round = self.start_round + server_round
        task_id, _ = self.task_for_round(global_round)

        active_results = [
            fit_res for _, fit_res in results
            if fit_res.num_examples > 0 and not fit_res.metrics.get("skipped", False)
        ]
        total_examples = sum(fit_res.num_examples for fit_res in active_results)
        if total_examples > 0:
            train_loss = sum(
                fit_res.metrics.get("train_loss", 0.0) * fit_res.num_examples
                for fit_res in active_results
            ) / total_examples
            train_acc = sum(
                fit_res.metrics.get("train_accuracy", 0.0) * fit_res.num_examples
                for fit_res in active_results
            ) / total_examples
        else:
            train_loss, train_acc = float("nan"), float("nan")

        self.latest_fit_metrics = {
            "task_id": task_id,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_acc),
            "participating_clients": len(results),
            "active_clients": len(active_results),
            "failed_clients": len(failures),
        }

        if params is not None:
            self.latest_parameters = params
            os.makedirs(CKPT_DIR, exist_ok=True)
            state = ndarrays_to_state_dict(
                self.template_model, fl.common.parameters_to_ndarrays(params))
            path = os.path.join(CKPT_DIR, f"round_{global_round:03d}.pth")
            torch.save({
                "round": global_round,
                "task_id": task_id,
                "model_state_dict": state,
            }, path)
            logger.info(f"[Round {global_round}] global checkpoint saved -> {path}")
        return params, metrics


def load_checkpoint(path: str, model) -> int:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        return int(ckpt.get("round", 0))
    model.load_state_dict(ckpt)
    return 0


def main():
    parser = argparse.ArgumentParser(description="CNN1D CAN incremental Flower server")
    parser.add_argument("--mode", choices=["train", "resume", "test"], default="train")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint bat buoc voi resume/test")
    parser.add_argument("--num-clients", type=int, default=10)
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--task-rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--address", type=str, default="0.0.0.0:8081")
    parser.add_argument("--test-file", type=str, default=DEFAULT_TEST)
    parser.add_argument("--test-max-samples", type=int, default=1_000_000,
                        help="So mau global test dung moi round (0 = dung het)")
    parser.add_argument("--test-batch-size", type=int, default=4096)
    args = parser.parse_args()

    if args.mode in ("resume", "test") and not args.checkpoint:
        parser.error(f"--mode {args.mode} can --checkpoint")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN1D_IDS(input_len=INPUT_LEN, num_classes=NUM_GLOBAL_CLASSES, dropout=0.15)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"CNN1D model: {n_params:,} trainable params | device: {device}")

    start_round = 0
    if args.mode in ("resume", "test"):
        start_round = load_checkpoint(args.checkpoint, model)
        logger.info(f"Loaded checkpoint '{args.checkpoint}' (round {start_round})")

    test_loader, y_test = load_global_test(
        args.test_file,
        args.test_max_samples,
        args.test_batch_size,
    )
    criterion = make_criterion(y_test, device)
    model.to(device)

    total_rounds = args.num_tasks * args.task_rounds

    if args.mode == "test":
        strategy_helper = IncrementalFedAvg(
            template_model=model,
            local_epochs=args.local_epochs,
            start_round=0,
            num_tasks=args.num_tasks,
            task_rounds=args.task_rounds,
        )
        task_id, task_round = strategy_helper.task_for_round(max(start_round, 1))
        eval_metrics = evaluate_on_global_test(model, test_loader, criterion, device)
        log_and_save_metrics(start_round, task_id, task_round, eval_metrics, {})
        return

    num_rounds = total_rounds - start_round
    if num_rounds <= 0:
        logger.error(f"Checkpoint da o round {start_round} >= total_rounds {total_rounds}.")
        return

    strategy = IncrementalFedAvg(
        template_model=model,
        local_epochs=args.local_epochs,
        start_round=start_round,
        num_tasks=args.num_tasks,
        task_rounds=args.task_rounds,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=args.num_clients,
        min_evaluate_clients=args.num_clients,
        min_available_clients=args.num_clients,
        initial_parameters=fl.common.ndarrays_to_parameters(get_model_parameters(model)),
    )

    def evaluate_fn(server_round: int, parameters, config):
        global_round = start_round + server_round
        if global_round <= 0:
            return None
        model.load_state_dict(ndarrays_to_state_dict(model, parameters))
        model.to(device)
        task_id, task_round = strategy.task_for_round(global_round)
        eval_metrics = evaluate_on_global_test(model, test_loader, criterion, device)
        log_and_save_metrics(
            global_round,
            task_id,
            task_round,
            eval_metrics,
            strategy.latest_fit_metrics,
        )
        return eval_metrics["eval_loss"], eval_metrics

    strategy.evaluate_fn = evaluate_fn

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    if strategy.latest_parameters is not None:
        ndarrays = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
        model.load_state_dict(ndarrays_to_state_dict(model, ndarrays))
        torch.save(model.state_dict(), "cnn1d_can_il_global.pth")
        logger.info("Saved final global model -> cnn1d_can_il_global.pth")


if __name__ == "__main__":
    main()
