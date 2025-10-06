"""Train and evaluate the Dynamic Graph Diffusion Neural Network."""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Data.dataset_gen import DatasetConfig, MyDataset
from Model.dgdnn import DGDNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(frozen=True)
class ExperimentConfig:
    """All hyper-parameters required for training the model."""

    root: Path
    destination: Path
    market: str
    tickers: Sequence[str]
    train_range: Tuple[str, str]
    val_range: Tuple[str, str]
    test_range: Tuple[str, str]
    window: int = 19
    fast_approximation: bool = False
    layers: int = 6
    expansion_step: int = 7
    num_heads: int = 2
    embedding_hidden: int = 1024
    embedding_output: int = 256
    raw_feature_size: int = 64
    classes: int = 2
    learning_rate: float = 2e-4
    weight_decay: float = 1.5e-5
    epochs: int = 6000


def read_ticker_file(path: Path) -> List[str]:
    """Load ticker symbols from a CSV file."""

    tickers: List[str] = []
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        tickers = [row[0] for row in reader if row]
    return tickers


def filter_missing(base: Iterable[str], missing: Iterable[str]) -> List[str]:
    """Remove missing tickers while preserving order."""

    missing_set = set(missing)
    return [ticker for ticker in base if ticker not in missing_set]


def build_dataset(config: ExperimentConfig, mode: str, date_range: Tuple[str, str]) -> MyDataset:
    dataset_config = DatasetConfig(
        root=str(config.root),
        dest=str(config.destination),
        market=config.market,
        tickers=list(config.tickers),
        start=date_range[0],
        end=date_range[1],
        window=config.window,
        mode=mode,
        fast_approx=config.fast_approximation,
    )
    return MyDataset(dataset_config)


def build_model(config: ExperimentConfig, num_nodes: int, timestamp: int) -> DGDNN:
    diffusion_size = [timestamp * 5, 64, 128, 256, 256, 256, 128]
    embedding_size = [64 + 64, 128 + 256, 256 + 256, 256 + 256, 256 + 256, 128 + 256]

    emb_output = config.embedding_output
    raw_feature_size = config.raw_feature_size
    if config.num_heads != 2:
        scale = config.num_heads / 2.0
        emb_output = int(round(emb_output * scale))
        raw_feature_size = int(round(raw_feature_size * scale))
        diffusion_size = [diffusion_size[0]] + [int(round(x * scale)) for x in diffusion_size[1:]]
        embedding_size = [int(round(x * scale)) for x in embedding_size]

    model = DGDNN(
        diffusion_size,
        embedding_size,
        config.embedding_hidden,
        emb_output,
        raw_feature_size,
        config.classes,
        config.layers,
        num_nodes,
        config.expansion_step,
        config.num_heads,
        active=[True] * config.layers,
    )
    return model.to(device)


def train_epoch(model: DGDNN, dataset: MyDataset, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for sample in dataset:
        features = sample["X"].to(device)
        adjacency = sample["A"].to(device)
        target = sample["Y"].to(device).long()

        optimizer.zero_grad()
        logits = model(features, adjacency)
        loss = F.cross_entropy(logits, target)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        predictions = logits.argmax(dim=1)
        correct += int((predictions == target).sum())
        total += target.size(0)

    accuracy = correct / total if total else 0.0
    return total_loss, accuracy


def evaluate(model: DGDNN, dataset: MyDataset) -> Tuple[float, float, float]:
    model.eval()
    predictions: List[int] = []
    targets: List[int] = []

    with torch.no_grad():
        for sample in dataset:
            features = sample["X"].to(device)
            adjacency = sample["A"].to(device)
            target = sample["Y"].to(device).long()

            logits = model(features, adjacency)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(target.cpu().tolist())

    accuracy = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average="macro")
    mcc = matthews_corrcoef(targets, predictions)
    return accuracy, f1, mcc


def main() -> None:
    """Entry point for training and evaluation."""

    # The placeholders below mirror the original script paths.  Replace with the
    # paths on your system as necessary.
    nasdaq_path = Path("/content/drive/MyDrive/.../NASDAQ.csv")
    nyse_path = Path("/content/drive/MyDrive/.../NYSE.csv")
    nyse_missing_path = Path("/content/drive/MyDrive/.../NYSE_missing.csv")
    sse_path = Path("/content/drive/MyDrive/.../SSE-130-tickers.csv")

    nasdaq_tickers = read_ticker_file(nasdaq_path)
    nyse_tickers = read_ticker_file(nyse_path)
    nyse_missing = read_ticker_file(nyse_missing_path)
    sse_tickers = read_ticker_file(sse_path)

    nyse_filtered = filter_missing(nyse_tickers, nyse_missing)
    _ = (nyse_filtered, sse_tickers)  # Other markets can be incorporated if desired.

    experiment = ExperimentConfig(
        root=Path("/content/.../google_finance"),
        destination=Path("/content/.../data"),
        market="NASDAQ",
        tickers=nasdaq_tickers,
        train_range=("2013-01-01", "2014-12-31"),
        val_range=("2015-01-01", "2015-06-30"),
        test_range=("2015-07-01", "2017-12-31"),
    )

    train_dataset = build_dataset(experiment, "train", experiment.train_range)
    val_dataset = build_dataset(experiment, "validation", experiment.val_range)
    test_dataset = build_dataset(experiment, "test", experiment.test_range)

    timestamp = experiment.window
    model = build_model(experiment, num_nodes=len(experiment.tickers), timestamp=timestamp)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=experiment.learning_rate, weight_decay=experiment.weight_decay
    )

    for epoch in range(1, experiment.epochs + 1):
        loss, acc = train_epoch(model, train_dataset, optimizer)
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d}: loss={loss:.4f}, acc={acc:.4f}")

    val_metrics = evaluate(model, val_dataset)
    test_metrics = evaluate(model, test_dataset)

    print(
        "Validation -- Acc: {:.4f}, F1: {:.4f}, MCC: {:.4f}".format(*val_metrics)
    )
    print("Test -- Acc: {:.4f}, F1: {:.4f}, MCC: {:.4f}".format(*test_metrics))


if __name__ == "__main__":
    main()
