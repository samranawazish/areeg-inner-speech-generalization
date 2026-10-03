"""Protocol 2: zero-shot LOSO evaluation using EEGNet on ArEEG.

This file reuses the audited LOSO split from train_loso.py but trains EEGNet.

"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn

from models.eegnet import create_eegnet
from train_loso import (
    available_subjects,
    calculate_metrics,
    create_loso_split,
    load_filtered_data,
    make_loader,
    print_split_summary,
    run_epoch,
    set_seed,
)


def save_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train_fold(args: argparse.Namespace, test_subject: str) -> dict:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / test_subject
    output_dir.mkdir(parents=True, exist_ok=True)

    split = create_loso_split(
        test_subject=test_subject,
        validation_subject=args.validation_subject,
        normalization=args.normalization,
        data_path=args.data_path,
        metadata_path=args.metadata_path,
    )
    print_split_summary(split)

    train_loader = make_loader(
        split["X_train"], split["y_train"], args.batch_size, True
    )
    validation_loader = make_loader(
        split["X_validation"], split["y_validation"], args.batch_size, False
    )
    test_loader = make_loader(
        split["X_test"], split["y_test"], args.batch_size, False
    )

    model = create_eegnet(
        n_channels=8,
        n_classes=5,
        n_times=1200,
        drop_prob=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Device:", device)
    print("Model: eegnet")
    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    best_macro_f1 = -float("inf")
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    waiting = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_true, train_prediction = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        validation_loss, validation_true, validation_prediction = run_epoch(
            model, validation_loader, criterion, device
        )
        train_metrics = calculate_metrics(train_true, train_prediction)
        validation_metrics = calculate_metrics(
            validation_true, validation_prediction
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, acc {train_metrics['accuracy']:.3f} | "
            f"val loss {validation_loss:.4f}, "
            f"acc {validation_metrics['accuracy']:.3f}, "
            f"macro-F1 {validation_metrics['macro_f1']:.3f}"
        )

        current_f1 = validation_metrics["macro_f1"]
        improved = current_f1 > best_macro_f1 + args.minimum_delta
        tied_lower_loss = (
            abs(current_f1 - best_macro_f1) <= args.minimum_delta
            and validation_loss < best_validation_loss
        )
        if improved or tied_lower_loss:
            best_macro_f1 = current_f1
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1

        if waiting >= args.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")

    model.load_state_dict(best_state)
    save_rows(history, output_dir / "training_history.csv")
    test_loss, test_true, test_prediction = run_epoch(
        model, test_loader, criterion, device
    )
    test_metrics = calculate_metrics(test_true, test_prediction)
    label_names = [str(name) for name in split["label_names"]]
    label_ids = list(range(len(label_names)))
    matrix = confusion_matrix(test_true, test_prediction, labels=label_ids)
    np.savetxt(
        output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d"
    )
    report = classification_report(
        test_true,
        test_prediction,
        labels=label_ids,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "protocol": "zero_shot_loso",
            "model": "eegnet",
            "train_subjects": split["train_subjects"],
            "validation_subject": split["validation_subject"],
            "test_subject": test_subject,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_validation_macro_f1": best_macro_f1,
            "normalization": args.normalization,
            "channel_mean": split["channel_mean"],
            "channel_std": split["channel_std"],
            "label_names": label_names,
            "arguments": vars(args),
        },
        output_dir / "best_model.pt",
    )

    result = {
        "test_subject": test_subject,
        "validation_subject": split["validation_subject"],
        "model": "eegnet",
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_validation_loss),
        "best_validation_macro_f1": float(best_macro_f1),
        "test_loss": float(test_loss),
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "n_train": len(split["y_train"]),
        "n_validation": len(split["y_validation"]),
        "n_test": len(split["y_test"]),
        "seed": args.seed,
    }
    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation macro-F1: {best_macro_f1:.4f}")
    print("\nZero-shot EEGNet test results")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print("Results saved in:", output_dir)
    return result


def save_summary(results: list[dict], output_dir: str) -> Path:
    path = Path(output_dir) / "all_loso_folds_summary.csv"
    excluded = {
        "test_subject", "validation_subject", "model", "best_epoch",
        "n_train", "n_validation", "n_test", "seed",
    }
    metrics = [name for name in results[0] if name not in excluded]
    rows = list(results)
    mean_row = {
        "test_subject": "MEAN", "validation_subject": "", "model": "eegnet"
    }
    std_row = {
        "test_subject": "STD", "validation_subject": "", "model": "eegnet"
    }
    for name in metrics:
        values = np.asarray([float(row[name]) for row in results])
        mean_row[name] = float(values.mean())
        std_row[name] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    rows.extend([mean_row, std_row])
    save_rows(rows, path)
    return path


def train_all_folds(args: argparse.Namespace) -> None:
    _, _, metadata, _ = load_filtered_data(
        data_path=args.data_path, metadata_path=args.metadata_path
    )
    subjects = available_subjects(metadata)
    results = []
    for number, test_subject in enumerate(subjects, start=1):
        print("\n" + "=" * 72)
        print(f"EEGNet LOSO fold {number}/{len(subjects)}: test {test_subject}")
        print("=" * 72)
        results.append(train_fold(args, test_subject))
        summary_path = save_summary(results, args.output_dir)
        print("Updated summary:", summary_path)
    print("\nAll EEGNet LOSO folds completed.")
    print("Final summary:", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protocol 2 zero-shot LOSO using EEGNet."
    )
    parser.add_argument("--test-subject", default="sub0")
    parser.add_argument("--validation-subject", default=None)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument(
        "--normalization",
        choices=["train-channel", "per-trial", "none"],
        default="per-trial",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="protocol2_cross_subject/results/eegnet_zero_shot_seed42",
    )
    parser.add_argument("--data-path", default="areeg_filtered.npz")
    parser.add_argument("--metadata-path", default="areeg_metadata.csv")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.all_folds:
        if arguments.validation_subject is not None:
            raise ValueError(
                "Do not set --validation-subject with --all-folds."
            )
        train_all_folds(arguments)
    else:
        train_fold(arguments, arguments.test_subject)
