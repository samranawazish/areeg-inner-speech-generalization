"""Train EEG-Conformer with optional session-level CORAL alignment.

The participant's sessions are split chronologically by splits.py. Model
selection uses validation macro-F1; the test sessions are evaluated only once,
after the best validation checkpoint has been restored.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path
import time
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.eeg_conformer import create_eeg_conformer
from models.eegnet import create_eegnet
from models.deepconvnet import DeepConvNet
from splits import create_baseline_split, print_split_summary


def set_seed(seed: int) -> None:
    """Make a run as reproducible as reasonably possible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    session_labels: np.ndarray | None = None,
) -> DataLoader:
    """Convert NumPy EEG arrays into a PyTorch DataLoader."""
    tensors = [
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    ]

    if session_labels is not None:
        if len(session_labels) != len(y):
            raise ValueError(
                "Session labels do not align with EEG labels."
            )
        tensors.append(
            torch.from_numpy(session_labels).long()
        )

    dataset = TensorDataset(*tensors)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # safest setting for Windows
        pin_memory=False,
    )


def feature_covariance(features: torch.Tensor) -> torch.Tensor:
    """Calculate a differentiable feature covariance matrix."""
    if features.ndim != 2:
        features = features.flatten(start_dim=1)

    if features.size(0) < 2:
        raise ValueError(
            "At least two feature vectors are required for covariance."
        )

    centered = features - features.mean(
        dim=0,
        keepdim=True
    )

    return (
        centered.transpose(0, 1) @ centered
    ) / (features.size(0) - 1)


def coral_loss_across_sessions(
    features: torch.Tensor,
    session_labels: torch.Tensor,
) -> torch.Tensor:
    """Average Deep CORAL loss over usable training-session pairs."""
    covariances = []

    for session_id in torch.unique(session_labels):
        session_features = features[
            session_labels == session_id
        ]

        if session_features.size(0) >= 2:
            covariances.append(
                feature_covariance(session_features)
            )

    if len(covariances) < 2:
        return features.sum() * 0.0

    feature_dimension = features.flatten(
        start_dim=1
    ).size(1)

    pair_losses = []

    for first_index in range(len(covariances)):
        for second_index in range(
            first_index + 1,
            len(covariances)
        ):
            covariance_difference = (
                covariances[first_index]
                - covariances[second_index]
            )

            pair_losses.append(
                covariance_difference.pow(2).sum()
                / (4.0 * feature_dimension ** 2)
            )

    return torch.stack(pair_losses).mean()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    coral_weight: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Run one training or evaluation epoch."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_classification_loss = 0.0
    total_coral_loss = 0.0
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            inputs = batch[0]
            targets = batch[1]
            session_labels = (
                batch[2]
                if len(batch) == 3
                else None
            )

            inputs = inputs.to(device)
            targets = targets.to(device)

            if session_labels is not None:
                session_labels = session_labels.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            use_coral = (
                training
                and coral_weight > 0.0
            )

            if use_coral:
                if session_labels is None:
                    raise ValueError(
                        "CORAL training requires session labels."
                    )

                logits, features = model(
                    inputs,
                    return_features=True
                )

                batch_coral_loss = (
                    coral_loss_across_sessions(
                        features,
                        session_labels
                    )
                )
            else:
                logits = model(inputs)
                batch_coral_loss = logits.sum() * 0.0

            classification_loss = criterion(
                logits,
                targets
            )

            loss = (
                classification_loss
                + coral_weight * batch_coral_loss
            )

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            total_classification_loss += (
                classification_loss.item()
                * targets.size(0)
            )
            total_coral_loss += (
                batch_coral_loss.item()
                * targets.size(0)
            )
            all_targets.append(targets.detach().cpu().numpy())
            all_predictions.append(logits.argmax(dim=1).detach().cpu().numpy())

    targets_array = np.concatenate(all_targets)
    predictions_array = np.concatenate(all_predictions)
    mean_loss = total_loss / len(loader.dataset)
    mean_classification_loss = (
        total_classification_loss
        / len(loader.dataset)
    )
    mean_coral_loss = (
        total_coral_loss
        / len(loader.dataset)
    )

    return (
        mean_loss,
        targets_array,
        predictions_array,
        mean_classification_loss,
        mean_coral_loss,
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def save_history(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


SUBJECT_IDS = [f"sub{number}" for number in range(12)]


def train_subject(
    args: argparse.Namespace,
    subject_id: str,
) -> dict[str, int | float | str]:
    """Train and evaluate one independent subject-specific model."""
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir) / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    split = create_baseline_split(
        subject_id=subject_id,
        normalization=args.normalization,
        validation_only=args.validation_only,
    )
    print_split_summary(split, subject_id)

    if args.coral_weight < 0:
        raise ValueError(
            "coral-weight must be non-negative."
        )

    effective_coral_weight = (
        args.coral_weight
        if args.use_coral
        else 0.0
    )

    if args.model == "eegnet" and args.use_coral:
        raise ValueError(
            "CORAL is disabled for the initial EEGNet baseline. "
            "Run EEGNet without --use-coral."
        )

    train_loader = make_loader(
        split["X_train"],
        split["y_train"],
        args.batch_size,
        shuffle=True,
        session_labels=(
            split["train_session_labels"]
            if args.use_coral
            else None
        ),
    )
    validation_loader = make_loader(
        split["X_validation"],
        split["y_validation"],
        args.batch_size,
        shuffle=False,
    )
    test_loader = None
    if not args.validation_only:
        test_loader = make_loader(
            split["X_test"], split["y_test"], args.batch_size, shuffle=False
        )
    if args.model == "eeg-conformer":
        model = create_eeg_conformer(
            n_filters_time=args.filters,
            att_depth=args.attention_depth,
            att_heads=args.attention_heads,
            drop_prob=args.dropout,
        ).to(device)

    elif args.model == "eegnet":
        model = create_eegnet(
            drop_prob=args.dropout,
        ).to(device)

    elif args.model == "deepconvnet":
        input_shape = split["X_train"].shape
        print("DeepConvNet input shape:", input_shape)

        if len(input_shape) == 4:
            n_channels = input_shape[2]
            n_time_samples = input_shape[3]
        elif len(input_shape) == 3:
            n_channels = input_shape[1]
            n_time_samples = input_shape[2]
        else:
            raise ValueError(
                f"Unexpected EEG input shape: {input_shape}"
            )

        n_classes = len(np.unique(split["y_train"]))

        model = DeepConvNet(
            n_channels=n_channels,
            input_time_samples=n_time_samples,
            n_classes=n_classes,
            dropout=args.dropout,
        ).to(device)

    else:
        raise ValueError(
            f"Unknown model selected: {args.model}"
        )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    print(f"\nDevice: {device}")
    print(f"Model: {args.model}")
    print(f"CORAL enabled: {args.use_coral}")
    print(f"CORAL weight: {effective_coral_weight}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    best_validation_macro_f1 = -float("inf")
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    training_start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        (
            train_loss,
            train_true,
            train_pred,
            train_classification_loss,
            train_coral_loss,
        ) = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            coral_weight=effective_coral_weight,
        )
        (
            validation_loss,
            validation_true,
            validation_pred,
            _,
            _,
        ) = run_epoch(
            model, validation_loader, criterion, device
        )
        train_metrics = metrics(train_true, train_pred)
        validation_metrics = metrics(validation_true, validation_pred)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_classification_loss": train_classification_loss,
            "train_coral_loss": train_coral_loss,
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "validation_loss": validation_loss,
            "validation_accuracy": validation_metrics["accuracy"],
            "validation_balanced_accuracy": validation_metrics["balanced_accuracy"],
            "validation_macro_f1": validation_metrics["macro_f1"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f}, "
            f"class {train_classification_loss:.4f}, "
            f"CORAL {train_coral_loss:.6e}, "
            f"acc {train_metrics['accuracy']:.3f} | "
            f"val loss {validation_loss:.4f}, acc {validation_metrics['accuracy']:.3f}, "
            f"macro-F1 {validation_metrics['macro_f1']:.3f}"
        )

        current_validation_macro_f1 = validation_metrics["macro_f1"]
        macro_f1_improved = (
            current_validation_macro_f1
            > best_validation_macro_f1 + args.minimum_delta
        )
        same_macro_f1_but_lower_loss = (
            abs(current_validation_macro_f1 - best_validation_macro_f1)
            <= args.minimum_delta
            and validation_loss < best_validation_loss
        )

        if macro_f1_improved or same_macro_f1_but_lower_loss:
            best_validation_macro_f1 = current_validation_macro_f1
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping after epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")
    training_time_seconds = time.perf_counter() - training_start_time

    print(
        f"Training time: {training_time_seconds:.2f} seconds "
        f"({training_time_seconds / 60:.2f} minutes)"
    )
    model.load_state_dict(best_state)
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "subject_id": subject_id,
            "best_epoch": best_epoch,
            "best_validation_loss": best_validation_loss,
            "best_validation_macro_f1": best_validation_macro_f1,
            "label_names": split["label_names"].tolist(),
            "channel_mean": split["channel_mean"],
            "channel_std": split["channel_std"],
            "train_sessions": split["train_sessions"],
            "validation_sessions": split["validation_sessions"],
            "test_sessions": split["test_sessions"],
            "arguments": {
                **vars(args),
                "subject": subject_id,
            },
        },
        checkpoint_path,
    )
    save_history(history, output_dir / "training_history.csv")

    if args.validation_only:
        (
            validation_loss,
            validation_true,
            validation_pred,
            _,
            _,
        ) = run_epoch(
            model, validation_loader, criterion, device
        )
        validation_result = metrics(validation_true, validation_pred)

        result = {
            "subject_id": subject_id,
            "model": args.model,
            "training_time_seconds": float(training_time_seconds),
            "training_time_minutes": float(training_time_seconds / 60),
            "best_epoch": best_epoch,
            "validation_loss": float(validation_loss),
            "validation_accuracy": float(validation_result["accuracy"]),
            "validation_balanced_accuracy": float(
                validation_result["balanced_accuracy"]
            ),
            "validation_macro_f1": float(validation_result["macro_f1"]),
            "n_train": len(split["y_train"]),
            "n_validation": len(split["y_validation"]),
            "seed": args.seed,
        }

        (output_dir / "validation_metrics.json").write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print(f"\nBest epoch: {best_epoch}")
        print(f"Validation loss: {validation_loss:.4f}")
        print(f"Validation accuracy: {validation_result['accuracy']:.4f}")
        print(
            "Validation balanced accuracy: "
            f"{validation_result['balanced_accuracy']:.4f}"
        )
        print(f"Validation macro-F1: {validation_result['macro_f1']:.4f}")
        print("Test set: not evaluated")
        print(f"Results saved in: {output_dir}")
        return result

    # The test set is touched only here, after model selection is complete.
    (
        test_loss,
        test_true,
        test_pred,
        _,
        _,
    ) = run_epoch(
        model, test_loader, criterion, device
    )
    test_metrics = metrics(test_true, test_pred)
    label_names = [str(name) for name in split["label_names"]]
    labels = list(range(len(label_names)))
    matrix = confusion_matrix(test_true, test_pred, labels=labels)
    np.savetxt(output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d")

    report = classification_report(
        test_true,
        test_pred,
        labels=labels,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    result = {
        "subject_id": subject_id,
        "model": args.model,
        "best_epoch": best_epoch,
        "best_validation_loss": float(best_validation_loss),
        "best_validation_macro_f1": float(best_validation_macro_f1),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_balanced_accuracy": float(test_metrics["balanced_accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "n_train": len(split["y_train"]),
        "n_validation": len(split["y_validation"]),
        "n_test": len(split["y_test"]),
        "seed": args.seed,
    }

    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print(f"\nBest epoch: {best_epoch}")
    print(f"Best validation loss: {best_validation_loss:.4f}")
    print(f"Best validation macro-F1: {best_validation_macro_f1:.4f}")
    print("\nTest results")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(report)
    print(f"Results saved in: {output_dir}")

    return result


def save_subject_summary(
    results: list[dict[str, int | float | str]],
    output_dir: str,
) -> Path:
    """Save participant-level metrics and an aggregate mean/std row."""
    summary_path = Path(output_dir) / "all_subjects_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_metrics = {
        "subject_id",
        "model",
        "best_epoch",
        "n_train",
        "n_validation",
        "n_test",
        "seed",
    }
    numeric_metrics = [
        name
        for name in results[0]
        if name not in excluded_metrics
    ]
    rows = list(results)

    mean_row: dict[str, int | float | str] = {"subject_id": "MEAN"}
    std_row: dict[str, int | float | str] = {"subject_id": "STD"}
    for name in numeric_metrics:
        values = np.asarray([float(row[name]) for row in results])
        mean_row[name] = float(values.mean())
        std_row[name] = float(values.std(ddof=1)) if len(values) > 1 else 0.0

    rows.extend([mean_row, std_row])
    fieldnames = list(results[0].keys())
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return summary_path


def train_all_subjects(args: argparse.Namespace) -> None:
    """Train a separate model for every participant."""
    results = []

    for position, subject_id in enumerate(SUBJECT_IDS, start=1):
        print("\n" + "=" * 72)
        print(f"Subject-specific run {position}/{len(SUBJECT_IDS)}: {subject_id}")
        print("=" * 72)

        result = train_subject(args, subject_id)
        results.append(result)

        # Update after every participant so completed results survive interruption.
        summary_path = save_subject_summary(results, args.output_dir)
        print(f"Updated summary: {summary_path}")

    print("\nAll subject-specific runs completed.")
    print(f"Final summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EEG-Conformer or EEGNet on ArEEG."
    )
    parser.add_argument("--subject", default="sub0")
    parser.add_argument(
        "--model",
        choices=["eeg-conformer", "eegnet", "deepconvnet"],
        default="eeg-conformer",
        help="Neural-network architecture. Default: eeg-conformer.",
    )
    parser.add_argument(
        "--all-subjects",
        action="store_true",
        help="Train a separate subject-specific model for sub0 through sub11.",
    )
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--attention-depth", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument(
        "--normalization",
        choices=["train-channel", "per-trial", "none"],
        default="train-channel",
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Select and report on validation data without evaluating test data.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--use-coral",
        action="store_true",
        help="Align feature covariances across training sessions.",
    )
    parser.add_argument(
        "--coral-weight",
        type=float,
        default=0.1,
        help="Weight multiplying the source-session CORAL loss.",
    )
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()

    if arguments.all_subjects:
        train_all_subjects(arguments)
    else:
        train_subject(arguments, arguments.subject)
