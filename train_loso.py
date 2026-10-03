"""Protocol 2: zero-shot leave-one-subject-out (LOSO) decoding on ArEEG.

For every outer fold:
    - one participant is held out for final testing;
    - one different source participant is used for validation;
    - all remaining participants are used for training.

The held-out participant is never used for training, model selection, or
train-channel normalization. """

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import re
from pathlib import Path

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
from splits import (
    apply_channel_normalizer,
    apply_per_trial_channel_normalizer,
    fit_channel_normalizer,
    load_filtered_data,
)


def subject_number(subject_id: str) -> int:
    match = re.fullmatch(r"sub(\d+)", str(subject_id))
    if match is None:
        raise ValueError(f"Invalid participant identifier: {subject_id}")
    return int(match.group(1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def available_subjects(metadata) -> list[str]:
    subjects = metadata["subject_id"].astype(str).unique().tolist()
    return sorted(subjects, key=subject_number)


def choose_validation_subject(
    subjects: list[str],
    test_subject: str,
) -> str:
    """Select the next participant cyclically as the validation participant."""
    test_position = subjects.index(test_subject)
    return subjects[(test_position + 1) % len(subjects)]


def create_loso_split(
    test_subject: str,
    validation_subject: str | None = None,
    normalization: str = "per-trial",
    data_path: str = "areeg_filtered.npz",
    metadata_path: str = "areeg_metadata.csv",
) -> dict:
    """Create a leakage-controlled train/validation/test participant split."""
    X, y, metadata, label_names = load_filtered_data(
        data_path=data_path,
        metadata_path=metadata_path,
    )
    subjects = available_subjects(metadata)

    if test_subject not in subjects:
        raise ValueError(f"Test participant not found: {test_subject}")

    if validation_subject is None:
        validation_subject = choose_validation_subject(subjects, test_subject)

    if validation_subject not in subjects:
        raise ValueError(
            f"Validation participant not found: {validation_subject}"
        )
    if validation_subject == test_subject:
        raise ValueError("Validation and test participants must be different.")

    subject_values = metadata["subject_id"].astype(str).to_numpy()
    test_mask = subject_values == test_subject
    validation_mask = subject_values == validation_subject
    train_mask = ~(test_mask | validation_mask)

    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    test_indices = np.flatnonzero(test_mask)

    X_train, y_train = X[train_indices], y[train_indices]
    X_validation, y_validation = X[validation_indices], y[validation_indices]
    X_test, y_test = X[test_indices], y[test_indices]

    channel_mean = None
    channel_std = None

    if normalization == "train-channel":
        channel_mean, channel_std = fit_channel_normalizer(X_train)
        X_train = apply_channel_normalizer(X_train, channel_mean, channel_std)
        X_validation = apply_channel_normalizer(
            X_validation, channel_mean, channel_std
        )
        X_test = apply_channel_normalizer(X_test, channel_mean, channel_std)
    elif normalization == "per-trial":
        X_train = apply_per_trial_channel_normalizer(X_train)
        X_validation = apply_per_trial_channel_normalizer(X_validation)
        X_test = apply_per_trial_channel_normalizer(X_test)
    elif normalization in (None, "none"):
        pass
    else:
        raise ValueError(
            "normalization must be 'train-channel', 'per-trial', or 'none'."
        )

    train_subjects = sorted(
        set(subject_values[train_indices].tolist()), key=subject_number
    )

    # Explicit leakage audit.
    assert test_subject not in train_subjects
    assert validation_subject not in train_subjects
    assert not np.intersect1d(train_indices, validation_indices).size
    assert not np.intersect1d(train_indices, test_indices).size
    assert not np.intersect1d(validation_indices, test_indices).size
    assert len(train_indices) + len(validation_indices) + len(test_indices) == len(X)
    assert all(
        np.isfinite(array).all()
        for array in (X_train, X_validation, X_test)
    )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "train_subjects": train_subjects,
        "validation_subject": validation_subject,
        "test_subject": test_subject,
        "channel_mean": channel_mean,
        "channel_std": channel_std,
        "label_names": label_names,
    }


def print_split_summary(split: dict) -> None:
    print("\nProtocol 2: zero-shot LOSO")
    print("Training participants:", ", ".join(split["train_subjects"]))
    print("Validation participant:", split["validation_subject"])
    print("Held-out test participant:", split["test_subject"])
    print("\nArray shapes:")
    print("Train:", split["X_train"].shape, split["y_train"].shape)
    print(
        "Validation:",
        split["X_validation"].shape,
        split["y_validation"].shape,
    )
    print("Test:", split["X_test"].shape, split["y_test"].shape)
    print("Leakage audit: PASSED")


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).long(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    targets_all = []
    predictions_all = []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, targets)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * targets.size(0)
            targets_all.append(targets.detach().cpu().numpy())
            predictions_all.append(logits.argmax(dim=1).detach().cpu().numpy())

    return (
        total_loss / len(loader.dataset),
        np.concatenate(targets_all),
        np.concatenate(predictions_all),
    )


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def save_csv(rows: list[dict], path: Path) -> None:
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
    # Default size before optional 11-source refitting.
    refit_training_size = len(
        split["y_train"]
    )
    train_loader = make_loader(
        split["X_train"], split["y_train"], args.batch_size, True
    )
    validation_loader = make_loader(
        split["X_validation"],
        split["y_validation"],
        args.batch_size,
        False,
    )
    test_loader = make_loader(
        split["X_test"], split["y_test"], args.batch_size, False
    )

    model = create_eeg_conformer(
        n_filters_time=args.filters,
        att_depth=args.attention_depth,
        att_heads=args.attention_heads,
        drop_prob=args.dropout,
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    print("Device:", device)
    print("Model: EEG-Conformer")
    print(
        "Trainable parameters:",
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}",
    )

    best_macro_f1 = -float("inf")
    best_validation_loss = float("inf")
    best_epoch = 0
    best_state = None
    epochs_without_improvement = 0
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
        improved_f1 = current_f1 > best_macro_f1 + args.minimum_delta
        tied_f1_lower_loss = (
            abs(current_f1 - best_macro_f1) <= args.minimum_delta
            and validation_loss < best_validation_loss
        )

        if improved_f1 or tied_f1_lower_loss:
            best_macro_f1 = current_f1
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
        raise RuntimeError(
            "Training did not produce a checkpoint."
        )
    save_csv(
        history,
        output_dir / "training_history.csv"
    )

    refit_training_size = len(
        split["y_train"]
    )

    if args.refit_all_sources:

        print(
            "\nRefitting a fresh EEG-Conformer "
            "using all 11 source participants."
        )

        print(
            f"Selected number of epochs: "
            f"{best_epoch}"
        )

        X_refit = np.concatenate(
            [
                split["X_train"],
                split["X_validation"]
            ],
            axis=0
        )

        y_refit = np.concatenate(
            [
                split["y_train"],
                split["y_validation"]
            ],
            axis=0
        )

        refit_training_size = len(y_refit)

        if len(X_refit) != len(y_refit):
            raise ValueError(
                "Refit EEG and labels are not aligned."
            )

        if not np.isfinite(X_refit).all():
            raise ValueError(
                "Refit EEG contains NaN or infinity."
            )

        refit_loader = make_loader(
            X_refit,
            y_refit,
            args.batch_size,
            shuffle=True
        )

        # Reset the seed and initialize a completely fresh model.
        set_seed(args.seed)

        model = create_eeg_conformer(
            n_filters_time=args.filters,
            att_depth=args.attention_depth,
            att_heads=args.attention_heads,
            drop_prob=args.dropout,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )

        refit_history = []

        for refit_epoch in range(
            1,
            best_epoch + 1
        ):

            (
                refit_loss,
                refit_true,
                refit_prediction
            ) = run_epoch(
                model,
                refit_loader,
                criterion,
                device,
                optimizer
            )

            refit_metrics = calculate_metrics(
                refit_true,
                refit_prediction
            )
            refit_history.append({
                "epoch": refit_epoch,
                "train_loss": refit_loss,
                "train_accuracy": (
                    refit_metrics["accuracy"]
                ),
                "train_macro_f1": (
                    refit_metrics["macro_f1"]
                )
            })

            print(
                f"Refit epoch "
                f"{refit_epoch:03d}/{best_epoch} | "
                f"loss {refit_loss:.4f}, "
                f"accuracy "
                f"{refit_metrics['accuracy']:.3f}, "
                f"macro-F1 "
                f"{refit_metrics['macro_f1']:.3f}"
            )

        save_csv(
            refit_history,
            output_dir / "refit_history.csv"
        )

    else:

        # Use the original 10-source model.
        model.load_state_dict(
            best_state
        )

    # The held-out participant is evaluated only after model selection.
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

    prediction_rows = [
        {
            "dataset_index": int(index),
            "true_label_id": int(true_label),
            "true_label": label_names[int(true_label)],
            "predicted_label_id": int(predicted_label),
            "predicted_label": label_names[int(predicted_label)],
        }
        for index, true_label, predicted_label in zip(
            split["test_indices"], test_true, test_prediction
        )
    ]
    save_csv(prediction_rows, output_dir / "test_predictions.csv")

    checkpoint = {
        "model_state_dict": best_state,
        "model_state_dict": copy.deepcopy(
            model.state_dict()
        ),
        "refit_all_sources": args.refit_all_sources,
        "refit_training_size": refit_training_size,
        "protocol": "zero_shot_loso",
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
    }
    torch.save(checkpoint, output_dir / "best_model.pt")

    result = {
        "test_subject": test_subject,
        "refit_all_sources": args.refit_all_sources,
        "refit_training_size": refit_training_size,
        "validation_subject": split["validation_subject"],
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
    print("\nZero-shot held-out-participant results")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print("\nConfusion matrix:")
    print(matrix)
    print("\nClassification report:")
    print(report)
    print("Results saved in:", output_dir)
    return result


def save_summary(results: list[dict], output_dir: str) -> Path:
    summary_path = Path(output_dir) / "all_loso_folds_summary.csv"
    rows = list(results)
    excluded = {
        "test_subject",
        "validation_subject",
        "best_epoch",
        "n_train",
        "n_validation",
        "n_test",
        "seed",
        "refit_all_sources",
        "refit_training_size",
    }
    metrics = [name for name in results[0] if name not in excluded]
    mean_row = {"test_subject": "MEAN", "validation_subject": ""}
    std_row = {"test_subject": "STD", "validation_subject": ""}
    for name in metrics:
        values = np.asarray([float(row[name]) for row in results])
        mean_row[name] = float(values.mean())
        std_row[name] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    rows.extend([mean_row, std_row])
    save_csv(rows, summary_path)
    return summary_path


def train_all_folds(args: argparse.Namespace) -> None:
    _, _, metadata, _ = load_filtered_data(
        data_path=args.data_path,
        metadata_path=args.metadata_path,
    )
    subjects = available_subjects(metadata)
    results = []

    for fold_number, test_subject in enumerate(subjects, start=1):
        print("\n" + "=" * 72)
        print(f"LOSO fold {fold_number}/{len(subjects)}: test {test_subject}")
        print("=" * 72)
        result = train_fold(args, test_subject)
        results.append(result)
        summary_path = save_summary(results, args.output_dir)
        print("Updated summary:", summary_path)

    print("\nAll zero-shot LOSO folds completed.")
    print("Final summary:", summary_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protocol 2 zero-shot LOSO with EEG-Conformer."
    )
    parser.add_argument("--test-subject", default="sub0")
    parser.add_argument(
        "--validation-subject",
        default=None,
        help="Optional source validation participant; defaults to the next subject.",
    )
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
    parser.add_argument("--filters", type=int, default=32)
    parser.add_argument("--attention-depth", type=int, default=2)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--minimum-delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="protocol2_cross_subject/results/zero_shot_seed42",
    )
    parser.add_argument("--data-path", default="areeg_filtered.npz")
    parser.add_argument("--metadata-path", default="areeg_metadata.csv")
    parser.add_argument(
    "--refit-all-sources",
    action="store_true",
    help=(
        "After selecting the best epoch, initialize a fresh model "
        "and train it on all 11 source participants."
    ),
)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.all_folds:
        if arguments.validation_subject is not None:
            raise ValueError(
                "Do not set --validation-subject with --all-folds; "
                "each fold chooses a different source validation participant."
            )
        train_all_folds(arguments)
    else:
        train_fold(arguments, arguments.test_subject)