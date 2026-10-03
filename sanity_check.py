"""

This is a diagnostic test, not a scientific experiment. It uses only training
sessions and evaluates the model on the same 25 trials used for optimization.
"""

import argparse

import numpy as np
import torch
from torch import nn

from models.eeg_conformer import create_eeg_conformer
from splits import create_baseline_split
from train import make_loader, metrics, run_epoch, set_seed


def select_balanced_subset(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the first N training trials from every class."""
    selected_indices = []

    for class_id in sorted(np.unique(y)):
        class_indices = np.flatnonzero(y == class_id)
        if len(class_indices) < samples_per_class:
            raise ValueError(
                f"Class {class_id} has only {len(class_indices)} trials; "
                f"cannot select {samples_per_class}."
            )
        selected_indices.extend(class_indices[:samples_per_class].tolist())

    selected_indices = np.asarray(selected_indices, dtype=np.int64)
    return X[selected_indices], y[selected_indices]


def disable_dropout(model: nn.Module) -> None:
    """Set every dropout probability to zero for the memorization test."""
    for module in model.modules():
        if isinstance(
            module,
            (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d),
        ):
            module.p = 0.0


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    split = create_baseline_split(subject_id=args.subject)
    X_small, y_small = select_balanced_subset(
        split["X_train"],
        split["y_train"],
        samples_per_class=args.samples_per_class,
    )

    loader = make_loader(
        X_small,
        y_small,
        batch_size=len(y_small),
        shuffle=True,
    )

    model = create_eeg_conformer().to(device)
    disable_dropout(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.0,
    )

    print("Diagnostic: tiny-subset memorization")
    print("Subject:", args.subject)
    print("Device:", device)
    print("Subset shape:", X_small.shape)
    print("Class counts:", np.bincount(y_small))
    print("Dropout disabled: yes")

    final_accuracy = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, _, _ = run_epoch(
            model,
            loader,
            criterion,
            device,
            optimizer,
        )

        # Measure memorization in evaluation mode on the same training trials.
        evaluation_loss, targets, predictions = run_epoch(
            model,
            loader,
            criterion,
            device,
        )
        result = metrics(targets, predictions)
        final_accuracy = result["accuracy"]

        if epoch == 1 or epoch % 10 == 0 or final_accuracy >= args.target_accuracy:
            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"optimization loss {train_loss:.4f} | "
                f"evaluation loss {evaluation_loss:.4f} | "
                f"accuracy {final_accuracy:.3f}"
            )

        if final_accuracy >= args.target_accuracy:
            print("Sanity check passed: the model memorized the tiny subset.")
            return

    print(
        "Sanity check did not reach the target accuracy. "
        f"Final accuracy: {final_accuracy:.3f}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit EEG-Conformer on a tiny balanced training subset."
    )
    parser.add_argument("--subject", default="sub0")
    parser.add_argument("--samples-per-class", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--target-accuracy", type=float, default=0.96)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
