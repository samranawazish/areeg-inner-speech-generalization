"""Protocol 3 pilot: few-shot adaptation of population EEGNet on ArEEG.

For each target participant:
  * load the participant-specific zero-shot population checkpoint;
  * reserve the final chronological sessions for testing;
  * use the earliest 1 or 4 sessions for labelled calibration;
  * compare zero-shot, linear probing, and full fine-tuning.

Every adaptation condition starts from the same original population checkpoint.
The test sessions are evaluated only once after fixed-epoch adaptation.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
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

from models.eegnet import create_eegnet
from splits import apply_per_trial_channel_normalizer, get_subject_sessions
from train_loso import load_filtered_data, make_loader, run_epoch, set_seed


def metrics(y_true: np.ndarray, y_prediction: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_prediction)
        ),
        "macro_f1": float(
            f1_score(y_true, y_prediction, average="macro", zero_division=0)
        ),
    }


def save_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def load_target_data(
    X: np.ndarray,
    y: np.ndarray,
    metadata,
    target_subject: str,
    calibration_sessions: int,
    test_sessions: int,
) -> dict:
    """Create chronological calibration and fixed final-session test data."""
    sessions = get_subject_sessions(metadata, target_subject)
    if calibration_sessions < 1:
        raise ValueError("calibration_sessions must be at least 1.")
    if len(sessions) < calibration_sessions + test_sessions:
        raise ValueError(
            f"{target_subject} has {len(sessions)} sessions, but "
            f"{calibration_sessions + test_sessions} are required."
        )

    calibration_ids = sessions[:calibration_sessions]
    test_ids = sessions[-test_sessions:]
    if set(calibration_ids) & set(test_ids):
        raise RuntimeError("Calibration and test sessions overlap.")

    subject_values = metadata["subject_id"].astype(str).to_numpy()
    session_values = metadata["session_id"].astype(str).to_numpy()
    subject_mask = subject_values == target_subject
    calibration_indices = np.flatnonzero(
        subject_mask & np.isin(session_values, calibration_ids)
    )
    test_indices = np.flatnonzero(
        subject_mask & np.isin(session_values, test_ids)
    )
    if not len(calibration_indices) or not len(test_indices):
        raise RuntimeError("Calibration or test data are empty.")
    if np.intersect1d(calibration_indices, test_indices).size:
        raise RuntimeError("Trial-level leakage detected.")

    X_calibration = apply_per_trial_channel_normalizer(X[calibration_indices])
    X_test = apply_per_trial_channel_normalizer(X[test_indices])
    y_calibration = y[calibration_indices]
    y_test = y[test_indices]

    return {
        "X_calibration": X_calibration,
        "y_calibration": y_calibration,
        "X_test": X_test,
        "y_test": y_test,
        "calibration_indices": calibration_indices,
        "test_indices": test_indices,
        "calibration_sessions": calibration_ids,
        "test_sessions": test_ids,
    }


def load_population_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    arguments = checkpoint.get("arguments", {})
    dropout = float(arguments.get("dropout", 0.5))
    normalization = checkpoint.get(
        "normalization", arguments.get("normalization", "per-trial")
    )
    if normalization != "per-trial":
        raise ValueError(
            f"Expected a per-trial checkpoint, found {normalization!r}."
        )
    model = create_eegnet(
        n_channels=8,
        n_classes=5,
        n_times=1200,
        drop_prob=dropout,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint


def configure_adaptation(model: nn.Module, method: str) -> int:
    """Choose trainable parameters and return their number."""
    if method == "linear":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.final_layer.parameters():
            parameter.requires_grad = True
    elif method == "full":
        for parameter in model.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"Unknown adaptation method: {method}")
    count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if count == 0:
        raise RuntimeError("No parameters are trainable.")
    return count


def adaptation_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device,
    method: str,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Train one epoch; keep the frozen encoder in evaluation mode."""
    if method == "linear":
        model.eval()
        model.final_layer.train()
    else:
        model.train()

    total_loss = 0.0
    targets_all = []
    predictions_all = []
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
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


def evaluate_condition(
    model,
    X_test,
    y_test,
    batch_size,
    device,
) -> tuple[float, dict, np.ndarray, np.ndarray]:
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(X_test, y_test, batch_size, False)
    loss, true, prediction = run_epoch(model, loader, criterion, device)
    return loss, metrics(true, prediction), true, prediction


def save_evaluation(
    output_dir: Path,
    result: dict,
    y_true: np.ndarray,
    y_prediction: np.ndarray,
    test_indices: np.ndarray,
    label_names: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(label_names)))
    matrix = confusion_matrix(y_true, y_prediction, labels=labels)
    np.savetxt(
        output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d"
    )
    report = classification_report(
        y_true,
        y_prediction,
        labels=labels,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    rows = [
        {
            "dataset_index": int(index),
            "true_label_id": int(true),
            "true_label": label_names[int(true)],
            "predicted_label_id": int(prediction),
            "predicted_label": label_names[int(prediction)],
        }
        for index, true, prediction in zip(
            test_indices, y_true, y_prediction
        )
    ]
    save_rows(rows, output_dir / "test_predictions.csv")
    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )


def run_subject(args, target_subject, X, y, metadata, label_names) -> list[dict]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = (
        Path(args.checkpoint_root) / target_subject / "best_model.pt"
    )
    print("\n" + "=" * 72)
    print("Target participant:", target_subject)
    print("Checkpoint:", checkpoint_path)
    print("Device:", device)
    results = []

    # All conditions use the same fixed final test sessions.
    reference = load_target_data(
        X, y, metadata, target_subject,
        max(args.calibration_sessions), args.test_sessions,
    )
    print("Fixed test sessions:")
    for session in reference["test_sessions"]:
        print(" ", session)

    # Matched zero-shot baseline on the final two sessions.
    set_seed(args.seed)
    zero_model, checkpoint = load_population_model(checkpoint_path, device)
    zero_loss, zero_metrics, zero_true, zero_prediction = evaluate_condition(
        zero_model,
        reference["X_test"],
        reference["y_test"],
        args.batch_size,
        device,
    )
    zero_result = {
        "target_subject": target_subject,
        "method": "zero_shot",
        "calibration_sessions": 0,
        "calibration_trials": 0,
        "test_sessions": args.test_sessions,
        "test_trials": len(reference["y_test"]),
        "trainable_parameters": 0,
        "adaptation_epochs": 0,
        "test_loss": float(zero_loss),
        "test_accuracy": zero_metrics["accuracy"],
        "test_balanced_accuracy": zero_metrics["balanced_accuracy"],
        "test_macro_f1": zero_metrics["macro_f1"],
        "seed": args.seed,
    }
    save_evaluation(
        Path(args.output_dir) / target_subject / "zero_shot",
        zero_result, zero_true, zero_prediction,
        reference["test_indices"], label_names,
    )
    results.append(zero_result)
    print(
        f"Zero-shot | accuracy {zero_metrics['accuracy']:.3f}, "
        f"macro-F1 {zero_metrics['macro_f1']:.3f}"
    )

    for number_of_sessions in args.calibration_sessions:
        target = load_target_data(
            X, y, metadata, target_subject,
            number_of_sessions, args.test_sessions,
        )
        if not np.array_equal(target["test_indices"], reference["test_indices"]):
            raise RuntimeError("Test trials changed across calibration conditions.")
        print(f"\nCalibration sessions: {number_of_sessions}")
        print("Calibration trials:", len(target["y_calibration"]))
        print("Class counts:", Counter(target["y_calibration"].tolist()))

        for method in args.methods:
            set_seed(args.seed)
            model, _ = load_population_model(checkpoint_path, device)
            trainable_parameters = configure_adaptation(model, method)
            loader = make_loader(
                target["X_calibration"],
                target["y_calibration"],
                args.batch_size,
                True,
            )
            criterion = nn.CrossEntropyLoss()
            learning_rate = (
                args.linear_learning_rate
                if method == "linear"
                else args.full_learning_rate
            )
            optimizer = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=learning_rate,
                weight_decay=args.weight_decay,
            )
            history = []
            for epoch in range(1, args.adaptation_epochs + 1):
                loss, true, prediction = adaptation_epoch(
                    model, loader, criterion, optimizer, device, method
                )
                epoch_metrics = metrics(true, prediction)
                history.append(
                    {
                        "epoch": epoch,
                        "train_loss": loss,
                        "train_accuracy": epoch_metrics["accuracy"],
                        "train_macro_f1": epoch_metrics["macro_f1"],
                    }
                )

            test_loss, test_metrics, test_true, test_prediction = (
                evaluate_condition(
                    model,
                    target["X_test"],
                    target["y_test"],
                    args.batch_size,
                    device,
                )
            )
            condition_name = f"{method}_{number_of_sessions}session"
            if number_of_sessions != 1:
                condition_name += "s"
            condition_dir = (
                Path(args.output_dir) / target_subject / condition_name
            )
            condition_dir.mkdir(parents=True, exist_ok=True)
            save_rows(history, condition_dir / "adaptation_history.csv")
            torch.save(
                {
                    "model_state_dict": copy.deepcopy(model.state_dict()),
                    "target_subject": target_subject,
                    "method": method,
                    "calibration_sessions": target["calibration_sessions"],
                    "test_sessions": target["test_sessions"],
                    "source_checkpoint": str(checkpoint_path),
                    "arguments": vars(args),
                },
                condition_dir / "adapted_model.pt",
            )
            result = {
                "target_subject": target_subject,
                "method": method,
                "calibration_sessions": number_of_sessions,
                "calibration_trials": len(target["y_calibration"]),
                "test_sessions": args.test_sessions,
                "test_trials": len(target["y_test"]),
                "trainable_parameters": trainable_parameters,
                "adaptation_epochs": args.adaptation_epochs,
                "test_loss": float(test_loss),
                "test_accuracy": test_metrics["accuracy"],
                "test_balanced_accuracy": test_metrics["balanced_accuracy"],
                "test_macro_f1": test_metrics["macro_f1"],
                "seed": args.seed,
            }
            save_evaluation(
                condition_dir, result, test_true, test_prediction,
                target["test_indices"], label_names,
            )
            results.append(result)
            print(
                f"{method}, {number_of_sessions} session(s) | "
                f"trainable {trainable_parameters:,} | "
                f"accuracy {test_metrics['accuracy']:.3f}, "
                f"macro-F1 {test_metrics['macro_f1']:.3f}"
            )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Protocol 3 EEGNet few-shot adaptation pilot."
    )
    parser.add_argument("--subjects", nargs="+", default=["sub0", "sub3", "sub7"])
    parser.add_argument(
        "--methods", nargs="+", choices=["linear", "full"],
        default=["linear", "full"],
    )
    parser.add_argument(
        "--calibration-sessions", nargs="+", type=int, default=[1, 4]
    )
    parser.add_argument("--test-sessions", type=int, default=2)
    parser.add_argument("--adaptation-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--linear-learning-rate", type=float, default=1e-3)
    parser.add_argument("--full-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--checkpoint-root",
        default="protocol2_cross_subject/results/eegnet_zero_shot_seed42",
    )
    parser.add_argument(
        "--output-dir", default="protocol3_adaptation/pilot_seed42"
    )
    parser.add_argument("--data-path", default="areeg_filtered.npz")
    parser.add_argument("--metadata-path", default="areeg_metadata.csv")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if any(value < 1 for value in arguments.calibration_sessions):
        raise ValueError("Calibration-session values must be positive.")
    if arguments.test_sessions < 1:
        raise ValueError("test-sessions must be positive.")
    X_all, y_all, metadata_all, names = load_filtered_data(
        data_path=arguments.data_path,
        metadata_path=arguments.metadata_path,
    )
    label_names = [str(name) for name in names]
    all_results = []
    for subject in arguments.subjects:
        all_results.extend(
            run_subject(
                arguments, subject, X_all, y_all, metadata_all, label_names
            )
        )
    output_root = Path(arguments.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    save_rows(all_results, output_root / "pilot_summary.csv")
    print("\nProtocol 3 pilot completed.")
    print("Summary:", output_root / "pilot_summary.csv")
