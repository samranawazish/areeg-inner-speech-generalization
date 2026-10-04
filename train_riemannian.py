"""Leakage-controlled Riemannian baseline for ArEEG.

Each participant is modeled independently. Earlier sessions form the training
set, the following sessions select logistic-regression C, and the final
sessions are evaluated only after model selection. The shared chronological
split and normalization are provided by splits.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from splits import create_baseline_split, print_split_summary


SUBJECT_IDS = [f"sub{number}" for number in range(12)]


def create_pipeline(C: float, seed: int) -> Pipeline:
    """Construct the covariance, tangent-space and classifier pipeline."""
    return Pipeline(
        steps=[
            (
                "covariance",
                Covariances(estimator="oas"),
            ),
            (
                "tangent_space",
                TangentSpace(metric="riemann"),
            ),
            (
                "standardize",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calculate the metrics used by the neural baselines."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def evaluate(
    pipeline: Pipeline,
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[float, dict[str, float], np.ndarray]:
    """Evaluate a fitted pipeline without changing it."""
    probabilities = pipeline.predict_proba(X)
    classes = pipeline.named_steps["classifier"].classes_
    predictions = classes[probabilities.argmax(axis=1)]
    loss = log_loss(
        y,
        probabilities,
        labels=np.arange(5),
    )
    return loss, calculate_metrics(y, predictions), predictions


def select_regularization(
    split: dict,
    c_values: list[float],
    seed: int,
) -> tuple[Pipeline, float, list[dict[str, float]]]:
    """Select C using validation macro-F1 and loss as the tie-breaker."""
    best_pipeline = None
    best_c = None
    best_macro_f1 = -float("inf")
    best_loss = float("inf")
    rows = []

    for C in c_values:
        pipeline = create_pipeline(C=C, seed=seed)
        pipeline.fit(split["X_train"], split["y_train"])

        validation_loss, validation_metrics, _ = evaluate(
            pipeline,
            split["X_validation"],
            split["y_validation"],
        )

        rows.append(
            {
                "C": C,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_metrics["accuracy"],
                "validation_balanced_accuracy": validation_metrics[
                    "balanced_accuracy"
                ],
                "validation_macro_f1": validation_metrics["macro_f1"],
            }
        )

        print(
            f"C={C:g} | val loss {validation_loss:.4f}, "
            f"acc {validation_metrics['accuracy']:.3f}, "
            f"macro-F1 {validation_metrics['macro_f1']:.3f}"
        )

        macro_f1 = validation_metrics["macro_f1"]
        improved = macro_f1 > best_macro_f1 + 1e-12
        tied_but_lower_loss = (
            abs(macro_f1 - best_macro_f1) <= 1e-12
            and validation_loss < best_loss
        )

        if improved or tied_but_lower_loss:
            best_pipeline = pipeline
            best_c = C
            best_macro_f1 = macro_f1
            best_loss = validation_loss

    if best_pipeline is None or best_c is None:
        raise RuntimeError("No valid Riemannian model was selected.")

    return best_pipeline, best_c, rows


def save_rows(rows: list[dict], path: Path) -> None:
    """Save a list of dictionaries as CSV."""
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def train_subject(
    args: argparse.Namespace,
    subject_id: str,
) -> dict[str, int | float | str]:
    """Fit and evaluate one subject-specific Riemannian model."""
    output_dir = Path(args.output_dir) / subject_id
    output_dir.mkdir(parents=True, exist_ok=True)

    split = create_baseline_split(
        subject_id=subject_id,
        normalization=args.normalization,
        validation_only=args.validation_only,
    )
    print_split_summary(split, subject_id)

    pipeline, best_c, selection_rows = select_regularization(
        split=split,
        c_values=args.c_values,
        seed=args.seed,
    )
    save_rows(selection_rows, output_dir / "validation_search.csv")

    validation_loss, validation_metrics, _ = evaluate(
        pipeline,
        split["X_validation"],
        split["y_validation"],
    )

    print(f"\nSelected C: {best_c:g}")
    print(f"Validation loss: {validation_loss:.4f}")
    print(f"Validation accuracy: {validation_metrics['accuracy']:.4f}")
    print(
        "Validation balanced accuracy: "
        f"{validation_metrics['balanced_accuracy']:.4f}"
    )
    print(f"Validation macro-F1: {validation_metrics['macro_f1']:.4f}")

    common_result = {
        "subject_id": subject_id,
        "model": "riemannian_tangent_logreg",
        "selected_C": best_c,
        "validation_loss": float(validation_loss),
        "validation_accuracy": float(validation_metrics["accuracy"]),
        "validation_balanced_accuracy": float(
            validation_metrics["balanced_accuracy"]
        ),
        "validation_macro_f1": float(validation_metrics["macro_f1"]),
        "n_train": len(split["y_train"]),
        "n_validation": len(split["y_validation"]),
        "seed": args.seed,
    }

    if args.validation_only:
        (output_dir / "validation_metrics.json").write_text(
            json.dumps(common_result, indent=2),
            encoding="utf-8",
        )
        print("Test set: not evaluated")
        print(f"Results saved in: {output_dir}")
        return common_result

    test_loss, test_metrics, test_predictions = evaluate(
        pipeline,
        split["X_test"],
        split["y_test"],
    )

    label_names = [str(name) for name in split["label_names"]]
    labels = list(range(len(label_names)))
    matrix = confusion_matrix(
        split["y_test"],
        test_predictions,
        labels=labels,
    )
    np.savetxt(
        output_dir / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
    )

    report = classification_report(
        split["y_test"],
        test_predictions,
        labels=labels,
        target_names=label_names,
        digits=4,
        zero_division=0,
    )
    (output_dir / "classification_report.txt").write_text(
        report,
        encoding="utf-8",
    )

    result = {
        **common_result,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_balanced_accuracy": float(
            test_metrics["balanced_accuracy"]
        ),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "n_test": len(split["y_test"]),
    }

    (output_dir / "test_metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

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
    """Save participant rows followed by aggregate mean and standard deviation."""
    summary_path = Path(output_dir) / "all_subjects_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    excluded = {
        "subject_id",
        "model",
        "n_train",
        "n_validation",
        "n_test",
        "seed",
    }
    numeric_metrics = [name for name in results[0] if name not in excluded]
    rows = list(results)
    mean_row: dict[str, int | float | str] = {"subject_id": "MEAN"}
    std_row: dict[str, int | float | str] = {"subject_id": "STD"}

    for name in numeric_metrics:
        values = np.asarray([float(row[name]) for row in results])
        mean_row[name] = float(values.mean())
        std_row[name] = (
            float(values.std(ddof=1)) if len(values) > 1 else 0.0
        )

    rows.extend([mean_row, std_row])
    save_rows(rows, summary_path)
    return summary_path


def train_all_subjects(args: argparse.Namespace) -> None:
    """Run the same independent baseline for all 12 participants."""
    results = []

    for position, subject_id in enumerate(SUBJECT_IDS, start=1):
        print("\n" + "=" * 72)
        print(f"Subject-specific run {position}/{len(SUBJECT_IDS)}: {subject_id}")
        print("=" * 72)
        results.append(train_subject(args, subject_id))
        summary_path = save_subject_summary(results, args.output_dir)
        print(f"Updated summary: {summary_path}")

    print("\nAll subject-specific runs completed.")
    print(f"Final summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Riemannian tangent-space baseline on ArEEG."
    )
    parser.add_argument("--subject", default="sub0")
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help="Tune and report validation results without loading test data.",
    )
    parser.add_argument(
        "--normalization",
        choices=["train-channel", "per-trial", "none"],
        default="per-trial",
    )
    parser.add_argument(
        "--c-values",
        type=float,
        nargs="+",
        default=[0.01, 0.1, 1.0, 10.0, 100.0],
        help="Logistic-regression C values selected using validation macro-F1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="riemannian_results")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.all_subjects:
        train_all_subjects(arguments)
    else:
        train_subject(arguments, arguments.subject)
