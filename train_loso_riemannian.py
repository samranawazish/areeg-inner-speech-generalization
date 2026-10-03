"""Protocol 2 zero-shot LOSO Riemannian baseline for ArEEG.

Uses exactly the participant split and leakage audit defined in train_loso.py.
Logistic-regression C is selected on the validation participant. The unseen
test participant is evaluated once after selection.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
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

from train_loso import (
    available_subjects,
    create_loso_split,
    load_filtered_data,
    print_split_summary,
)


def create_pipeline(C: float, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("covariance", Covariances(estimator="oas")),
            ("tangent_space", TangentSpace(metric="riemann")),
            ("standardize", StandardScaler()),
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


def calculate_metrics(y_true, y_prediction) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_prediction)
        ),
        "macro_f1": float(
            f1_score(y_true, y_prediction, average="macro", zero_division=0)
        ),
    }


def evaluate(pipeline, X, y):
    probabilities = pipeline.predict_proba(X)
    classes = pipeline.named_steps["classifier"].classes_
    predictions = classes[probabilities.argmax(axis=1)]
    loss = float(log_loss(y, probabilities, labels=np.arange(5)))
    return loss, calculate_metrics(y, predictions), predictions


def save_rows(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def select_regularization(split, c_values, seed):
    best_pipeline = None
    best_c = None
    best_macro_f1 = -float("inf")
    best_loss = float("inf")
    rows = []

    for C in c_values:
        pipeline = create_pipeline(C, seed)
        pipeline.fit(split["X_train"], split["y_train"])
        validation_loss, validation_metrics, _ = evaluate(
            pipeline, split["X_validation"], split["y_validation"]
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
        tied_lower_loss = (
            abs(macro_f1 - best_macro_f1) <= 1e-12
            and validation_loss < best_loss
        )
        if improved or tied_lower_loss:
            best_pipeline = pipeline
            best_c = C
            best_macro_f1 = macro_f1
            best_loss = validation_loss

    if best_pipeline is None:
        raise RuntimeError("No Riemannian pipeline was selected.")
    return best_pipeline, best_c, rows


def train_fold(args, test_subject: str) -> dict:
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

    pipeline, best_c, search_rows = select_regularization(
        split, args.c_values, args.seed
    )
    save_rows(search_rows, output_dir / "validation_search.csv")

    validation_loss, validation_metrics, _ = evaluate(
        pipeline, split["X_validation"], split["y_validation"]
    )
    test_loss, test_metrics, test_predictions = evaluate(
        pipeline, split["X_test"], split["y_test"]
    )

    label_names = [str(name) for name in split["label_names"]]
    label_ids = list(range(len(label_names)))
    matrix = confusion_matrix(
        split["y_test"], test_predictions, labels=label_ids
    )
    np.savetxt(
        output_dir / "confusion_matrix.csv", matrix, delimiter=",", fmt="%d"
    )
    report = classification_report(
        split["y_test"],
        test_predictions,
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
            split["test_indices"], split["y_test"], test_predictions
        )
    ]
    save_rows(prediction_rows, output_dir / "test_predictions.csv")

    with (output_dir / "best_model.pkl").open("wb") as file:
        pickle.dump(
            {
                "pipeline": pipeline,
                "protocol": "zero_shot_loso",
                "model": "riemannian_tangent_logreg",
                "train_subjects": split["train_subjects"],
                "validation_subject": split["validation_subject"],
                "test_subject": test_subject,
                "selected_C": best_c,
                "normalization": args.normalization,
                "channel_mean": split["channel_mean"],
                "channel_std": split["channel_std"],
                "label_names": label_names,
            },
            file,
        )

    result = {
        "test_subject": test_subject,
        "validation_subject": split["validation_subject"],
        "model": "riemannian_tangent_logreg",
        "selected_C": float(best_c),
        "validation_loss": validation_loss,
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_balanced_accuracy": validation_metrics[
            "balanced_accuracy"
        ],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "test_loss": test_loss,
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

    print(f"\nSelected C: {best_c:g}")
    print(f"Validation macro-F1: {validation_metrics['macro_f1']:.4f}")
    print("\nZero-shot held-out-participant results")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {test_metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {test_metrics['macro_f1']:.4f}")
    print("\nConfusion matrix:")
    print(matrix)
    print("Results saved in:", output_dir)
    return result


def save_summary(results: list[dict], output_dir: str) -> Path:
    summary_path = Path(output_dir) / "all_loso_folds_summary.csv"
    excluded = {
        "test_subject",
        "validation_subject",
        "model",
        "selected_C",
        "n_train",
        "n_validation",
        "n_test",
        "seed",
    }
    metrics = [name for name in results[0] if name not in excluded]
    rows = list(results)
    mean_row = {
        "test_subject": "MEAN",
        "validation_subject": "",
        "model": "riemannian_tangent_logreg",
    }
    std_row = {
        "test_subject": "STD",
        "validation_subject": "",
        "model": "riemannian_tangent_logreg",
    }
    for name in metrics:
        values = np.asarray([float(row[name]) for row in results])
        mean_row[name] = float(values.mean())
        std_row[name] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    rows.extend([mean_row, std_row])
    save_rows(rows, summary_path)
    return summary_path


def train_all_folds(args) -> None:
    _, _, metadata, _ = load_filtered_data(
        data_path=args.data_path, metadata_path=args.metadata_path
    )
    subjects = available_subjects(metadata)
    results = []
    for fold_number, test_subject in enumerate(subjects, start=1):
        print("\n" + "=" * 72)
        print(f"Riemannian LOSO fold {fold_number}/{len(subjects)}: {test_subject}")
        print("=" * 72)
        results.append(train_fold(args, test_subject))
        summary_path = save_summary(results, args.output_dir)
        print("Updated summary:", summary_path)
    print("\nAll Riemannian LOSO folds completed.")
    print("Final summary:", summary_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Protocol 2 zero-shot Riemannian LOSO baseline."
    )
    parser.add_argument("--test-subject", default="sub0")
    parser.add_argument("--validation-subject", default=None)
    parser.add_argument("--all-folds", action="store_true")
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
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="protocol2_cross_subject/results/riemannian_zero_shot",
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
