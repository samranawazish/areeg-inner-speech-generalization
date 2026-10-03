"""Create final tables and figures for ArEEG Protocol 1.

Expected project folders:
    final_control_seed42, final_control_seed123, final_control_seed2026
    final_coral1_seed42, final_coral1_seed123, final_coral1_seed2026
    eegnet_final_seed42, eegnet_final_seed123, eegnet_final_seed2026
    riemannian_final

The script reads result files only. It writes all outputs to a separate
protocol1_analysis directory and never modifies training results.
"""

from __future__ import annotations

import argparse
import re
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


LABELS = ["Down", "Left", "Right", "Select", "Up"]
MODEL_ORDER = [
    "EEG-Conformer",
    "EEG-Conformer + CORAL",
    "EEGNet",
    "Riemannian",
]
MODEL_COLORS = {
    "EEG-Conformer": "#4C78A8",
    "EEG-Conformer + CORAL": "#F58518",
    "EEGNet": "#54A24B",
    "Riemannian": "#B279A2",
}
EXPERIMENT_PATTERNS = {
    "EEG-Conformer": "final_control_seed*",
    "EEG-Conformer + CORAL": "final_coral1_seed*",
    "EEGNet": "eegnet_final_seed*",
    "Riemannian": "riemannian_final",
}


def participant_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove aggregate rows and retain sub0 through sub11."""
    mask = frame["subject_id"].astype(str).str.fullmatch(r"sub\d+")
    result = frame.loc[mask].copy()
    result["subject_number"] = (
        result["subject_id"].str.extract(r"(\d+)").astype(int)
    )
    return result.sort_values("subject_number").reset_index(drop=True)


def extract_seed(path: Path, frame: pd.DataFrame) -> int:
    """Obtain a seed from the CSV, falling back to the directory name."""
    if "seed" in frame and frame["seed"].notna().any():
        return int(frame["seed"].dropna().iloc[0])
    match = re.search(r"seed[_-]?(\d+)", path.name)
    return int(match.group(1)) if match else 42


def find_summary_file(directory: Path) -> Path:
    """Find a final summary by its schema, regardless of its filename."""
    candidates = []

    for path in sorted(directory.glob("*.csv")):
        try:
            columns = set(pd.read_csv(path, nrows=0).columns)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError):
            continue

        if {"subject_id", "test_accuracy"}.issubset(columns):
            candidates.append(path)

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        raise FileNotFoundError(
            f"No final summary CSV containing subject_id and test_accuracy "
            f"was found directly inside {directory}."
        )

    raise ValueError(
        f"Several possible final summaries were found in {directory}: "
        + ", ".join(path.name for path in candidates)
    )


def discover_experiments(project_root: Path) -> dict[str, list[Path]]:
    """Find the final experiment directories without assuming seed order."""
    discovered: dict[str, list[Path]] = {}
    missing = []

    for model, pattern in EXPERIMENT_PATTERNS.items():
        paths = sorted(
            path
            for path in project_root.glob(pattern)
            if path.is_dir()
        )
        expected = 1 if model == "Riemannian" else 3
        if len(paths) != expected:
            missing.append(
                f"{model}: expected {expected} directories matching "
                f"'{pattern}', found {len(paths)}"
            )
        discovered[model] = paths

    if missing:
        raise FileNotFoundError(
            "Final result directories are incomplete:\n- "
            + "\n- ".join(missing)
        )

    for directories in discovered.values():
        for directory in directories:
            find_summary_file(directory)

    return discovered


def load_summaries(
    experiments: dict[str, list[Path]],
) -> pd.DataFrame:
    """Load participant-level summary rows from every final run."""
    frames = []
    for model, directories in experiments.items():
        for directory in directories:
            summary_path = find_summary_file(directory)
            frame = participant_rows(
                pd.read_csv(summary_path)
            )
            if len(frame) != 12:
                raise ValueError(
                    f"{directory} contains {len(frame)} participant rows; "
                    "expected 12."
                )
            frame["analysis_model"] = model
            frame["analysis_seed"] = extract_seed(directory, frame)
            frame["source_directory"] = directory.name
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_confusions(
    experiments: dict[str, list[Path]],
) -> list[dict]:
    """Load and validate every 5-by-5 test confusion matrix."""
    records = []
    for model, directories in experiments.items():
        for directory in directories:
            summary_path = find_summary_file(directory)
            summary = participant_rows(
                pd.read_csv(summary_path)
            )
            seed = extract_seed(directory, summary)
            for subject in summary["subject_id"]:
                path = directory / subject / "confusion_matrix.csv"
                if not path.exists():
                    raise FileNotFoundError(f"Missing confusion matrix: {path}")
                matrix = np.loadtxt(path, delimiter=",", dtype=int)
                if matrix.shape != (5, 5):
                    raise ValueError(
                        f"{path} has shape {matrix.shape}; expected (5, 5)."
                    )
                if (matrix < 0).any():
                    raise ValueError(f"Negative count found in {path}.")
                records.append(
                    {
                        "model": model,
                        "seed": seed,
                        "subject_id": subject,
                        "matrix": matrix,
                    }
                )
    return records


def validation_macro_column(frame: pd.DataFrame) -> pd.Series:
    """Support neural and classical summary naming conventions."""
    for column in [
        "best_validation_macro_f1",
        "validation_macro_f1",
    ]:
        if column in frame and frame[column].notna().any():
            values = frame[column]
            if values.isna().any():
                raise ValueError(
                    f"Validation metric column '{column}' is only "
                    "partially populated."
                )
            return values
    raise KeyError("No validation macro-F1 column was found.")


def make_performance_table(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize participant-seed runs for each model."""
    rows = []
    for model in MODEL_ORDER:
        group = data[data["analysis_model"] == model]
        validation_macro = validation_macro_column(group)
        row = {
            "model": model,
            "runs": len(group),
            "subjects": group["subject_id"].nunique(),
            "seeds": group["analysis_seed"].nunique(),
            "validation_macro_f1_mean": validation_macro.mean(),
            "validation_macro_f1_sd": validation_macro.std(ddof=1),
        }
        for metric in [
            "test_accuracy",
            "test_balanced_accuracy",
            "test_macro_f1",
            "test_loss",
        ]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std(ddof=1)
        row["macro_f1_generalization_gap"] = (
            group["test_macro_f1"].mean() - validation_macro.mean()
        )
        rows.append(row)
    return pd.DataFrame(rows)


def make_participant_table(data: pd.DataFrame) -> pd.DataFrame:
    """Average repeated neural seeds before comparing participants."""
    table = (
        data.groupby(["subject_id", "subject_number", "analysis_model"])[
            ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]
        ]
        .mean()
        .reset_index()
        .sort_values(["subject_number", "analysis_model"])
    )
    return table


def make_seed_table(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize variation across random seeds for neural models."""
    neural = data[data["analysis_model"] != "Riemannian"]
    return (
        neural.groupby(["analysis_model", "analysis_seed"])[
            ["test_accuracy", "test_balanced_accuracy", "test_macro_f1"]
        ]
        .mean()
        .reset_index()
        .rename(
            columns={
                "analysis_model": "model",
                "analysis_seed": "seed",
            }
        )
    )


def confusion_tables(
    records: list[dict],
    output_dir: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Create class-recall and row-normalized confusion tables."""
    recall_rows = []
    normalized = {}

    for model in MODEL_ORDER:
        matrices = [
            record["matrix"]
            for record in records
            if record["model"] == model
        ]
        aggregate = np.sum(matrices, axis=0)
        row_totals = aggregate.sum(axis=1, keepdims=True)
        normalized_matrix = np.divide(
            aggregate,
            row_totals,
            out=np.zeros_like(aggregate, dtype=float),
            where=row_totals > 0,
        )
        normalized[model] = normalized_matrix

        slug = (
            model.lower()
            .replace(" + ", "_")
            .replace("-", "_")
            .replace(" ", "_")
        )
        pd.DataFrame(
            aggregate,
            index=LABELS,
            columns=LABELS,
        ).to_csv(output_dir / f"confusion_counts_{slug}.csv")
        pd.DataFrame(
            normalized_matrix,
            index=LABELS,
            columns=LABELS,
        ).to_csv(output_dir / f"confusion_normalized_{slug}.csv")

        run_recalls = np.asarray(
            [
                np.divide(
                    np.diag(matrix),
                    matrix.sum(axis=1),
                    out=np.zeros(5, dtype=float),
                    where=matrix.sum(axis=1) > 0,
                )
                for matrix in matrices
            ]
        )
        for index, label in enumerate(LABELS):
            recall_rows.append(
                {
                    "model": model,
                    "command": label,
                    "recall": normalized_matrix[index, index],
                    "run_recall_mean": run_recalls[:, index].mean(),
                    "run_recall_sd": run_recalls[:, index].std(ddof=1),
                    "zero_recall_runs": int(
                        np.sum(run_recalls[:, index] == 0)
                    ),
                    "number_of_runs": len(matrices),
                }
            )

    return pd.DataFrame(recall_rows), normalized


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm-adjust a family of p-values without extra dependencies."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def make_statistical_tests(
    participant_table: pd.DataFrame,
) -> pd.DataFrame:
    """Run exploratory paired tests on participant-averaged outcomes."""
    wide = participant_table.pivot(
        index="subject_id",
        columns="analysis_model",
        values=["test_accuracy", "test_macro_f1"],
    )
    rows = []

    for model in MODEL_ORDER:
        differences = wide[("test_accuracy", model)] - 0.20
        result = wilcoxon(differences, alternative="greater")
        rows.append(
            {
                "family": "accuracy_vs_chance",
                "metric": "test_accuracy",
                "comparison": f"{model} > 0.20",
                "mean_difference": differences.mean(),
                "statistic": result.statistic,
                "p_value": result.pvalue,
            }
        )

    for metric in ["test_accuracy", "test_macro_f1"]:
        for first, second in combinations(MODEL_ORDER, 2):
            differences = wide[(metric, first)] - wide[(metric, second)]
            result = wilcoxon(differences, alternative="two-sided")
            rows.append(
                {
                    "family": f"pairwise_{metric}",
                    "metric": metric,
                    "comparison": f"{first} - {second}",
                    "mean_difference": differences.mean(),
                    "statistic": result.statistic,
                    "p_value": result.pvalue,
                }
            )

    table = pd.DataFrame(rows)
    table["p_value_holm"] = np.nan
    for family, indices in table.groupby("family").groups.items():
        table.loc[indices, "p_value_holm"] = holm_adjust(
            table.loc[indices, "p_value"].tolist()
        )
    table["exploratory"] = True
    return table


def save_figure(fig: plt.Figure, output_dir: Path, name: str) -> None:
    """Save both publication PNG and editable vector PDF."""
    fig.savefig(output_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_model_performance(
    performance: pd.DataFrame,
    output_dir: Path,
) -> None:
    metrics = [
        ("test_accuracy_mean", "Accuracy"),
        ("test_balanced_accuracy_mean", "Balanced accuracy"),
        ("test_macro_f1_mean", "Macro-F1"),
    ]
    x = np.arange(len(MODEL_ORDER))
    width = 0.23
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, (column, label) in enumerate(metrics):
        values = performance.set_index("model").loc[MODEL_ORDER, column] * 100
        ax.bar(x + (offset - 1) * width, values, width, label=label)
    ax.axhline(20, color="black", linestyle="--", linewidth=1, label="Chance accuracy")
    ax.set_ylabel("Performance (%)")
    ax.set_xticks(x, MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylim(0, 32)
    ax.set_title("Protocol 1 held-out session performance")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_model_performance")


def plot_generalization_gap(
    performance: pd.DataFrame,
    output_dir: Path,
) -> None:
    indexed = performance.set_index("model").loc[MODEL_ORDER]
    validation = indexed["validation_macro_f1_mean"].to_numpy() * 100
    test = indexed["test_macro_f1_mean"].to_numpy() * 100
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    ax.bar(x - width / 2, validation, width, label="Validation")
    ax.bar(x + width / 2, test, width, label="Test")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_xticks(x, MODEL_ORDER, rotation=15, ha="right")
    ax.set_ylim(0, max(validation.max(), test.max()) + 6)
    ax.set_title("Validation-to-test generalization")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_generalization_gap")


def plot_command_recall(
    recall_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    x = np.arange(len(LABELS))
    width = 0.19
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for position, model in enumerate(MODEL_ORDER):
        group = recall_table.set_index(["model", "command"])
        values = [group.loc[(model, label), "recall"] * 100 for label in LABELS]
        ax.bar(
            x + (position - 1.5) * width,
            values,
            width,
            label=model,
            color=MODEL_COLORS[model],
        )
    ax.axhline(20, color="black", linestyle="--", linewidth=1)
    ax.set_ylabel("Recall (%)")
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 40)
    ax.set_title("Command recall on held-out sessions")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_command_recall")


def plot_confusions(
    normalized: dict[str, np.ndarray],
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), constrained_layout=True)
    image = None
    for ax, model in zip(axes.flat, MODEL_ORDER):
        matrix = normalized[model] * 100
        image = ax.imshow(matrix, vmin=0, vmax=35, cmap="Blues")
        ax.set_title(model)
        ax.set_xticks(range(5), LABELS, rotation=35, ha="right")
        ax.set_yticks(range(5), LABELS)
        ax.set_xlabel("Predicted command")
        ax.set_ylabel("True command")
        for row in range(5):
            for column in range(5):
                color = "white" if matrix[row, column] >= 22 else "black"
                ax.text(
                    column,
                    row,
                    f"{matrix[row, column]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )
    fig.colorbar(image, ax=axes, label="Row-normalized percentage")
    fig.suptitle("Protocol 1 normalized confusion matrices", fontsize=14)
    save_figure(fig, output_dir, "figure_confusion_matrices")


def plot_participants(
    participant_table: pd.DataFrame,
    output_dir: Path,
) -> None:
    wide = participant_table.pivot(
        index="subject_number",
        columns="analysis_model",
        values="test_accuracy",
    ).sort_index()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for model in MODEL_ORDER:
        ax.plot(
            wide.index,
            wide[model] * 100,
            marker="o",
            linewidth=1.5,
            label=model,
            color=MODEL_COLORS[model],
        )
    ax.axhline(20, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Participant")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xticks(wide.index, [f"sub{i}" for i in wide.index], rotation=45)
    ax.set_ylim(0, max(35, float((wide * 100).max().max()) + 3))
    ax.set_title("Participant-level cross-session accuracy")
    ax.legend(frameon=False, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_dir, "figure_participant_accuracy")


def main(args: argparse.Namespace) -> None:
    project_root = Path(args.project_root).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = discover_experiments(project_root)
    summaries = load_summaries(experiments)
    records = load_confusions(experiments)

    performance = make_performance_table(summaries)
    participants = make_participant_table(summaries)
    seeds = make_seed_table(summaries)
    recall, normalized = confusion_tables(records, output_dir)
    tests = make_statistical_tests(participants)

    performance.to_csv(output_dir / "table_model_performance.csv", index=False)
    participants.to_csv(output_dir / "table_participant_performance.csv", index=False)
    seeds.to_csv(output_dir / "table_seed_variability.csv", index=False)
    recall.to_csv(output_dir / "table_command_recall.csv", index=False)
    tests.to_csv(output_dir / "table_statistical_tests.csv", index=False)

    plot_model_performance(performance, output_dir)
    plot_generalization_gap(performance, output_dir)
    plot_command_recall(recall, output_dir)
    plot_confusions(normalized, output_dir)
    plot_participants(participants, output_dir)

    print("\nProtocol 1 analysis complete")
    print("Output directory:", output_dir)
    print("Confusion matrices analyzed:", len(records))
    print("\nFinal model table:")
    print(
        performance[
            [
                "model",
                "test_accuracy_mean",
                "test_balanced_accuracy_mean",
                "test_macro_f1_mean",
                "macro_f1_generalization_gap",
            ]
        ].to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate final Protocol 1 tables and figures."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="ArEEG project directory containing the final result folders.",
    )
    parser.add_argument(
        "--output-dir",
        default="protocol1_analysis",
        help="New analysis directory created inside the project root.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
