from Utilities.Extractor import process_eeg_with_metadata

import os
import re
import pandas as pd


def subject_number(subject_name):
    """
    Extract the number from a participant folder.

    Examples:
        sub2  -> 2
        sub10 -> 10
    """

    match = re.fullmatch(
        r"sub(\d+)",
        subject_name
    )

    if match:
        return int(match.group(1))

    return float("inf")


def get_subject_folders(full_dataset_path):
    """
    Return participant folders in numerical order.
    """

    if not os.path.isdir(full_dataset_path):
        raise FileNotFoundError(
            f"Dataset directory not found: "
            f"{full_dataset_path}"
        )

    subjects = [
        subject
        for subject in os.listdir(full_dataset_path)
        if os.path.isdir(
            os.path.join(
                full_dataset_path,
                subject
            )
        )
        and re.fullmatch(r"sub\d+", subject)
    ]

    return sorted(
        subjects,
        key=subject_number
    )


def LoadAllSubjectsCSV(full_dataset_path="CSVData"):
    """
    Load participants one at a time.

    Returns:
        subject_id, X, y, metadata
    """

    for subject_id in get_subject_folders(
        full_dataset_path
    ):

        subject_path = os.path.join(
            full_dataset_path,
            subject_id
        )

        X, y, metadata = process_eeg_with_metadata(
            subject_folder=subject_path
        )

        yield subject_id, X, y, metadata

if __name__ == "__main__":

    dataset_path = "CSVData"
    audit_rows = []
    all_metadata = []

    for subject_id in get_subject_folders(dataset_path):

        subject_path = os.path.join(
            dataset_path,
            subject_id
        )

        X, y, metadata = process_eeg_with_metadata(
            subject_folder=subject_path
        )

        assert len(X) == len(y) == len(metadata)
        assert X.shape[1:] == (8, 1200)

        padded = metadata[
            metadata["padding_samples"] > 0
        ]

        incomplete_sessions = (
            metadata
            .groupby("session_id")
            .size()
        )

        incomplete_sessions = incomplete_sessions[
            incomplete_sessions != 25
        ]

        audit_rows.append({
            "subject_id": subject_id,
            "sessions": metadata["session_id"].nunique(),
            "trials": len(metadata),
            "padded_trials": len(padded),
            "maximum_padding": (
                padded["padding_samples"].max()
                if len(padded) > 0
                else 0
            ),
            "median_padding": (
                padded["padding_samples"].median()
                if len(padded) > 0
                else 0
            ),
            "padding_over_125": (
                padded["padding_samples"] > 125
            ).sum(),
            "incomplete_sessions": len(
                incomplete_sessions
            )
        })

        all_metadata.append(metadata)

    audit_table = pd.DataFrame(audit_rows)

    complete_metadata = pd.concat(
        all_metadata,
        ignore_index=True
    )

    print("\nDataset quality summary")
    print(audit_table.to_string(index=False))

    print("\nOverall trials:", len(complete_metadata))

    print(
        "Overall padded trials:",
        (
            complete_metadata["padding_samples"] > 0
        ).sum()
    )

    print(
        "Trials with more than 125 padded samples:",
        (
            complete_metadata["padding_samples"] > 125
        ).sum()
    )

    print("\nTwenty most severely padded trials")

    print(
        complete_metadata
        .sort_values(
            "padding_samples",
            ascending=False
        )
        [
            [
                "subject_id",
                "session_id",
                "trial_id",
                "label",
                "original_samples",
                "padding_samples"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )