# -*- coding: utf-8 -*-

"""
Data splitting, session labels and leakage-controlled normalization for ArEEG.

Initial baseline:
- One participant
- Earlier sessions for training
- Subsequent sessions for validation
- Final sessions for testing
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


FILTERED_DATA_PATH = "areeg_filtered.npz"
METADATA_PATH = "areeg_metadata.csv"


def load_filtered_data(
    data_path=FILTERED_DATA_PATH,
    metadata_path=METADATA_PATH
):
    """
    Load filtered EEG, labels and aligned metadata.
    """

    data_path = Path(data_path)
    metadata_path = Path(metadata_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Filtered EEG not found: {data_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {metadata_path}"
        )

    with np.load(data_path) as data:

        X = data["X"].astype(
            np.float32,
            copy=True
        )

        y = data["y"].astype(
            np.int64,
            copy=True
        )

        if "label_names" in data.files:
            label_names = data[
                "label_names"
            ].copy()
        else:
            label_names = np.array([
                "Down",
                "Left",
                "Right",
                "Select",
                "Up"
            ])

    metadata = pd.read_csv(
        metadata_path
    )

    if len(X) != len(y):
        raise ValueError(
            "EEG and label lengths differ."
        )

    if len(y) != len(metadata):
        raise ValueError(
            "Labels and metadata lengths differ."
        )

    if X.shape[1:] != (8, 1200):
        raise ValueError(
            f"Unexpected EEG shape: {X.shape}"
        )

    if not np.array_equal(
        y,
        metadata["label_id"].to_numpy(
            dtype=np.int64
        )
    ):
        raise ValueError(
            "NPZ labels do not align with metadata."
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "Filtered EEG contains NaN or infinity."
        )

    return X, y, metadata, label_names


def get_subject_sessions(
    metadata,
    subject_id
):
    """
    Return a participant's session IDs in chronological order.
    """

    subject_metadata = metadata[
        metadata["subject_id"] == subject_id
    ].copy()

    if subject_metadata.empty:
        raise ValueError(
            f"Participant not found: {subject_id}"
        )

    session_table = (
        subject_metadata[
            [
                "recording_day",
                "session_id"
            ]
        ]
        .drop_duplicates()
        .sort_values(
            [
                "recording_day",
                "session_id"
            ]
        )
    )

    return session_table[
        "session_id"
    ].tolist()


def fit_channel_normalizer(X_train):
    """
    Calculate one mean and standard deviation for every channel.

    Statistics are calculated using training data only.

    Input:
        X_train shape: (trials, channels, samples)

    Returns:
        channel_mean shape: (1, channels, 1)
        channel_std shape: (1, channels, 1)
    """

    channel_mean = X_train.mean(
        axis=(0, 2),
        keepdims=True,
        dtype=np.float64
    )

    channel_std = X_train.std(
        axis=(0, 2),
        keepdims=True,
        dtype=np.float64
    )

    minimum_std = 1e-8

    channel_std = np.maximum(
        channel_std,
        minimum_std
    )

    return (
        channel_mean.astype(np.float32),
        channel_std.astype(np.float32)
    )


def apply_channel_normalizer(
    X,
    channel_mean,
    channel_std
):
    """
    Apply previously calculated channel statistics.
    """

    X_normalized = (
        X - channel_mean
    ) / channel_std

    return X_normalized.astype(
        np.float32
    )


def apply_per_trial_channel_normalizer(X):
    """Standardize each channel within each trial using time samples only."""

    trial_mean = X.mean(
        axis=2,
        keepdims=True,
        dtype=np.float64
    )

    trial_std = X.std(
        axis=2,
        keepdims=True,
        dtype=np.float64
    )

    trial_std = np.maximum(
        trial_std,
        1e-8
    )

    return (
        (X - trial_mean) / trial_std
    ).astype(np.float32)


def create_subject_session_split(
    X,
    y,
    metadata,
    subject_id="sub0",
    validation_sessions=2,
    test_sessions=2,
    normalization="train-channel",
    include_test=True
):
    """
    Create a chronological session-separated split.

    The final sessions are used for testing.
    The sessions immediately before them are used for validation.
    All earlier sessions are used for training.
    """

    sessions = get_subject_sessions(
        metadata,
        subject_id
    )

    number_of_sessions = len(sessions)

    required_sessions = (
        validation_sessions
        + test_sessions
        + 1
    )

    if number_of_sessions < required_sessions:
        raise ValueError(
            f"{subject_id} has {number_of_sessions} sessions, "
            f"but at least {required_sessions} are required."
        )

    train_end = (
        number_of_sessions
        - validation_sessions
        - test_sessions
    )

    validation_end = (
        number_of_sessions
        - test_sessions
    )

    train_session_ids = sessions[
        :train_end
    ]

    validation_session_ids = sessions[
        train_end:validation_end
    ]

    test_session_ids = sessions[
        validation_end:
    ]

    subject_mask = (
        metadata["subject_id"].to_numpy()
        == subject_id
    )

    session_values = metadata[
        "session_id"
    ].to_numpy()

    train_mask = (
        subject_mask
        & np.isin(
            session_values,
            train_session_ids
        )
    )

    validation_mask = (
        subject_mask
        & np.isin(
            session_values,
            validation_session_ids
        )
    )

    test_mask = (
        subject_mask
        & np.isin(
            session_values,
            test_session_ids
        )
    )

    train_indices = np.flatnonzero(
        train_mask
    )

    validation_indices = np.flatnonzero(
        validation_mask
    )

    test_indices = np.flatnonzero(
        test_mask
    )

    if len(train_indices) == 0:
        raise ValueError(
            "Training split is empty."
        )

    if len(validation_indices) == 0:
        raise ValueError(
            "Validation split is empty."
        )

    if len(test_indices) == 0:
        raise ValueError(
            "Test split is empty."
        )

    X_train = X[train_indices]
    y_train = y[train_indices]

    train_session_values = session_values[
        train_indices
    ]

    session_to_id = {
        session_id: session_number
        for session_number, session_id in enumerate(
            train_session_ids
        )
    }

    train_session_labels = np.asarray(
        [
            session_to_id[session_id]
            for session_id in train_session_values
        ],
        dtype=np.int64
    )

    X_validation = X[
        validation_indices
    ]

    y_validation = y[
        validation_indices
    ]

    X_test = X[test_indices] if include_test else None
    y_test = y[test_indices] if include_test else None

    channel_mean = None
    channel_std = None

    if normalization == "train-channel":

        channel_mean, channel_std = (
            fit_channel_normalizer(
                X_train
            )
        )

        X_train = apply_channel_normalizer(
            X_train,
            channel_mean,
            channel_std
        )

        X_validation = apply_channel_normalizer(
            X_validation,
            channel_mean,
            channel_std
        )

        if include_test:
            X_test = apply_channel_normalizer(
                X_test,
                channel_mean,
                channel_std
            )

    elif normalization == "per-trial":

        X_train = apply_per_trial_channel_normalizer(
            X_train
        )

        X_validation = apply_per_trial_channel_normalizer(
            X_validation
        )

        if include_test:
            X_test = apply_per_trial_channel_normalizer(
                X_test
            )

    elif normalization in (None, "none"):
        pass

    else:
        raise ValueError(
            "normalization must be 'train-channel', "
            "'per-trial', or 'none'."
        )

    if not np.isfinite(X_train).all():
        raise ValueError(
            "Training data contain invalid values."
        )

    if not np.isfinite(X_validation).all():
        raise ValueError(
            "Validation data contain invalid values."
        )

    if include_test and not np.isfinite(X_test).all():
        raise ValueError(
            "Test data contain invalid values."
        )

    if not (
        len(X_train)
        == len(y_train)
        == len(train_session_labels)
    ):
        raise ValueError(
            "Training EEG, labels and session labels are not aligned."
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "train_session_labels": train_session_labels,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,

        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,

        "train_sessions": train_session_ids,
        "validation_sessions": validation_session_ids,
        "test_sessions": test_session_ids,

        "channel_mean": channel_mean,
        "channel_std": channel_std
    }


def print_split_summary(
    split,
    subject_id
):
    """
    Display split sizes, sessions and class distributions.
    """

    print(f"\nParticipant: {subject_id}")

    print("\nTraining sessions:")
    for session in split["train_sessions"]:
        print(" ", session)

    print("\nValidation sessions:")
    for session in split["validation_sessions"]:
        print(" ", session)

    print("\nTest sessions:")
    for session in split["test_sessions"]:
        print(" ", session)

    print("\nArray shapes:")
    print(
        "Train:",
        split["X_train"].shape,
        split["y_train"].shape
    )

    print(
        "Validation:",
        split["X_validation"].shape,
        split["y_validation"].shape
    )

    if split["X_test"] is not None:
        print(
            "Test:",
            split["X_test"].shape,
            split["y_test"].shape
        )
    else:
        print("Test: held out and not loaded")

    print("\nClass distributions:")

    print(
        "Train:",
        Counter(
            split["y_train"].tolist()
        )
    )

    print(
        "Validation:",
        Counter(
            split["y_validation"].tolist()
        )
    )

    if split["y_test"] is not None:
        print(
            "Test:",
            Counter(
                split["y_test"].tolist()
            )
        )

    print("\nNormalized statistics:")

    print(
        "Training mean:",
        float(split["X_train"].mean())
    )

    print(
        "Training standard deviation:",
        float(split["X_train"].std())
    )

    print(
        "Validation mean:",
        float(split["X_validation"].mean())
    )

    if split["X_test"] is not None:
        print(
            "Test mean:",
            float(split["X_test"].mean())
        )


def create_baseline_split(
    subject_id="sub0",
    data_path=FILTERED_DATA_PATH,
    metadata_path=METADATA_PATH,
    normalization="train-channel",
    validation_only=False
):
    """
    Load the dataset and create a subject-specific session split.
    """

    X, y, metadata, label_names = (
        load_filtered_data(
            data_path=data_path,
            metadata_path=metadata_path
        )
    )

    split = create_subject_session_split(
        X=X,
        y=y,
        metadata=metadata,
        subject_id=subject_id,
        validation_sessions=2,
        test_sessions=2,
        normalization=normalization,
        include_test=not validation_only
    )

    split["label_names"] = label_names

    return split


if __name__ == "__main__":

    target_subject = "sub0"

    baseline_split = create_baseline_split(
        subject_id=target_subject
    )

    print_split_summary(
        baseline_split,
        target_subject
    )
