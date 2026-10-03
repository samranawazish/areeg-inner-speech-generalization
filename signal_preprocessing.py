# -*- coding: utf-8 -*-

"""
Signal preprocessing for the ArEEG dataset.

Processing:
1. Load prepared EEG and metadata.
2. Replace minor zero-padding using reflection.
3. Apply a 50 Hz notch filter.
4. Apply a 0.5–30 Hz band-pass filter.
5. Validate and save the filtered EEG.

Normalization is not performed here. It must be fitted using
training data only after the experimental split is created.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal


SAMPLING_FREQUENCY = 250
NOTCH_FREQUENCY = 50.0
LOW_CUTOFF = 0.5
HIGH_CUTOFF = 30.0
FILTER_ORDER = 4


def load_prepared_data(
    data_path="areeg_prepared_unfiltered.npz",
    metadata_path="areeg_metadata.csv"
):
    """
    Load prepared EEG arrays and their aligned metadata.

    Returns:
        X: EEG array with shape (trials, channels, samples)
        y: Numeric labels with shape (trials,)
        metadata: One metadata row per trial
        label_names: Class names ordered by numeric label
    """

    data_path = Path(data_path)
    metadata_path = Path(metadata_path)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Prepared EEG file not found: {data_path}"
        )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {metadata_path}"
        )

    with np.load(data_path) as prepared:
        X = prepared["X"].astype(
            np.float32,
            copy=True
        )

        y = prepared["y"].astype(
            np.int64,
            copy=True
        )

        if "label_names" in prepared.files:
            label_names = prepared[
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

    metadata = pd.read_csv(metadata_path)

    return X, y, metadata, label_names


def validate_eeg(
    X,
    y,
    metadata,
    stage_name="EEG data"
):
    """
    Validate array shapes, metadata alignment and numerical values.
    """

    if X.ndim != 3:
        raise ValueError(
            f"{stage_name}: expected a three-dimensional "
            f"EEG array, received shape {X.shape}."
        )

    if X.shape[1:] != (8, 1200):
        raise ValueError(
            f"{stage_name}: expected shape "
            f"(trials, 8, 1200), received {X.shape}."
        )

    if len(X) != len(y):
        raise ValueError(
            f"{stage_name}: EEG and label lengths differ."
        )

    if len(y) != len(metadata):
        raise ValueError(
            f"{stage_name}: labels and metadata lengths differ."
        )

    if "label_id" in metadata.columns:
        metadata_labels = metadata[
            "label_id"
        ].to_numpy(dtype=np.int64)

        if not np.array_equal(y, metadata_labels):
            raise ValueError(
                f"{stage_name}: numeric labels do not align "
                f"with the metadata."
            )

    if not np.isfinite(X).all():
        raise ValueError(
            f"{stage_name}: EEG contains NaN or infinity."
        )

    if set(np.unique(y)) != {0, 1, 2, 3, 4}:
        raise ValueError(
            f"{stage_name}: unexpected labels "
            f"{np.unique(y)}."
        )

    print(f"\n{stage_name}")
    print("Shape:", X.shape)
    print("Data type:", X.dtype)
    print("Minimum:", float(X.min()))
    print("Maximum:", float(X.max()))
    print("Mean:", float(X.mean()))
    print("Standard deviation:", float(X.std()))


def repair_minor_padding(X, metadata):
    """
    Replace artificial zero-padding at the end of valid trials
    using reflected EEG samples.

    Severely incomplete trials were already excluded during
    dataset preparation.
    """

    if "valid_samples" not in metadata.columns:
        raise ValueError(
            "Metadata does not contain valid_samples."
        )

    X_repaired = X.copy()

    repaired_trials = 0

    for trial_index in range(len(X_repaired)):

        valid_samples = int(
            metadata.iloc[trial_index][
                "valid_samples"
            ]
        )

        total_samples = X_repaired.shape[2]

        if valid_samples >= total_samples:
            continue

        if valid_samples < 2:
            raise ValueError(
                f"Trial {trial_index} contains only "
                f"{valid_samples} valid samples."
            )

        missing_samples = (
            total_samples - valid_samples
        )

        valid_signal = X_repaired[
            trial_index,
            :,
            :valid_samples
        ]

        repaired_signal = np.pad(
            valid_signal,
            pad_width=(
                (0, 0),
                (0, missing_samples)
            ),
            mode="reflect"
        )

        X_repaired[
            trial_index
        ] = repaired_signal

        repaired_trials += 1

    print(
        f"Minor padding repaired in "
        f"{repaired_trials} trials."
    )

    return X_repaired


def design_filters(
    sampling_frequency=SAMPLING_FREQUENCY,
    notch_frequency=NOTCH_FREQUENCY,
    low_cutoff=LOW_CUTOFF,
    high_cutoff=HIGH_CUTOFF,
    filter_order=FILTER_ORDER
):
    """
    Create the notch and band-pass filters.
    """

    if sampling_frequency <= 0:
        raise ValueError(
            "Sampling frequency must be positive."
        )

    if not 0 < low_cutoff < high_cutoff:
        raise ValueError(
            "Cutoff frequencies must satisfy "
            "0 < low_cutoff < high_cutoff."
        )

    nyquist = sampling_frequency / 2

    if high_cutoff >= nyquist:
        raise ValueError(
            "The high cutoff must be below the "
            "Nyquist frequency."
        )

    if not 0 < notch_frequency < nyquist:
        raise ValueError(
            "The notch frequency must be between "
            "0 and the Nyquist frequency."
        )

    notch_b, notch_a = signal.iirnotch(
        w0=notch_frequency,
        Q=30,
        fs=sampling_frequency
    )

    bandpass_sos = signal.butter(
        N=filter_order,
        Wn=[low_cutoff, high_cutoff],
        btype="bandpass",
        fs=sampling_frequency,
        output="sos"
    )

    return notch_b, notch_a, bandpass_sos


def filter_eeg_trials(
    X,
    sampling_frequency=SAMPLING_FREQUENCY,
    notch_frequency=NOTCH_FREQUENCY,
    low_cutoff=LOW_CUTOFF,
    high_cutoff=HIGH_CUTOFF,
    filter_order=FILTER_ORDER,
    batch_size=128
):
    """
    Apply zero-phase notch and band-pass filters along time.

    EEG is processed in batches to limit memory usage.
    """

    if batch_size < 1:
        raise ValueError(
            "batch_size must be at least 1."
        )

    notch_b, notch_a, bandpass_sos = design_filters(
    sampling_frequency=sampling_frequency,
    notch_frequency=notch_frequency,
    low_cutoff=low_cutoff,
    high_cutoff=high_cutoff,
    filter_order=filter_order
)

    X_filtered = np.empty(
        X.shape,
        dtype=np.float32
    )

    number_of_trials = len(X)

    for start in range(
        0,
        number_of_trials,
        batch_size
    ):

        end = min(
            start + batch_size,
            number_of_trials
        )

        print(
            f"Filtering trials {start}–{end - 1} "
            f"of {number_of_trials - 1}"
        )

        batch = X[start:end].astype(
            np.float64,
            copy=False
        )

        batch = signal.filtfilt(
            notch_b,
            notch_a,
            batch,
            axis=-1
        )

        batch = signal.sosfiltfilt(
            bandpass_sos,
            batch,
            axis=-1
        )

        X_filtered[start:end] = batch.astype(
            np.float32
        )

    return X_filtered


def preprocess_dataset(
    X,
    y,
    metadata
):
    """
    Run the complete signal-preprocessing pipeline.
    """

    validate_eeg(
        X,
        y,
        metadata,
        stage_name="Before signal preprocessing"
    )

    X_repaired = repair_minor_padding(
        X,
        metadata
    )

    X_filtered = filter_eeg_trials(
        X_repaired
    )

    validate_eeg(
        X_filtered,
        y,
        metadata,
        stage_name="After signal preprocessing"
    )

    return X_filtered, y, metadata


def preprocess_and_save(
    input_data="areeg_prepared_unfiltered.npz",
    input_metadata="areeg_metadata.csv",
    output_data="areeg_filtered.npz"
):
    """
    Load, preprocess and save ArEEG.

    This function can later be called automatically by train.py.
    """

    X, y, metadata, label_names = (
        load_prepared_data(
            data_path=input_data,
            metadata_path=input_metadata
        )
    )

    X_filtered, y, metadata = (
        preprocess_dataset(
            X,
            y,
            metadata
        )
    )

    np.savez_compressed(
        output_data,
        X=X_filtered,
        y=y,
        label_names=label_names
    )

    print(f"\nSaved filtered EEG: {output_data}")

    return X_filtered, y, metadata


if __name__ == "__main__":
    preprocess_and_save()