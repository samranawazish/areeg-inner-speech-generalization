import os
import numpy as np
import pandas as pd

from dataloader import get_subject_folders
from Utilities.Extractor import process_eeg_with_metadata


LABEL_TO_ID = {
    "Down": 0,
    "Left": 1,
    "Right": 2,
    "Select": 3,
    "Up": 4
}


def prepare_dataset(dataset_path="CSVData"):

    all_X = []
    all_y = []
    all_metadata = []

    for subject_id in get_subject_folders(dataset_path):

        subject_path = os.path.join(
            dataset_path,
            subject_id
        )

        X, y_text, metadata = process_eeg_with_metadata(
            subject_folder=subject_path
        )

        # Keep only trials satisfying the quality rule.
        quality_mask = (
            metadata["padding_samples"].to_numpy()
            <= 125
        )

        X = X[quality_mask]
        y_text = y_text[quality_mask]

        metadata = (
            metadata.loc[quality_mask]
            .reset_index(drop=True)
        )

        y_numeric = np.asarray(
            [
                LABEL_TO_ID[label]
                for label in y_text
            ],
            dtype=np.int64
        )

        metadata["label_id"] = y_numeric
        metadata["included"] = True

        all_X.append(
            X.astype(np.float32)
        )

        all_y.append(y_numeric)
        all_metadata.append(metadata)

    X_all = np.concatenate(
        all_X,
        axis=0
    )

    y_all = np.concatenate(
        all_y,
        axis=0
    )

    metadata_all = pd.concat(
        all_metadata,
        ignore_index=True
    )

    assert len(X_all) == len(y_all)
    assert len(y_all) == len(metadata_all)
    assert X_all.shape[1:] == (8, 1200)
    assert np.array_equal(
        y_all,
        metadata_all["label_id"].to_numpy()
    )

    assert np.isfinite(X_all).all()

    assert set(np.unique(y_all)) == {
        0, 1, 2, 3, 4
    }
    return X_all, y_all, metadata_all


def prepare_and_save(
    dataset_path="CSVData",
    output_data="areeg_prepared_unfiltered.npz",
    output_metadata="areeg_metadata.csv"
):
    """
    Prepare the complete ArEEG dataset and save it.

    This function can be called directly by train.py.
    """

    X, y, metadata = prepare_dataset(
        dataset_path
    )

    print("Prepared EEG shape:", X.shape)
    print("Prepared labels shape:", y.shape)
    print("Metadata shape:", metadata.shape)

    print("\nNumeric class counts:")
    print(
        pd.Series(y)
        .value_counts()
        .sort_index()
    )

    print("\nLabel mapping:")
    print(LABEL_TO_ID)

    print("\nTrials per subject:")
    print(
        metadata["subject_id"]
        .value_counts()
    )

    np.savez_compressed(
        output_data,
        X=X,
        y=y,
        label_names=np.array([
            "Down",
            "Left",
            "Right",
            "Select",
            "Up"
        ])
    )

    metadata.to_csv(
        output_metadata,
        index=False
    )

    print("\nSaved:")
    print(output_data)
    print(output_metadata)

    return X, y, metadata


if __name__ == "__main__":
    prepare_and_save()