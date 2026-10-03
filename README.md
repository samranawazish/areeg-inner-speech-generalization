# ArEEG Inner-Speech Generalization Benchmark

A reproducible comparison of EEGNet, EEG-Conformer, DeepConvNet, and a Riemannian tangent-space baseline for decoding five Arabic inner-speech commands from low-density EEG. The main assignment experiment is chronological within-participant cross-session generalization. Additional scripts investigate leave-one-subject-out generalization, few-shot adaptation, and CORAL session alignment.

## Research question

How well do conventional geometry-based methods and neural networks generalize to later recording sessions when only eight EEG channels and a few hundred labelled trials per participant are available?

This project is connected to work on robust inner-speech EEG decoding across sessions and participants. The comparison is deliberately leakage-controlled: training, validation, and test sets are separated by recording session rather than by randomly mixing trials.

## Dataset

- **Name:** ArEEG: Arabic Inner Speech EEG Dataset
- **Official dataset:** [OpenNeuro ds005262, version 1.0.1](https://openneuro.org/datasets/ds005262/versions/1.0.1)
- **Dataset DOI:** [10.18112/openneuro.ds005262.v1.0.1](https://doi.org/10.18112/openneuro.ds005262.v1.0.1)
- **Dataset license:** Creative Commons Zero (CC0)
- **Dataset version used:** 1.0.1
- **Publication:** [Metwalli et al., 2025, Scientific Data](https://doi.org/10.1038/s41597-025-05387-w)
- **Official code repository:** [Eslam21/ArEEG-an-Open-Access-Arabic-Inner-Speech-EEG-Dataset](https://github.com/Eslam21/ArEEG-an-Open-Access-Arabic-Inner-Speech-EEG-Dataset)
- **Participants:** 12 native Arabic speakers
- **Task:** five-class classification: `Down`, `Left`, `Right`, `Select`, `Up`
- **Acquisition:** 8 EEG channels, sampled at 250 Hz

The publication reports 4,650 trials. The local extractor recovered 4,606 trials, and the quality-controlled benchmark retained **4,575 trials**. This difference is reported rather than hidden; `sanity_check.py` should be run after preparation to print the exact retained counts and exclusions for the local copy.

### Prepared-data representation

| Item | Value |
|---|---:|
| Retained trials | 4,575 |
| Input channels | 8 |
| Samples per trial | 1,200 |
| Values per trial | 9,600 |
| Classes | 5 |
| Approximate trials per class | 915 |
| Label mapping | Down=0, Left=1, Right=2, Select=3, Up=4 |

The classes are approximately balanced. The data audit must also confirm array shapes, finite values, duplicate trial identifiers, session ordering, and participant/session overlap before training.

## Repository flow

```text
Raw ArEEG CSV sessions
        |
        v
Utilities/Extractor.py
        |
        +--> dataloader.py
        +--> prepare_dataset.py
                  |
                  v
       prepared NPZ + metadata CSV
                  |
                  v
      signal_preprocessing.py
                  |
                  v
               splits.py
                  |
        +---------+------------------+
        |                            |
        v                            v
     train.py              train_riemannian.py
 EEGNet / Conformer /       covariance + tangent
 DeepConvNet / CORAL        space + logistic model
        |                            |
        +-------------+--------------+
                      v
            analyze_protocol1.py
                      |
                      v
             tables and figures
```

## Files

| Path | Purpose |
|---|---|
| `dataloader.py` | Loads ArEEG sessions through the original extraction utilities. |
| `Utilities/Extractor.py` | Reads raw recordings and their metadata. Required by the loader. |
| `Utilities/Preprocessing.py` | Original ArEEG preprocessing helpers used by the extraction path. |
| `prepare_dataset.py` | Creates the prepared EEG array and metadata table. |
| `signal_preprocessing.py` | Shared signal-processing functions. |
| `sanity_check.py` | Audits shapes, labels, missing/non-finite values, duplicates, and counts. |
| `splits.py` | Creates chronological, session-disjoint train/validation/test partitions. |
| `models/eegnet.py` | EEGNet architecture. |
| `models/eeg_conformer.py` | EEG-Conformer wrapper. |
| `models/deepconvnet.py` | DeepConvNet architecture. |
| `train.py` | Trains EEGNet, EEG-Conformer, DeepConvNet, and optional Conformer+CORAL. |
| `train_riemannian.py` | Trains the covariance/tangent-space logistic-regression baseline. |
| `analyze_protocol1.py` | Aggregates seeds/participants and creates comparison tables and figures. |
| `train_loso.py` | Protocol 2: leave-one-subject-out EEG-Conformer experiment. |
| `train_loso_eegnet.py` | Protocol 2: leave-one-subject-out EEGNet experiment. |
| `train_loso_riemannian.py` | Protocol 2: leave-one-subject-out Riemannian experiment. |
| `train_adaptation.py` | Protocol 3: few-shot target-participant adaptation. |
| `train_riemannian_ablation.py` | Riemannian ablation comparing tangent-space features with flattened covariance features. |

Raw EEG files, virtual environments, caches, temporary timing runs, and neural-network checkpoints are intentionally excluded from GitHub. Compact CSV/JSON summaries and final figures are retained so reported results can be checked without downloading model weights.

## Experimental protocol

### Protocol 1: chronological cross-session generalization

Each participant is modelled separately. Sessions are ordered chronologically:

- **Training:** all earlier sessions
- **Validation:** the next two sessions
- **Test:** the final two sessions

For a typical participant with 15 sessions, this gives 11 training, 2 validation, and 2 test sessions. Splitting by whole sessions prevents trials from the same recording session appearing in more than one partition. Hyperparameters and early stopping use validation data only; test sessions are evaluated after model selection.

### Shared preprocessing and seeds

The neural models were evaluated with seeds 42, 123, and 2026.

## Installation

Python 3.11 or a tested compatible Python version is recommended. 

```bash
python -m venv .venv

# Windows Git Bash
source .venv/Scripts/activate

# Linux/macOS
# source .venv/bin/activate

python -m pip install -r requirements.txt
```

PyTorch and TorchAudio must be installed as a compatible version pair. A mismatch can produce a TorchAudio DLL error before model selection begins.

## Data preparation

Download OpenNeuro dataset `ds005262`, version `1.0.1`, and place the raw files in the location described by `data/README.md` or pass the path expected by the loader.

```bash
python prepare_dataset.py
python sanity_check.py
```

Before training, confirm that the audit reports:

- 12 participants and five stable class IDs;
- arrays shaped as trials x 8 channels x 1,200 samples;
- no participant/session overlap between train, validation, and test;
- no missing labels or non-finite model inputs after quality control;
- no duplicate trial IDs across partitions;

## Reproduce Protocol 1

Run from the repository root. The output-directory names below match the patterns consumed by `analyze_protocol1.py`.

### EEG-Conformer

```bash
for seed in 42 123 2026; do
  python train.py --model eeg-conformer --all-subjects \
    --normalization per-trial --seed "$seed" \
    --output-dir "final_control_seed${seed}"
done
```

### EEGNet

```bash
for seed in 42 123 2026; do
  python train.py --model eegnet --all-subjects \
    --normalization per-trial --seed "$seed" \
    --output-dir "eegnet_final_seed${seed}"
done
```

### DeepConvNet

```bash
for seed in 42 123 2026; do
  python train.py --model deepconvnet --all-subjects \
    --normalization per-trial --seed "$seed" \
    --output-dir "deepconvnet_final_seed${seed}"
done
```

### Riemannian tangent-space baseline

```bash
for seed in 42 123 2026; do
  python train_riemannian.py --all-subjects \
    --normalization per-trial --seed "$seed" \
    --output-dir "riemannian_final_seed${seed}"
done
```

### CORAL component analysis

```bash
for seed in 42 123 2026; do
  python train.py --model eeg-conformer --all-subjects \
    --normalization per-trial --use-coral --coral-weight 0.1 \
    --seed "$seed" --output-dir "final_coral1_seed${seed}"
done
```

### Aggregate the results

Before running the analysis, confirm that `EXPERIMENT_PATTERNS` includes all four assignment models and matches the output directories above. 
```bash
python analyze_protocol1.py
```
The analysis should read existing result files only and write compact tables/figures without modifying training outputs.

## Current results

Accuracy and macro-F1 are means across participants and then seeds. The `+/-` values are standard deviations across the three seed-level means. Chance accuracy is 20% for five balanced classes.

| Method | Test accuracy | Macro-F1 | Model size | Runtime status |
|---|---:|---:|---:|---|
| Riemannian tangent-space + logistic regression | **23.43%** | **18.99%** | 36 tangent features; about 185 classifier coefficients |29.49 s |
| EEGNet | 20.16% +/- 1.35 | 17.58% +/- 0.66 | 4,197 trainable parameters | 12.03 s |
| EEG-Conformer | 18.99% +/- 0.68 | 16.84% +/- 0.60 | 650,437 trainable parameters | 35.17 s |
| DeepConvNet | 18.23% +/- 0.72 | 14.07% +/- 0.64 | 282,505 trainable parameters | 31.91 s |

## Why the Riemannian baseline performed best

The most important data property is the combination of low spatial resolution and limited labelled data per participant. ArEEG contains only eight EEG channels and approximately 250–425 training trials in each participant-specific chronological split.

The Riemannian pipeline estimates an \(8 \times 8\) covariance matrix for each trial and maps it into a compact 36-dimensional tangent-space representation. These features summarize relationships between EEG channels while substantially reducing the dimensionality of the original \(8 \times 1{,}200\) signal.

A regularized linear classifier can learn from this compact representation with less data than the higher-capacity neural networks require. Consequently, the Riemannian pipeline achieved the highest cross-session accuracy of 23.43%, exceeding the five-class chance level of 20%. However, the improvement remains modest, confirming that cross-session Arabic inner-speech decoding is a challenging and noisy classification problem.

## Ablation studies

Two controlled comparisons were conducted to examine the contribution of specific pipeline components.

### 1. Riemannian representation ablation

The tangent-space transformation was removed while keeping the covariance features and the remaining evaluation procedure unchanged.

| Riemannian condition | Accuracy | Macro-F1 | Test loss |
|---|---:|---:|---:|
| Full tangent-space pipeline | **23.43%** | 18.99% | **1.786** |
| Flat covariance without tangent mapping | 22.08% | **19.34%** | 2.927 |
| Change after removing tangent mapping | −1.34 points | +0.35 points | +1.141 |

Removing tangent-space mapping reduced accuracy by 1.34 percentage points and increased test loss. The full pipeline achieved higher participant-level accuracy for 7 of the 12 participants, while two participants were tied and flat covariance performed better for three.

However, macro-F1 increased slightly without tangent mapping, and the paired accuracy difference was not statistically significant using the Wilcoxon signed-rank test (\(p = 0.064\)). Therefore, the ablation provides partial support for the importance of tangent-space mapping, but it does not prove that this component alone explains the Riemannian pipeline’s advantage.

### 2. Conformer–CORAL component analysis

A second comparison evaluated whether explicit alignment of session-level feature distributions improved the EEG-Conformer.

| Conformer condition | Accuracy | Macro-F1 |
|---|---:|---:|
| Without CORAL | 18.99% | 16.84% |
| With CORAL, weight = 0.1 | 18.94% | 15.90% |
| Change with CORAL | −0.05 points | −0.94 points |

Adding CORAL did not improve cross-session performance. Accuracy decreased slightly, while macro-F1 decreased by 0.94 percentage points. This suggests that aligning second-order feature statistics alone was insufficient to address the session shift in this dataset.

Together, the two analyses indicate that the Riemannian pipeline’s advantage is more consistent with its compact covariance-based representation and suitability for limited training data than with distribution alignment alone.

Protocol 2: Cross-subject generalization

A leave-one-subject-out evaluation tested whether models trained on 11 participants could classify an entirely unseen participant. Performance remained close to the 20% chance level: Riemannian achieved 20.79%, EEGNet 20.11%, and EEG-Conformer 20.10% accuracy. This demonstrates substantial variability between participants.

Protocol 3: Few-shot adaptation

A pretrained population EEGNet was adapted using one or four early sessions from the target participant and evaluated on later sessions. Adaptation improved development accuracy from 22.00% to 24.00%, but confirmatory accuracy decreased from 18.99% to 17.42%. Therefore, the observed development improvement did not generalize to the confirmatory evaluation.
## Reproducibility notes

- Run commands from the repository root.
- Use the same prepared data, chronological splits, per-trial normalization, and seeds for all methods.
- Fit any data-dependent preprocessing using training data only.
- Use validation macro-F1 for model selection.
- Evaluate the held-out test sessions only after model selection.
- Results in this repository come from the authors' own runs.

## Contributions

| Contributor | Contribution |
|---|---|
| Samra | Led the project design and dataset preparation and audit; implemented the cross-session, cross-subject, and adaptation protocols; ran the EEGNet, EEG-Conformer, and CORAL experiments; consolidated and interpreted the results; and prepared the repository documentation. |
| Alya | Contributed to EEG data preprocessing; ran and validated the DeepConvNet and Riemannian classical baseline experiments with ablation; analyzed the resulting outputs; and prepared presentation slides and repository documentation.|


## Use of AI tools

OpenAI ChatGPT/Codex was used to assist with code debugging, repository organization, README editing, result-table formatting, and presentation organization. All experiment commands were executed by the author, and all reported numeric results were checked against locally generated CSV/JSON outputs. AI-generated suggestions were reviewed before inclusion.

