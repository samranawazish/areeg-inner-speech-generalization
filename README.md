# ArEEG Inner-Speech Generalization Benchmark

A reproducible comparison of EEGNet, EEG-Conformer, DeepConvNet, and a Riemannian tangent-space baseline for decoding five Arabic inner-speech commands from low-density EEG. The main assignment experiment is **chronological within-participant cross-session generalization**. Additional scripts investigate leave-one-subject-out generalization, few-shot adaptation, and CORAL session alignment.

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

Raw EEG files, virtual environments, caches, temporary timing runs, and neural-network checkpoints are intentionally excluded from GitHub. Compact CSV/JSON summaries and final figures are retained so reported results can be checked without downloading model weights.

## Experimental protocol

### Protocol 1: chronological cross-session generalization

Each participant is modelled separately. Sessions are ordered chronologically:

- **Training:** all earlier sessions
- **Validation:** the next two sessions
- **Test:** the final two sessions

For a typical participant with 15 sessions, this gives 11 training, 2 validation, and 2 test sessions. Splitting by whole sessions prevents trials from the same recording session appearing in more than one partition. Hyperparameters and early stopping use validation data only; test sessions are evaluated after model selection.

### Shared preprocessing and seeds

For the assignment comparison, all four algorithms must be run with the explicit option:

```bash
--normalization per-trial
```

Do not rely on script defaults: `train.py` and `train_riemannian.py` currently have different default normalization choices. The neural models were evaluated with seeds **42, 123, and 2026**. Run the Riemannian baseline with the same three seeds as well, even if its deterministic solver returns identical results.

## Installation

Python 3.11 or a tested compatible Python version is recommended. Create a clean environment rather than committing a local virtual-environment directory.

```bash
python -m venv .venv

# Windows Git Bash
source .venv/Scripts/activate

# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch and TorchAudio must be installed as a compatible version pair. A mismatch can produce a TorchAudio DLL error before model selection begins.

## Data preparation

The data are not redistributed in this repository. Download OpenNeuro dataset `ds005262`, version `1.0.1`, and place the raw files in the location described by `data/README.md` or pass the path expected by the loader.

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
- approximately balanced classes.

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

Before running the analysis, confirm that `EXPERIMENT_PATTERNS` includes all four assignment models and matches the output directories above. The earlier analysis script did not include DeepConvNet and expected a single `riemannian_final` directory; that must be updated to `deepconvnet_final_seed*` and `riemannian_final_seed*` when the three Riemannian runs are available.

```bash
python analyze_protocol1.py
```

The analysis should read existing result files only and write compact tables/figures without modifying training outputs.

## Current results

Accuracy and macro-F1 are means across participants and then seeds. The `+/-` values are standard deviations across the three seed-level means. Chance accuracy is 20% for five balanced classes.

| Method | Test accuracy | Macro-F1 | Model size | Runtime status |
|---|---:|---:|---:|---|
| Riemannian tangent-space + logistic regression | **23.43%** | **18.99%** | 36 tangent features; about 185 classifier coefficients | **29.49 s measured**, all 12 participants, seed 42 |
| EEGNet | 20.16% +/- 1.35 | 17.58% +/- 0.66 | 4,197 trainable parameters | 12.03 s measured on sub0; about 2.4 min extrapolated to 12 participants |
| EEG-Conformer | 18.99% +/- 0.68 | 16.84% +/- 0.60 | 650,437 trainable parameters | 35.17 s measured on sub0; about 7.0 min extrapolated to 12 participants |
| DeepConvNet | 18.23% +/- 0.72 | 14.07% +/- 0.64 | 282,505 trainable parameters | 31.91 s measured on sub0; about 6.4 min extrapolated to 12 participants |

Runtime comparisons are CPU wall-clock measurements. The neural full-cohort values are labelled estimates because they are twelve times the measured sub0 runtime; they must not be presented as directly measured full-cohort times. For the final assignment table, either time all methods on the same participant or time every full 12-participant command on the same machine.

## Why the Riemannian baseline performed best

The key data property is the combination of **low spatial resolution and limited labelled data per participant**: only eight channels and roughly 250-425 training trials for each chronological subject-specific split. Under these conditions, estimating an 8 x 8 covariance matrix and mapping it to a 36-dimensional tangent-space vector gives a compact summary of spatial relationships. A regularized linear classifier can learn from this representation with much less data than the neural networks require. The result is modest but consistent with a data-limited, noisy inner-speech task: Riemannian accuracy is above the 20% chance level, while the larger neural models do not obtain a clear advantage.

This is a data-based explanation, not a claim that Riemannian methods are universally superior.

## Ablation status

The completed Conformer/CORAL comparison is a controlled component analysis:

| Conformer condition | Accuracy | Macro-F1 |
|---|---:|---:|
| Without CORAL | 18.99% | 16.84% |
| With CORAL, weight 0.1 | 18.94% | 15.90% |

CORAL did not improve this setting, so distribution alignment alone does not explain the Riemannian advantage. However, this comparison does **not directly test** the main explanation that the winning compact representation is more data-efficient.

To satisfy the assignment wording strongly, run a training-data ablation at 25%, 50%, and 100% of the chronological training sessions while leaving validation/test sessions unchanged. The prediction is that the Riemannian model's lead will be largest at 25%, and that neural models will narrow the gap as training data increase. Do not insert ablation numbers until the experiment has actually been run.

## Reproducibility notes

- Run commands from the repository root.
- Use the same prepared data, chronological splits, per-trial normalization, and seeds for all methods.
- Fit any data-dependent preprocessing using training data only.
- Use validation macro-F1 for model selection.
- Evaluate the held-out test sessions only after model selection.
- Record Python/package versions, CPU/GPU, operating system, command, seed, start time, end time, and wall-clock duration.
- Report failed or excluded trials rather than silently dropping them.
- Results in this repository come from the authors' own runs; they are not copied from the ArEEG paper.

## Contributions

| Contributor | Contribution |
|---|---|
| Samra | Dataset preparation and audit; experimental design; implementation and execution of the four-model benchmark; result analysis; documentation and presentation. |

If this is a group submission, add every member's full name and exact contribution here. The Git history must also show meaningful activity from every group member.

## Use of AI tools

OpenAI ChatGPT/Codex was used to assist with code debugging, repository organization, README editing, result-table formatting, and presentation organization. All experiment commands were executed by the author, and all reported numeric results were checked against locally generated CSV/JSON outputs. AI-generated suggestions were reviewed before inclusion.

## Citation

```bibtex
@article{Metwalli2025ArEEG,
  title   = {ArEEG: an Open-Access Arabic Inner Speech EEG Dataset},
  author  = {Metwalli, Donia and Kiroles, Antony E. and Radwan, Yousef A. and Mohamed, Eslam Ahmed and Barakat, Mariam and Ahmed, Anas and Omar, Amr M. and Selim, Sahar},
  journal = {Scientific Data},
  volume  = {12},
  number  = {1},
  year    = {2025},
  doi     = {10.1038/s41597-025-05387-w}
}
```

This repository uses or builds on [PyTorch](https://pytorch.org/), [Braindecode](https://braindecode.org/), [PyRiemann](https://pyriemann.readthedocs.io/), [scikit-learn](https://scikit-learn.org/), and the official ArEEG loading/preprocessing code. See `requirements.txt`, source-file headers, and `LICENSE` for details.

## Licenses

- **ArEEG dataset:** CC0 through OpenNeuro.
- **Repository code:** see `LICENSE`. If code derived from the official ArEEG GPL-3.0 repository is retained, this repository must use a GPL-compatible license and preserve attribution.
