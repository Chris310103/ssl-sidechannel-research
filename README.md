# Self-Supervised Learning for Side-Channel Analysis

This project evaluates fixed 1-D self-supervised learning (SSL) baselines on
ASCAD power traces. The purpose is to compare representation-learning methods
for side-channel analysis, not to expose a large hyperparameter-search surface.

Each experiment pre-trains an encoder on unlabeled profiling traces, saves the
encoder checkpoint, and can use its representations in a downstream profiling
or non-profiling attack.

## Experiment design

The training entry point is [`src/train.py`](src/train.py). It accepts only the
experiment identity and data/run controls:

- SSL method
- input dataset and output directory
- trace window
- number of unlabeled profiling traces

Architecture, loss, optimizer, and augmentation choices are fixed baselines in
[`SSL_METHOD_PRESETS`](src/methods/ssl_factory.py). They are intentionally not
CLI options. This ensures that the model trained by `train_ssl_model` has the
same shape and settings as the one reconstructed by `build_ssl_model` for
checkpoint evaluation.

To adopt a different baseline later, edit one method entry in
`src/methods/ssl_factory.py` and record the change with the experiment output.
Do not add a new argparse option for an individual model parameter.

## Supported methods and fixed baselines

| Method | Encoder and method setup | Training setup |
| --- | --- | --- |
| `simclr` | Shared 1-D CNN, mean+max pooling, 320-wide projector hidden layer, 128-d projection | 100 epochs, batch 64, Adam LR 0.001, weight decay 1e-4; independent random shift/denoise/noise views |
| `ts2vec` | Shared 1-D CNN, mean+max pooling | 10 epochs, batch 64, AdamW LR 0.001; hierarchical temporal cropping and timestamp masking |
| `ts_tcc` | Shared 1-D CNN with a 256-d temporal context module | 100 epochs, batch 64, AdamW LR 0.0003; weak/strong random views |
| `byol` | Shared 1-D CNN, mean+max pooling, 128-d projection, 512-d hidden layers, EMA 0.996 | 100 epochs, batch 128, Adam LR 0.0003; independent random shift/noise views |
| `lhf_bootstrap` | Shared 1-D CNN with MLP and temporal-convolution bootstrap heads | 100 epochs, batch 128, AdamW LR 0.0003; random shift/denoise/noise views |
| `mae` | Trace transformer, mean+max pooling, patch size 5, mask ratio 0.30 | 100 epochs, batch 128, AdamW LR 0.0001 |
| `time_mae` | Trace transformer, mean+max pooling, patch size 10, mask ratio 0.60, target EMA 0.996 | 100 epochs, batch 128, AdamW LR 0.0001 |

The shared CNN is `shared_cnn_v1`. It is used for the temporal SSL methods
because TS2Vec and TS-TCC need a sequence of temporal features; the separate
CNN-best implementation does not provide a suitable temporal sequence at the
700-sample trace length.

### Random augmentation policy

For methods using random views, the policy itself is fixed. For example,
SimCLR always uses the `random_shift`, `denoise` and `gaussian_noise` family. Each view
independently samples which transforms to apply with probability 0.5, so the
view realization is random while the experiment definition remains fixed.

## Data contract

Training accepts ASCAD HDF5 (`.h5`) or NumPy archive (`.npz`) inputs.

- HDF5 training data are read from `Profiling_traces`; test data are read from
  `Attack_traces`.
- An NPZ training archive must contain `X_train` (and normally `y_train`,
  `plaintext`, and `key`). A test archive must contain `X_test`.
- `--trace_window` uses `START_END` notation, such as `0_700`.
- Traces remain raw after loading. The selected window is applied exactly once
  inside the SSL training pipeline, after view augmentation.

## Installation

Create a Python environment and install the project dependencies.
There is not yet a pinned dependency file, so install the required libraries
explicitly:

```bash
pip install torch numpy h5py scikit-learn matplotlib
```

Use a CUDA-enabled PyTorch build when training on an NVIDIA GPU. Device
selection is handled by `src.utils.get_device`.

## Train a baseline

Run commands from the repository root:

```bash
python -m src.train \
  --method simclr \
  --input data/npz_data/ASCAD/ATM_AES_v1_fixed_key/ASCAD.npz \
  --trace_window 0_700 \
  --n_train 500 \
  --output_dir outputs/checkpoints
```

The available method names are:

```text
simclr  ts2vec  ts_tcc  byol  lhf_bootstrap  mae  time_mae
```

`--n_train` controls only how many unlabeled profiling traces are used. It is
not a model hyperparameter. Epoch count, batch size, learning rate, and
augmentation policy come from the selected fixed baseline.

## Outputs and reproducibility

A run writes the following files to `--output_dir`:

- `<run>.pt`: model weights, method name, trace window, input length, and the
  complete fixed baseline preset;
- `<run>_loss.npy`: one loss value per epoch;
- `<run>_config.json`: run controls, method metadata, and the saved preset.

The default seed is 47. The run name includes the method, trace window, trace
count, and fixed epoch count.

## Evaluation

The repository contains two downstream evaluation entry points:

- `src/test.py`: non-profiling KNN distinguisher evaluation;
- `src/test_profiling_attack.py`: profiling attack with a logistic-regression
  classifier and key-rank evaluation.

Both reconstruct the encoder from the checkpoint method name through
`build_ssl_model`, so they use the fixed baseline rather than re-reading
architecture values from command-line arguments.

> Note: these evaluation scripts currently depend on the legacy
> `src.datasets.ascad_loader` and `src.utils.cli_parsers` helpers. Ensure those
> helpers are present in your checkout before running the evaluation commands.

## Repository layout

```text
src/
  train.py                 Fixed-baseline SSL training entry point
  methods/ssl_factory.py   Single source of truth for method presets
  methods/                 Method implementations and SSL training loops
  models/model_zoo.py      Shared 1-D CNN and trace-transformer backbones
  datasets/data_loader.py  HDF5/NPZ ASCAD loading and trace-window parsing
  test.py                  Non-profiling KNN evaluation
  test_profiling_attack.py Profiling attack evaluation
weekly_updates/            Historical experiment notes and figures
outputs/                   Generated checkpoints, losses, and evaluation artifacts
```

## Research record

The fixed values capture the current ASCAD 1-D baselines recorded in
`weekly_updates/`. Keep a new result, changed preset, dataset revision, and
random seed together so comparisons remain reproducible.
