# singlecell-CVAE

A conditional variational autoencoder (CVAE) for generating single-cell transcriptional profiles of *Streptococcus pneumoniae* under cefepime treatment, conditioned on antibiotic dose and exposure time.

## Contents

- [Overview](#overview)
- [Model](#model)
- [Data](#data)
- [Installation](#installation)
- [Usage](#usage)
- [Repository structure](#repository-structure)

## Overview

Given a dose and timepoint, the model generates synthetic single-cell expression profiles. The encoder is trained adversarially to produce condition-invariant (basal) latent representations, and condition information is introduced only after latent sampling. This allows the model to generate cells for arbitrary dose–time combinations, including conditions absent from the training set.

## Model

![Model architecture](figures/model.png)

### Architecture

| Component | Description |
| --- | --- |
| Encoder | Two-layer MLP (ReLU) mapping log1p(CPM) expression to latent mean $\mu$ and log-variance $\log\sigma^2$ |
| Condition embeddings | Separate single-layer MLPs (SiLU) embed scalar dose and time into the latent dimension |
| Composition | Sequential FiLM layers apply condition-dependent scale and shift to the latent sample: $h = \gamma(c) \odot z + \beta(c)$ |
| Decoder | Two-layer MLP with Softplus output, mapping the composed latent to log1p(CPM) expression |
| Adversarial regressors | Linear regressors predict dose and time from $z$ through a gradient reversal layer |

### Objective

The model is trained end-to-end by minimizing

$$
\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \, D_{\text{KL}}\big(q(z \mid x) \,\|\, \mathcal{N}(0, I)\big) + \mathcal{L}_{\text{adv}},
$$

where

- $\mathcal{L}_{\text{recon}}$ is the mean squared error between the input and the reconstruction in log1p(CPM) space, with the reconstruction renormalized to a fixed library size of $10^6$;
- $\beta$ weights the KL divergence term;
- $\mathcal{L}_{\text{adv}}$ is the sum of the dose and time regression MSEs. Gradients from this term are reversed and scaled by $\lambda$ before reaching the encoder, so that the regressors learn to predict the condition while the encoder learns to remove condition information from $z$.

### Inference

1. Sample $z \sim \mathcal{N}(0, I)$.
2. Embed the target dose and time.
3. Compose the embeddings with $z$ via FiLM.
4. Decode to log1p(CPM) and convert to CPM.

### Default hyperparameters

| Parameter | Value |
| --- | --- |
| Hidden dimension | 256 |
| Latent dimension | 32 |
| Batch size | 16 |
| Epochs | 150 |
| Optimizer | Adam, learning rate $10^{-3}$ |
| KL weight $\beta$ | 0.1 |
| Adversarial weight $\lambda$ | 0.1 |
| Random seed | 111 |

## Data

The model is trained on approximately 80,000 *S. pneumoniae* scRNA-seq profiles collected across multiple cefepime doses (expressed as multiples of the MIC) and timepoints (0–4 h). Raw counts are converted to counts per million (CPM) and log1p-transformed before training.

**Evaluation split.** All cells from the 2× MIC, 2 h condition are held out as the test set to assess generalization to an unseen dose–time combination. The remaining cells are split 80/20 into training and validation sets.

**Bulk comparison.** The evaluation notebook additionally compares pseudobulk profiles derived from the single-cell data with bulk RNA-seq (TPM) profiles from a separate experiment.

> **Note:** The datasets are not included in this repository. See [Usage](#usage) for the expected data layout.

### Expected input format

`src/sc_data.get_sc_data` reads every file in a directory, where each file:

- is a tab-separated matrix with genes as rows and cells as columns;
- is named with a condition label encoding dose, drug, and timepoint (e.g. `2CEF2hr`), which is parsed by `src/metadata.py`.

## Installation

Requires Python 3.11 or later.

```bash
git clone https://github.com/eddykang06/singlecell-CVAE.git
cd singlecell-CVAE

conda create -n singlecell-cvae python=3.11 -y
conda activate singlecell-cvae

pip install -r requirements.txt
```

For GPU support, install the [PyTorch](https://pytorch.org/get-started/locally/) build matching your CUDA version before installing the requirements.

## Usage

The notebooks run locally or on Google Colab. Locally, they read from `data/` at the repository root (override with the `DATA_DIR` environment variable); on Colab, they read from Google Drive. The expected layout is:

```text
data/
├── single_cell/           # scRNA-seq count matrices, one file per condition
├── fcnts_timezero/        # Bulk featureCounts output (evaluation only)
└── CVAE_checkpoint.pt     # Written by 01-train.ipynb, read by 02-eval.ipynb
```

| Notebook | Purpose |
| --- | --- |
| [`01-train.ipynb`](notebooks/01-train.ipynb) | Loads data, constructs the train/validation/test split, trains the CVAE, plots loss curves, and saves a checkpoint |
| [`02-eval.ipynb`](notebooks/02-eval.ipynb) | Compares pseudobulk with bulk RNA-seq, generates cells for the held-out condition, computes pseudobulk Pearson and Spearman correlations against real cells, and visualizes the latent space with PCA |

### Minimal example

```python
from src.sc_data import get_sc_data
from src.train import train_cvae

data = get_sc_data(path="data/single_cell")
num_genes = data.columns.str.contains("SP").sum()

# train_df and val_df: see notebooks/01-train.ipynb for the split
model, *losses = train_cvae(
    train_df=train_df,
    val_df=val_df,
    batch_size=16,
    epochs=150,
    lr=1e-3,
    model_params={"input_dim": num_genes, "hidden_dim": 256, "latent_dim": 32},
    device="cuda",
    kl_weight=0.1,
    adv_weight=0.1,
)

# Generate 1,000 cells at 2x MIC, 2 h (returned in CPM)
model.eval()
cells = model.generate_samples(num_samples=1000, d=2, t=2)
```

## Repository structure

```text
singlecell-CVAE/
├── figures/
│   └── model.png          # Model architecture diagram
├── notebooks/
│   ├── 01-train.ipynb     # Training pipeline
│   └── 02-eval.ipynb      # Inference and evaluation
├── src/
│   ├── bulk_data.py       # Bulk RNA-seq loading (featureCounts → TPM)
│   ├── eda.py             # Pseudobulk vs. bulk comparison
│   ├── eval.py            # Latent-space PCA and evaluation plots
│   ├── metadata.py        # Condition-label parsing (drug, dose, timepoint)
│   ├── sc_data.py         # Single-cell loading, CPM normalization, data loaders
│   ├── train.py           # Loss functions and training loop
│   └── vae.py             # CVAE, condition embeddings, FiLM, adversarial components
└── requirements.txt
```
