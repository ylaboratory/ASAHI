<img src="assets/asahi_logo.png" alt="ASAHI icon" width="150"/>


**ASAHI**: Attention-based Self-supervised Architecture for High-fidelity methylome Inference

This is the code for *Recovering high-fidelity DNA methylation profiles from low-depth whole-genome bisulfite sequencing data with ASAHI*.


## Overview

Whole-genome bisulfite sequencing (WGBS) libraries are often sequenced at low depth to limit cost, leading to missing or noisy per-CpG methylation estimates. Minimum read-depth thresholds are commonly recommended for reliable downstream analyses, such as differentially methylated region (DMR) calling. However, filtering poorly covered CpGs can discard informative sites, whereas retaining them risks introducing unreliable measurements. Existing denoising and imputation methods can improve low-depth methylomes but often fail to preserve genuine biological variation. We introduce ASAHI, a slice-based, self-supervised transformer model that jointly denoises and imputes genome-wide methylation profiles from low-depth WGBS data. Using low-depth profiles simulated from a large, deeply sequenced cohort spanning 39 cell types, we demonstrate that ASAHI outperforms state-of-the-art baselines across sequencing depths when evaluated on unseen donors and cell types. ASAHI also generalizes to an independent WGBS dataset without any retraining. Crucially, ASAHI restores methylation variance toward matched high-depth reference levels and better preserves methylation patterns at cell-type-specific DMRs. These results establish ASAHI as a framework for recovering high-fidelity methylation profiles from shallow WGBS libraries, enabling more reliable biological discovery.

This repository provides:
1. The ASAHI model, training and inference code.
2. The pipeline from FASTQ to the per-chromosome matrices the model reads.
3. Example ENCODE data for chromosome 21 at full depth and 4 low depths.
4. The trained models behind the paper's ENCODE results, for chromosomes 1, 11 and 21.
5. Example notebooks for data preparation, training and inference.


## Quick Navigation

- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Data and Preprocessing](#data-and-preprocessing)
- [Running the Examples](#running-the-examples)
  - [Data creation](#data-creation)
  - [Inference with a Pretrained Model](#inference-with-a-pretrained-model)
  - [Training a Model](#training-a-model)
- [Reproducibility Notes](#reproducibility-notes)
- [Citation](#citation)

---

## Installation

```bash
# Clone the repository:
git clone https://github.com/ylaboratory/ASAHI.git
cd ASAHI

# Create and activate a Python environment:
conda create -n asahi python=3.11
conda activate asahi

# Install dependencies (add jupyter to run the notebooks):
pip install -r requirements.txt
pip install jupyter
```

> This repository is not a published Python package.
All scripts and notebooks assume execution from the repository root.

## Repository Structure
```text
ASAHI/
├── asahi/                  # Model, training, inference and data-processing code
├── assets/                 # FASTQ -> strand-merged CpG files, and read downsampling
├── checkpoints/            # Trained models, one per chromosome
├── configs/                # Training configurations
├── data/
│   ├── encode_subset/      # Strand-merged CpG files, 12 ENCODE libraries, chr21
│   ├── encode_pt/          # chr21 matrices at full depth
│   ├── encode_pt_frac*/    # chr21 matrices at 4 low depths
│   └── metadata/           # CpG index, encoded chr21 genome, ENCODE accessions, example split
├── notebooks/              # Example notebooks
├── scripts/                # Command-line training, inference, scoring and data building
├── results/                # Output storage
└── requirements.txt
```

## Data and Preprocessing

### Bioinformatics pipeline

- `assets/bioinfopipe_pairedWGBS.sh` goes from FASTQ to strand-merged CpG files: Trim Galore, Bismark alignment, deduplication, methylation extraction and `coverage2cytosine --merge_CpG`. Commands for single-end data are in its header. 3 of the 12 ENCODE libraries are single-end.
- `assets/downsample_bam.sh` makes the low-depth versions: `samtools view` on the deduplicated BAM at fractions 0.1, 0.2, 0.4 and 0.6, followed by the same methylation calling. Run the first script with `--keep-intermediates` so the BAM is kept.

Accessions, biosamples, donors and FASTQ links are in `data/metadata/encode_libraries.tsv`.

### Matrices

`scripts/build_matrices.py` turns one directory of per-sample CpG files into the input the model reads:

```text
data/<dataset>/chrom_centric/true/
├── FractionalMethylation/<chrom>.fm    # betas in [0, 1]
└── ReadDepth/<chrom>.cov               # read depth
```

Both are tab-separated, with one row per canonical hg38 CpG (index column `start`, 0-based), one column per library, and empty cells where a CpG was not observed. They may be gzipped.

### Reference metadata

Located under `data/metadata/`:

- `hg38_cpg_py0idx.csv.gz`: canonical CpGs of chr1, chr11 and chr21.
- `hg38_encoded_chr21.npz`: encoded chr21 sequence. Other chromosomes can be made with `scripts/encode_genome.py` from an hg38 FASTA.
- `encode_libraries.tsv`: the 12 ENCODE libraries.
- `encode_demo_split.json`: the training, validation and held-out libraries of the example model.

## Running the Examples

### Data creation

`notebooks/data_prep.ipynb` builds the chr21 full-depth matrices from `data/encode_subset/` and checks them against the included ones.

```bash
python scripts/build_matrices.py --cov-dir data/encode_subset --dataset encode_pt_rebuilt --chrom chr21
```

### Inference with a Pretrained Model

`notebooks/example_infer.ipynb` predicts all 12 ENCODE libraries at 3x with `checkpoints/asahi_chr21.pth` and scores them against full depth. This takes about 7 minutes on a GPU.

```bash
python scripts/inference.py --checkpoint checkpoints/asahi_chr21.pth --chrom chr21 \
    --anchor-dataset encode_pt_frac0.1_seed11 --support-dataset encode_pt \
    --out results/encode_pt_frac0.1_seed11_chr21.csv

python scripts/evaluate.py --preds results/encode_pt_frac0.1_seed11_chr21.csv --chrom chr21 \
    --input-dataset encode_pt_frac0.1_seed11 --gt-dataset encode_pt
```

The output CSV has one row per CpG (`start`) and one column per library, with predicted betas in [0, 1].

### Training a Model

`notebooks/example_train.ipynb` trains an example model on ENCODE chr21 with early stopping, saves it under `checkpoints/`, and predicts the 2 held-out libraries.

```bash
python scripts/train.py --config example_train
```

Each epoch takes about 30 seconds on a GPU. A GPU is strongly recommended.

## Reproducibility Notes

- Exact Python dependencies are specified in `requirements.txt`.
- Inference draws support rows from a per-item seed, so its output does not depend on the number of workers. Small differences in the last digits can occur between hardware.
- A training seed fixes the initial weights and the slice composition. GPU training is not guaranteed to be bit-identical between runs.
- The DNA branch reads an encoded hg38 sequence. The included chr21 was checked to be identical to the encoding of an Ensembl-style GRCh38 FASTA with headers such as `21`, for which `scripts/encode_genome.py` takes `--prepend-chr`.

## Citation
Coming Soon
