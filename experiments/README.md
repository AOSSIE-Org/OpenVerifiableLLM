## Experiments

This directory contains small reproducible experiments used to validate assumptions behind the **OpenVerifiableLLM deterministic training pipeline**.

The goal of these experiments is to verify that:

- preprocessing produces deterministic outputs
- dataset tampering can be detected using Merkle roots
- small reproducible datasets can be used for testing the pipeline

These experiments are **not part of the main pipeline**. They are intended for testing ideas and validating reproducibility guarantees.

---

## Directory Structure

experiments/
│
├── data_subset/
│ ├── sample_wiki_generate.py
│ ├── sample_wiki.xml.bz2
│ └── tampered_sample_wiki.xml.bz2
│
├── preprocessing_determinism/
│ └── test_preprocessing.py
│
├── merkle_verification/
│ └── test_merkle.py
│
└── README.md

---

## Experiments includes

### 1. Preprocessing Determinism

Verifies that running the preprocessing pipeline multiple times on the same dataset produces identical outputs.

The experiment compares:

- `processed_sha256`
- `processed_merkle_root`
- `environment_hash`

If these values match across runs, the preprocessing step is deterministic.

Run:

```bash
python -m experiments.preprocessing_determinism.test_preprocessing experiments/data_subset/sample_wiki.xml.bz2
```

**Expected Results** -

```bash
Run 1 hash: ...
Run 2 hash: ...

Deterministic preprocessing confirmed 🎉
```

### 2. Merkle Root Tamper Detection

Tests whether dataset tampering is detected by comparing Merkle roots.

Two datasets are used:

sample_wiki.xml.bz2 (original)

tampered_sample_wiki.xml.bz2 (modified)

The experiment compares:

raw_merkle_root

processed_merkle_root

If either root differs, the tampering is successfully detected.

Run:

```bash
python -m experiments.merkle_verification.test_merkle --path1 experiments/data_subset/sample_wiki.xml.bz2 --path2 experiments/data_subset/tampered_sample_wiki.xml.bz2
```

**Expected Results** -

```bash
Run 1 RAW Merkle root: ...
Run 2 RAW Merkle root: ...

Tampering detected 🎉
```

### 3. Dataset Subset

The data_subset directory contains a minimal Wikipedia XML example used for quick experimentation without downloading full dumps.

This allows experiments to run quickly while still exercising the preprocessing pipeline.
