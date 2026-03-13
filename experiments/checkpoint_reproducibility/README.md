# Checkpoint Reproducibility Experiment

## Motivation
During discussion in Discord, the question was raised whether checkpoint hashing alone could verify the integrity of a training run.

Before building a full verification pipeline, we validate the assumption that identical deterministic training runs produce identical model checkpoints.

## Experiments
**A — Deterministic run** Two training runs with identical configuration.  
*Expected:* identical checkpoint hashes.

**B — Seed variation** Training runs with different seeds.  
*Expected:* different checkpoint hashes.

**C — Checkpoint tampering** Manually modify checkpoint.  
*Expected:* verification fails.

## Model
Tiny feed-forward network (to keep runtime small).

## Dataset
Synthetic deterministic dataset.

## Run
```bash
python run_experiment.py