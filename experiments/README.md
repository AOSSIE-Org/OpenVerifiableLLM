# Experiments

This directory contains reproducibility experiments for the OpenVerifiableLLM project.

Purpose:
To validate assumptions about deterministic training, checkpoint reproducibility,
and dataset sampling.

Experiments included:

checkpoint_reproducibility
Tests whether identical training runs produce identical checkpoint hashes.

determinism_tests
Tests whether setting fixed seeds results in deterministic outputs.

dataset_subset_tests
Tests reproducible sampling of dataset subsets.