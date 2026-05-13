"""
openverifiablellm.eval
======================

Evaluation framework for OpenVerifiableLLM.

Provides pluggable evaluators for perplexity, bias, and benchmark tasks,
all built on top of the abstract ``BaseEvaluator`` interface.

Available evaluators
--------------------
- :class:`~openverifiablellm.eval.perplexity.PerplexityEvaluator`
    Measures cross-entropy perplexity on a held-out text corpus.
- :class:`~openverifiablellm.eval.bias.BiasEvaluator`
    Bias-testing stub (WinoBias / BBQ — integration pending).
- :class:`~openverifiablellm.eval.benchmarks.BenchmarkEvaluator`
    MMLU / factual-accuracy stub (lm-eval-harness — integration pending).
"""

from .base import BaseEvaluator
from .benchmarks import BenchmarkEvaluator
from .bias import BiasEvaluator
from .perplexity import PerplexityEvaluator

__all__ = [
    "BaseEvaluator",
    "PerplexityEvaluator",
    "BiasEvaluator",
    "BenchmarkEvaluator",
]
