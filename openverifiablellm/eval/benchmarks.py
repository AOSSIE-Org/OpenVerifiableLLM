"""
openverifiablellm.eval.benchmarks
====================================

Standard LLM benchmark evaluation stub for OpenVerifiableLLM.

This module provides the :class:`BenchmarkEvaluator` class, which wraps
established NLP benchmarks commonly used to compare language models.

Planned benchmarks
------------------
* **MMLU** (Hendrycks et al., 2021) — Massive Multitask Language Understanding.
  57 academic subjects, 4-way multiple-choice.  Evaluates broad knowledge and
  reasoning across STEM, humanities, social sciences, and more.

* **TriviaQA** (Joshi et al., 2017) — Factual accuracy benchmark with
  trivia-style questions and supporting evidence passages.  A random subset
  (e.g., 1 000 questions) is used for fast evaluation.

Integration is pending a stable lm-eval-harness dependency.  The class
skeleton is provided now so that downstream code can import and type-check
:class:`BenchmarkEvaluator` without error.

TODO
----
* Integrate MMLU via ``lm_eval.tasks``::

      lm_eval --model hf \
              --model_args pretrained=<model_path> \
              --tasks mmlu \
              --device cpu \
              --output_path results/

* Integrate TriviaQA via ``lm_eval.tasks`` or HuggingFace ``datasets``.
* Cache downloaded datasets locally to avoid redundant network traffic.
* Return metrics: ``mmlu_accuracy``, ``triviaqa_exact_match``,
  ``per_subject_accuracy`` dict.
"""

import logging
from typing import Any, Callable, Dict

from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class BenchmarkEvaluator(BaseEvaluator):
    """Evaluate a language model on standard NLP benchmarks (stub).

    Parameters
    ----------
    benchmark : {"mmlu", "triviaqa"}
        Which benchmark to run.
    n_samples : int or None
        Number of examples to evaluate on.  ``None`` means the full benchmark.
        Set a small value (e.g., 100) for rapid iteration during development.
    name : str
        Evaluator name used in reports.

    Notes
    -----
    This class is intentionally a stub.  Calling :meth:`evaluate` will raise
    :class:`NotImplementedError` until the benchmark integration is complete.
    See module docstring for the planned implementation.
    """

    SUPPORTED_BENCHMARKS = ("mmlu", "triviaqa")

    def __init__(
        self,
        benchmark: str = "mmlu",
        n_samples: int = None,
        name: str = "benchmark",
    ) -> None:
        super().__init__(name=name)

        if benchmark not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark '{benchmark}'. "
                f"Choose from: {self.SUPPORTED_BENCHMARKS}"
            )

        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be a positive integer or None")

        self.benchmark = benchmark
        self.n_samples = n_samples

    def evaluate(
        self,
        model: Callable[..., Any],
        tokenizer: Callable[..., Any],
    ) -> Dict[str, float]:
        """Run the benchmark evaluation.

        .. note::
            Not yet implemented.  Raises :class:`NotImplementedError`.

        Parameters
        ----------
        model :
            Language model callable.
        tokenizer :
            Tokenizer callable.

        Raises
        ------
        NotImplementedError
            Always, until MMLU/TriviaQA integration is complete.
        """
        # TODO: implement MMLU via lm-eval-harness task registry
        # TODO: implement TriviaQA via HuggingFace datasets + exact-match scorer
        raise NotImplementedError(
            f"BenchmarkEvaluator ({self.benchmark}) is not yet implemented. "
            "See openverifiablellm/eval/benchmarks.py for the integration plan."
        )
