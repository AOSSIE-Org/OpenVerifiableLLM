"""
openverifiablellm.eval.bias
=============================

Bias evaluation stub for OpenVerifiableLLM.

This module provides the :class:`BiasEvaluator` class, which is intended to
measure social bias in a language model using established benchmarks.

Planned benchmarks
------------------
* **WinoBias** (Zhao et al., 2018) — coreference-resolution pairs that reveal
  occupational gender bias.  Each example has a pro-stereotypical and an
  anti-stereotypical version; a fair model should perform equally on both.

* **BBQ** (Parrish et al., 2022) — a question-answering dataset covering nine
  social-bias dimensions (age, disability status, gender identity, nationality,
  physical appearance, race/ethnicity, religion, SES, sexual orientation).

Integration is pending a stable lm-eval-harness dependency.  The class
skeleton is provided now so that downstream code can import and type-check
:class:`BiasEvaluator` without error.

TODO
----
* Integrate WinoBias evaluation via HuggingFace ``datasets``.
* Integrate BBQ via ``lm_eval.tasks`` (lm-eval-harness).
* Implement ``_score_pair()`` helper that forwards pro/anti pairs through the
  model and computes the accuracy gap.
* Return bias metrics: ``gender_bias_score``, ``bbq_accuracy``,
  ``per_category_bias`` dict.
"""

import logging
from typing import Any, Callable, Dict

from .base import BaseEvaluator

logger = logging.getLogger(__name__)


class BiasEvaluator(BaseEvaluator):
    """Evaluate social bias in a language model (stub).

    Parameters
    ----------
    benchmark : {"winobias", "bbq"}
        Which bias benchmark to use.
    name : str
        Evaluator name used in reports.

    Notes
    -----
    This class is intentionally a stub.  Calling :meth:`evaluate` will raise
    :class:`NotImplementedError` until the benchmark integration is complete.
    See module docstring for the planned implementation.
    """

    SUPPORTED_BENCHMARKS = ("winobias", "bbq")

    def __init__(
        self,
        benchmark: str = "winobias",
        name: str = "bias",
    ) -> None:
        super().__init__(name=name)

        if benchmark not in self.SUPPORTED_BENCHMARKS:
            raise ValueError(
                f"Unsupported benchmark '{benchmark}'. "
                f"Choose from: {self.SUPPORTED_BENCHMARKS}"
            )

        self.benchmark = benchmark

    def evaluate(
        self,
        model: Callable[..., Any],
        tokenizer: Callable[..., Any],
    ) -> Dict[str, float]:
        """Run bias evaluation.

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
            Always, until WinoBias/BBQ integration is complete.
        """
        # TODO: implement WinoBias via HuggingFace datasets
        # TODO: implement BBQ via lm-eval-harness task registry
        raise NotImplementedError(
            f"BiasEvaluator ({self.benchmark}) is not yet implemented. "
            "See openverifiablellm/eval/bias.py for the integration plan."
        )
