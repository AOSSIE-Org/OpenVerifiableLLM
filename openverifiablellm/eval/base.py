"""
openverifiablellm.eval.base
============================

Abstract base class for all evaluation strategies.

All concrete evaluators must subclass :class:`BaseEvaluator` and implement
:meth:`evaluate`, which receives a model callable and a tokenizer callable
and returns a flat ``dict`` of metric names to scalar values.

Example
-------
::

    class MyEvaluator(BaseEvaluator):
        def evaluate(self, model, tokenizer):
            # ... compute metrics ...
            return {"my_metric": 42.0}
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict


class BaseEvaluator(ABC):
    """Abstract base class for LLM evaluators.

    Parameters
    ----------
    name : str
        Human-readable identifier for this evaluator (used in reports).
    """

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def evaluate(
        self,
        model: Callable[..., Any],
        tokenizer: Callable[..., Any],
    ) -> Dict[str, float]:
        """Run the evaluation and return a metric dictionary.

        Parameters
        ----------
        model :
            A callable that accepts token sequences and returns log-probabilities
            or logits.  The exact signature is determined by the concrete
            evaluator subclass.
        tokenizer :
            A callable that maps a string to a sequence of integer token IDs.

        Returns
        -------
        dict
            Mapping of metric name → scalar value.  All values must be
            JSON-serialisable floats.
        """
