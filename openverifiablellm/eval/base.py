from abc import ABC, abstractmethod
from typing import List

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol, runtime_checkable


@runtime_checkable
class Model(Protocol):
    """Structural type for a language model callable."""

    def __call__(self, input_ids: List[int]) -> List[List[float]]: ...


@runtime_checkable
class Tokenizer(Protocol):
    """Structural type for a tokenizer."""

    def encode(self, text: str) -> List[int]: ...


class BaseEvaluator(ABC):
    """Abstract base class for all dataset evaluators."""

    @abstractmethod
    def evaluate(self, model: Model, tokenizer: Tokenizer) -> dict:
        """
        Evaluate a language model using the given tokenizer.

        Parameters
        ----------
        model : callable
            Callable accepting a sequence of token IDs and returning a
            2-D sequence of logits with shape ``(len(input_ids), vocab_size)``.
        tokenizer : object
            Object with an ``encode(text: str) -> list[int]`` method.

        Returns
        -------
        dict
            Benchmark-specific evaluation results.
        """
