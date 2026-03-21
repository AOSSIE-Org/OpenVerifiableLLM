from abc import ABC, abstractmethod


class BaseEvaluator(ABC):
    """Abstract base class for all dataset evaluators."""

    @abstractmethod
    def evaluate(self, model, tokenizer) -> dict:
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
