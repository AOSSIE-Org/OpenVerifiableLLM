"""
openverifiablellm/eval/perplexity.py

Perplexity evaluator for language models.
"""

import math
from typing import List, Optional

from .base import BaseEvaluator


class PerplexityEvaluator(BaseEvaluator):
    """
    Evaluates language-model perplexity on a HuggingFace benchmark dataset.

    Perplexity is computed with a teacher-forced sliding-window approach:
    for each token position *i* the model receives tokens ``[0 .. i-1]``
    and the negative log-probability of token ``[i]`` is accumulated.
    The final perplexity is ``exp(mean_NLL)``.

    Parameters
    ----------
    benchmark : str
        HuggingFace dataset identifier.  Default ``"wikitext"``.
    n_samples : int or None
        Maximum number of non-empty samples to evaluate.  ``None`` means
        evaluate the whole dataset.  Default ``50``.
    stride : int
        Window stride used when the sequence exceeds the model's context
        window.  Default ``512``.
    """

    def __init__(
        self,
        benchmark: str = "wikitext",
        n_samples: Optional[int] = 50,
        stride: int = 512,
        split: Optional[str] = None,
    ):
        self.benchmark = benchmark
        self.n_samples = n_samples
        self.stride = stride
        self.split = split

    # ------------------------------------------------------------------
    # Mock helpers
    # ------------------------------------------------------------------

    @staticmethod
    def uniform_model(vocab_size: int = 1000):
        """
        Return a mock model that produces uniform (all-zero) logits.

        Useful for unit testing: because all logits are equal, the
        log-softmax is ``-log(vocab_size)`` at every position, giving a
        predictable perplexity of exactly ``vocab_size``.

        Parameters
        ----------
        vocab_size : int
            Vocabulary size of the mock model.  Default ``1000``.

        Returns
        -------
        callable
            ``model(input_ids) -> list[list[float]]`` of shape
            ``(len(input_ids), vocab_size)``.
        """

        def _model(input_ids):
            return [[0.0] * vocab_size for _ in input_ids]

        return _model

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_sentence_perplexity(model, token_ids: List[int]) -> float:
        """
        Compute the perplexity of *token_ids* under *model*.

        Parameters
        ----------
        model : callable
            ``model(input_ids) -> 2-D sequence`` of shape
            ``(len(input_ids), vocab_size)``.
        token_ids : list[int]
            Tokenised sentence.

        Returns
        -------
        float
            Perplexity (≥ 1).  Returns ``float("inf")`` for sequences
            shorter than 2 tokens.
        """
        if len(token_ids) < 2:
            return float("inf")

        inputs = token_ids[:-1]
        targets = token_ids[1:]

        logits_batch = model(inputs)  # shape: (n-1, vocab_size)

        if len(logits_batch) != len(targets):
            raise ValueError(
                f"Model returned {len(logits_batch)} logit vectors but expected "
                f"{len(targets)} (one per target token)."
            )

        nll_sum = 0.0
        for logits, target in zip(logits_batch, targets):
            # numerically-stable log-softmax
            max_l = max(logits)
            exp_shifted = [math.exp(v - max_l) for v in logits]
            log_sum = math.log(sum(exp_shifted))
            log_prob_target = (logits[target] - max_l) - log_sum
            nll_sum -= log_prob_target

        return math.exp(nll_sum / len(targets))

    @staticmethod
    def compute_sequence_perplexity(model, token_ids: List[int], stride: int = 512) -> float:
        """
        Compute perplexity over a (possibly long) sequence using non-overlapping
        stride-sized windows.

        The sequence is partitioned into windows of *stride* tokens.  Each
        window contributes its token predictions to a pooled NLL.  The final
        perplexity is ``exp(total_NLL / total_scored_tokens)``.

        For sequences shorter than *stride* + 1 tokens the result is
        identical to :meth:`compute_sentence_perplexity`.

        Parameters
        ----------
        model : callable
            ``model(input_ids) -> 2-D sequence`` of shape
            ``(len(input_ids), vocab_size)``.
        token_ids : list[int]
            Tokenised sequence.
        stride : int
            Number of tokens scored per window.  Default ``512``.

        Returns
        -------
        float
            Perplexity (≥ 1).  Returns ``float("inf")`` for sequences
            shorter than 2 tokens.
        """
        if len(token_ids) < 2:
            return float("inf")

        nll_sum = 0.0
        n_scored = 0
        n = len(token_ids)

        for start in range(0, n - 1, stride):
            end = min(start + stride + 1, n)
            window = token_ids[start:end]
            if len(window) < 2:
                break
            inputs = window[:-1]
            targets = window[1:]
            logits_batch = model(inputs)
            if len(logits_batch) != len(targets):
                raise ValueError(
                    f"Model returned {len(logits_batch)} logit vectors but expected "
                    f"{len(targets)} (one per target token)."
                )
            for logits, target in zip(logits_batch, targets):
                max_l = max(logits)
                exp_shifted = [math.exp(v - max_l) for v in logits]
                log_sum = math.log(sum(exp_shifted))
                nll_sum -= (logits[target] - max_l) - log_sum
                n_scored += 1

        return math.exp(nll_sum / n_scored) if n_scored > 0 else float("inf")

    # ------------------------------------------------------------------
    # BaseEvaluator interface
    # ------------------------------------------------------------------

    def evaluate(self, model, tokenizer) -> dict:
        """
        Compute mean perplexity on *self.benchmark*.

        Parameters
        ----------
        model : callable
            Callable as described in :meth:`compute_sentence_perplexity`.
        tokenizer : object
            Object with ``encode(text: str) -> list[int]``.

        Returns
        -------
        dict
            ``{"perplexity": float}`` — mean perplexity across evaluated
            sentences.
        """
        import datasets as hf_datasets  # deferred; runtime dep

        if self.split is not None:
            ds = hf_datasets.load_dataset(self.benchmark, split=self.split, streaming=True)
        else:
            _splits_to_try = ("test", "validation", "train")
            for _s in _splits_to_try:
                try:
                    ds = hf_datasets.load_dataset(self.benchmark, split=_s, streaming=True)
                    break
                except Exception:
                    continue
            else:
                raise ValueError(
                    f"Dataset {self.benchmark!r} has none of the expected splits: "
                    f"{_splits_to_try}. Pass split= explicitly."
                )
        scores = []
        for row in ds:
            text = row.get("text", "")
            if not text.strip():
                continue
            if self.n_samples is not None and len(scores) >= self.n_samples:
                break
            token_ids = tokenizer.encode(text)
            scores.append(self.compute_sequence_perplexity(model, token_ids, self.stride))

        mean_ppl = float(sum(scores) / len(scores)) if scores else float("inf")
        return {"perplexity": mean_ppl}
