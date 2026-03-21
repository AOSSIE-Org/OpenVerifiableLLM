"""
openverifiablellm.eval.perplexity
===================================

Token-level perplexity measurement on a held-out text corpus.

Perplexity is defined as::

    PPL = exp( -1/N * sum_i log P(token_i | context_i) )

where *N* is the total number of tokens in the evaluation corpus.  This is the
standard metric for language-model quality and is directly relevant to Wikipedia
pre-training: a lower perplexity indicates that the model assigns higher
probability to held-out Wikipedia text.

Usage
-----
::

    from openverifiablellm.eval.perplexity import PerplexityEvaluator

    evaluator = PerplexityEvaluator(text="The quick brown fox ...", stride=128)
    results = evaluator.evaluate(model=my_model, tokenizer=my_tokenizer)
    print(results)  # {"perplexity": 45.3, "nll_bits_per_byte": 2.1, "n_tokens": 512}

The ``model`` callable must accept a list of integer token IDs and return a
list of per-token log-probabilities (log P(token | prefix)) for every position.
This signature is intentionally simple so that tiny mock models work in tests
without requiring a GPU or a full transformer stack.
"""

import logging
import math
from typing import Callable, Dict, List, Sequence

from .base import BaseEvaluator

logger = logging.getLogger(__name__)

# Maximum sequence length forwarded through the model in a single call.
# Keeping this small allows evaluation on CPU with tiny mock models.
DEFAULT_MAX_LENGTH: int = 512

# Stride used for the sliding-window approach when the corpus is longer than
# max_length.  Tokens in the overlap zone are scored only once (by the later
# window), which avoids inflating perplexity near window boundaries.
DEFAULT_STRIDE: int = 256


def _sliding_window_nll(
    token_ids: List[int],
    model: Callable[[List[int]], List[float]],
    max_length: int,
    stride: int,
) -> tuple:
    """Compute total negative log-likelihood using a sliding window.

    Parameters
    ----------
    token_ids :
        Full list of token IDs for the evaluation corpus.
    model :
        Callable mapping a token-ID list to per-token log-probabilities.
        ``model(ids)[i]`` is log P(ids[i] | ids[:i]).
    max_length :
        Maximum number of tokens forwarded to the model at once.
    stride :
        How many tokens the window advances between calls.

    Returns
    -------
    (total_nll, n_scored_tokens) : tuple of (float, int)
        Aggregate negative log-likelihood and the number of tokens scored.
    """
    if max_length <= 0:
        raise ValueError("max_length must be a positive integer")
    if stride <= 0:
        raise ValueError("stride must be a positive integer")
    if stride > max_length:
        raise ValueError("stride must not exceed max_length")

    n = len(token_ids)
    if n == 0:
        return 0.0, 0

    total_nll = 0.0
    n_scored = 0
    start = 0

    while start < n:
        end = min(start + max_length, n)
        window = token_ids[start:end]

        # The first token in the very first window has no context → skip it.
        # For subsequent windows the overlap is max_length - stride tokens;
        # we only score the *new* tokens (the last stride tokens of the window).
        if start == 0:
            # Score positions 1 … end-1 (position 0 has no left context)
            log_probs: List[float] = model(window)
            for pos in range(1, len(window)):
                total_nll -= log_probs[pos]
                n_scored += 1
        else:
            # Only score tokens beyond the overlap
            log_probs = model(window)
            overlap = max_length - stride
            for pos in range(overlap, len(window)):
                total_nll -= log_probs[pos]
                n_scored += 1

        if end == n:
            break
        start += stride

    return total_nll, n_scored


class PerplexityEvaluator(BaseEvaluator):
    """Evaluate a language model's perplexity on a held-out text corpus.

    Parameters
    ----------
    text : str
        The held-out text to evaluate on.  Typically a few thousand words of
        Wikipedia-style prose.
    max_length : int
        Maximum context window passed to the model per forward call.
        Default: 512.
    stride : int
        Sliding window stride.  Must be ≤ ``max_length``.
        Default: 256.
    name : str
        Evaluator name used in reports.
    """

    def __init__(
        self,
        text: str,
        max_length: int = DEFAULT_MAX_LENGTH,
        stride: int = DEFAULT_STRIDE,
        name: str = "perplexity",
    ) -> None:
        super().__init__(name=name)

        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string")
        if max_length <= 0:
            raise ValueError("max_length must be a positive integer")
        if stride <= 0:
            raise ValueError("stride must be a positive integer")
        if stride > max_length:
            raise ValueError("stride must not exceed max_length")

        self.text = text
        self.max_length = max_length
        self.stride = stride

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model: Callable[[Sequence[int]], Sequence[float]],
        tokenizer: Callable[[str], Sequence[int]],
    ) -> Dict[str, float]:
        """Compute perplexity metrics.

        Parameters
        ----------
        model :
            ``model(token_ids) → log_probs``

            Receives a list of integer token IDs and returns a list of the same
            length where ``log_probs[i]`` is the natural-log probability of
            ``token_ids[i]`` given all preceding tokens.  The value at position
            0 is ignored (no left context).
        tokenizer :
            ``tokenizer(text) → token_ids``

            Maps a raw string to a list of integer token IDs.

        Returns
        -------
        dict with keys:
            - ``"perplexity"`` – exp(mean NLL) — lower is better.
            - ``"nll_bits_per_byte"`` – NLL in bits per *byte* of input text,
              a length-independent measure.
            - ``"n_tokens"`` – number of tokens scored.
            - ``"n_bytes"`` – length of the input text in UTF-8 bytes.
        """
        token_ids: List[int] = list(tokenizer(self.text))

        if len(token_ids) == 0:
            logger.warning("Tokenizer produced empty output; returning infinite perplexity")
            return {
                "perplexity": float("inf"),
                "nll_bits_per_byte": float("inf"),
                "n_tokens": 0,
                "n_bytes": len(self.text.encode("utf-8")),
            }

        total_nll, n_scored = _sliding_window_nll(
            token_ids=token_ids,
            model=model,
            max_length=self.max_length,
            stride=self.stride,
        )

        if n_scored == 0:
            # Edge case: single-token corpus — nothing to score.
            logger.warning("No tokens were scored (corpus too short); returning perplexity=1.0")
            return {
                "perplexity": 1.0,
                "nll_bits_per_byte": 0.0,
                "n_tokens": len(token_ids),
                "n_bytes": len(self.text.encode("utf-8")),
            }

        mean_nll = total_nll / n_scored
        perplexity = math.exp(mean_nll)

        n_bytes = len(self.text.encode("utf-8"))
        # Convert nats → bits (log2(e) ≈ 1.4427); divide by byte count.
        nll_bits_per_byte = (total_nll * math.log2(math.e)) / n_bytes if n_bytes > 0 else 0.0

        logger.info(
            "Perplexity: %.4f  (NLL=%.4f, tokens=%d, bytes=%d)",
            perplexity,
            mean_nll,
            n_scored,
            n_bytes,
        )

        return {
            "perplexity": perplexity,
            "nll_bits_per_byte": nll_bits_per_byte,
            "n_tokens": n_scored,
            "n_bytes": n_bytes,
        }

    # ------------------------------------------------------------------
    # Convenience helpers (static, so they are easy to unit-test in isolation)
    # ------------------------------------------------------------------

    @staticmethod
    def uniform_model(vocab_size: int) -> Callable[[Sequence[int]], List[float]]:
        """Return a trivial model that assigns uniform probability to all tokens.

        Useful for smoke tests.  The perplexity of a uniform model on any
        corpus should equal ``vocab_size`` exactly.

        Parameters
        ----------
        vocab_size : int
            Size of the vocabulary.

        Returns
        -------
        Callable
        """
        if vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")

        log_prob = math.log(1.0 / vocab_size)

        def _model(token_ids: Sequence[int]) -> List[float]:
            return [log_prob] * len(token_ids)

        return _model
