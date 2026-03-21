"""
openverifiablellm/eval/bias/wino_bias.py

Gender-bias evaluator using the WinoBias benchmark.
"""

from typing import Optional

from ..base import BaseEvaluator
from ..perplexity import PerplexityEvaluator


class WinoBiasEvaluator(BaseEvaluator):
    """
    Evaluates gender bias in a language model using the WinoBias benchmark.

    For each sentence pair (pro-stereotype / anti-stereotype) the model's
    perplexity is computed via the same sliding-window method used by
    :class:`PerplexityEvaluator`.  A lower ``bias_score`` indicates a less
    biased model.

    Parameters
    ----------
    n_samples : int or None
        Maximum number of sentences to load from each WinoBias split.
        ``None`` evaluates the full dataset.  Default ``None``.
    """

    def __init__(self, n_samples: Optional[int] = None):
        self.n_samples = n_samples

    def evaluate(self, model, tokenizer) -> dict:
        """
        Compute stereotype and anti-stereotype perplexity scores.

        Loads ``type1_pro`` (pro-stereotype) and ``type1_anti``
        (anti-stereotype) splits of WinoBias and measures how much more
        easily the model predicts gender-stereotypical sentences than
        counter-stereotypical ones.

        Parameters
        ----------
        model : callable
            ``model(input_ids) -> 2-D sequence`` of shape
            ``(len(input_ids), vocab_size)``, as described in
            :meth:`PerplexityEvaluator.compute_sentence_perplexity`.
        tokenizer : object
            Object with ``encode(text: str) -> list[int]``.

        Returns
        -------
        dict
            A dictionary with the following keys:

            * **stereotype_score** (*float*) — mean perplexity on
              pro-stereotype sentences.
            * **anti_stereotype_score** (*float*) — mean perplexity on
              anti-stereotype sentences.
            * **bias_score** (*float*) —
              ``abs(stereotype_score - anti_stereotype_score)``;
              lower means less biased.
        """
        import datasets as hf_datasets  # deferred; runtime dep

        pro_ds = hf_datasets.load_dataset("wino_bias", "type1_pro", split="test")
        anti_ds = hf_datasets.load_dataset("wino_bias", "type1_anti", split="test")

        def _score_split(dataset) -> float:
            scores = []
            for i, row in enumerate(dataset):
                if self.n_samples is not None and i >= self.n_samples:
                    break
                tokens = row.get("tokens", [])
                text = " ".join(tokens) if isinstance(tokens, list) else str(tokens)
                if not text.strip():
                    continue
                token_ids = tokenizer.encode(text)
                scores.append(
                    PerplexityEvaluator.compute_sentence_perplexity(model, token_ids)
                )
            return float(sum(scores) / len(scores)) if scores else float("inf")

        stereotype_score = _score_split(pro_ds)
        anti_stereotype_score = _score_split(anti_ds)
        bias_score = abs(stereotype_score - anti_stereotype_score)

        return {
            "stereotype_score": stereotype_score,
            "anti_stereotype_score": anti_stereotype_score,
            "bias_score": bias_score,
        }
