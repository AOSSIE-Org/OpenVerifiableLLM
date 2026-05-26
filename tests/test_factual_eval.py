"""
tests/test_factual_eval.py

Tests for WikipediaFactualEvaluator.

Run with:
    pytest tests/test_factual_eval.py -v
"""

import math
import random

import pytest

from openverifiablellm.eval.factual import WikipediaFactualEvaluator
from openverifiablellm.eval.perplexity import PerplexityEvaluator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_WIKI_TEXT = (
    "Albert Einstein was born in Germany.\n"
    "He developed the Theory of Relativity.\n"
    "Marie Curie was born in Poland.\n"
    "\n"
    "Isaac Newton discovered gravity in England.\n"
    "Newton worked at Cambridge University.\n"
)


class _MockTokenizer:
    """Tokenizer that maps each character to its ASCII code modulo 100."""

    def encode(self, text: str) -> list:
        return [ord(c) % 100 for c in text.replace(" ", "_")]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def wiki_file(tmp_path):
    p = tmp_path / "wiki_clean.txt"
    p.write_text(_WIKI_TEXT, encoding="utf-8")
    return p


@pytest.fixture()
def mock_model():
    return PerplexityEvaluator.uniform_model(vocab_size=100)


@pytest.fixture()
def mock_tokenizer():
    return _MockTokenizer()


@pytest.fixture()
def evaluator(wiki_file):
    return WikipediaFactualEvaluator(wiki_text_path=wiki_file)


# ---------------------------------------------------------------------------
# _extract_passages
# ---------------------------------------------------------------------------


def test_extract_passages_returns_pairs(wiki_file):
    random.seed(42)
    pairs = WikipediaFactualEvaluator._extract_passages(wiki_file, n_samples=None)
    assert len(pairs) > 0
    for pair in pairs:
        assert "original" in pair
        assert "counterfactual" in pair


def test_counterfactual_differs_from_original(wiki_file):
    random.seed(42)
    pairs = WikipediaFactualEvaluator._extract_passages(wiki_file, n_samples=None)
    assert len(pairs) > 0
    for pair in pairs:
        assert pair["original"] != pair["counterfactual"]


def test_n_samples_limits_pairs(wiki_file):
    random.seed(42)
    pairs = WikipediaFactualEvaluator._extract_passages(wiki_file, n_samples=2)
    assert len(pairs) <= 2


# ---------------------------------------------------------------------------
# _substitute_entity
# ---------------------------------------------------------------------------


def test_substitute_entity_replaces_entity():
    random.seed(42)
    sentence = "Albert Einstein was born in Germany"
    candidates = ["Albert Einstein", "Germany", "Marie Curie", "Poland"]
    result = WikipediaFactualEvaluator._substitute_entity(sentence, candidates)
    assert result is not None
    assert result != sentence
    assert "Albert Einstein" not in result


def test_substitute_entity_returns_none_when_no_entity():
    # All-lowercase sentence has no capitalized sequences
    result = WikipediaFactualEvaluator._substitute_entity(
        "the cat sat on the mat", ["Germany", "Poland"]
    )
    assert result is None


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_returns_correct_keys(evaluator, mock_model, mock_tokenizer):
    result = evaluator.evaluate(mock_model, mock_tokenizer)
    assert set(result.keys()) == {
        "factual_perplexity",
        "counterfactual_perplexity",
        "factual_score",
    }


def test_evaluate_scores_are_finite(evaluator, mock_model, mock_tokenizer):
    result = evaluator.evaluate(mock_model, mock_tokenizer)
    assert math.isfinite(result["factual_perplexity"])
    assert math.isfinite(result["counterfactual_perplexity"])
    assert math.isfinite(result["factual_score"])


def test_factual_score_is_difference(evaluator, mock_model, mock_tokenizer):
    """factual_score must equal mean(cf_ppl - factual_ppl) per pair,
    which by linearity of expectation equals counterfactual_perplexity
    minus factual_perplexity."""
    result = evaluator.evaluate(mock_model, mock_tokenizer)
    expected = result["counterfactual_perplexity"] - result["factual_perplexity"]
    assert abs(result["factual_score"] - expected) < 1e-9


def test_n_samples_limits_pairs_via_evaluate(wiki_file, mock_model, mock_tokenizer):
    """n_samples=2 must cause evaluate() to process at most 2 pairs."""
    ev = WikipediaFactualEvaluator(wiki_text_path=wiki_file, n_samples=2)
    # Verify by checking _extract_passages directly with n_samples=2
    random.seed(42)
    pairs = WikipediaFactualEvaluator._extract_passages(wiki_file, n_samples=2)
    assert len(pairs) <= 2
    # evaluate() must still complete without error
    result = ev.evaluate(mock_model, mock_tokenizer)
    assert set(result.keys()) == {
        "factual_perplexity",
        "counterfactual_perplexity",
        "factual_score",
    }


def test_determinism(evaluator, mock_model, mock_tokenizer):
    """Calling evaluate() twice on the same input must return identical results."""
    result1 = evaluator.evaluate(mock_model, mock_tokenizer)
    result2 = evaluator.evaluate(mock_model, mock_tokenizer)
    assert result1 == result2
