"""
tests/test_eval.py

Tests for the evaluator module (BiasEvaluator, PerplexityEvaluator).

Run with:
    pytest tests/test_eval.py -v
"""

import math
from unittest.mock import patch

import pytest

from openverifiablellm.eval.bias import BiasEvaluator
from openverifiablellm.eval.perplexity import PerplexityEvaluator

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Tokenizer that maps each character to its ASCII code modulo 100."""

    def encode(self, text: str) -> list:
        return [ord(c) % 100 for c in text.replace(" ", "_")]


def _make_dataset(sentences):
    """Return a list of row dicts matching the WinoBias ``tokens`` field."""
    return [{"tokens": s.split()} for s in sentences]


PRO_SENTENCES = [
    "The doctor examined the patient",
    "The engineer fixed the machine",
]
ANTI_SENTENCES = [
    "The nurse examined the patient",
    "The secretary fixed the machine",
]


def _patch_load_dataset(pro_data, anti_data):
    """Patch ``datasets.load_dataset`` to return pre-built lists."""

    def _load(name, config=None, split=None):
        if config == "type1_pro":
            return pro_data
        return anti_data

    return patch("datasets.load_dataset", side_effect=_load)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    """Uniform model: all-zero logits → perplexity == vocab_size for any input."""
    return PerplexityEvaluator.uniform_model(vocab_size=100)


@pytest.fixture
def mock_tokenizer():
    return _MockTokenizer()


@pytest.fixture
def bias_evaluator():
    return BiasEvaluator(n_samples=2)


# ---------------------------------------------------------------------------
# PerplexityEvaluator.uniform_model
# ---------------------------------------------------------------------------


def test_uniform_model_output_shape():
    model = PerplexityEvaluator.uniform_model(vocab_size=50)
    out = model([1, 2, 3])
    assert len(out) == 3
    assert len(out[0]) == 50


def test_uniform_model_all_zero_logits():
    model = PerplexityEvaluator.uniform_model(vocab_size=10)
    out = model([0, 1])
    assert all(v == 0.0 for row in out for v in row)


def test_uniform_model_perplexity_equals_vocab_size():
    vocab_size = 100
    model = PerplexityEvaluator.uniform_model(vocab_size=vocab_size)
    token_ids = list(range(10))
    ppl = PerplexityEvaluator.compute_sentence_perplexity(model, token_ids)
    assert abs(ppl - vocab_size) < 1e-6


# ---------------------------------------------------------------------------
# BiasEvaluator — initialisation
# ---------------------------------------------------------------------------


def test_bias_evaluator_invalid_benchmark_raises():
    with pytest.raises(ValueError):
        BiasEvaluator(benchmark="nonexistent_benchmark")


def test_bias_evaluator_valid_init():
    ev = BiasEvaluator()
    assert ev.benchmark == "wino_bias"


def test_bias_evaluator_n_samples_stored():
    ev = BiasEvaluator(n_samples=5)
    assert ev.n_samples == 5


# ---------------------------------------------------------------------------
# BiasEvaluator.evaluate() — patched load_dataset
# ---------------------------------------------------------------------------


def test_evaluate_does_not_raise(bias_evaluator, mock_model, mock_tokenizer):
    """evaluate() must complete without raising NotImplementedError or any error."""
    pro = _make_dataset(PRO_SENTENCES)
    anti = _make_dataset(ANTI_SENTENCES)
    with _patch_load_dataset(pro, anti):
        result = bias_evaluator.evaluate(mock_model, mock_tokenizer)
    assert isinstance(result, dict)


def test_evaluate_returns_exactly_three_keys(bias_evaluator, mock_model, mock_tokenizer):
    pro = _make_dataset(PRO_SENTENCES)
    anti = _make_dataset(ANTI_SENTENCES)
    with _patch_load_dataset(pro, anti):
        result = bias_evaluator.evaluate(mock_model, mock_tokenizer)
    assert set(result.keys()) == {"stereotype_score", "anti_stereotype_score", "bias_score"}


def test_evaluate_bias_score_equals_abs_diff(bias_evaluator, mock_model, mock_tokenizer):
    pro = _make_dataset(PRO_SENTENCES)
    anti = _make_dataset(ANTI_SENTENCES)
    with _patch_load_dataset(pro, anti):
        result = bias_evaluator.evaluate(mock_model, mock_tokenizer)
    expected = abs(result["stereotype_score"] - result["anti_stereotype_score"])
    assert abs(result["bias_score"] - expected) < 1e-9


def test_evaluate_scores_are_finite(bias_evaluator, mock_model, mock_tokenizer):
    pro = _make_dataset(PRO_SENTENCES)
    anti = _make_dataset(ANTI_SENTENCES)
    with _patch_load_dataset(pro, anti):
        result = bias_evaluator.evaluate(mock_model, mock_tokenizer)
    assert math.isfinite(result["stereotype_score"])
    assert math.isfinite(result["anti_stereotype_score"])
    assert math.isfinite(result["bias_score"])


# ---------------------------------------------------------------------------
# n_samples limits dataset consumption
# ---------------------------------------------------------------------------


def test_n_samples_limits_dataset(mock_model, mock_tokenizer):
    """With n_samples=2, rows beyond index 1 must never be processed.

    Rows beyond index 1 are single-character strings ("a"), which tokenise
    to exactly one token and yield infinite perplexity.  If n_samples works
    correctly, only the first two (multi-token) rows are consumed and the
    returned bias_score is finite.
    """
    # Append single-char rows that yield inf perplexity if reached
    bad_rows = ["a"] * 10
    pro = _make_dataset(PRO_SENTENCES + bad_rows)
    anti = _make_dataset(ANTI_SENTENCES + bad_rows)

    ev = BiasEvaluator(n_samples=len(PRO_SENTENCES))  # == 2

    with _patch_load_dataset(pro, anti):
        result = ev.evaluate(mock_model, mock_tokenizer)

    assert math.isfinite(result["bias_score"])
