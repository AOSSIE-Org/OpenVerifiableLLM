"""
tests/test_eval.py
==================

Tests for the openverifiablellm.eval package.

All tests run on CPU with tiny mock models — no GPU required.

Run with:
    pytest tests/test_eval.py -v
"""

import math

import pytest

from openverifiablellm.eval import (
    BaseEvaluator,
    BenchmarkEvaluator,
    BiasEvaluator,
    PerplexityEvaluator,
)
from openverifiablellm.eval.perplexity import _sliding_window_nll

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

SAMPLE_TEXT = (
    "Wikipedia is a free online encyclopedia created by volunteers. "
    "Anyone can edit an article at any time. "
    "It is one of the most visited websites in the world. "
)

VOCAB_SIZE = 256  # byte-level, conveniently small


def byte_tokenizer(text: str):
    """Encode text as UTF-8 bytes (each byte is one token)."""
    return list(text.encode("utf-8"))


def uniform_log_probs(vocab_size: int):
    """Return a model that outputs uniform log-probabilities for every token."""
    lp = math.log(1.0 / vocab_size)

    def _model(token_ids):
        return [lp] * len(token_ids)

    return _model


# ---------------------------------------------------------------------------
# BaseEvaluator — interface contract
# ---------------------------------------------------------------------------


class TestBaseEvaluator:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            BaseEvaluator(name="x")  # type: ignore[abstract]

    def test_concrete_subclass_must_implement_evaluate(self):
        """A subclass that forgets to implement evaluate() is not instantiable."""

        class Incomplete(BaseEvaluator):
            pass

        with pytest.raises(TypeError):
            Incomplete(name="incomplete")  # type: ignore[abstract]

    def test_concrete_subclass_is_instantiable(self):
        class Minimal(BaseEvaluator):
            def evaluate(self, model, tokenizer):
                return {"score": 1.0}

        ev = Minimal(name="minimal")
        assert ev.name == "minimal"
        assert ev.evaluate(None, None) == {"score": 1.0}

    def test_base_evaluator_import_alias(self):
        # The package re-export must be the same class object as the base module class.
        from openverifiablellm.eval.base import BaseEvaluator as _Direct

        assert BaseEvaluator is _Direct


# ---------------------------------------------------------------------------
# PerplexityEvaluator — construction validation
# ---------------------------------------------------------------------------


class TestPerplexityEvaluatorConstruction:
    def test_valid_construction(self):
        ev = PerplexityEvaluator(text=SAMPLE_TEXT)
        assert ev.text == SAMPLE_TEXT

    def test_empty_text_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            PerplexityEvaluator(text="")

    def test_non_string_text_raises(self):
        with pytest.raises(ValueError):
            PerplexityEvaluator(text=123)  # type: ignore[arg-type]

    def test_zero_max_length_raises(self):
        with pytest.raises(ValueError, match="max_length"):
            PerplexityEvaluator(text=SAMPLE_TEXT, max_length=0)

    def test_zero_stride_raises(self):
        with pytest.raises(ValueError, match="stride"):
            PerplexityEvaluator(text=SAMPLE_TEXT, stride=0)

    def test_stride_exceeds_max_length_raises(self):
        with pytest.raises(ValueError, match="stride"):
            PerplexityEvaluator(text=SAMPLE_TEXT, max_length=64, stride=128)

    def test_name_stored(self):
        ev = PerplexityEvaluator(text=SAMPLE_TEXT, name="test_ppl")
        assert ev.name == "test_ppl"


# ---------------------------------------------------------------------------
# PerplexityEvaluator — evaluate() correctness
# ---------------------------------------------------------------------------


class TestPerplexityEvaluatorEvaluate:
    def test_returns_required_keys(self):
        ev = PerplexityEvaluator(text=SAMPLE_TEXT)
        result = ev.evaluate(
            model=uniform_log_probs(VOCAB_SIZE),
            tokenizer=byte_tokenizer,
        )
        for key in ("perplexity", "nll_bits_per_byte", "n_tokens", "n_bytes"):
            assert key in result, f"Missing key: {key}"

    def test_uniform_model_perplexity_equals_vocab_size(self):
        """For a uniform model, PPL must equal vocab_size (within float tolerance)."""
        ev = PerplexityEvaluator(text=SAMPLE_TEXT, max_length=512, stride=256)
        result = ev.evaluate(
            model=uniform_log_probs(VOCAB_SIZE),
            tokenizer=byte_tokenizer,
        )
        assert abs(result["perplexity"] - VOCAB_SIZE) < 1e-6, (
            f"Expected PPL≈{VOCAB_SIZE}, got {result['perplexity']}"
        )

    def test_n_bytes_matches_utf8_length(self):
        ev = PerplexityEvaluator(text=SAMPLE_TEXT)
        result = ev.evaluate(
            model=uniform_log_probs(VOCAB_SIZE),
            tokenizer=byte_tokenizer,
        )
        assert result["n_bytes"] == len(SAMPLE_TEXT.encode("utf-8"))

    def test_n_tokens_positive(self):
        ev = PerplexityEvaluator(text=SAMPLE_TEXT, max_length=64, stride=32)
        result = ev.evaluate(
            model=uniform_log_probs(VOCAB_SIZE),
            tokenizer=byte_tokenizer,
        )
        assert result["n_tokens"] > 0

    def test_perplexity_is_finite(self):
        ev = PerplexityEvaluator(text=SAMPLE_TEXT)
        result = ev.evaluate(
            model=uniform_log_probs(VOCAB_SIZE),
            tokenizer=byte_tokenizer,
        )
        assert math.isfinite(result["perplexity"])

    def test_lower_surprise_gives_lower_perplexity(self):
        """A model that perfectly predicts each token must have PPL=1."""
        # A model that returns log(1.0) = 0.0 for every token
        perfect_model = lambda ids: [0.0] * len(ids)  # noqa: E731

        ev = PerplexityEvaluator(text=SAMPLE_TEXT)
        result = ev.evaluate(model=perfect_model, tokenizer=byte_tokenizer)
        assert abs(result["perplexity"] - 1.0) < 1e-9

    def test_uniform_model_factory(self):
        model = PerplexityEvaluator.uniform_model(vocab_size=100)
        log_probs = model([0, 1, 2])
        assert len(log_probs) == 3
        expected_lp = math.log(1.0 / 100)
        for lp in log_probs:
            assert abs(lp - expected_lp) < 1e-12

    def test_uniform_model_zero_vocab_raises(self):
        with pytest.raises(ValueError, match="vocab_size"):
            PerplexityEvaluator.uniform_model(vocab_size=0)

    def test_nll_bits_per_byte_non_negative(self):
        ev = PerplexityEvaluator(text=SAMPLE_TEXT)
        result = ev.evaluate(
            model=uniform_log_probs(VOCAB_SIZE),
            tokenizer=byte_tokenizer,
        )
        assert result["nll_bits_per_byte"] >= 0.0


# ---------------------------------------------------------------------------
# _sliding_window_nll — internal unit tests
# ---------------------------------------------------------------------------


class TestSlidingWindowNll:
    def _uniform_model(self, vocab_size=VOCAB_SIZE):
        lp = math.log(1.0 / vocab_size)
        return lambda ids: [lp] * len(ids)

    def test_empty_token_list_returns_zero(self):
        nll, n = _sliding_window_nll([], self._uniform_model(), max_length=64, stride=32)
        assert nll == 0.0
        assert n == 0

    def test_single_token_nothing_scored(self):
        nll, n = _sliding_window_nll([42], self._uniform_model(), max_length=64, stride=32)
        # Position 0 has no left context — not scored
        assert n == 0

    def test_two_tokens_scores_one(self):
        nll, n = _sliding_window_nll([1, 2], self._uniform_model(), max_length=64, stride=32)
        assert n == 1

    def test_invalid_max_length_raises(self):
        with pytest.raises(ValueError, match="max_length"):
            _sliding_window_nll([1, 2, 3], self._uniform_model(), max_length=0, stride=1)

    def test_invalid_stride_raises(self):
        with pytest.raises(ValueError, match="stride"):
            _sliding_window_nll([1, 2, 3], self._uniform_model(), max_length=4, stride=0)

    def test_stride_exceeds_max_length_raises(self):
        with pytest.raises(ValueError, match="stride"):
            _sliding_window_nll([1, 2, 3], self._uniform_model(), max_length=2, stride=4)

    def test_nll_proportional_to_n_tokens_for_uniform_model(self):
        """For a uniform model NLL = n_tokens * log(vocab_size)."""
        text = SAMPLE_TEXT * 3  # ~500 bytes, enough to span multiple windows
        token_ids = byte_tokenizer(text)
        model = self._uniform_model(VOCAB_SIZE)
        nll, n = _sliding_window_nll(token_ids, model, max_length=128, stride=64)
        expected_nll_per_token = math.log(VOCAB_SIZE)
        assert abs(nll / n - expected_nll_per_token) < 1e-9


# ---------------------------------------------------------------------------
# BiasEvaluator — stub contract
# ---------------------------------------------------------------------------


class TestBiasEvaluator:
    def test_valid_construction_winobias(self):
        ev = BiasEvaluator(benchmark="winobias")
        assert ev.benchmark == "winobias"

    def test_valid_construction_bbq(self):
        ev = BiasEvaluator(benchmark="bbq")
        assert ev.benchmark == "bbq"

    def test_invalid_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            BiasEvaluator(benchmark="unknown_benchmark")

    def test_evaluate_raises_not_implemented(self):
        ev = BiasEvaluator()
        with pytest.raises(NotImplementedError):
            ev.evaluate(model=None, tokenizer=None)

    def test_is_base_evaluator_subclass(self):
        from openverifiablellm.eval.base import BaseEvaluator

        assert issubclass(BiasEvaluator, BaseEvaluator)


# ---------------------------------------------------------------------------
# BenchmarkEvaluator — stub contract
# ---------------------------------------------------------------------------


class TestBenchmarkEvaluator:
    def test_valid_construction_mmlu(self):
        ev = BenchmarkEvaluator(benchmark="mmlu")
        assert ev.benchmark == "mmlu"

    def test_valid_construction_triviaqa(self):
        ev = BenchmarkEvaluator(benchmark="triviaqa")
        assert ev.benchmark == "triviaqa"

    def test_invalid_benchmark_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            BenchmarkEvaluator(benchmark="glue")

    def test_n_samples_none_allowed(self):
        ev = BenchmarkEvaluator(n_samples=None)
        assert ev.n_samples is None

    def test_n_samples_positive_allowed(self):
        ev = BenchmarkEvaluator(n_samples=100)
        assert ev.n_samples == 100

    def test_n_samples_zero_raises(self):
        with pytest.raises(ValueError, match="n_samples"):
            BenchmarkEvaluator(n_samples=0)

    def test_evaluate_raises_not_implemented(self):
        ev = BenchmarkEvaluator()
        with pytest.raises(NotImplementedError):
            ev.evaluate(model=None, tokenizer=None)

    def test_is_base_evaluator_subclass(self):
        from openverifiablellm.eval.base import BaseEvaluator

        assert issubclass(BenchmarkEvaluator, BaseEvaluator)
