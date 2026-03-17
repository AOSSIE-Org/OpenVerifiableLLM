import bz2
import json

import pytest

from openverifiablellm import utils
from openverifiablellm.config import BenchmarkConfig
from openverifiablellm.contamination import (
    build_bloom_filter,
    check_contamination,
    get_ngrams,
)

# ------------------------------------------------------------------ #
#  get_ngrams tests                                                   #
# ------------------------------------------------------------------ #

class TestGetNgrams:
    """Tests for n-gram generation."""

    def test_basic_generation(self):
        # 15 tokens → with n=13 we expect 3 n-grams
        text = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen"
        ngrams = get_ngrams(text, n=13)
        assert len(ngrams) == 3
        # each n-gram should have 13 space-separated tokens
        for ng in ngrams:
            assert len(ng.split()) == 13

    def test_short_text_returns_empty(self):
        text = "hello world"
        assert get_ngrams(text, n=13) == []

    def test_exact_n_tokens(self):
        text = " ".join(f"word{i}" for i in range(13))
        ngrams = get_ngrams(text, n=13)
        assert len(ngrams) == 1

    def test_case_insensitive(self):
        text = " ".join(f"Word{i}" for i in range(15))
        ngrams = get_ngrams(text, n=13)
        for ng in ngrams:
            assert ng == ng.lower()

    def test_punctuation_stripped(self):
        text = "Hello, world! This is a test: of punctuation-removal; really? yes."
        ngrams = get_ngrams(text, n=5)
        for ng in ngrams:
            assert "," not in ng
            assert "!" not in ng
            assert "?" not in ng

    def test_empty_string(self):
        assert get_ngrams("", n=13) == []

    def test_custom_n(self):
        text = "a b c d e f"
        assert len(get_ngrams(text, n=3)) == 4  # 6 - 3 + 1
        assert len(get_ngrams(text, n=6)) == 1

    def test_invalid_n_raises(self):
        text = "some valid text"
        with pytest.raises(ValueError, match="n must be a positive integer"):
            get_ngrams(text, n=0)
        with pytest.raises(ValueError, match="n must be a positive integer"):
            get_ngrams(text, n=-1)


# ------------------------------------------------------------------ #
#  Bloom filter tests                                                 #
# ------------------------------------------------------------------ #

class TestBloomFilter:
    def _make_config(self, tmp_path, **overrides):
        defaults = dict(
            benchmarks=[],
            bloom_capacity=1000,
            bloom_error_rate=0.001,
            filter_path=tmp_path / "test_filter.bin",
            n=5,
        )
        defaults.update(overrides)
        return BenchmarkConfig(**defaults)

    def test_insert_and_lookup(self, tmp_path):
        config = self._make_config(tmp_path)
        texts = ["one two three four five six seven eight"]  # 8 tokens → 4 5-grams
        bloom = build_bloom_filter(texts, config)

        ngrams = get_ngrams(texts[0], n=5)
        for ng in ngrams:
            assert ng in bloom

    def test_negative_lookup(self, tmp_path):
        config = self._make_config(tmp_path)
        texts = ["alpha bravo charlie delta echo foxtrot golf hotel"]
        bloom = build_bloom_filter(texts, config)

        # Completely unrelated n-gram
        assert "zzz yyy xxx www vvv" not in bloom

    def test_serialisation_roundtrip(self, tmp_path):
        config = self._make_config(tmp_path)
        texts = ["one two three four five six seven"]
        build_bloom_filter(texts, config)
        assert config.filter_path.exists()

        # Loading again should read from disk
        bloom2 = build_bloom_filter(texts, config)
        for ng in get_ngrams(texts[0], n=5):
            assert ng in bloom2


# ------------------------------------------------------------------ #
#  check_contamination tests                                          #
# ------------------------------------------------------------------ #

class TestCheckContamination:
    def _build_filter(self, benchmark_texts, tmp_path, n=5):
        config = BenchmarkConfig(
            benchmarks=[],
            bloom_capacity=1000,
            bloom_error_rate=0.001,
            filter_path=tmp_path / "test_filter.bin",
            n=n,
        )
        return build_bloom_filter(benchmark_texts, config)

    def test_contaminated_text_detected(self, tmp_path):
        benchmark = "what is the capital of france and why is it important"
        bloom = self._build_filter([benchmark], tmp_path)
        # The chunk contains the benchmark verbatim
        chunk = "Let me tell you what is the capital of france and why is it important to know"
        assert check_contamination(chunk, bloom, [benchmark], n=5) is True

    def test_clean_text_passes(self, tmp_path):
        benchmark = "what is the capital of france and why is it important"
        bloom = self._build_filter([benchmark], tmp_path)
        chunk = "the quick brown fox jumps over the lazy dog repeatedly"
        assert check_contamination(chunk, bloom, [benchmark], n=5) is False

    def test_case_insensitive_detection(self, tmp_path):
        benchmark = "What Is The Capital Of France And Why Is It Important"
        bloom = self._build_filter([benchmark], tmp_path)
        chunk = "WHAT IS THE CAPITAL OF FRANCE AND WHY IS IT IMPORTANT to know"
        assert check_contamination(chunk, bloom, [benchmark], n=5) is True

    def test_short_chunk_not_flagged(self, tmp_path):
        benchmark = "alpha bravo charlie delta echo foxtrot golf hotel india juliet oskar tango zulu"
        bloom = self._build_filter([benchmark], tmp_path, n=13)
        chunk = "alpha bravo"
        assert check_contamination(chunk, bloom, [benchmark], n=13) is False


# ------------------------------------------------------------------ #
#  Manifest contamination metadata                                    #
# ------------------------------------------------------------------ #

class TestManifestContamination:
    def test_manifest_includes_contamination_fields(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        raw_file = tmp_path / "raw.txt"
        raw_file.write_text("dummy data")

        processed_file = tmp_path / "processed.txt"
        processed_file.write_text("cleaned data")

        metadata = {
            "enabled": True,
            "n_gram_size": 13,
            "benchmarks_used": ["gsm8k", "cais/mmlu"],
            "redacted_chunks": 7,
        }
        
        utils.generate_manifest(
            raw_file,
            processed_file,
            contamination_metadata=metadata,
        )

        manifest_file = tmp_path / "data" / "dataset_manifest.json"
        manifest = json.loads(manifest_file.read_text())

        assert "contamination" in manifest
        assert manifest["contamination"]["benchmarks_used"] == ["gsm8k", "cais/mmlu"]
        assert manifest["contamination"]["redacted_chunks"] == 7
        assert manifest["contamination"]["enabled"] is True

    def test_manifest_omits_contamination_when_not_used(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)

        raw_file = tmp_path / "raw.txt"
        raw_file.write_text("dummy data")

        processed_file = tmp_path / "processed.txt"
        processed_file.write_text("cleaned data")

        utils.generate_manifest(raw_file, processed_file)

        manifest_file = tmp_path / "data" / "dataset_manifest.json"
        manifest = json.loads(manifest_file.read_text())

        assert "contamination" not in manifest
        assert "contamination_checks_passed" not in manifest


# ------------------------------------------------------------------ #
#  Backward compatibility                                             #
# ------------------------------------------------------------------ #

class TestBackwardCompat:
    @pytest.fixture
    def simple_xml(self, tmp_path):
        xml_content = """<?xml version="1.0"?>
        <mediawiki>
          <page>
            <revision>
              <text>Hello [[World]]</text>
            </revision>
          </page>
        </mediawiki>
        """
        input_file = tmp_path / "simplewiki.xml.bz2"
        with bz2.open(input_file, "wt", encoding="utf-8") as f:
            f.write(xml_content)
        return input_file

    def test_extract_text_from_xml_no_contamination(self, simple_xml, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        utils.extract_text_from_xml(simple_xml)

        processed_file = tmp_path / "data" / "processed" / "wiki_clean.txt"
        assert processed_file.exists()
        assert "Hello World" in processed_file.read_text()

    def test_extract_fails_if_only_bloom_filter_provided(self, simple_xml):
        from rbloom import Bloom
        dummy_bloom = Bloom(100, 0.01)
        with pytest.raises(ValueError, match="Both 'bloom_filter' and 'benchmark_texts' must"):
            utils.extract_text_from_xml(simple_xml, bloom_filter=dummy_bloom)

    def test_extract_fails_if_only_benchmark_texts_provided(self, simple_xml):
        with pytest.raises(ValueError, match="Both 'bloom_filter' and 'benchmark_texts' must"):
            utils.extract_text_from_xml(simple_xml, benchmark_texts=["some text"])