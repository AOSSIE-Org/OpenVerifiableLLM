"""
test_merkle.py
==============
pytest suite for IncrementalMerkleTree.

The critical correctness property we verify
--------------------------------------------
Given the same ordered sequence of N strings, a *batch* Merkle tree
(built by collecting all leaf hashes up-front, then reducing level by
level) and our *incremental* Merkle tree (appending one leaf at a time
via the Merkle Frontier) MUST produce byte-for-byte identical root hashes.

If that invariant ever breaks, the streaming pipeline cannot be trusted
as a drop-in replacement for the legacy in-memory approach.

Run with:
    pip install -e ".[dev]"
    pytest tests/test_merkle.py -v
"""

import hashlib
from typing import List, Optional

import pytest

from openverifiablellm.incremental_merkle import IncrementalMerkleTree


# ===========================================================================
# Reference implementation: a classic batch Merkle tree
# ===========================================================================

def _sha256_bytes(data: bytes) -> bytes:
    """Return raw 32-byte SHA-256 digest."""
    return hashlib.sha256(data).digest()


def _combine(left: bytes, right: bytes) -> bytes:
    """Combine two 32-byte node hashes: parent = SHA-256(left || right)."""
    return _sha256_bytes(left + right)


def batch_merkle_root(texts: List[str]) -> Optional[str]:
    """
    Build a standard, batch Merkle tree from a list of strings and return
    the root hash as a hex string.

    This is the canonical reference implementation used to verify the
    incremental version.  It stores ALL leaf hashes in memory and reduces
    them level-by-level exactly as the legacy code does.

    Odd-number-of-nodes rule: when a level has an odd count, the last node
    is duplicated so every parent has exactly two children.  This matches
    the rule used in ``IncrementalMerkleTree.get_root_hash()``.

    Parameters
    ----------
    texts : list of str
        Ordered list of article texts (or any strings).

    Returns
    -------
    str | None
        64-char hex root hash, or ``None`` if *texts* is empty.
    """
    if not texts:
        return None

    # Leaf level: hash each string
    level: List[bytes] = [
        _sha256_bytes(t.encode("utf-8")) for t in texts
    ]

    # Reduce level-by-level until only the root remains
    while len(level) > 1:
        next_level: List[bytes] = []
        for i in range(0, len(level), 2):
            left = level[i]
            # Duplicate the last node if the level has an odd count
            right = level[i + 1] if i + 1 < len(level) else left
            next_level.append(_combine(left, right))
        level = next_level

    return level[0].hex()


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def hundred_strings() -> List[str]:
    """
    A deterministic list of 100 unique strings.

    Uses the f-string ``"article_{i:03d}"`` pattern so the content is
    predictable and reproducible across test runs.
    """
    return [f"article_{i:03d}: The quick brown fox jumps over the lazy dog." for i in range(100)]


# ===========================================================================
# Core correctness test (the PRIMARY deliverable)
# ===========================================================================

class TestIncrementalVsBatch:
    """
    Verify that IncrementalMerkleTree produces the same root as the
    batch reference implementation for the same input sequence.
    """

    def test_root_hash_matches_batch_100_strings(
        self, hundred_strings: List[str]
    ) -> None:
        """
        PRIMARY TEST: IncrementalMerkleTree root must exactly equal the
        batch Merkle root for the same 100 strings.

        This is the definitive correctness gate for the streaming pipeline.
        """
        # Build batch reference root
        expected_root = batch_merkle_root(hundred_strings)
        assert expected_root is not None, "batch_merkle_root should not return None for non-empty input"

        # Build incremental root
        tree = IncrementalMerkleTree()
        for text in hundred_strings:
            tree.append_leaf(text)

        actual_root = tree.get_root_hash()
        assert actual_root is not None, "IncrementalMerkleTree.get_root_hash() must not return None"

        assert actual_root == expected_root, (
            f"Root hash mismatch!\n"
            f"  Batch root      : {expected_root}\n"
            f"  Incremental root: {actual_root}\n"
            "The streaming pipeline is NOT a safe replacement for the "
            "legacy in-memory approach until this test passes."
        )

    def test_root_hash_matches_batch_single_string(self) -> None:
        """Single leaf: root must equal SHA-256 of that leaf's text."""
        texts = ["only one article here"]
        expected = batch_merkle_root(texts)

        tree = IncrementalMerkleTree()
        tree.append_leaf(texts[0])

        assert tree.get_root_hash() == expected

    def test_root_hash_matches_batch_two_strings(self) -> None:
        """Two leaves: tests the first combine() call in both implementations."""
        texts = ["alpha", "beta"]
        expected = batch_merkle_root(texts)

        tree = IncrementalMerkleTree()
        for t in texts:
            tree.append_leaf(t)

        assert tree.get_root_hash() == expected

    def test_root_hash_matches_batch_power_of_two(self) -> None:
        """
        Power-of-two leaf count (8 leaves): the tree is perfectly balanced;
        the frontier should collapse to a single node (the root itself).
        """
        texts = [f"leaf_{i}" for i in range(8)]
        expected = batch_merkle_root(texts)

        tree = IncrementalMerkleTree()
        for t in texts:
            tree.append_leaf(t)

        # For exactly 2^k leaves, the frontier should have exactly 1 node
        assert len(tree._frontier) == 1, (
            "For a power-of-two leaf count the frontier should collapse to 1 node"
        )
        assert tree.get_root_hash() == expected

    def test_root_hash_matches_batch_odd_leaf_count(self) -> None:
        """Odd leaf count (7): tests the 'duplicate last node' codepath."""
        texts = [f"article_{i}" for i in range(7)]
        expected = batch_merkle_root(texts)

        tree = IncrementalMerkleTree()
        for t in texts:
            tree.append_leaf(t)

        assert tree.get_root_hash() == expected

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 15, 16, 17, 31, 32, 33, 64, 100, 128, 255, 256])
    def test_root_hash_matches_batch_parametric(self, n: int) -> None:
        """
        Parametric sweep over various leaf counts including edge cases,
        powers of two, and powers-of-two ± 1.
        """
        texts = [f"string_{i:04d}" for i in range(n)]
        expected = batch_merkle_root(texts)

        tree = IncrementalMerkleTree()
        for t in texts:
            tree.append_leaf(t)

        assert tree.get_root_hash() == expected, (
            f"Root hash mismatch for n={n}"
        )


# ===========================================================================
# Empty-tree behaviour
# ===========================================================================

class TestEmptyTree:
    def test_get_root_hash_returns_none_when_empty(self) -> None:
        """An empty tree has no root — get_root_hash() must return None."""
        tree = IncrementalMerkleTree()
        assert tree.get_root_hash() is None

    def test_leaf_count_zero_when_empty(self) -> None:
        tree = IncrementalMerkleTree()
        assert tree.leaf_count == 0

    def test_frontier_empty_when_empty(self) -> None:
        tree = IncrementalMerkleTree()
        assert tree._frontier == {}


# ===========================================================================
# Leaf count tracking
# ===========================================================================

class TestLeafCount:
    def test_leaf_count_increments(self) -> None:
        tree = IncrementalMerkleTree()
        for i in range(50):
            tree.append_leaf(f"leaf_{i}")
            assert tree.leaf_count == i + 1

    def test_leaf_count_matches_hundred(self, hundred_strings: List[str]) -> None:
        tree = IncrementalMerkleTree()
        for t in hundred_strings:
            tree.append_leaf(t)
        assert tree.leaf_count == 100


# ===========================================================================
# Frontier size invariant
# ===========================================================================

class TestFrontierInvariant:
    """
    The frontier size equals the number of set bits in the binary
    representation of the leaf count (popcount / Hamming weight).
    """

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 100, 128, 255, 256])
    def test_frontier_size_equals_popcount(self, n: int) -> None:
        tree = IncrementalMerkleTree()
        for i in range(n):
            tree.append_leaf(f"x_{i}")

        expected_frontier_nodes = bin(n).count("1")
        assert len(tree._frontier) == expected_frontier_nodes, (
            f"After {n} leaves (binary: {bin(n)}), frontier should have "
            f"{expected_frontier_nodes} node(s), got {len(tree._frontier)}"
        )


# ===========================================================================
# Determinism / reproducibility
# ===========================================================================

class TestDeterminism:
    def test_same_input_same_root(self, hundred_strings: List[str]) -> None:
        """Two trees built from the same input must produce identical roots."""
        tree1 = IncrementalMerkleTree()
        tree2 = IncrementalMerkleTree()
        for t in hundred_strings:
            tree1.append_leaf(t)
            tree2.append_leaf(t)
        assert tree1.get_root_hash() == tree2.get_root_hash()

    def test_different_order_different_root(self) -> None:
        """Order matters: reversed input must produce a different root."""
        texts = [f"item_{i}" for i in range(10)]

        tree_fwd = IncrementalMerkleTree()
        tree_rev = IncrementalMerkleTree()
        for t in texts:
            tree_fwd.append_leaf(t)
        for t in reversed(texts):
            tree_rev.append_leaf(t)

        assert tree_fwd.get_root_hash() != tree_rev.get_root_hash()

    def test_extra_leaf_changes_root(self) -> None:
        """Appending one more leaf must change the root hash."""
        texts = [f"article_{i}" for i in range(10)]

        tree_a = IncrementalMerkleTree()
        for t in texts:
            tree_a.append_leaf(t)
        root_a = tree_a.get_root_hash()

        tree_b = IncrementalMerkleTree()
        for t in texts:
            tree_b.append_leaf(t)
        tree_b.append_leaf("one_more_article")
        root_b = tree_b.get_root_hash()

        assert root_a != root_b

    def test_get_root_hash_is_non_destructive(self) -> None:
        """Calling get_root_hash() multiple times must return the same value."""
        tree = IncrementalMerkleTree()
        for i in range(20):
            tree.append_leaf(f"leaf_{i}")

        roots = {tree.get_root_hash() for _ in range(5)}
        assert len(roots) == 1, "get_root_hash() must be idempotent"


# ===========================================================================
# Hash format sanity checks
# ===========================================================================

class TestHashFormat:
    def test_root_hash_is_64_char_hex(self) -> None:
        """SHA-256 produces 32 bytes → 64 lowercase hex characters."""
        tree = IncrementalMerkleTree()
        tree.append_leaf("hello world")
        root = tree.get_root_hash()
        assert root is not None
        assert len(root) == 64
        assert root == root.lower()
        # Must be valid hex
        int(root, 16)

    def test_single_leaf_root_equals_sha256_of_text(self) -> None:
        """
        For a single leaf the root hash must equal SHA-256(text.encode()).
        There is no combining step — the leaf hash IS the root.
        """
        text = "wikipedia article about Python"
        expected = hashlib.sha256(text.encode("utf-8")).hexdigest()

        tree = IncrementalMerkleTree()
        tree.append_leaf(text)

        assert tree.get_root_hash() == expected


# ===========================================================================
# repr() smoke test
# ===========================================================================

class TestRepr:
    def test_repr_contains_leaf_count(self) -> None:
        tree = IncrementalMerkleTree()
        for i in range(7):
            tree.append_leaf(f"t_{i}")
        r = repr(tree)
        assert "leaves=7" in r
        assert "IncrementalMerkleTree" in r
