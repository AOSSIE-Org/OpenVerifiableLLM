"""
incremental_merkle.py
=====================
An append-only, O(log N) space Merkle Tree using the "Merkle Frontier".

Background: The Merkle Frontier Algorithm
------------------------------------------
A classical Merkle tree requires ALL leaf hashes to be stored so the
tree can be reconstructed level by level.  For N leaves that is O(N)
memory — completely unacceptable when N is in the millions.

The Merkle Frontier (also called a "Merkle accumulator" in certificate
transparency literature) solves this by exploiting one structural
property of binary Merkle trees:

    The only nodes you ever need to recompute the root are the
    **rightmost unpaired nodes at each depth** — the "frontier".

Concretely: after inserting N leaves the frontier contains at most
⌈log₂ N⌉ hashes — one per set bit in the binary representation of N.

Example with N = 5  (binary: 101)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Leaf indices: L0 L1 L2 L3 L4

Full tree structure:
                       ROOT
                      /    \\
                  H(0-3)   H(4)
                 /     \\
             H(0-1)   H(2-3)
             /   \\    /   \\
            L0   L1  L2   L3

After appending L0…L4 the frontier is:
    depth 2  →  H(0-3)   (covers 4 leaves — a complete sub-tree)
    depth 0  →  L4        (a single unpaired leaf at the right edge)

To get the root we just fold right:  root = H( H(0-3) || L4 )

Space used by frontier: 2 hashes  (⌈log₂ 5⌉ = 3, but only 2 bits are
set in 5 = 0b101, so there are only 2 frontier nodes).

Root computation
----------------
``get_root_hash()`` folds the frontier from the **lowest depth** to the
**highest depth**, combining nodes pair-wise:

    current = frontier[lowest_depth]
    for d in range(lowest_depth + 1, max_depth + 1):
        if frontier[d] exists:
            current = sha256( frontier[d] || current )
        else:
            current = sha256( current || current )   # duplicate (odd node)

The final ``current`` value is the root hash.

Usage
-----
    from openverifiablellm.incremental_merkle import IncrementalMerkleTree

    tree = IncrementalMerkleTree()
    for text in stream_text_from_xml("dump.xml.bz2"):
        tree.append_leaf(text)

    root = tree.get_root_hash()
    print(f"Merkle root: {root}")
"""

import hashlib
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(data: bytes) -> bytes:
    """Return the raw 32-byte SHA-256 digest of *data*."""
    return hashlib.sha256(data).digest()


def _combine(left: bytes, right: bytes) -> bytes:
    """
    Combine two 32-byte node hashes into a parent hash.

    The canonical MediaWiki / Bitcoin-style combination:
        parent = SHA256( left_bytes || right_bytes )

    Parameters
    ----------
    left, right:
        Raw 32-byte digest values (NOT hex strings).

    Returns
    -------
    bytes
        32-byte digest of the concatenation.
    """
    return _sha256_bytes(left + right)


# ---------------------------------------------------------------------------
# IncrementalMerkleTree
# ---------------------------------------------------------------------------


class IncrementalMerkleTree:
    """
    An append-only Merkle tree with O(log N) time and space per operation.

    State
    -----
    _frontier : Dict[int, bytes]
        Maps ``depth`` → 32-byte hash.
        Depth 0 = leaf level.  Higher depth = closer to the root.
        The frontier holds at most one node per depth; a node at depth d
        represents a **complete** subtree of height d (covering 2**d leaves).

    _leaf_count : int
        Total number of leaves appended so far.  Used only for logging/info.

    Invariant
    ---------
    After appending N leaves, ``_frontier`` contains exactly the nodes
    corresponding to the set bits in the binary representation of N.
    For example N = 6 = 0b110 → frontier has nodes at depth 1 and depth 2.
    """

    def __init__(self) -> None:
        # depth → 32-byte node hash.  Only "complete" subtree roots are stored.
        self._frontier: Dict[int, bytes] = {}
        self._leaf_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def append_leaf(self, text_chunk: str) -> None:
        """
        Hash *text_chunk* and insert it as the next leaf in the tree.

        Algorithm (O(log N) time, O(log N) space)
        ------------------------------------------
        1. Compute the SHA-256 hash of the UTF-8-encoded text.
        2. Start at depth 0 (leaf level) with the new hash as ``node``.
        3. While there is already a node stored at the current depth:
             a. Combine the stored node (left) with our new node (right).
             b. Remove the stored node from the frontier (the slot is now
                "consumed" into a higher level).
             c. Move one level up and continue with the combined hash.
        4. Store the final unconsumed node at its depth in the frontier.

        This mirrors how a binary counter increments: each carry bit
        propagates up until it finds an empty slot.

        Parameters
        ----------
        text_chunk:
            Arbitrary Unicode string.  Encoded to UTF-8 before hashing.
        """
        # Step 1: hash the raw text to get a 32-byte leaf digest
        new_node: bytes = _sha256_bytes(text_chunk.encode("utf-8"))

        # Step 2-4: propagate carries up the tree, exactly like binary addition
        depth: int = 0
        while depth in self._frontier:
            # Combine: existing left sibling || new right node → parent
            left_sibling: bytes = self._frontier.pop(depth)
            new_node = _combine(left_sibling, new_node)
            depth += 1

        # No existing node at this depth — park the new node here
        self._frontier[depth] = new_node
        self._leaf_count += 1

    def get_root_hash(self) -> Optional[str]:
        """
        Compute and return the current Merkle root hash as a hex string.

        This method is **non-destructive** — the frontier is not modified.

        Algorithm (O(log N))
        --------------------
        The frontier decomposes N leaves into complete power-of-two subtrees,
        one per set bit of N.  To collapse them into a single root we must
        replicate the same "odd-node duplication" rule used by the classic
        batch builder:

            When a level has an **odd** number of nodes, the rightmost node
            is paired with itself: parent = combine(node, node).

        Concretely, the frontier nodes sit at various depths d₀ < d₁ < … < dₖ
        (sorted ascending).  We fold them right-to-left (lowest depth first),
        promoting each partial subtree to the next depth by self-combining
        before merging it with the larger complete subtree on its left:

            accumulator = frontier[d₀]

            for each successive depth dᵢ (i = 1 … k):
                # Promote accumulator from its current depth up to dᵢ
                # by repeatedly self-combining (mirroring the batch tree's
                # odd-node duplication at each intermediate level).
                while current_depth < dᵢ:
                    accumulator = combine(accumulator, accumulator)
                    current_depth += 1

                # Merge: the complete subtree at dᵢ is on the LEFT
                accumulator = combine(frontier[dᵢ], accumulator)

        Why self-combine?
        -----------------
        In the batch tree, after all full pairs are consumed at level L,
        any leftover (odd) node is duplicated before ascending.  The frontier
        encodes exactly those "leftover" nodes.  If frontier[d₀] exists and
        the next frontier node is at d₁ > d₀+1, the batch tree would have
        duplicated the depth-d₀ subtree (d₁-d₀) times to produce a depth-d₁
        right child before combining with the depth-d₁ left sibling.

        Edge cases
        ----------
        * Zero leaves        → returns ``None``.
        * Single leaf        → returns SHA-256(leaf text).
        * Power-of-two count → frontier has 1 node, returned directly.

        Returns
        -------
        str or None
            64-character lowercase hex string, or ``None`` if empty.
        """
        if not self._frontier:
            return None

        # Sort depths ascending: smallest (rightmost partial) to largest (leftmost complete)
        sorted_depths = sorted(self._frontier.keys())

        # Seed the accumulator with the rightmost (lowest depth) frontier node
        accumulator: bytes = self._frontier[sorted_depths[0]]
        current_depth: int = sorted_depths[0]

        for target_depth in sorted_depths[1:]:
            # ----------------------------------------------------------------
            # Promote accumulator to target_depth by self-combining.
            # This mirrors the batch tree's "duplicate the odd node" rule:
            # at each intermediate level the partial right-edge subtree is
            # paired with itself before ascending one more level.
            # ----------------------------------------------------------------
            while current_depth < target_depth:
                accumulator = _combine(accumulator, accumulator)
                current_depth += 1

            # ----------------------------------------------------------------
            # Merge: the complete subtree in the frontier at target_depth
            # sits to the LEFT of our right-edge accumulator.
            # ----------------------------------------------------------------
            accumulator = _combine(self._frontier[target_depth], accumulator)
            # After the merge current_depth advances by one more level
            current_depth += 1

        return accumulator.hex()

    # ------------------------------------------------------------------
    # Convenience / introspection
    # ------------------------------------------------------------------

    @property
    def frontier_size(self) -> int:
        """Number of nodes currently stored in the frontier.

        Equals ``bin(leaf_count).count('1')`` — one node per set bit in the
        binary representation of the leaf count.  This is the canonical public
        way to observe frontier occupancy without accessing ``_frontier`` directly.
        """
        return len(self._frontier)

    @property
    def leaf_count(self) -> int:
        """Total number of leaves that have been appended."""
        return self._leaf_count

    @property
    def frontier_depth(self) -> int:
        """Current maximum depth of the frontier (0 if empty)."""
        return max(self._frontier.keys(), default=0)

    def __repr__(self) -> str:
        return (
            f"IncrementalMerkleTree("
            f"leaves={self._leaf_count}, "
            f"frontier_nodes={len(self._frontier)}, "
            f"max_depth={self.frontier_depth}"
            f")"
        )
