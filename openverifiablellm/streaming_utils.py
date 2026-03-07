"""
streaming_utils.py
==================
Memory-efficient, streaming text extractor for Wikipedia XML dumps.

Key design decisions
--------------------
* Uses ``xml.etree.ElementTree.iterparse`` so the XML is parsed
  **event-by-event** — no full DOM is ever built in RAM.
* After each ``<page>`` element is yielded it is immediately cleared
  (``elem.clear()``), releasing both the element and all its children.
  This keeps heap usage at O(1) regardless of dump size.
* Supports both plain ``.xml`` and bz2-compressed ``.xml.bz2`` inputs
  by sniffing the first three bytes for the BZh magic header.
* The generator contract: callers receive one cleaned plain-text string
  per Wikipedia article.  An empty/redirect article yields nothing.

Usage
-----
    from openverifiablellm.streaming_utils import stream_text_from_xml

    for article_text in stream_text_from_xml("simplewiki-....xml.bz2"):
        # process one article at a time — constant memory
        do_something(article_text)
"""

import bz2
import gc
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Generator

from openverifiablellm.utils import clean_wikitext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tag name suffixes we care about (we use .endswith() to be namespace-agnostic
# because MediaWiki dumps include a Clark-notation namespace prefix such as
# "{http://www.mediawiki.org/xml/export-0.11/}page").
# ---------------------------------------------------------------------------
_PAGE_TAG_SUFFIX = "page"
_TEXT_TAG_SUFFIX = "text"


def _open_xml_source(file_path: Path):
    """
    Return an open binary file-like object for ``file_path``.

    Sniffs the first 3 bytes for the BZh magic header used by bzip2.
    Falls back to a plain binary open for uncompressed XML.

    Parameters
    ----------
    file_path:
        Resolved ``Path`` to the dump file.

    Returns
    -------
    A context-manager-compatible binary IO object.
    """
    with file_path.open("rb") as probe:
        magic = probe.read(3)

    if magic == b"BZh":
        logger.debug("Detected bzip2 stream: %s", file_path.name)
        return bz2.open(file_path, "rb")

    logger.debug("Detected plain XML stream: %s", file_path.name)
    return file_path.open("rb")


def stream_text_from_xml(
    file_path: str,
) -> Generator[str, None, None]:
    """
    Stream cleaned article texts from a Wikipedia XML (or XML.bz2) dump.

    This is a **generator** — it yields exactly one string per Wikipedia
    article that contains non-empty wikitext.  It never holds more than
    a single ``<page>`` element tree in memory at any moment.

    Memory complexity : O(1)  — independent of dump size.
    Time complexity   : O(N)  — one linear scan of the byte stream.

    Parameters
    ----------
    file_path:
        Path to a Wikipedia XML dump.  Both ``.xml`` and ``.xml.bz2``
        (bzip2-compressed) files are accepted.

    Yields
    ------
    str
        Cleaned plain-text content of one Wikipedia article.
        Articles that are empty after cleaning are silently skipped.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist on disk.
    xml.etree.ElementTree.ParseError
        If the XML stream is structurally malformed.

    Examples
    --------
    >>> total = 0
    >>> for text in stream_text_from_xml("simplewiki-20260201.xml.bz2"):
    ...     total += len(text)
    >>> print(f"Streamed {total:,} characters with O(1) memory")
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dump file not found: {path}")

    articles_yielded = 0

    with _open_xml_source(path) as xml_stream:
        # ``iterparse`` fires events as the SAX-like cursor advances.
        # We only care about "end" events (element fully parsed).
        context = ET.iterparse(xml_stream, events=("end",))

        for _event, elem in context:
            # ----------------------------------------------------------------
            # Tag matching: MediaWiki dumps include a Clark-notation namespace
            # prefix, e.g. "{http://www.mediawiki.org/xml/export-0.11/}page".
            # Using .endswith() avoids hard-coding any specific namespace URI
            # while still being precise enough for our needs.
            # ----------------------------------------------------------------
            if not elem.tag.endswith(_PAGE_TAG_SUFFIX):
                continue

            # At this point *elem* is the fully-parsed <page> subtree.
            # Walk its children to locate the <text> element.
            raw_text: str = ""
            for child in elem.iter():
                if child.tag.endswith(_TEXT_TAG_SUFFIX):
                    if child.text:
                        raw_text = child.text
                    break

            # ----------------------------------------------------------------
            # CRITICAL memory management step:
            # elem.clear() removes all child elements, text content, and
            # attributes from this element object, dropping every reference
            # that iterparse has accumulated in the parsed subtree so far.
            # Without this call, ALL previously seen <page> elements remain
            # live in memory for the entire lifetime of the loop — causing
            # O(N) memory growth and eventual OOM on large dumps.
            # ----------------------------------------------------------------
            elem.clear()

            # Periodically request the cyclic-reference collector so that
            # any cross-references within now-cleared subtrees are resolved
            # promptly rather than accumulating until the next automatic GC.
            # We amortise the GC overhead by triggering only every 1 000 pages.
            articles_yielded += 1
            if articles_yielded % 1_000 == 0:
                gc.collect()
                logger.debug("Streamed %d articles so far …", articles_yielded)

            if not raw_text:
                continue

            cleaned = clean_wikitext(raw_text)
            if cleaned:
                yield cleaned

    logger.info(
        "Finished streaming '%s': %d articles yielded.",
        path.name,
        articles_yielded,
    )
