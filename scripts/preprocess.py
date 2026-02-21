import bz2
import re
import defusedxml.ElementTree as ET
from pathlib import Path
import sys
from .generate_manifest import generate_manifest


def clean_wikitext(text: str) -> str:
    """
    Basic deterministic wikitext cleaning.

    NOTE:
    This implementation intentionally uses regex-based approximations
    for performance and determinism. It does NOT fully parse MediaWiki syntax.

    Known limitations:
    - Nested templates like {{Infobox | birth={{Date|1990|1|1}}}}
    are not fully handled. The non-greedy template regex may leave
    stray closing braces (e.g., "}}") in deeply nested structures.
    - Self-closing references such as <ref name="foo"/> are only partially
    handled. While generic tag stripping removes the tag itself,
    complex edge cases may not be fully normalized.
    - This is not a complete MediaWiki parser and should not be relied
    upon for perfectly structured wikitext normalization.

    These trade-offs are acceptable for v1 deterministic preprocessing.
    """
    text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)
    text = re.sub(r"<ref.*?>.*?</ref>", "", text, flags=re.DOTALL)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", text)
    text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_text_from_xml(input_path):
    input_path = Path(input_path)

    # Fixed output path
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "wiki_clean.txt"

    with bz2.open(input_path, "rb") as f:
        context = ET.iterparse(f, events=("end",))

        with open(output_path, "w", encoding="utf-8") as out:
            for _, elem in context:
                if elem.tag.endswith("page"):
                    text_elem = elem.find(".//{*}text")

                    if text_elem is not None and text_elem.text:
                        cleaned = clean_wikitext(text_elem.text)
                        if cleaned:
                            out.write(cleaned + "\n\n")

                    elem.clear()

    print(f"Preprocessing complete. Output saved to {output_path}")

    # Automatically generate manifest
    generate_manifest(input_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <input_dump>")
        sys.exit(1)

    extract_text_from_xml(sys.argv[1])
