# import bz2
# import re
# import xml.etree.ElementTree as ET
# from pathlib import Path
# import sys


# def clean_wikitext(text: str) -> str:
#     # Remove templates {{...}}
#     text = re.sub(r"\{\{.*?\}\}", "", text, flags=re.DOTALL)

#     # Remove references <ref>...</ref>
#     text = re.sub(r"<ref.*?>.*?</ref>", "", text, flags=re.DOTALL)

#     # Remove HTML tags
#     text = re.sub(r"<.*?>", "", text)

#     # Convert [[Link|Text]] → Text
#     text = re.sub(r"\[\[.*?\|(.*?)\]\]", r"\1", text)

#     # Convert [[Link]] → Link
#     text = re.sub(r"\[\[(.*?)\]\]", r"\1", text)

#     # Remove multiple spaces/newlines
#     text = re.sub(r"\s+", " ", text)

#     return text.strip()


# def extract_text_from_xml(input_path, output_path):
#     Path(output_path).parent.mkdir(parents=True, exist_ok=True)

#     with bz2.open(input_path, "rb") as f:
#         context = ET.iterparse(f, events=("end",))

#         with open(output_path, "w", encoding="utf-8") as out:
#             for event, elem in context:
#                 if elem.tag.endswith("page"):
#                     text_elem = elem.find(".//{*}text")

#                     if text_elem is not None and text_elem.text:
#                         cleaned = clean_wikitext(text_elem.text)
#                         if cleaned:
#                             out.write(cleaned + "\n\n")

#                     elem.clear()


# if __name__ == "__main__":
#     input_path = sys.argv[1]
#     output_path = sys.argv[2]
#     extract_text_from_xml(input_path, output_path)
#     print("Preprocessing complete.")

import bz2
import re
import xml.etree.ElementTree as ET
from pathlib import Path
import sys
from generate_manifest import generate_manifest


def clean_wikitext(text: str) -> str:
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
            for event, elem in context:
                if elem.tag.endswith("page"):
                    text_elem = elem.find(".//{*}text")

                    if text_elem is not None and text_elem.text:
                        cleaned = clean_wikitext(text_elem.text)
                        if cleaned:
                            out.write(cleaned + "\n\n")

                    elem.clear()

    print(f"Preprocessing complete. Output saved to {output_path}")

    # 🔥 Automatically generate manifest
    generate_manifest(input_path, output_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess.py <input_dump>")
        sys.exit(1)

    extract_text_from_xml(sys.argv[1])
