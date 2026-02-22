## run via-
## python -m examples.demo_util examples\sample_wiki.xml.bz2

import sys
import logging
from openverifiablellm.utils import extract_text_from_xml

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m examples.demo_util <input_dump>")
        sys.exit(1)
        
    logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
    )
    extract_text_from_xml(sys.argv[1])