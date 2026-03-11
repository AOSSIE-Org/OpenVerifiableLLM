import sys
import json
import logging
from pathlib import Path
from openverifiablellm.utils import extract_text_from_xml

logger = logging.getLogger(__name__)

"""
Experiment to test Deterministic preprocessing, by compairing generated hash on 2 runs.

Run with:
    python -m experiments.preprocessing_determinism.test_preprocessing experiments/data_subset/sample_wiki.xml.bz2
"""
MANIFEST_PATH = Path("data/dataset_manifest.json")

def run(input_path):
    # Run preprocessing
    extract_text_from_xml(input_path)
    
    #read genertaed manifest
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)
        
    return {
        "processed_sha256": manifest["processed_sha256"],
        "processed_merkle_root": manifest["processed_merkle_root"],
        "environment_hash": manifest["environment_hash"],
    }


if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        print("Usage: python -m experiments.preprocessing_determinism.test_preprocessing <input_dump>")
        sys.exit(1)
        
    logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
    )
    
    result1= run(sys.argv[1])
    result2= run(sys.argv[1])
    
    print(f"\nRun 1 hash: {result1['processed_sha256']}")
    print(f"Run 2 hash: {result2['processed_sha256']}")
        
    if (
        result1["processed_sha256"] == result2["processed_sha256"]
        and result1["processed_merkle_root"] == result2["processed_merkle_root"]
        and result1["environment_hash"] == result2["environment_hash"]
    ):
        print("\nDeterministic preprocessing confirmed🎉")
    else:
        print("Hash didn't match❌")