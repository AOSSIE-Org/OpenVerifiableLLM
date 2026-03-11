import argparse
import logging
import json
from pathlib import Path
from openverifiablellm.utils import extract_text_from_xml

logger = logging.getLogger(__name__)

"""
Experiment: Tamper Detection via Merkle Root Comparison

Run with:
python -m experiments.merkle_verification.test_merkle --path1 experiments/data_subset/sample_wiki.xml.bz2 --path2 experiments/data_subset/tampered_sample_wiki.xml.bz2

"""
MANIFEST_PATH = Path("data/dataset_manifest.json")

def run(path1):
    """Run preprocessing and return processed Merkle root."""
    extract_text_from_xml(path1)
    
    #read genertaed manifest
    with MANIFEST_PATH.open() as f:
        manifest = json.load(f)
        
    return {
        "raw_merkle_root": manifest["raw_merkle_root"],
        "processed_merkle_root": manifest["processed_merkle_root"]
    }

if __name__ == "__main__":
    
    parser= argparse.ArgumentParser(
        description= "Test tamper detection using Merkle root"
    )
    
    parser.add_argument("--path1",required=True,help="Original dataset")
    parser.add_argument("--path2",required=True,help="Tampered dataset")
    
    args= parser.parse_args()
    
    logging.basicConfig(
        level= logging.INFO,
        format="%(levelname)s - %(message)s"
    )
    
    root1 = run(args.path1)
    root2 = run(args.path2)

    print(f"\nRun 1 RAW Merkle root: {root1['raw_merkle_root']}")
    print(f"Run 2 RAW Merkle root: {root2['raw_merkle_root']}")
    
    print(f"\nRun 1 processed Merkle root: {root1['processed_merkle_root']}")
    print(f"Run 2 processed Merkle root: {root2['processed_merkle_root']}")

    if (
        root1["raw_merkle_root"] != root2["raw_merkle_root"] 
        or root1["processed_merkle_root"] != root2["processed_merkle_root"]
    ):
        print("\nTampering detected 🎉 (Merkle roots differ)")
    else:
        print("\nUnexpected result ❌ (Merkle roots identical)")