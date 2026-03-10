import argparse
import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd, description, *, cwd=None):
    logger.info(f"--- Starting: {description} ---")
    logger.info(f"Command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
        logger.info(f"--- Completed: {description} ---\n")
    except subprocess.CalledProcessError as e:
        logger.error(f"--- Failed: {description} ---")
        logger.error(f"Error executing command: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-End Open-Everything Verifiable LLM Data Pipeline"
    )
    parser.add_argument(
        "dump_path", type=str, help="Path to the Wikipedia XML dump (e.g., .xml or .xml.bz2)"
    )

    args = parser.parse_args()
    dump_path = Path(args.dump_path).expanduser().resolve()

    if not dump_path.exists():
        logger.error(f"Input dump file not found: {dump_path}")
        sys.exit(1)

    # Paths
    project_root = Path(__file__).resolve().parent.parent
    clean_txt_path = project_root / "data" / "processed" / "wiki_clean.txt"
    tokenizer_dir = project_root / "data" / "tokenizer"
    tokens_bin_path = project_root / "data" / "tokens.bin"

    # Step 1: Preprocessing (Extract and Clean XML)
    cmd1 = [sys.executable, "-m", "openverifiablellm.utils", str(dump_path)]
    run_command(cmd1, "Step 1: Data Preprocessing (XML to Clean Text)", cwd=project_root)

    # Step 2: Train Tokenizer
    python_code = f"from openverifiablellm.tokenizer import train_tokenizer; train_tokenizer(r'{clean_txt_path}', r'{tokenizer_dir}')"
    cmd2 = [sys.executable, "-c", python_code]
    run_command(cmd2, "Step 2: Train BPE Tokenizer", cwd=project_root)

    # Step 3: Tokenize Data
    cmd3 = [
        sys.executable,
        "-m",
        "openverifiablellm.tokenize_data",
        "--text_path",
        str(clean_txt_path),
        "--tokenizer_path",
        str(tokenizer_dir),
        "--output_path",
        str(tokens_bin_path),
    ]
    run_command(cmd3, "Step 3: Tokenize Dataset to Binary format", cwd=project_root)

    # Step 4: Deterministic Training
    cmd4 = [
        sys.executable,
        "-m",
        "openverifiablellm.train",
        "--data_path",
        str(tokens_bin_path),
        "--output_dir",
        str(project_root / "data" / "checkpoints"),
        "--max_steps",
        "10",  # Just run 10 steps max for the full pipeline demo to ensure it works
        "--log_interval",
        "5",
        "--save_interval",
        "10",
    ]
    run_command(cmd4, "Step 4: Deterministic Training Loop", cwd=project_root)

    logger.info("=====================================================")
    logger.info(" Pipeline execution completed successfully!")
    logger.info("=====================================================")
    logger.info(f" Clean Text:      {clean_txt_path}")
    logger.info(f" Tokenizer Dir:   {tokenizer_dir}")
    logger.info(f" Binary Tokens:   {tokens_bin_path}")
    logger.info(f" Checkpoints:     {project_root / 'data' / 'checkpoints'}")


if __name__ == "__main__":
    main()
