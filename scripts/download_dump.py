"""
download_dump.py
================
Dedicated script for downloading Wikipedia XML dump files.

Downloads the specified Wikipedia dump and verifies its integrity
using an MD5 checksum fetched from Wikimedia's checksum file.

Usage
-----
# Download latest Simple English Wikipedia (default)
    python scripts/download_dump.py

# Download a specific wiki
    python scripts/download_dump.py --wiki simplewiki

# Download a specific dated snapshot
    python scripts/download_dump.py --wiki simplewiki --date 20260201

# Choose output directory
    python scripts/download_dump.py --output-dir data/raw

# Skip checksum verification (not recommended)
    python scripts/download_dump.py --no-verify
"""

import argparse
import hashlib
import logging
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)

# Constants

WIKIMEDIA_BASE = "https://dumps.wikimedia.org"
DUMP_FILENAME_TEMPLATE = "{wiki}-{date}-pages-articles.xml.bz2"
CHECKSUM_FILENAME_TEMPLATE = "{wiki}-{date}-md5sums.txt"

DEFAULT_WIKI = "simplewiki"
DEFAULT_DATE = "latest"
DEFAULT_OUTPUT_DIR = Path(".")


# Helpers


def _build_urls(wiki: str, date: str) -> tuple[str, str]:
    """
    Return (dump_url, checksum_url) for the given wiki and date.

    When date is 'latest', Wikimedia redirects to the most recent snapshot.
    """
    dump_filename = DUMP_FILENAME_TEMPLATE.format(wiki=wiki, date=date)
    checksum_filename = CHECKSUM_FILENAME_TEMPLATE.format(wiki=wiki, date=date)

    base = f"{WIKIMEDIA_BASE}/{wiki}/{date}"
    dump_url = f"{base}/{dump_filename}"
    check_url = f"{base}/{checksum_filename}"

    return dump_url, check_url


def _download_file(url: str, dest: Path) -> None:
    """
    Download *url* to *dest*, showing a simple progress indicator.
    Resumes are not supported — the file is always written fresh.
    """
    logger.info("Downloading: %s", url)
    logger.info("Destination: %s", dest)

    dest.parent.mkdir(parents=True, exist_ok=True)

    def _progress(block_count, block_size, total_size):
        if total_size > 0:
            downloaded = block_count * block_size
            pct = min(downloaded / total_size * 100, 100)
            mb_done = downloaded / 1_048_576
            mb_total = total_size / 1_048_576
            print(
                f"\r  {pct:5.1f}%  {mb_done:.1f} / {mb_total:.1f} MB",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_progress)
    except Exception as exc:
        # Clean up partial file so a retry starts fresh
        if dest.exists():
            dest.unlink()
        raise RuntimeError(f"Download failed: {exc}") from exc

    print()  # newline after progress bar
    logger.info("Download complete: %s (%.1f MB)", dest, dest.stat().st_size / 1_048_576)


def _fetch_expected_md5(checksum_url: str, dump_filename: str) -> str | None:
    """
    Fetch Wikimedia's MD5 checksum file and extract the hash for *dump_filename*.
    Returns None if the file or the specific entry cannot be found.
    """
    try:
        logger.info("Fetching checksums from: %s", checksum_url)
        with urllib.request.urlopen(checksum_url, timeout=30) as resp:
            content = resp.read().decode("utf-8")
    except Exception as exc:
        logger.warning("Could not fetch checksum file (%s) — skipping verification.", exc)
        return None

    # Format: "<md5hash>  <filename>\n"
    for line in content.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and parts[1].strip() == dump_filename:
            return parts[0].strip()

    logger.warning("Checksum entry for '%s' not found in checksum file.", dump_filename)
    return None


def _compute_md5(file_path: Path) -> str:
    """Compute the MD5 hash of a file in streaming fashion."""
    md5 = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(8192):
            md5.update(chunk)
    return md5.hexdigest()


def _verify_checksum(dest: Path, checksum_url: str) -> bool:
    """
    Download the Wikimedia MD5 checksum file and verify *dest* against it.
    Returns True if verification passes or if the checksum cannot be fetched
    (non-fatal — a warning is logged).
    Returns False only if we got a checksum and it does not match.
    """
    expected = _fetch_expected_md5(checksum_url, dest.name)
    if expected is None:
        logger.error("Checksum verification failed: expected hash unavailable.")
        return False

    logger.info("Verifying checksum ...")
    actual = _compute_md5(dest)

    if actual == expected:
        logger.info("Checksum OK  ✓  (MD5: %s)", actual)
        return True

    logger.error(
        "Checksum MISMATCH — file may be corrupt.\n  expected : %s\n  actual   : %s",
        expected,
        actual,
    )
    return False


# Public API


def download_dump(
    wiki: str = DEFAULT_WIKI,
    date: str = DEFAULT_DATE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    verify: bool = True,
) -> Path:
    """
    Download a Wikipedia XML dump and (optionally) verify its MD5 checksum.

    Parameters
    ----------
    wiki :
        Wiki identifier, e.g. 'simplewiki', 'enwiki'.
    date :
        Snapshot date string 'YYYYMMDD', or 'latest' for the most recent dump.
    output_dir :
        Directory where the .bz2 file will be saved.
    verify :
        If True, fetch Wikimedia's MD5 checksum file and verify the download.

    Returns
    -------
    Path
        Absolute path to the downloaded dump file.

    Raises
    ------
    RuntimeError
        If the download fails or the checksum does not match.
    """
    dump_filename = DUMP_FILENAME_TEMPLATE.format(wiki=wiki, date=date)
    dest = Path(output_dir).resolve() / dump_filename

    dump_url, checksum_url = _build_urls(wiki, date)

    # Skip download if file already exists and checksum passes
    if dest.exists():
        logger.info("File already exists: %s", dest)
        if verify:
            ok = _verify_checksum(dest, checksum_url)
            if ok:
                logger.info("Existing file is valid — skipping download.")
                return dest
            logger.warning("Existing file failed checksum — re-downloading.")
            dest.unlink()
        else:
            logger.info("Skipping checksum — using existing file.")
            return dest

    _download_file(dump_url, dest)

    if verify:
        ok = _verify_checksum(dest, checksum_url)
        if not ok:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                "Downloaded file failed MD5 verification. The file has been removed. Please retry."
            )

    return dest


# CLI


def main(argv=None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download a Wikipedia XML dump from Wikimedia.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--wiki",
        default=DEFAULT_WIKI,
        help="Wiki identifier (default: simplewiki)",
    )
    parser.add_argument(
        "--date",
        default=DEFAULT_DATE,
        help="Snapshot date YYYYMMDD or 'latest' (default: latest)",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save the dump file (default: current directory)",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip MD5 checksum verification after download",
    )

    args = parser.parse_args(argv)

    try:
        dest = download_dump(
            wiki=args.wiki,
            date=args.date,
            output_dir=Path(args.output_dir),
            verify=not args.no_verify,
        )
        print(f"\nDump ready at: {dest}")
        print("\nNext step — run preprocessing from the repository root:")
        print(f'  python -m openverifiablellm.utils "{dest}"')

    except RuntimeError as exc:
        logger.error("%s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nDownload cancelled.")
        sys.exit(1)


if __name__ == "__main__":
    main()
