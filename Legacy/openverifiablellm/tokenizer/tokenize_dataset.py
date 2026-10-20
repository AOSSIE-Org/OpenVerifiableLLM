from datetime import datetime, timezone
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Dict, List, Optional, Union

import numpy as np

from openverifiablellm.manifest_chain import (
    get_parent_manifest_hash,
    verify_manifest_chain_link,
)
from openverifiablellm.utils import (
    compute_merkle_root,
    compute_sha256,
)
from openverifiablellm.verify import (
    CheckResult,
    CheckStatus,
    VerificationReport,
)

from .base import BaseTokenizer
from .factory import load_tokenizer
from .train import hash_tokenizer_config

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE_BYTES = 1048576  # 1 MB
SUPPORTED_DTYPES = {"uint16", "uint32"}
DTYPE_MAP = {
    "uint16": np.dtype("<u2"),
    "uint32": np.dtype("<u4"),
}
DTYPE_RANGES = {
    "uint16": (0, 65535),
    "uint32": (0, 4294967295),
}


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, sort_keys=True)


def tokenize_dataset(
    input_file: Union[str, Path],
    tokenizer: Union[BaseTokenizer, str, Path],
    output_file: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    previous_manifest_path: Optional[Union[str, Path]] = None,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    dtype: str = "uint32",
    write_manifest: bool = True,
) -> Dict[str, Any]:
    """
    Tokenize a text dataset in a deterministic, memory-efficient streaming fashion.

    Parameters
    ----------
    input_file : str or Path
        Path to the preprocessed dataset text file.
    tokenizer : BaseTokenizer or str or Path
        Tokenizer instance with encode() method, or path to trained tokenizer directory.
    output_file : str or Path
        Path where tokenized binary dataset (.bin) will be written.
    manifest_path : str or Path, optional
        Path where tokenized dataset manifest will be written. Defaults to output_file.parent / "tokenized_manifest.json".
    previous_manifest_path : str or Path, optional
        Path to the previous manifest (e.g. preprocessing dataset_manifest.json) to establish a cryptographic chain.
    chunk_size_bytes : int, default 1MB
        Byte size of each chunk for Merkle tree construction.
    dtype : str, default "uint32"
        Data type for stored token IDs ("uint16" or "uint32").
    write_manifest : bool, default True
        Whether to write the generated manifest JSON to disk.

    Returns
    -------
    dict
        Tokenization manifest dictionary with hashes, Merkle root, and metadata.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)

    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset file not found: {input_path}")

    def _is_same_file(p1: Path, p2: Path) -> bool:
        if p1.exists() and p2.exists():
            try:
                if p1.samefile(p2):
                    return True
            except OSError:
                pass
        try:
            return p1.resolve() == p2.resolve()
        except OSError:
            return p1.absolute() == p2.absolute()

    if _is_same_file(input_path, output_path):
        raise ValueError(f"Input file and output file cannot be the same file: {input_path}")

    target_manifest: Optional[Path] = None
    if write_manifest:
        target_manifest = Path(manifest_path) if manifest_path is not None else output_path.parent / "tokenized_manifest.json"
        if _is_same_file(input_path, target_manifest):
            raise ValueError(f"Input file and manifest file cannot be the same file: {target_manifest}")
        if _is_same_file(output_path, target_manifest):
            raise ValueError(f"Output file and manifest file cannot be the same file: {target_manifest}")

    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype: '{dtype}'. Supported dtypes are: {sorted(SUPPORTED_DTYPES)}"
        )

    if chunk_size_bytes <= 0:
        raise ValueError("chunk_size_bytes must be > 0")

    tokenizer_dir: Optional[Path] = None
    if isinstance(tokenizer, (str, Path)):
        tokenizer_dir = Path(tokenizer)
        tok_instance = load_tokenizer(tokenizer_dir)
    elif isinstance(tokenizer, BaseTokenizer):
        tok_instance = tokenizer
        if hasattr(tokenizer, "tokenizer_dir") and tokenizer.tokenizer_dir:
            tokenizer_dir = Path(tokenizer.tokenizer_dir)
    else:
        tok_instance = tokenizer

    if not hasattr(tok_instance, "encode") or not callable(getattr(tok_instance, "encode")):
        raise TypeError(
            f"Tokenizer instance must implement a callable encode() method, got {type(tok_instance).__name__}"
        )

    if write_manifest:
        if tokenizer_dir is None or not tokenizer_dir.is_dir():
            raise ValueError(
                "A valid tokenizer artifact directory must be provided when write_manifest=True "
                "to compute tokenizer provenance hashes. Pass write_manifest=False if skipping manifest."
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use explicit little-endian byte ordering for guaranteed cross-platform reproducibility
    np_dtype = DTYPE_MAP[dtype]
    min_val, max_val = DTYPE_RANGES[dtype]

    total_tokens = 0
    total_bytes = 0

    logger.info("Starting deterministic streaming tokenization: %s -> %s", input_path, output_path)

    tmp_out = tempfile.NamedTemporaryFile(
        dir=output_path.parent,
        prefix=f".{output_path.name}.tmp_",
        delete=False,
    )
    tmp_out_path = Path(tmp_out.name)
    input_hasher = hashlib.sha256()

    try:
        with input_path.open("rb") as fin_raw, tmp_out as fout:
            st = os.fstat(fin_raw.fileno())
            if not stat.S_ISREG(st.st_mode):
                raise ValueError(f"Input path must be a regular file: {input_path}")

            text_stream = io.TextIOWrapper(fin_raw, encoding="utf-8", errors="strict")
            for line in text_stream:
                text = line.strip()
                if not text:
                    continue

                encoded = tok_instance.encode(text)
                if isinstance(encoded, list):
                    token_ids = encoded
                elif hasattr(encoded, "ids"):
                    token_ids = encoded.ids
                else:
                    raise TypeError(
                        f"Tokenizer.encode() returned unsupported type: {type(encoded).__name__}. "
                        "Expected list of ints or object with 'ids' attribute."
                    )

                if not token_ids:
                    continue

                for tid in token_ids:
                    if not isinstance(tid, (int, np.integer)) or isinstance(tid, bool):
                        raise TypeError(f"Token ID must be an integer, got {type(tid).__name__}: {tid}")
                    if tid < min_val or tid > max_val:
                        raise ValueError(
                            f"Token ID {tid} is outside allowable range [{min_val}, {max_val}] for {dtype}"
                        )

                arr = np.array(token_ids, dtype=np_dtype)
                raw_bytes = arr.tobytes()
                fout.write(raw_bytes)

                total_tokens += len(token_ids)
                total_bytes += len(raw_bytes)

            fout.flush()
            os.fsync(fout.fileno())

            # Read and hash input_path through the same opened file handle
            fin_raw.seek(0)
            while block := fin_raw.read(65536):
                input_hasher.update(block)
            input_sha256 = input_hasher.hexdigest()

        tokenized_sha256 = compute_sha256(file_path=tmp_out_path)
        merkle_root = compute_merkle_root(tmp_out_path, chunk_size=chunk_size_bytes)

        if output_path.is_symlink():
            output_path.unlink()
        elif output_path.exists():
            if _is_same_file(input_path, output_path):
                raise ValueError(f"Input file and output file cannot be the same file: {input_path}")

        os.replace(tmp_out_path, output_path)
    except Exception:
        if tmp_out_path.exists():
            try:
                tmp_out_path.unlink()
            except OSError:
                pass
        raise

    parent_manifest_hash: Optional[str] = None
    if previous_manifest_path is not None:
        prev_path = Path(previous_manifest_path)
        if prev_path.is_file():
            parent_manifest_hash = get_parent_manifest_hash(prev_path)
        else:
            logger.warning("Previous manifest path specified but not found: %s", prev_path)

    tok_config: Optional[Dict[str, Any]] = None
    if tokenizer_dir is not None and tokenizer_dir.is_dir():
        tok_config = hash_tokenizer_config(tokenizer_dir)

    manifest_data: Dict[str, Any] = {
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dataset_path": str(input_path),
        "input_dataset_sha256": input_sha256,
        "tokenized_dataset_path": str(output_path),
        "tokenized_dataset_sha256": tokenized_sha256,
        "merkle_root": merkle_root,
        "chunk_size_bytes": chunk_size_bytes,
        "total_tokens": total_tokens,
        "total_bytes": total_bytes,
        "dtype": dtype,
        "tokenizer_config": tok_config,
        "parent_manifest_hash": parent_manifest_hash,
    }

    if write_manifest and target_manifest is not None:
        target_manifest.parent.mkdir(parents=True, exist_ok=True)
        tmp_mf = tempfile.NamedTemporaryFile(
            dir=target_manifest.parent,
            prefix=f".{target_manifest.name}.tmp_",
            mode="w",
            encoding="utf-8",
            delete=False,
        )
        tmp_mf_path = Path(tmp_mf.name)
        try:
            with tmp_mf as f_mf:
                f_mf.write(_canonical_json(manifest_data))
                f_mf.flush()
                os.fsync(f_mf.fileno())
            if target_manifest.is_symlink():
                target_manifest.unlink()
            os.replace(tmp_mf_path, target_manifest)
        except Exception:
            if tmp_mf_path.exists():
                try:
                    tmp_mf_path.unlink()
                except OSError:
                    pass
            raise
        logger.info("Saved tokenized dataset manifest to %s", target_manifest)

    return manifest_data


def verify_tokenized_dataset(
    tokenized_file: Union[str, Path],
    manifest_path: Union[str, Path],
    tokenizer_path: Optional[Union[str, Path]] = None,
    input_file: Optional[Union[str, Path]] = None,
    previous_manifest_path: Optional[Union[str, Path]] = None,
) -> VerificationReport:
    """
    Verify tokenized dataset binary artifact against its cryptographic manifest.

    Validates:
    - Tokenized binary file existence
    - Byte-level SHA256 integrity
    - Merkle root verification over chunk hashes
    - Tokenizer config hashes (if tokenizer_path provided)
    - Input dataset SHA256 integrity (if input_file provided)
    - Parent manifest chain linkage (if previous_manifest_path provided)

    Returns
    -------
    VerificationReport
        Detailed verification report with individual check results.
    """
    tok_file = Path(tokenized_file)
    mf_path = Path(manifest_path)

    report = VerificationReport(
        input_dump=str(tok_file),
        manifest_path=str(mf_path),
        previous_manifest_path=str(previous_manifest_path) if previous_manifest_path else None,
    )

    if not mf_path.is_file():
        report.add(
            CheckResult(
                name="manifest_exists",
                status=CheckStatus.FAIL,
                expected=str(mf_path),
                actual="FileNotFound",
                detail=f"Tokenized manifest not found at {mf_path}",
            )
        )
        return report

    try:
        manifest_data = json.loads(mf_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        report.add(
            CheckResult(
                name="manifest_json_valid",
                status=CheckStatus.FAIL,
                detail=f"Malformed manifest JSON: {e}",
            )
        )
        return report

    # 1. Check tokenized file existence
    if not tok_file.is_file():
        report.add(
            CheckResult(
                name="tokenized_file_exists",
                status=CheckStatus.FAIL,
                expected=str(tok_file),
                actual="FileNotFound",
                detail=f"Tokenized dataset binary not found at {tok_file}",
            )
        )
        return report
    else:
        report.add(CheckResult(name="tokenized_file_exists", status=CheckStatus.PASS))

    # 2. Check tokenized dataset SHA256
    expected_sha = manifest_data.get("tokenized_dataset_sha256")
    actual_sha = compute_sha256(file_path=tok_file)
    if expected_sha and actual_sha == expected_sha:
        report.add(CheckResult(name="tokenized_sha256", status=CheckStatus.PASS))
    else:
        report.add(
            CheckResult(
                name="tokenized_sha256",
                status=CheckStatus.FAIL,
                expected=expected_sha,
                actual=actual_sha,
                detail="Tokenized binary SHA256 does not match recorded manifest hash",
            )
        )

    # 3. Check Merkle root
    expected_merkle = manifest_data.get("merkle_root")
    chunk_size = manifest_data.get("chunk_size_bytes", DEFAULT_CHUNK_SIZE_BYTES)
    actual_merkle = compute_merkle_root(tok_file, chunk_size=chunk_size)
    if expected_merkle and actual_merkle == expected_merkle:
        report.add(CheckResult(name="merkle_root", status=CheckStatus.PASS))
    else:
        report.add(
            CheckResult(
                name="merkle_root",
                status=CheckStatus.FAIL,
                expected=expected_merkle,
                actual=actual_merkle,
                detail=f"Merkle root mismatch using chunk size {chunk_size} bytes",
            )
        )

    # 4. Optional: check input dataset SHA256
    if input_file is not None:
        in_path = Path(input_file)
        expected_in_sha = manifest_data.get("input_dataset_sha256")
        if not in_path.is_file():
            report.add(
                CheckResult(
                    name="input_dataset_sha256",
                    status=CheckStatus.FAIL,
                    expected=expected_in_sha,
                    actual="FileNotFound",
                    detail=f"Input file not found at {in_path}",
                )
            )
        else:
            actual_in_sha = compute_sha256(file_path=in_path)
            if expected_in_sha and actual_in_sha == expected_in_sha:
                report.add(CheckResult(name="input_dataset_sha256", status=CheckStatus.PASS))
            else:
                report.add(
                    CheckResult(
                        name="input_dataset_sha256",
                        status=CheckStatus.FAIL,
                        expected=expected_in_sha,
                        actual=actual_in_sha,
                        detail="Input dataset text SHA256 mismatch",
                    )
                )

    # 5. Optional: check tokenizer configuration hashes
    if tokenizer_path is not None:
        tok_dir = Path(tokenizer_path)
        expected_tok_config = manifest_data.get("tokenizer_config")
        if not tok_dir.is_dir():
            report.add(
                CheckResult(
                    name="tokenizer_config",
                    status=CheckStatus.FAIL,
                    detail=f"Tokenizer directory not found: {tok_dir}",
                )
            )
        else:
            try:
                actual_tok_config = hash_tokenizer_config(tok_dir)
                if expected_tok_config and actual_tok_config == expected_tok_config:
                    report.add(CheckResult(name="tokenizer_config", status=CheckStatus.PASS))
                else:
                    report.add(
                        CheckResult(
                            name="tokenizer_config",
                            status=CheckStatus.FAIL,
                            expected=str(expected_tok_config),
                            actual=str(actual_tok_config),
                            detail="Tokenizer configuration hashes do not match recorded config",
                        )
                    )
            except Exception as e:
                report.add(
                    CheckResult(
                        name="tokenizer_config",
                        status=CheckStatus.FAIL,
                        detail=f"Failed to hash tokenizer directory: {e}",
                    )
                )

    # 6. Optional: check cryptographic manifest chain link
    if previous_manifest_path is not None:
        prev_path = Path(previous_manifest_path)
        if not prev_path.is_file():
            report.add(
                CheckResult(
                    name="parent_manifest_chain",
                    status=CheckStatus.FAIL,
                    detail=f"Previous manifest file not found: {prev_path}",
                )
            )
        else:
            is_valid = verify_manifest_chain_link(prev_path, manifest_data)
            if is_valid:
                report.add(CheckResult(name="parent_manifest_chain", status=CheckStatus.PASS))
            else:
                expected_parent_hash = manifest_data.get("parent_manifest_hash")
                actual_parent_hash = get_parent_manifest_hash(prev_path)
                report.add(
                    CheckResult(
                        name="parent_manifest_chain",
                        status=CheckStatus.FAIL,
                        expected=expected_parent_hash,
                        actual=actual_parent_hash,
                        detail="Parent manifest cryptographic link mismatch",
                    )
                )

    return report
