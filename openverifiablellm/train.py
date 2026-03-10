import argparse
import logging
import os
from dataclasses import asdict

import numpy as np

# Require cuBLAS determinism config before CUDA initializes
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import random
from pathlib import Path
from typing import Tuple

import torch

from openverifiablellm.environment import generate_environment_fingerprint
from openverifiablellm.model import ModelConfig, VerifiableLLM
from openverifiablellm.utils import compute_merkle_root

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int):
    """
    Sets strictly deterministic seed across all layers of the stack.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":16:8", ":4096:8"):
        logger.warning("CUBLAS_WORKSPACE_CONFIG not set optimally. CUDA determinism may fail.")

    logger.info(f"Set deterministic seed to {seed} (Deterministic algorithms enabled)")


class DeterministicDataLoader:
    """
    A DataLoader that reads from a uint16 memory-mapped binary token file.
    Deterministic behavior is guaranteed based on the provided random state.
    """

    def __init__(
        self, data_path: Path, batch_size: int, seq_len: int, np_rng: np.random.RandomState
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.np_rng = np_rng

        # Memory map the array as little-endian uint16 ('<u2')
        self.data = np.memmap(data_path, dtype="<u2", mode="r")
        self.total_tokens = len(self.data)
        logger.info(f"Loaded {self.total_tokens} tokens from {data_path}")

    def get_batch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate random start indices
        max_start = self.total_tokens - self.seq_len
        if max_start <= 0:
            raise ValueError(f"Need at least seq_len + 1 tokens, got {self.total_tokens}.")
        ix = self.np_rng.randint(0, max_start, size=(self.batch_size,))

        x = np.stack([self.data[i : i + self.seq_len].astype(np.int64) for i in ix])
        y = np.stack([self.data[i + 1 : i + self.seq_len + 1].astype(np.int64) for i in ix])

        x_tensor = torch.from_numpy(x).to(device)
        y_tensor = torch.from_numpy(y).to(device)
        return x_tensor, y_tensor


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, output_dir: Path
):
    """
    Save the model and optimizer state. Then, compute the verifiable merkle root of the checkpoint.
    """
    checkpoint_path = output_dir / f"checkpoint_iter_{iteration:06d}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "model_config": asdict(model.config),
        "environment_fingerprint": generate_environment_fingerprint(),
    }

    torch.save(checkpoint, checkpoint_path)

    # Compute verifiable Merkle root of the checkpoint file
    m_root = compute_merkle_root(checkpoint_path)
    logger.info(f"Saved Checkpoint to {checkpoint_path} | Merkle Root: {m_root}")


def train(args):
    set_deterministic_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"Using device: {device}")

    # Init Configuration
    config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.seq_len,
    )

    model = VerifiableLLM(config)

    # Verifiable pure deterministic initialization
    logger.info(f"Initializing model deterministically with seed {args.seed}")
    model.init_weights_deterministically(args.seed)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    # Setup data loader
    np_rng = np.random.RandomState(args.seed)  # Separate deterministic generator for data
    data_loader = DeterministicDataLoader(
        Path(args.data_path), batch_size=args.batch_size, seq_len=args.seq_len, np_rng=np_rng
    )

    # Validate max token ID against vocab_size
    max_token_id = int(data_loader.data.max())
    if max_token_id >= args.vocab_size:
        raise ValueError(
            f"tokens.bin contains token id {max_token_id} which is >= vocab_size {args.vocab_size}. "
            f"Use the correct --vocab_size matching your tokenizer."
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model.train()

    logger.info("Starting deterministic training loop...")
    for step in range(1, args.max_steps + 1):
        x, y = data_loader.get_batch(device)

        optimizer.zero_grad(set_to_none=True)

        logits, loss = model(x, targets=y)
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        if step % args.log_interval == 0:
            logger.info(f"Step {step:05d} | Loss: {loss.item():.4f}")

        if step % args.save_interval == 0 or step == args.max_steps:
            save_checkpoint(model, optimizer, step, output_dir)

    logger.info("Training completed deterministically.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic LLM Training Loop")
    parser.add_argument("--data_path", type=str, required=True, help="Path to binary tokens.bin")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=256, help="Sequence length")
    parser.add_argument("--max_steps", type=int, default=1000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")

    # Small default config for testing purposes
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Tokenizer vocabulary size (must match the tokenizer used to produce tokens.bin)",
    )
    parser.add_argument("--cpu", action="store_true", help="Force CPU training")

    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval")
    parser.add_argument("--save_interval", type=int, default=500, help="Checkpoint save interval")

    args = parser.parse_args()

    train(args)
