"""The token ids, batch layout and seeding that ovl.py shares with the data preparation.

Copied unchanged from ovl_pipeline (data.py and training.py) at the commit that prepared the
released data, 3f882ad. simple/rebuild_prep.sh checks out that commit to rebuild the data.
"""
import random

import numpy as np
import torch

PAD, BOS, EOS, USER, ASSISTANT, OFFSET = range(6)


def stream_arrays(directory):
    return (np.memmap(directory / "tokens.u16", dtype="<u2", mode="r"),
            np.memmap(directory / "mask.u8", dtype=np.uint8, mode="r"))


def _collate(windows, context):
    shape = (len(windows), context)
    out = {"inputs": torch.full(shape, PAD, dtype=torch.int64), "targets": torch.full(shape, PAD, dtype=torch.int64),
           "mask": torch.zeros(shape, dtype=torch.bool), "target_ids": torch.full(shape, -1, dtype=torch.int64)}
    for row, window in enumerate(windows):
        for name, arr in zip(out, window):
            out[name][row, :len(arr)] = torch.from_numpy(arr.copy())
    return out


def configure(seed):
    # Initialization arithmetic remains on CPU. The explicit GPU wrapper must
    # configure CUDA first; production admission remains a separate closed gate.
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float32)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
