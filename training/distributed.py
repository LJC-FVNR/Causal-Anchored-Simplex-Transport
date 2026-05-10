from __future__ import annotations

import os
from typing import Tuple

import torch
import torch.distributed as dist


def setup_distributed() -> Tuple[torch.device, int, int, int, bool]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
        else:
            device = torch.device("cpu")
        distributed = True
    else:
        rank = 0
        world_size = 1
        local_rank = 0
        distributed = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device, rank, world_size, local_rank, distributed


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
