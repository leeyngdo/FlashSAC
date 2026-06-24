"""Data-parallel multi-GPU helpers for FlashSAC.

The training entrypoint is launched with ``torchrun`` (one process per GPU). Each rank
runs its own simulator + replay buffer and the learner averages gradients across ranks,
so the optimizer step is mathematically equivalent to a single large-batch update.

Every function in this module is a no-op when training in a single process (``WORLD_SIZE``
unset or ``1``), so the single-GPU code path is unchanged.
"""

import os
from datetime import timedelta
from typing import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn


def local_rank() -> int:
    """Local GPU id within the node (set by ``torchrun``; ``0`` otherwise)."""
    return int(os.environ.get("LOCAL_RANK", "0"))


def rank() -> int:
    """Global rank across all nodes (``0`` when not distributed)."""
    return int(os.environ.get("RANK", "0"))


def world_size() -> int:
    """Total number of processes (``1`` when not distributed)."""
    return int(os.environ.get("WORLD_SIZE", "1"))


def is_main() -> bool:
    """Whether this process is the main logging/checkpointing rank."""
    return rank() == 0


def is_distributed() -> bool:
    """Whether more than one process is participating in training."""
    return world_size() > 1


def _distributed_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def resolve_device_type(device_type: str) -> str:
    """Resolve a device string to this process's local GPU.

    In distributed mode, any CUDA device string maps to ``cuda:{local_rank}`` so each
    rank binds its own GPU. In single-process mode, explicit CUDA indices are honored.
    Anything else falls back to ``"cpu"``.
    """
    if device_type.startswith("cuda"):
        if is_distributed():
            return f"cuda:{local_rank()}"
        if ":" in device_type:
            return device_type
        return f"cuda:{local_rank()}"
    return "cpu"


def init_process_group() -> None:
    """Initialize the default process group when running distributed.

    Must be called once per process. The CUDA device is pinned to ``LOCAL_RANK`` first so
    NCCL binds the correct GPU. A generous timeout keeps a slow rank-0 evaluation from
    tripping the collective watchdog.
    """
    if not is_distributed() or dist.is_initialized():
        return

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank())
    dist.init_process_group(
        backend="nccl" if use_cuda else "gloo",
        init_method="env://",
        world_size=world_size(),
        rank=rank(),
        timeout=timedelta(minutes=30),
    )


@torch.no_grad()
def broadcast_parameters_(modules: Iterable[nn.Module], src: int = 0) -> None:
    """Broadcast all parameters and buffers of each module from ``src`` in place.

    Guarantees every rank starts from identical weights regardless of per-rank seeding.
    No-op when not distributed.
    """
    if not _distributed_ready():
        return
    for module in modules:
        for tensor in list(module.parameters()) + list(module.buffers()):
            dist.broadcast(tensor.data, src=src)


@torch.no_grad()
def all_reduce_grads_average_(optimizer: torch.optim.Optimizer) -> None:
    """Average ``.grad`` of every optimizer parameter across ranks, in place.

    Call after ``backward()`` and before ``optimizer.step()``. Under AMP, call this
    before ``GradScaler.unscale_`` so an overflow on any rank is visible to every rank.
    No-op when training in a single process.
    """
    if not _distributed_ready():
        return
    num_ranks = dist.get_world_size()

    grads = [p.grad for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
    if not grads:
        return

    # Flatten first to avoid many small NCCL calls and to keep this backend-agnostic
    # (ReduceOp.AVG support differs across distributed backends / PyTorch versions).
    flat = torch.cat([grad.reshape(-1) for grad in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= float(num_ranks)

    offset = 0
    for grad in grads:
        numel = grad.numel()
        grad.copy_(flat[offset : offset + numel].view_as(grad))
        offset += numel


def barrier() -> None:
    """Synchronize all ranks. No-op when not distributed."""
    if _distributed_ready():
        dist.barrier()


def cleanup() -> None:
    """Destroy the default process group if one was created."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
