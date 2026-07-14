from typing import Any

import gymnasium as gym

from ..types import NDArray
from .base_buffer import BaseBuffer, Batch  # noqa
from .numpy_buffer import NpyUniformBuffer
from .torch_buffer import (
    MemoryEfficientTorchUniformBuffer,
    TorchGeometricBuffer,
    TorchUniformBuffer,
)

__all__ = [
    "BaseBuffer",
    "Batch",
    "MemoryEfficientTorchUniformBuffer",
    "NpyUniformBuffer",
    "TorchUniformBuffer",
    "TorchGeometricBuffer",
    "create_buffer",
]


def create_buffer(
    buffer_class_type: str,
    buffer_type: str,
    observation_space: gym.spaces.Space[NDArray],
    action_space: gym.spaces.Space[NDArray],
    n_step: int,
    gamma: float,
    max_length: int,
    min_length: int,
    sample_batch_size: int,
    **kwargs: Any,
) -> BaseBuffer:
    if buffer_class_type == "numpy":
        if buffer_type == "uniform":
            return NpyUniformBuffer(
                observation_space=observation_space,
                action_space=action_space,
                n_step=n_step,
                gamma=gamma,
                max_length=max_length,
                min_length=min_length,
                sample_batch_size=sample_batch_size,
            )
        else:
            raise NotImplementedError
    elif buffer_class_type == "jax":
        raise NotImplementedError
    elif buffer_class_type == "torch":
        common = dict(
            observation_space=observation_space,
            action_space=action_space,
            n_step=n_step,
            gamma=gamma,
            max_length=max_length,
            min_length=min_length,
            sample_batch_size=sample_batch_size,
            device_type=kwargs["device_type"],
            obs_storage_dtype=kwargs.get("obs_storage_dtype"),
        )
        optimize_memory_usage = kwargs.get("optimize_memory_usage", False)
        if buffer_type == "uniform":
            buffer_cls = MemoryEfficientTorchUniformBuffer if optimize_memory_usage else TorchUniformBuffer
            return buffer_cls(**common)
        elif buffer_type in ("geometric", "exponential"):
            if optimize_memory_usage:
                raise NotImplementedError("memory-efficient geometric buffer is not implemented")
            return TorchGeometricBuffer(
                **common,
                geom_alpha=kwargs.get("geom_alpha", 10.0),
            )
        else:
            raise NotImplementedError(f"Unknown torch buffer_type: {buffer_type}")
    else:
        raise ValueError(f"Invalid buffer class type: {buffer_class_type}")
