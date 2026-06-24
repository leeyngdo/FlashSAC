from typing import Any, Union

from flash_rl.common.logger import NoOpLogger, TensorboardTrainerLogger, WandbTrainerLogger  # noqa

TrainerLogger = Union[WandbTrainerLogger, TensorboardTrainerLogger, NoOpLogger]


def create_logger(cfg: Any, is_main: bool = True) -> TrainerLogger:
    # In multi-GPU training only the main rank logs; other ranks discard metrics.
    if not is_main:
        return NoOpLogger()

    logger_type = getattr(cfg, "logger_type", "wandb")

    if logger_type == "wandb":
        return WandbTrainerLogger(cfg)
    elif logger_type == "tensorboard":
        return TensorboardTrainerLogger(cfg)
    else:
        raise ValueError


__all__ = [
    "WandbTrainerLogger",
    "TensorboardTrainerLogger",
    "NoOpLogger",
    "create_logger",
]
