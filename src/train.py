"""Train a DynaConTalk model: python src/train.py -cn dynacontalk_{edit,speech,face} [overrides]."""
from typing import List, Optional

import hydra
import lightning.pytorch as L
import rootutils
import torch
from lightning.pytorch import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

# adds the project root to PYTHONPATH and sets PROJECT_ROOT (used in configs/paths)
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig):
    torch.set_float32_matmul_precision("high")
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=False)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    # The released recipes assume a fixed effective batch (per-GPU batch x GPUs x grad
    # accumulation). Changing the GPU count without rebalancing the other two changes
    # the number of optimizer steps per epoch, so the epoch-based LR schedule no longer
    # matches the published runs.
    expected = cfg.get("expected_effective_batch")
    if expected:
        effective = int(cfg.data.batch_size) * int(trainer.world_size) * int(trainer.accumulate_grad_batches)
        if effective != int(expected):
            raise ValueError(
                f"effective batch = data.batch_size({cfg.data.batch_size}) x GPUs({trainer.world_size}) "
                f"x trainer.accumulate_grad_batches({trainer.accumulate_grad_batches}) = {effective}, "
                f"but this recipe expects {expected}. Adjust the three so their product is {expected}, "
                f"or set expected_effective_batch=null to train with a different batch on purpose."
            )
        log.info(f"Effective batch {effective} (matches the released recipe)")

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    log.info("Starting training!")
    ckpt_path = cfg.get("ckpt_path")
    if ckpt_path:
        # resuming needs the full checkpoint (optimizer and scheduler state), not weights only
        original_torch_load = torch.load
        torch.load = lambda *args, **kwargs: original_torch_load(*args, **{**kwargs, "weights_only": False})
        try:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        finally:
            torch.load = original_torch_load
    else:
        trainer.fit(model=model, datamodule=datamodule)
    return dict(trainer.callback_metrics), object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="dynacontalk_speech")
def main(cfg: DictConfig) -> Optional[float]:
    extras(cfg)
    train(cfg)


if __name__ == "__main__":
    main()
