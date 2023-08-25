from typing import Tuple
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from segment_anything.finetuning.config import Config
from segment_anything.finetuning.trainer.datamodule import SamDatamodule
from segment_anything.finetuning.trainer.sam_module import GuidedSamFinetuner
from segment_anything.finetuning.trainer.utils import create_experiment


def train(config: Config) -> None:

    seed_everything(config.seed)

    experiment_dir, checkpoints_path = create_experiment(config)

    finetune_callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_loss',
            patience=config.early_stop_patience,
            verbose=False,
            mode='min',
        ),
        ModelCheckpoint(
            dirpath=checkpoints_path,
            filename='{epoch:02d}_{val_loss:.4f}',
            save_top_k=5,
            monitor='val_loss',
            mode='min',
        ),
    ]

    trainer = Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        max_epochs=config.max_epochs,
        callbacks=finetune_callbacks,
        logger=,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm='value',
        deterministic=True,
        log_every_n_steps=100,
    )
    datamodule = SamDatamodule(config)

    if config.module_checkpoint_path:
        model = GuidedSamFinetuner.load_from_checkpoint(
            config.module_checkpoint_path,
            config=config,
        )
    else:
        model = GuidedSamFinetuner(config)

    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    config = Config()
    train(config)
