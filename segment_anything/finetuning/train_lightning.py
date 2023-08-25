from clearml import Task
from configs.base import Config
from configs import config
from typing import Tuple
from lightning import Trainer, seed_everything
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from new_src.callbacks.logger import ClearmlImageLogger, ClearmlLogger
from new_src.callbacks.model import SaveJitModel, FeatureExtractorFreeze, FeatureExtractorFreezeConvolutions
from new_src.trainer.datamodule import FacesDataModule
from new_src.trainer.module import VerificationModelTuning, VerificationModelTraining
from new_src.trainer.utils import create_experiment


def train(config: Config) -> None:

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
        SaveJitModel(checkpoints_path, experiment_dir),
    ]

    trainer = Trainer(
        accelerator=config.accelerator,
        devices=config.devices,
        strategy=config.strategy,
        max_epochs=config.feature_extractor_epochs,
        callbacks=finetune_callbacks,
        logger=,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm='value',
        deterministic=True,
        log_every_n_steps=100,
    )
    datamodule = FacesDataModule(config)
    model = VerificationModelTuning.load_from_checkpoint(
        best_model_path,
        config=config,
        class_counts=datamodule.class_counts,
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    config = Config()
    train(config)
