import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from segment_anything.finetuning.config import Config


class SamDatamodule(L.LightningDataModule):

    def __init__(self, config: Config, val_fold=0):
        super().__init__()

        self.config = config

        self.train_df = pd.read_csv(config.train_dataset_path)
        self.val_df = self.train_df[self.train_df['fold'] == val_fold]
        self.train_df = self.train_df[self.train_df['fold'] != val_fold]

        self.test_df = pd.read_csv(config.test_dataset_path)

    def setup(self, stage: str):
        """

        called on every process in DDP
        """
        self.train_dataset = self.config.dataset(
            df=self.train_df,
            model_input_size=self.config.model_input_size,
            preprocess_function=self.config.model_preprocess_function,
            augmentations=self.config.augmentations,
        )

        self.val_dataset = self.config.dataset(
            df=self.val_df,
            model_input_size=self.config.model_input_size,
            preprocess_function=self.config.model_preprocess_function,
            augmentations=None,
        )

        self.test_dataset = self.config.dataset(
            df=self.test_df,
            model_input_size=self.config.model_input_size,
            preprocess_function=self.config.model_preprocess_function,
            augmentations=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            pin_memory=True,
        )
