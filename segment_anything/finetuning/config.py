from dataclasses import dataclass
from typing import Callable, Optional

import albumentations as A
from torch.utils.data import Dataset

from segment_anything.finetuning.data.dataset import GuidedSegmentationDataset


# todo переписать на hydra!

@dataclass
class Config:
    model_type: str = "vit_h"
    checkpoint: str = "../../../weights/sam_vit_h_4b8939.pth"
    module_checkpoint_path: Optional[str] = None
    experiments_dir: str = ''
    experiment_name: str = 'example'

    # images_dir: str = '../../../datasets/nails/images'
    # mask_dir: str = '../../../datasets/nails/masks'

    train_dataset_path = ''  # with folds
    test_dataset_path = ''
    dataset: Dataset = GuidedSegmentationDataset

    metrics_interval: int = 100

    model_input_size: int = 1024
    model_preprocess_function: Callable = None
    augmentations: A.Compose = None

    batch_size: int = 2
    early_stop_patience: int = 10
