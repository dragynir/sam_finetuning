from dataclasses import dataclass
from typing import Callable

import albumentations as A
from torch.utils.data import Dataset

from segment_anything.finetuning.data.dataset import GuidedSegmentationDataset


# todo переписать на hydra!

@dataclass
class Config:
    model_type: str = "vit_h"
    checkpoint: str = "../../../weights/sam_vit_h_4b8939.pth"

    # images_dir: str = '../../../datasets/nails/images'
    # mask_dir: str = '../../../datasets/nails/masks'

    train_dataset_path = ''  # with folds
    test_dataset_path = ''
    dataset: Dataset = GuidedSegmentationDataset

    model_input_size: int = 1024
    model_preprocess_function: Callable = None
    batch_size: int = 2
    augmentations: A.Compose = None
