from dataclasses import dataclass
import albumentations as A


@dataclass
class Config:
    model_type: str = "vit_h"
    checkpoint: str = "../../../weights/sam_vit_h_4b8939.pth"

    images_dir: str = '../../../datasets/nails/images'
    mask_dir: str = '../../../datasets/nails/masks'
    model_input_size: int = 1024
    batch_size: int = 4
    augmentations: A.Compose = None
