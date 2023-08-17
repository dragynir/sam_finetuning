from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

import torch
import os
import cv2
from segment_anything.utils.transforms import ResizeLongestSide


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir,
        mask_dir,
        model_input_size,
        preprocess_function,
        augmentations=None,
    ):
        self.images_dir = images_dir
        self.mask_dir = mask_dir
        self.augmentations = augmentations
        self.images = os.listdir(self.images_dir)
        self.transform = ResizeLongestSide(model_input_size)
        self.model_input_size = model_input_size
        self.preprocess_function = preprocess_function

    def __len__(self):
        return len(self.images)

    def read_image(self, image_path):
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    def prepare_image(self, image: np.ndarray) -> torch.Tensor:
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        return self.preprocess_function(input_image_torch)

    def prepare_mask(self, image: np.ndarray) -> torch.Tensor:
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image[:, :, 0])
        return self.preprocess_function(input_image_torch, normalize=False)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = self.read_image(image_path)
        mask = self.read_image(mask_path)
        mask = (mask > 126).astype(image.dtype)

        if self.augmentations:
            transformed = self.augmentations(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image_tensor = self.prepare_image(image)
        mask_tensor = self.prepare_mask(mask)

        return image_tensor, mask_tensor
