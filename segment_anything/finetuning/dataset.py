from typing import Tuple

import numpy as np
from torch.utils.data import Dataset

import torch
import os
import cv2

from segment_anything.finetuning.transforms import build_all_layer_point_grids
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
        self.points_grid = build_all_layer_point_grids(n_per_side=32, n_layers=0, scale_per_layer=1)

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

    def prepare_coords(self, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        points_scale = np.array(image_size)[None, ::-1]
        points_for_image = self.points_grid[0] * points_scale

        point_coords = self.transform.apply_coords(points_for_image, image_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
        labels_torch = torch.ones(coords_torch.shape[0], dtype=torch.int)
        return coords_torch, labels_torch

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        image_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = self.read_image(image_path)
        mask = self.read_image(mask_path)
        mask = (mask > 126).astype(image.dtype)

        if self.augmentations:
            transformed = self.augmentations(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        coords_tensor, labels_tensor = self.prepare_coords(image.shape[:2])
        image_tensor = self.prepare_image(image)
        mask_tensor = self.prepare_mask(mask)

        return image_tensor, mask_tensor, coords_tensor, labels_tensor
