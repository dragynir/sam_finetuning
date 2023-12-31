from typing import Tuple, Dict

import numpy as np
from torch.utils.data import Dataset

import torch
import os
import cv2
import pandas as pd

from segment_anything.finetuning.transforms import build_all_layer_point_grids
from segment_anything.utils.transforms import ResizeLongestSide


class SegmentationDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        model_input_size,
        preprocess_function,
        augmentations=None,
    ):
        self.df = df
        self.augmentations = augmentations
        self.transform = ResizeLongestSide(model_input_size)
        self.model_input_size = model_input_size
        self.preprocess_function = preprocess_function

    def __len__(self):
        return len(self.df)

    def read_image(self, image_path):
        return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    def prepare_image(self, image: np.ndarray) -> torch.Tensor:
        input_image = self.transform.apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()
        return self.preprocess_function(input_image_torch)

    def prepare_mask(self, mask: np.ndarray) -> torch.Tensor:
        input_mask = self.transform.apply_image(mask)
        input_mask_torch = torch.as_tensor(input_mask[:, :, 0], dtype=torch.float)
        return self.preprocess_function(input_mask_torch, normalize=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        image_path = self.df.iloc[idx, :]['image_path']
        mask_path = self.df.iloc[idx, :]['mask_path']

        image = self.read_image(image_path)
        mask = self.read_image(mask_path)
        mask = (mask > 126).astype(image.dtype)  # for bad jpeg masks

        if self.augmentations:
            transformed = self.augmentations(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']

        image_tensor = self.prepare_image(image)
        mask_tensor = self.prepare_mask(mask)

        return {'image': image_tensor, 'mask': mask_tensor}


class GuidedSegmentationDataset(SegmentationDataset):
    # TODO rename to GuidedSegmentationDataset
    # and add boxes guide training https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/
    # также добавить дефолтный бокс как с центральной точкой
    # все точки на изображении сеткой - по умолчанию не сегментирует ничего при их подаче
    # self.points_grid = build_all_layer_point_grids(n_per_side=32, n_layers=0, scale_per_layer=1)[0]
    # points in the center of image
    # self.points_grid = np.array([[0.5, 0.5]])
    # if points is None:
    #   points_scale = np.array(image_size)[None, ::-1]
    #     points = self.points_grid[0] * points_scale
    def prepare_coords(self, points_for_image: np.array, image_size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """"
        :param points_for_image: relative coords np.ndarray with shape (num_points, 2) and range [0, 1]
        """
        point_coords = self.transform.apply_coords(points_for_image, image_size)
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float)
        labels_torch = torch.ones(coords_torch.shape[0], dtype=torch.int)
        return coords_torch, labels_torch

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        image_path = self.df.iloc[idx, :]['image_path']
        mask_path = self.df.iloc[idx, :]['mask_path']
        points = self.df.loc[self.df[idx], 'points']

        image = self.read_image(image_path)
        mask = self.read_image(mask_path)
        mask = (mask > 126).astype(image.dtype)  # for bad jpeg masks

        if self.augmentations:
            # https://albumentations.ai/docs/getting_started/keypoints_augmentation/
            transformed = self.augmentations(image=image, mask=mask, keypoints=points)
            image = transformed['image']
            mask = transformed['mask']
            points = transformed.get('keypoints', None)

        image_size = image.shape[:2]
        points_tensor, points_labels_tensor = self.prepare_coords(points, image_size)
        image_tensor = self.prepare_image(image)
        mask_tensor = self.prepare_mask(mask)

        return {'image': image_tensor, 'mask': mask_tensor, 'points': points_tensor, 'points_labels': points_labels_tensor}
