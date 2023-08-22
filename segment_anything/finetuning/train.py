import sys
from typing import List

import torch
from torch.utils.data import DataLoader

from segment_anything.finetuning.config import Config

sys.path.append("..")


from segment_anything.finetuning.dataset import SegmentationDataset
from segment_anything import sam_model_registry, SamPredictor

import cv2
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.functional import threshold, normalize


# пишу тут код, далее проверяю на датасете с масками, железо беру с kaggle


def main():
    config = Config()
    model = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
    model.prompt_encoder.set_batch_size(config.batch_size)
    model.train()

    dataset = SegmentationDataset(
        images_dir=config.images_dir,
        mask_dir=config.mask_dir,
        model_input_size=config.model_input_size,
        preprocess_function=model.preprocess,
        augmentations=config.augmentations
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)


    # Делать grid из точек не правильно т к сетка тогда просто будет выделять все;
    # Надо пустой грид задавать, либо осмысленный


    # check dataset
    # for batch in dataloader:
    #     images, masks = batch
    #     break

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for batch in dataloader:
        images, masks, coords, coords_labels = batch

        with torch.no_grad():
            image_embeddings = model.image_encoder(images)
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=None,  #(coords, coords_labels),
                boxes=None,
                masks=None,
            )
        # похоже тут надо передавать равномерное поле из точек или бокс на все изображение!!!

        low_res_masks, iou_predictions = model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        binary_masks = low_res_masks
        # binary_masks = normalize(threshold(low_res_masks, 0.0, 0))
        # need to upscale masks here to mask.shape
        loss = criterion(binary_masks, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    main()
