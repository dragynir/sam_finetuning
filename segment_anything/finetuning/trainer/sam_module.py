from typing import Dict

import pytorch_lightning as pl
from datasets import load_metric

import torch
from torch import nn
from torch.nn import functional as F

import numpy as np

from segment_anything import sam_model_registry
from segment_anything.finetuning.config import Config


class GuidedSamFinetuner(pl.LightningModule):
    # https://blog.roboflow.com/how-to-train-segformer-on-a-custom-dataset-with-pytorch-lightning/
    def __init__(self, config: Config):
        super(GuidedSamFinetuner, self).__init__()
        self.metrics_interval = config.metrics_interval

        self.model = sam_model_registry[config.model_type](checkpoint=config.checkpoint)
        self.model.prompt_encoder.set_batch_size(config.batch_size)
        self.criterion = torch.nn.BCEWithLogitsLoss()  # TODO change to more complex loss

        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")

    def forward(
        self,
        images: torch.Tensor,
        masks: torch.Tensor,
        points: torch.Tensor,
        points_labels: torch.Tensor,
    ) -> torch.Tensor:

        with torch.no_grad():
            # disable gradients for image encoder
            image_embeddings = self.model.image_encoder(images)
            sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                points=(points, points_labels),
                boxes=None,
                masks=None,
            )

        dense_points_encoding = self.model.prompt_encoder.get_dense_pe()
        low_res_masks, iou_predictions = self.model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=dense_points_encoding,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        binary_masks = F.interpolate(
            low_res_masks,
            (self.model.image_encoder.img_size, self.model.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        return torch.squeeze(binary_masks, dim=1)  # add activation ???

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> Dict[str, torch.Tensor]:

        images, masks = batch['image'], batch['masks']
        points, points_labels = batch['points'], batch['points_labels']

        binary_masks = self(images, masks, points, points_labels)
        # Поменять функцию потерь на focal loss + з статьи что-то
        loss = self.criterion(binary_masks, masks)

        self.train_mean_iou.add_batch(
            predictions=binary_masks.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        return {'loss': loss}

    def on_train_epoch_end(self):

        metrics = self.train_mean_iou.compute(
            num_labels=1,  # one segmentation mask for sam output
            reduce_labels=False,
        )

        metrics = {"train_mean_iou": metrics["mean_iou"], "train_mean_accuracy": metrics["mean_accuracy"]}

        for k, v in metrics.items():
            self.log(k, v)
        return metrics

    def validation_step(self, batch, batch_nb):

        images, masks = batch['image'], batch['masks']
        points, points_labels = batch['points'], batch['points_labels']

        binary_masks = self(images, masks, points, points_labels)
        loss = self.criterion(binary_masks, masks)

        self.val_mean_iou.add_batch(
            predictions=binary_masks.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        return {'val_loss': loss}

    def on_validation_epoch_end(self):

        metrics = self.val_mean_iou.compute(
            num_labels=1,  # one segmentation mask for sam output
            reduce_labels=False,
        )

        metrics = {"val_mean_iou": metrics["mean_iou"], "val_mean_accuracy": metrics["mean_accuracy"]}

        for k, v in metrics.items():
            self.log(k, v)

        return metrics

    # def configure_optimizers(self):
    #     return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)

    def configure_optimizers(self):
        """Настройка оптимизатора и lr_scheduler."""
        optimizer = self.config.optimizer(
            [
                {'params': self.feature_extractor.parameters()}, {'params': self.arc_face_head.kernel},
            ],
            lr=self.config.lr * self.config.reduce_lr_factor,
        )

        scheduler = self.config.scheduler(optimizer, **self.config.scheduler_kwargs)
        lr_scheduler_config = {
            'scheduler': scheduler,
            'monitor': 'val_loss',
            'name': 'learning_rate',
            'interval': 'epoch',
            'frequency': 1,
        }
        return [optimizer], [lr_scheduler_config]
