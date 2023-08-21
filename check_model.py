import sys

from segment_anything.finetuning.transforms import build_all_layer_point_grids

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def main():
    sam_checkpoint = "../weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    predictor = SamPredictor(sam)

    image = cv2.imread('assets/img.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    # Выводы:
    # Делать grid из точек не правильно т к сетка тогда просто будет выделять все;
    # Надо пустой грид задавать, либо осмысленный.

    # Если совсем точек не подавать - отдает пустую маску.


    input_point = build_all_layer_point_grids(n_per_side=32, n_layers=0, scale_per_layer=1)
    # points_scale = np.array(image.shape[:2])[None, ::-1]
    # input_point = input_point[0] * points_scale
    # input_label = np.ones(input_point.shape[0])

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_points(input_point, input_label, plt.gca())
    plt.axis('on')
    plt.show()

    # TODO - сравнить маски на выходе из predict и train.py
    masks, scores, logits = predictor.predict(
        point_coords=None,  #input_point,
        point_labels=None,  #input_label,
        multimask_output=False,
    )

    print(masks.shape)

    plt.figure()
    plt.imshow(masks.squeeze())
    plt.show()


if __name__ == '__main__':
    main()
