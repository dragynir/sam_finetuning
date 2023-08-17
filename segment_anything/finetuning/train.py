import sys

sys.path.append("..")
from segment_anything import sam_model_registry, SamPredictor

import cv2
import numpy as np
import matplotlib.pyplot as plt


# пишу тут код, далее проверяю на датасете с масками, железо беру с kaggle
def main():
    sam_checkpoint = "../weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    # sam.to(device=device)
    predictor = SamPredictor(sam)


if __name__ == '__main__':
    main()
