import math
import torch.nn as nn
import random
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, List
import torchvision.transforms.functional as F


class RandomCrop:
    def __init__(self, size, scale, eye_region):
        super().__init__()
        self.eye_region = eye_region
        self.size = size
        self.scale = scale
        self.ratio = (1., 1.)

    @staticmethod
    def param(scale: List[float]) -> Tuple[int, int, int, int]:
        width = 224; height = 224
        area = height * width
        log_ratio = torch.log(torch.tensor((1., 1.)))
        for _ in range(1):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(
                torch.empty(1).uniform_(log_ratio[0], log_ratio[1])
            ).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

    @staticmethod
    def getboxs(box: List[float], eye_region: List[float]) -> Tuple[int, int, int, int]:
        i, j, h, w = box
        for _ in range(1):
            lx1, ly1, lx2, ly2, rx1, ry1, rx2, ry2 = eye_region
            if j < lx1 and j+w > lx2 and i < ly1 and i+h > ly2:
                # print('left')
                return i, j, h, w
            elif j < rx1 and j+w > rx2 and i < ry1 and i+h > ry2:
                # print('right')
                return i, j, h, w
        prob = np.random.random_sample()
        if prob <= 0.5:
            x = int(random.uniform(max(ly2-ly1, lx2-lx1), min(224-ly1, 224-lx1)))
            # print('full_eye_left')
            return ly1, lx1, x, x
        else:
            x = int(random.uniform(max(ry2-ry1, rx2-rx1), min(224-ry1, 224-rx1)))
            # print('full_eye_right')
            return ry1, rx1, x, x

    def __call__(self, img):
        box = self.param(scale=self.scale)
        i, j, h, w = self.getboxs(box, self.eye_region)
        return F.resized_crop(img, i, j, h, w, self.size)


class Centercrop:
    def __init__(self, size, scale):
        super().__init__()
        self.size = size
        self.scale = scale
        self.newsize = int(self.size * random.uniform(self.scale[0], self.scale[1]))

    def __call__(self, img):
        # print("center")
        img = F.center_crop(img, self.newsize)
        return F.resize(img, self.size)