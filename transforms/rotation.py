from typing import Tuple, Union

import numpy as np
import torchvision.transforms as T
from PIL.Image import Image
from torch import Tensor
from torchvision.transforms import functional as TF


class RandomRotate:

    def __init__(self, prob: float = 0.5, angle: int = 20):
        self.prob = prob
        self.angle = angle

    def __call__(self, image: Union[Image, Tensor]) -> Union[Image, Tensor]:
        """Rotates the image with a given probability.

        Args:
            image:
                PIL image or tensor which will be rotated.

        Returns:
            Rotated image or original image.

        """
        prob = np.random.random_sample()
        if prob < self.prob:
            image = TF.rotate(image, self.angle)
        return image


class RandomRotateDegrees:

    def __init__(self, prob: float, degrees: Union[float, Tuple[float, float]]):
        self.transform = T.RandomApply([T.RandomRotation(degrees=degrees)], p=prob)

    def __call__(self, image: Union[Image, Tensor]) -> Union[Image, Tensor]:
        return self.transform(image)


def random_rotation_transform(
    rr_prob: float,
    rr_degrees: Union[None, float, Tuple[float, float]],
) -> Union[RandomRotate, T.RandomApply]:
    if rr_degrees is None:
        # Random rotation by 90 degrees.
        return RandomRotate(prob=rr_prob, angle=20)
    else:
        # Random rotation with random angle defined by rr_degrees.
        return RandomRotateDegrees(prob=rr_prob, degrees=rr_degrees)
