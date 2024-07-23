# Copyright (c) 2020. Lightly AG and its affiliates.
# All Rights Reserved

from typing import Tuple, Union
from warnings import warn

import numpy as np
from PIL import ImageFilter
from PIL.Image import Image


class GaussianBlur:

    def __init__(
        self,
        kernel_size: Union[float, None] = None,
        prob: float = 0.5,
        scale: Union[float, None] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
    ):
        if scale != None or kernel_size != None:
            warn(
                "The 'kernel_size' and 'scale' arguments of the GaussianBlur augmentation will be deprecated.  "
                "Please use the 'sigmas' parameter instead.",
                DeprecationWarning,
            )
        self.prob = prob
        self.sigmas = sigmas

    def __call__(self, sample: Image) -> Image:
        prob = np.random.random_sample()
        if prob < self.prob:
            # choose randomized std for Gaussian filtering
            sigma = np.random.uniform(self.sigmas[0], self.sigmas[1])
            # PIL GaussianBlur https://github.com/python-pillow/Pillow/blob/76478c6865c78af10bf48868345db2af92f86166/src/PIL/ImageFilter.py#L154 label the
            # sigma parameter of the gaussian filter as radius. Before, the radius of the patch was passed as the argument.
            # The issue was addressed here https://github.com/lightly-ai/lightly/issues/1051 and solved by AurelienGauffre.
            return sample.filter(ImageFilter.GaussianBlur(radius=sigma))
        # return original image
        return sample
