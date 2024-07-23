from typing import List, Sequence, Union
import random
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms as T


class MultiViewTransform:

    def __init__(self, transforms: Sequence[T.Compose]):
        self.transforms = transforms

    def __call__(self, image: Union[Tensor, Image]) -> Union[List[Tensor], List[Image]]:
        return [transform(image) for transform in self.transforms]
