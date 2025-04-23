__all__ = (
    "ImageLike",
)

from PIL.Image import Image as PILImage
from torch import Tensor

type ImageLike = PILImage | Tensor
