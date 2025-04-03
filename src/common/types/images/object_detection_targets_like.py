__all__ = (
    "ObjectDetectionTargetsLike",
)

from typing import NotRequired, TypedDict

from torch import Tensor
# Torchvision doesn't have type hints
from torchvision import tv_tensors  # type: ignore


class _BoundingBoxesAndMasks(TypedDict):
    boxes: NotRequired[tv_tensors.BoundingBoxes | None]
    masks: NotRequired[Tensor | None]


type ObjectDetectionTargetsLike = tv_tensors.BoundingBoxes | _BoundingBoxesAndMasks
