__all__ = (
    "Unfold2dProps",
)

from dataclasses import dataclass

from utils import DataclassHelpersMixin

from ..pad2d import Pad2dPropsWithSame


@dataclass
class Unfold2dProps(DataclassHelpersMixin):
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    dilation: int | tuple[int, int] = 1
    padding_props: Pad2dPropsWithSame | None = None
