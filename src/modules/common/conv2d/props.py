__all__ = (
    "Conv2dProps",
    "Conv2dNoChannelsProps",
    "Conv2dNoChannelsFixedPaddingProps",
)

from dataclasses import dataclass

from common.enums import PaddingMode
from utils import DataclassHelpersMixin

from ..pad2d import Pad2dPropsWithSame


@dataclass
class Conv2dProps(DataclassHelpersMixin):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    dilation: int | tuple[int, int] = 1
    padding_props: Pad2dPropsWithSame | None = None


@dataclass
class Conv2dNoChannelsProps(DataclassHelpersMixin):
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    dilation: int | tuple[int, int] = 1
    padding_props: Pad2dPropsWithSame | None = None


@dataclass
class Conv2dNoChannelsFixedPaddingProps(DataclassHelpersMixin):
    kernel_size: int | tuple[int, int]
    stride: int | tuple[int, int] = 1
    dilation: int | tuple[int, int] = 1
    padding_mode: PaddingMode = PaddingMode.ZEROS
