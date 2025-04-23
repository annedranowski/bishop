__all__ = (
    "Pad2dProps",
    "Pad2dPropsWithSame",
)

from dataclasses import dataclass
from typing import Literal

from common.enums import PaddingMode
from utils import DataclassHelpersMixin


@dataclass
class Pad2dProps(DataclassHelpersMixin):
    mode: PaddingMode = PaddingMode.ZEROS
    padding: int | tuple[int, int] | tuple[int, int, int, int] = 0


@dataclass
class Pad2dPropsWithSame(DataclassHelpersMixin):
    mode: PaddingMode = PaddingMode.ZEROS

    padding: (
        int
        | tuple[int, int]
        | tuple[int, int, int, int]
        | Literal["same"]
    ) = 0
