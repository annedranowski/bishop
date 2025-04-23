__all__ = (
    "PaddingMode",
)

from enum import Enum


class PaddingMode(Enum):
    ZEROS = "zeros"
    REFLECT = "reflect"
    REPLICATE = "replicate"
    CIRCULAR = "circular"
