__all__ = (
    "EncoderLayerProps",
)

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Sequence

from torch import nn

from utils import DataclassHelpersMixin

from ...common import Conv2dNoChannelsFixedPaddingProps
from ..attention import AttentionNoChannelsProps


@dataclass
class EncoderLayerProps(DataclassHelpersMixin):
    embedding_size: int
    attn_props: AttentionNoChannelsProps
    feedforward_conv_options: Conv2dNoChannelsFixedPaddingProps
    hidden: int | Sequence[int] = field(default_factory=lambda: [])
    Activation: Callable[[], nn.Module] = nn.GELU
