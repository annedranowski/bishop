__all__ = (
    "AttentionProps",
    "AttentionNoChannelsProps",
)

from dataclasses import dataclass

from utils import DataclassHelpersMixin

from ...common import Conv2dNoChannelsFixedPaddingProps


@dataclass
class AttentionProps(DataclassHelpersMixin):
    in_channels: int
    out_channels: int
    heads: int
    key_size: int
    key_conv_props: Conv2dNoChannelsFixedPaddingProps
    value_size: int
    value_conv_props: Conv2dNoChannelsFixedPaddingProps
    attn_conv_props: Conv2dNoChannelsFixedPaddingProps
    out_conv_props: Conv2dNoChannelsFixedPaddingProps


@dataclass
class AttentionNoChannelsProps(DataclassHelpersMixin):
    heads: int
    key_size: int
    key_conv_props: Conv2dNoChannelsFixedPaddingProps
    value_size: int
    value_conv_props: Conv2dNoChannelsFixedPaddingProps
    attn_conv_props: Conv2dNoChannelsFixedPaddingProps
    out_conv_props: Conv2dNoChannelsFixedPaddingProps
