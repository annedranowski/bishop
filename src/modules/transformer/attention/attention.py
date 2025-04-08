__all__ = (
    "Attention",
)

import typing

import torch
from torch import Tensor, nn

from utils import expand_size_2

from ...common import (
    Conv2d,
    Conv2dNoChannelsFixedPaddingProps,
    Pad2dPropsWithSame,
    Unfold2d,
)


class Attention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int,
        key_size: int,
        key_conv_props: Conv2dNoChannelsFixedPaddingProps,
        value_size: int,
        value_conv_props: Conv2dNoChannelsFixedPaddingProps,
        attn_conv_props: Conv2dNoChannelsFixedPaddingProps,
        out_conv_props: Conv2dNoChannelsFixedPaddingProps,
    ) -> None:
        super().__init__()  # type: ignore

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.key_size = key_size
        self.key_conv_props = key_conv_props
        self.value_size = value_size
        self.value_conv_props = value_conv_props
        self.attn_conv_props = attn_conv_props
        self.out_conv_props = out_conv_props

        self.key_conv = self._create_same_padding_conv2d(
            in_channels,
            key_size * heads,
            key_conv_props,
        )

        self.query_conv = self._create_same_padding_conv2d(
            in_channels,
            key_size * heads,
            key_conv_props,
        )

        self.value_conv = self._create_same_padding_conv2d(
            in_channels,
            value_size * heads,
            value_conv_props,
        )

        self.attn_unfold = self._create_same_padding_unfold2d(
            attn_conv_props,
        )

        self.out_conv = self._create_same_padding_conv2d(
            value_size * heads,
            out_channels,
            out_conv_props,
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() != 4:
            raise ValueError(
                f"Expected 4D input of shape (batch, channel, height, width), got {input.dim()}D"
            )

        samples = input.shape[0]

        key: Tensor = self.key_conv(input)
        query: Tensor = self.query_conv(input)
        value: Tensor = self.value_conv(input)

        attn_shape = key.shape[-2:]
        attn_kernel_size = expand_size_2(self.attn_conv_props.kernel_size)
        attn_kernel_area = attn_kernel_size[0] * attn_kernel_size[1]

        key_for_scoring = (
            typing.cast(Tensor, self.attn_unfold(key))
            .reshape(
                samples,
                self.heads,
                self.key_size,
                attn_kernel_area,
                *attn_shape,
            )
            .permute(0, 1, 4, 5, 3, 2)
        )

        query_for_scoring = (
            query
            .reshape(
                samples,
                self.heads,
                self.key_size,
                1,
                *attn_shape,
            )
            .permute(0, 1, 4, 5, 3, 2)
        )

        score_scale_factor = torch.sqrt(torch.tensor(self.key_size).float())
        score_logits = query_for_scoring @ key_for_scoring.transpose(-1, -2)
        scaled_score_logits = score_logits / score_scale_factor
        score = torch.softmax(scaled_score_logits, dim=-1)

        unfolded_value = (
            typing.cast(Tensor, self.attn_unfold(value))
            .reshape(
                samples,
                self.heads,
                self.value_size,
                attn_kernel_area,
                *attn_shape,
            )
            .permute(0, 1, 4, 5, 3, 2)
        )

        attention = (
            (score @ unfolded_value)
            .squeeze(-2)
            .permute(0, 1, 4, 2, 3)
            .reshape(
                samples,
                self.value_size * self.heads,
                *attn_shape,
            )
        )

        output = self.out_conv(attention)

        return output

    def _create_same_padding_conv2d(
        self,
        in_channels: int,
        out_channels: int,
        props: Conv2dNoChannelsFixedPaddingProps,
    ) -> Conv2d:
        props_dict = props.asdict()
        padding_mode = props_dict.pop("padding_mode")

        return Conv2d(
            in_channels,
            out_channels,
            **props_dict,
            padding_props=Pad2dPropsWithSame(
                mode=padding_mode,
                padding="same",
            ),
        )

    def _create_same_padding_unfold2d(self, props: Conv2dNoChannelsFixedPaddingProps) -> Unfold2d:
        props_dict = props.asdict()
        padding_mode = props_dict.pop("padding_mode")

        return Unfold2d(
            **props_dict,
            padding_props=Pad2dPropsWithSame(
                mode=padding_mode,
                padding="same",
            ),
        )
