__all__ = (
    "EncoderLayer",
)

from collections.abc import Callable
from typing import Sequence

from torch import Tensor, nn

from ...common import (
    Conv2d,
    Conv2dNoChannelsFixedPaddingProps,
    Pad2dPropsWithSame,
)

from ..attention import Attention, AttentionNoChannelsProps


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        attn_props: AttentionNoChannelsProps,
        feedforward_conv_props: Conv2dNoChannelsFixedPaddingProps,
        hidden: int | Sequence[int] = [],
        Activation: Callable[[], nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()  # type: ignore

        self.attn_props = attn_props
        self.embedding_size = embedding_size
        self.feedforward_conv_props = feedforward_conv_props
        self.hidden = [hidden] if isinstance(hidden, int) else hidden

        self.attention = Attention(
            embedding_size,
            embedding_size,
            **attn_props.asdict(),
        )

        self.activation = Activation()

        self.hidden_layers = nn.ModuleList(
            self._create_same_padding_conv2d(
                in_channels,
                out_channels,
                feedforward_conv_props,
            )
            for in_channels, out_channels in zip(
                [self.embedding_size, *self.hidden],
                [*self.hidden, self.embedding_size],
            )
        )

    def forward(self, input: Tensor) -> Tensor:
        if input.dim() != 4:
            raise ValueError(
                f"Expected 4D input of shape (batch, channel, height, width), got {input.dim()}D"
            )

        attention_out: Tensor = input + self.attention(input)
        feedforward_residual = attention_out.detach().clone()

        for layer in self.hidden_layers:
            feedforward_residual = self.activation(layer(feedforward_residual))

        output = attention_out + feedforward_residual

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
