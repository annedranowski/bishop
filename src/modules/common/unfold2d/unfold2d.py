__all__ = (
    "Unfold2d",
)

import typing
from torch import Tensor, nn

from utils.image import expand_size_2, get_convolution_size_reduction

from ..pad2d import Pad2d, Pad2dProps, Pad2dPropsWithSame


class Unfold2d(nn.Module):
    def __init__(
        self,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        padding_props: Pad2dPropsWithSame | None = None,
    ) -> None:
        super().__init__()  # type: ignore

        self.kernel_size = expand_size_2(kernel_size)
        self.stride = expand_size_2(stride)
        self.dilation = expand_size_2(dilation)

        if padding_props:
            if padding_props.padding == "same":
                size_reduction = get_convolution_size_reduction(
                    self.kernel_size,
                    self.stride,
                    (0, 0),
                )

                reduction_height, reduction_width = size_reduction

                normalized_padding_props = typing.cast(
                    Pad2dProps,
                    padding_props.replace(
                        padding=(
                            reduction_height // 2,
                            reduction_height - reduction_height // 2,
                            reduction_width // 2,
                            reduction_width - reduction_width // 2,
                        ),
                    ),
                )
            else:
                normalized_padding_props = padding_props

            self.padding_layer = Pad2d(**normalized_padding_props.asdict())
        else:
            self.padding_layer = nn.Identity()

        self.unfold_layer = nn.Unfold(
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.unfold_layer(self.padding_layer(input))
