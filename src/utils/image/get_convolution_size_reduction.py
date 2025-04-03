__all__ = (
    "get_convolution_size_reduction",
)

from typing import Sequence


def get_convolution_size_reduction(
    kernel_size: Sequence[int],
    dilation: Sequence[int],
    padding: Sequence[int],
) -> Sequence[int]:
    if len(kernel_size) != len(dilation):
        raise ValueError(
            "kernel_size and dilation must have an equal number of dimensions"
        )

    if len(padding) == len(kernel_size):
        padding = [element for element in padding[::-1] for _ in range(2)]

    if len(padding) != len(kernel_size) * 2:
        raise ValueError(
            "kernel_size and padding must have an equal number of dimensions"
        )

    return [
        (kernel_size - 1) * dilation - padding_before - padding_after
        for (kernel_size, dilation, padding_before, padding_after) in zip(
            kernel_size[::-1],
            dilation[::-1],
            *([iter(padding)] * 2)
        )
    ]
