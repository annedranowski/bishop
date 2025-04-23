__all__ = (
    "expand_size_4",
)


def expand_size_4(
    shape: int | tuple[int, int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if isinstance(shape, int):
        return (shape, shape, shape, shape)
    elif len(shape) == 2:
        height, width = shape

        return (height, height, width, width)
    else:
        return shape
