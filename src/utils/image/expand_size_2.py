__all__ = (
    "expand_size_2",
)


def expand_size_2(shape: int | tuple[int, int]) -> tuple[int, int]:
    return (
        (shape, shape)
        if isinstance(shape, int)
        else shape
    )
