__all__ = (
    "Pad2d",
)

from collections.abc import Callable

from torch import Tensor, nn

from common.enums import PaddingMode
from utils import expand_size_4


class Pad2d(nn.Module):
    _padding_classes_by_mode: dict[PaddingMode, Callable[[tuple[int, int, int, int]], nn.Module]] = {
        PaddingMode.ZEROS: nn.ZeroPad2d,
        PaddingMode.REFLECT: nn.ReflectionPad2d,
        PaddingMode.REPLICATE: nn.ReplicationPad2d,
        PaddingMode.CIRCULAR: nn.CircularPad2d,
    }

    def __init__(
        self,
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        mode: PaddingMode = PaddingMode.ZEROS,
    ) -> None:
        super().__init__()  # type: ignore

        expanded_padding = expand_size_4(padding)
        reversed_padding = expanded_padding[2:] + expanded_padding[:2]

        self.mode = mode
        self.padding = reversed_padding
        self.inner_class = self._get_padding_class(self.mode)(self.padding)

    def forward(self, input: Tensor) -> Tensor:
        return self.inner_class(input)

    def _get_padding_class(self, mode: PaddingMode) -> Callable[[tuple[int, int, int, int]], nn.Module]:
        return self._padding_classes_by_mode[mode]
