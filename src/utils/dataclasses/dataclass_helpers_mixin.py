__all__ = (
    "DataclassHelpersMixin",
)

import dataclasses
from typing import Any, Self


class DataclassHelpersMixin:
    def asdict(self) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            # DataclassInstance is not in typing yet
            for field in dataclasses.fields(self)  # type: ignore
        }

    def astuple(self) -> tuple[Any, ...]:
        return tuple(
            getattr(self, field.name)
            for field in dataclasses.fields(self)  # type: ignore
        )

    def asdict_recursive(self) -> dict[str, Any]:
        return dataclasses.asdict(self)  # type: ignore

    def astuple_recursive(self) -> tuple[Any, ...]:
        return dataclasses.astuple(self)  # type: ignore

    def replace(self, **changes: Any) -> Self:
        return dataclasses.replace(self, **changes)  # type: ignore
