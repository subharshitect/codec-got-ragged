"""Small tqdm wrapper with a no-dependency fallback."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import TypeVar


T = TypeVar("T")


class _NoProgress:
    def __init__(self, iterable: Iterable[T] | None = None, **_: object) -> None:
        self.iterable = iterable

    def __iter__(self) -> Iterator[T]:
        return iter(self.iterable or [])

    def __enter__(self) -> "_NoProgress":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def update(self, _: int = 1) -> None:
        return None


try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = _NoProgress
