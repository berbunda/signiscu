"""Общие утилиты для прогресс-баров (stderr + tqdm при наличии)."""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")


def progress_to_stderr() -> bool:
    """Показывать интерактивный tqdm только когда stderr — терминал."""
    return sys.stderr.isatty()


def tqdm_labeled(iterable_: Iterable[T], *, desc: str, unit: str, total: int | None = None) -> Iterable[T]:
    disabled = not progress_to_stderr()
    try:
        from tqdm import tqdm
    except ImportError:
        return iterable_

    return tqdm(
        iterable_,
        desc=desc,
        unit=unit,
        total=total,
        file=sys.stderr,
        disable=disabled,
        leave=True,
        ascii=True,
    )


@contextmanager
def scenedetect_progress_compat(*, enabled: bool) -> Iterator[None]:
    """
    На время вызова PySceneDetect подменяет tqdm и шаблон подписи в scenedetect.scene_manager,
    чтобы полоса совпадала с tqdm_labeled (stderr, ascii, leave, отключение без TTY).
    """
    if not enabled:
        yield
        return
    import scenedetect.scene_manager as sm

    orig_tqdm = sm.tqdm
    orig_desc = sm.PROGRESS_BAR_DESCRIPTION

    def wrapped_tqdm(*args, **kwargs):
        try:
            from tqdm import tqdm
        except ImportError:
            return orig_tqdm(*args, **kwargs)
        k = dict(kwargs)
        k["file"] = sys.stderr
        k["ascii"] = True
        k["leave"] = True
        k["dynamic_ncols"] = False
        k["disable"] = (not progress_to_stderr()) or k.get("disable", False)
        if k.get("unit") == "frames":
            k["unit"] = "кадр"
        return tqdm(*args, **k)

    sm.tqdm = wrapped_tqdm
    sm.PROGRESS_BAR_DESCRIPTION = "[generate] Сцены: PySceneDetect | срезов: %d"
    try:
        yield
    finally:
        sm.tqdm = orig_tqdm
        sm.PROGRESS_BAR_DESCRIPTION = orig_desc


__all__ = ["progress_to_stderr", "scenedetect_progress_compat", "tqdm_labeled"]
