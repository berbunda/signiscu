"""Модели данных для нарезки и кандидатов (без привязки к формату файла)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClipSpec:
    """Описание одного клипа."""

    name: str
    start: str
    end: str


@dataclass(frozen=True)
class Project:
    """Задача нарезки: исходное видео, каталог вывода, список клипов."""

    input_video: Path
    output_dir: Path
    clips: tuple[ClipSpec, ...]


__all__ = ["ClipSpec", "Project"]
