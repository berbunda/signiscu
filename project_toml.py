"""Загрузка project.toml (tomllib)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProjectTomlError(ValueError):
    """Ошибка разбора или структуры project.toml."""


@dataclass(frozen=True)
class ProjectToml:
    """Пути и отладка из project.toml (относительные пути — от каталога файла).

    input_video — один видеофайл или каталог с файлами (только файлы этого уровня, без рекурсии).
    """

    path: Path
    input_video: Path | None
    input_candidate_file: Path | None
    output_candidate_file: Path | None
    output_clips_dir: Path | None
    debug_enabled: bool
    debug_log_file: Path | None


def _table(root: dict[str, Any], name: str) -> dict[str, Any]:
    t = root.get(name)
    if t is None:
        return {}
    if not isinstance(t, dict):
        raise ProjectTomlError(f"Секция [{name}] должна быть таблицей.")
    return t


def _path_field(base: Path, d: dict[str, Any], key: str) -> Path | None:
    if key not in d:
        return None
    v = d[key]
    if v is None:
        return None
    if isinstance(v, bool):
        raise ProjectTomlError(f"Поле {key!r}: неверный тип.")
    s = str(v).strip()
    if not s:
        return None
    p = Path(s)
    if not p.is_absolute():
        p = (base / p).resolve()
    else:
        p = p.expanduser().resolve()
    return p


def load_project_toml(path: Path) -> ProjectToml:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл проекта не найден: {path}")

    base = path.parent
    try:
        with path.open("rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ProjectTomlError(f"Некорректный TOML в {path}: {e}") from e

    if not isinstance(data, dict):
        raise ProjectTomlError("Корень project.toml должен быть таблицей.")

    inp = _table(data, "input")
    out = _table(data, "output")
    dbg = _table(data, "debug")

    log_path = _path_field(base, dbg, "log_file")
    enabled = dbg.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ProjectTomlError('Поле [debug] enabled должно быть true/false.')

    return ProjectToml(
        path=path,
        input_video=_path_field(base, inp, "video"),
        input_candidate_file=_path_field(base, inp, "candidate_file"),
        output_candidate_file=_path_field(base, out, "candidate_file"),
        output_clips_dir=_path_field(base, out, "clips_dir"),
        debug_enabled=enabled,
        debug_log_file=log_path,
    )


__all__ = ["ProjectToml", "ProjectTomlError", "load_project_toml"]
