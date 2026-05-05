"""Загрузка config.toml (tomllib)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from settings import SceneDetectionSettings, ToolSettings


class ConfigLoadError(ValueError):
    """Ошибка разбора или структуры config.toml."""


def _table(root: dict[str, Any], name: str) -> dict[str, Any]:
    t = root.get(name)
    if t is None:
        return {}
    if not isinstance(t, dict):
        raise ConfigLoadError(f"Секция [{name}] должна быть таблицей.")
    return t


def _str_opt(val: Any) -> str | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return None
    s = str(val).strip()
    return s or None


def _float(d: dict[str, Any], key: str, default: float) -> float:
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return float(v)
    raise ConfigLoadError(f"Поле {key!r}: ожидается число.")


def _bool(d: dict[str, Any], key: str, default: bool) -> bool:
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, bool):
        return v
    raise ConfigLoadError(f"Поле {key!r}: ожидается логическое значение.")


def load_config_toml(explicit_path: Path | None, default_path: Path) -> ToolSettings:
    """
    Прочитать объединённые секции [tools], [general], [audio_analysis], [scene_detection].
    Отсутствующий файл — значения по умолчанию; отсутствующие ключи — дефолты.
    """
    path = explicit_path if explicit_path is not None else default_path
    data: dict[str, Any] = {}
    if path.is_file():
        try:
            with path.open("rb") as f:
                raw = tomllib.load(f)
        except tomllib.TOMLDecodeError as e:
            raise ConfigLoadError(f"Некорректный TOML в {path}: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigLoadError("Корень config.toml должен быть таблицей.")
        data = raw

    tools = _table(data, "tools")
    general = _table(data, "general")
    audio = _table(data, "audio_analysis")
    scene = _table(data, "scene_detection")

    ffmpeg_path = _str_opt(tools.get("ffmpeg_path")) or "ffmpeg"
    ffprobe_path = _str_opt(tools.get("ffprobe_path"))

    overwrite = _bool(general, "overwrite", True)

    scene_block = SceneDetectionSettings(
        enabled=_bool(scene, "enabled", False),
        threshold=_float(scene, "threshold", 27.0),
        min_scene_seconds=_float(scene, "min_scene_seconds", 2.0),
        show_progress=_bool(scene, "show_progress", True),
        max_scene_seconds=_float(scene, "max_scene_seconds", 30.0),
    )

    tools_block = ToolSettings(
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        overwrite=overwrite,
        window_seconds=_float(audio, "window_seconds", 5.0),
        threshold=_float(audio, "threshold", 0.6),
        min_audio_coverage_ratio=_float(audio, "min_audio_coverage_ratio", 0.4),
        min_peak_score=_float(audio, "min_peak_score", 0.85),
        merge_gap_seconds=_float(audio, "merge_gap_seconds", 7.0),
        padding_before_seconds=_float(audio, "padding_before_seconds", 5.0),
        padding_after_seconds=_float(audio, "padding_after_seconds", 5.0),
        min_clip_seconds=_float(audio, "min_clip_seconds", 8.0),
        max_clip_seconds=_float(audio, "max_clip_seconds", 90.0),
        scene_detection=scene_block,
    )
    return tools_block


__all__ = ["ConfigLoadError", "load_config_toml"]
