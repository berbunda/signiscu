"""Загрузка config.toml (tomllib)."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from settings import (
    MotionAnalysisSettings,
    MotionBackendTuneSettings,
    SceneDetectionSettings,
    ToolSettings,
)


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


def _int(d: dict[str, Any], key: str, default: int) -> int:
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, bool):
        return default
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v)
    raise ConfigLoadError(f"Поле {key!r}: ожидается целое число.")


def _bool(d: dict[str, Any], key: str, default: bool) -> bool:
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, bool):
        return v
    raise ConfigLoadError(f"Поле {key!r}: ожидается логическое значение.")


def _motion_backend_tune(
    motion_tbl: dict[str, Any],
    subsection: str,
    *,
    default_motion_threshold: float,
    default_min_motion_peak_score: float,
) -> MotionBackendTuneSettings:
    sub = motion_tbl.get(subsection)
    if sub is None:
        sub = {}
    if not isinstance(sub, dict):
        raise ConfigLoadError(
            f"Секция [motion_analysis.{subsection}] должна быть таблицей."
        )
    return MotionBackendTuneSettings(
        motion_threshold=_float(sub, "motion_threshold", default_motion_threshold),
        min_motion_peak_score=_float(sub, "min_motion_peak_score", default_min_motion_peak_score),
    )


def _backend(d: dict[str, Any], key: str, default: str, allowed: frozenset[str]) -> str:
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, bool):
        raise ConfigLoadError(f"Поле {key!r}: ожидается строка.")
    s = str(v).strip().lower()
    if s not in allowed:
        opts = ", ".join(sorted(allowed))
        raise ConfigLoadError(f"Поле {key!r}: неизвестный backend {v!r}; допустимо: {opts}.")
    return s


def load_config_toml(explicit_path: Path | None, default_path: Path) -> ToolSettings:
    """
    Прочитать объединённые секции [tools], [general], [audio_analysis], [scene_detection], [motion_analysis].
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
    motion_tbl = _table(data, "motion_analysis")

    ffmpeg_path = _str_opt(tools.get("ffmpeg_path")) or "ffmpeg"
    ffprobe_path = _str_opt(tools.get("ffprobe_path"))

    overwrite = _bool(general, "overwrite", True)

    scene_block = SceneDetectionSettings(
        enabled=_bool(scene, "enabled", False),
        backend=_backend(scene, "backend", "ffmpeg", frozenset({"pyscenedetect", "ffmpeg"})),
        threshold=_float(scene, "threshold", 27.0),
        min_scene_seconds=_float(scene, "min_scene_seconds", 6.0),
        show_progress=_bool(scene, "show_progress", True),
        max_scene_seconds=_float(scene, "max_scene_seconds", 240.0),
        ffmpeg_scene_threshold=_float(scene, "ffmpeg_scene_threshold", 0.30),
        merge_short_scenes=_bool(scene, "merge_short_scenes", True),
        merge_short_scenes_target_seconds=_float(scene, "merge_short_scenes_target_seconds", 16.0),
    )
    motion_backend = _backend(
        motion_tbl,
        "backend",
        "optical_flow",
        frozenset({"optical_flow", "mediapipe_pose"}),
    )
    optical_flow_tune = _motion_backend_tune(
        motion_tbl,
        "optical_flow",
        default_motion_threshold=0.4,
        default_min_motion_peak_score=0.85,
    )
    mediapipe_pose_tune = _motion_backend_tune(
        motion_tbl,
        "mediapipe_pose",
        default_motion_threshold=0.05,
        default_min_motion_peak_score=0.15,
    )
    active_tune = (
        mediapipe_pose_tune if motion_backend == "mediapipe_pose" else optical_flow_tune
    )
    motion_block = MotionAnalysisSettings(
        enabled=_bool(motion_tbl, "enabled", False),
        backend=motion_backend,
        sample_fps=_float(motion_tbl, "sample_fps", 2.0),
        resize_width=_int(motion_tbl, "resize_width", 320),
        residual_percentile=_float(motion_tbl, "residual_percentile", 85.0),
        optical_flow=optical_flow_tune,
        mediapipe_pose=mediapipe_pose_tune,
        motion_threshold=active_tune.motion_threshold,
        min_motion_coverage_ratio=_float(motion_tbl, "min_motion_coverage_ratio", 0.3),
        min_motion_peak_score=active_tune.min_motion_peak_score,
        affect_selection=_bool(motion_tbl, "affect_selection", False),
        affect_score=_bool(motion_tbl, "affect_score", False),
        weight_motion=_float(motion_tbl, "weight_motion", 0.2),
        mediapipe_pose_model=_backend(
            motion_tbl,
            "mediapipe_pose_model",
            "lite",
            frozenset({"lite", "full", "heavy"}),
        ),
        mediapipe_min_detection_confidence=_float(
            motion_tbl, "mediapipe_min_detection_confidence", 0.5
        ),
        mediapipe_min_tracking_confidence=_float(
            motion_tbl, "mediapipe_min_tracking_confidence", 0.5
        ),
        mediapipe_visibility_threshold=_float(motion_tbl, "mediapipe_visibility_threshold", 0.5),
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
        motion_analysis=motion_block,
    )
    return tools_block


__all__ = ["ConfigLoadError", "load_config_toml"]
