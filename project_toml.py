"""Загрузка project.toml (tomllib)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ProjectTomlError(ValueError):
    """Ошибка разбора или структуры project.toml."""


def _table(root: dict[str, Any], name: str) -> dict[str, Any]:
    t = root.get(name)
    if t is None:
        return {}
    if not isinstance(t, dict):
        raise ProjectTomlError(f"Секция [{name}] должна быть таблицей.")
    return t


def _bool(d: dict[str, Any], key: str, default: bool) -> bool:
    if key not in d:
        return default
    v = d[key]
    if isinstance(v, bool):
        return v
    raise ProjectTomlError(f"Поле {key!r}: ожидается true/false.")


@dataclass(frozen=True)
class MediapipePoseMetricsSettings:
    """Экспериментальные MediaPipe Pose метрики ([motion_analysis.mediapipe_pose_metrics] в project.toml)."""

    enabled: bool = False
    pose_detection_ratio: bool = False
    avg_visible_landmarks: bool = False
    pose_visibility_score: bool = False
    landmark_dropout_ratio: bool = False
    pose_tracking_stability: bool = False
    upper_body_motion_score: bool = False
    lower_body_motion_score: bool = False
    torso_motion_score: bool = False
    head_motion_score: bool = False
    pose_motion_direction_variance: bool = False
    pose_motion_periodicity: bool = False


def load_mediapipe_pose_metrics_settings(root: dict[str, Any]) -> MediapipePoseMetricsSettings:
    motion = _table(root, "motion_analysis")
    sub = motion.get("mediapipe_pose_metrics")
    if sub is None:
        sub = {}
    if not isinstance(sub, dict):
        raise ProjectTomlError(
            "Секция [motion_analysis.mediapipe_pose_metrics] должна быть таблицей."
        )
    return MediapipePoseMetricsSettings(
        enabled=_bool(sub, "enabled", False),
        pose_detection_ratio=_bool(sub, "pose_detection_ratio", False),
        avg_visible_landmarks=_bool(sub, "avg_visible_landmarks", False),
        pose_visibility_score=_bool(sub, "pose_visibility_score", False),
        landmark_dropout_ratio=_bool(sub, "landmark_dropout_ratio", False),
        pose_tracking_stability=_bool(sub, "pose_tracking_stability", False),
        upper_body_motion_score=_bool(sub, "upper_body_motion_score", False),
        lower_body_motion_score=_bool(sub, "lower_body_motion_score", False),
        torso_motion_score=_bool(sub, "torso_motion_score", False),
        head_motion_score=_bool(sub, "head_motion_score", False),
        pose_motion_direction_variance=_bool(sub, "pose_motion_direction_variance", False),
        pose_motion_periodicity=_bool(sub, "pose_motion_periodicity", False),
    )


_SELECTION_MODES = frozenset({"filtered", "review_all_scenes"})


@dataclass(frozen=True)
class SelectionSettings:
    """Режим отбора кандидатов ([selection] в project.toml)."""

    mode: str = "filtered"


@dataclass(frozen=True)
class AudioAnalysisProjectSettings:
    """Управление audio analysis в review_all_scenes ([audio_analysis] в project.toml)."""

    enabled: bool = True


def load_selection_settings(root: dict[str, Any]) -> SelectionSettings:
    sel = _table(root, "selection")
    mode = "filtered"
    if "mode" in sel:
        v = sel["mode"]
        if isinstance(v, bool):
            raise ProjectTomlError("Поле selection.mode: ожидается строка.")
        s = str(v).strip().lower()
        if s not in _SELECTION_MODES:
            opts = ", ".join(sorted(_SELECTION_MODES))
            raise ProjectTomlError(f"Поле selection.mode: неизвестное значение {v!r}; допустимо: {opts}.")
        mode = s
    return SelectionSettings(mode=mode)


def load_audio_analysis_project_settings(root: dict[str, Any]) -> AudioAnalysisProjectSettings:
    audio = _table(root, "audio_analysis")
    return AudioAnalysisProjectSettings(enabled=_bool(audio, "enabled", True))


@dataclass(frozen=True)
class MetricsSettings:
    """Опциональные метрики кандидата ([metrics] в project.toml; отсутствие ключа = false)."""

    audio_median: bool = False
    audio_percentile: bool = False
    audio_stddev: bool = False
    audio_peak_density: bool = False
    audio_peak_duration: bool = False
    audio_entropy: bool = False
    motion_median: bool = False
    motion_percentile: bool = False
    motion_stddev: bool = False
    motion_peak_density: bool = False
    motion_peak_duration: bool = False
    motion_entropy: bool = False


def load_metrics_settings(root: dict[str, Any]) -> MetricsSettings:
    m = _table(root, "metrics")
    return MetricsSettings(
        audio_median=_bool(m, "audio_median", False),
        audio_percentile=_bool(m, "audio_percentile", False),
        audio_stddev=_bool(m, "audio_stddev", False),
        audio_peak_density=_bool(m, "audio_peak_density", False),
        audio_peak_duration=_bool(m, "audio_peak_duration", False),
        audio_entropy=_bool(m, "audio_entropy", False),
        motion_median=_bool(m, "motion_median", False),
        motion_percentile=_bool(m, "motion_percentile", False),
        motion_stddev=_bool(m, "motion_stddev", False),
        motion_peak_density=_bool(m, "motion_peak_density", False),
        motion_peak_duration=_bool(m, "motion_peak_duration", False),
        motion_entropy=_bool(m, "motion_entropy", False),
    )


@dataclass(frozen=True)
class ProjectToml:
    """Пути и отладка из project.toml (относительные пути — от каталога файла).

    input_video — один видеофайл или каталог с файлами (только файлы этого уровня, без рекурсии).
    input_candidate_file — один JSON кандидата или каталог с любыми именами (*.json); каталог включает пакетный cut.
    """

    path: Path
    input_video: Path | None
    input_candidate_file: Path | None
    output_candidate_file: Path | None
    output_clips_dir: Path | None
    debug_enabled: bool
    debug_log_file: Path | None
    metrics: MetricsSettings
    mediapipe_pose_metrics: MediapipePoseMetricsSettings
    selection: SelectionSettings
    audio_analysis: AudioAnalysisProjectSettings


def _normalize_windows_path_string(s: str) -> str:
    """
    На Windows буква диска должна быть латиницей A–Z.
    Частая ошибка в TOML: кириллическая «С» (U+0421 / U+0441) вместо латинской «C».
    """
    t = s.strip()
    if len(t) >= 2 and t[1] == ":" and t[0] in ("\u0421", "\u0441"):
        return "C" + t[1:]
    return t


def _path_field(base: Path, d: dict[str, Any], key: str) -> Path | None:
    if key not in d:
        return None
    v = d[key]
    if v is None:
        return None
    if isinstance(v, bool):
        raise ProjectTomlError(f"Поле {key!r}: неверный тип.")
    s = _normalize_windows_path_string(str(v).strip())
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
        metrics=load_metrics_settings(data),
        mediapipe_pose_metrics=load_mediapipe_pose_metrics_settings(data),
        selection=load_selection_settings(data),
        audio_analysis=load_audio_analysis_project_settings(data),
    )


__all__ = [
    "AudioAnalysisProjectSettings",
    "MediapipePoseMetricsSettings",
    "MetricsSettings",
    "ProjectToml",
    "ProjectTomlError",
    "SelectionSettings",
    "load_audio_analysis_project_settings",
    "load_mediapipe_pose_metrics_settings",
    "load_metrics_settings",
    "load_selection_settings",
    "load_project_toml",
]
