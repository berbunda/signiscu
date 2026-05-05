"""Настройки инструмента (значения после загрузки config.toml)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneDetectionSettings:
    enabled: bool
    threshold: float
    min_scene_seconds: float
    show_progress: bool
    max_scene_seconds: float


@dataclass(frozen=True)
class ToolSettings:
    ffmpeg_path: str
    ffprobe_path: str | None
    overwrite: bool
    window_seconds: float
    threshold: float
    min_audio_coverage_ratio: float
    min_peak_score: float
    merge_gap_seconds: float
    padding_before_seconds: float
    padding_after_seconds: float
    min_clip_seconds: float
    max_clip_seconds: float
    scene_detection: SceneDetectionSettings


__all__ = ["SceneDetectionSettings", "ToolSettings"]
