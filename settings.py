"""Настройки инструмента (значения после загрузки config.toml)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneDetectionSettings:
    enabled: bool
    backend: str
    threshold: float
    min_scene_seconds: float
    show_progress: bool
    max_scene_seconds: float
    ffmpeg_scene_threshold: float
    merge_short_scenes: bool
    merge_short_scenes_target_seconds: float


@dataclass(frozen=True)
class MotionBackendTuneSettings:
    motion_threshold: float
    min_motion_peak_score: float


@dataclass(frozen=True)
class MotionAnalysisSettings:
    enabled: bool
    backend: str
    sample_fps: float
    resize_width: int
    residual_percentile: float
    optical_flow: MotionBackendTuneSettings
    mediapipe_pose: MotionBackendTuneSettings
    motion_threshold: float
    min_motion_coverage_ratio: float
    min_motion_peak_score: float
    affect_selection: bool
    affect_score: bool
    weight_motion: float
    mediapipe_pose_model: str
    mediapipe_min_detection_confidence: float
    mediapipe_min_tracking_confidence: float
    mediapipe_visibility_threshold: float


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
    motion_analysis: MotionAnalysisSettings


__all__ = [
    "MotionAnalysisSettings",
    "MotionBackendTuneSettings",
    "SceneDetectionSettings",
    "ToolSettings",
]
