"""Обнаружение границ сцен через PySceneDetect (только границы, без нарезки видео)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from scenedetect import ContentDetector, detect

from settings import SceneDetectionSettings


@dataclass(frozen=True)
class SceneSegment:
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


def detect_scenes(video_path: Path | str, scene_settings: SceneDetectionSettings) -> list[SceneSegment]:
    """Границы сцен в секундах; сцены короче min_scene_seconds отбрасываются."""
    _, segments = detect_scenes_with_counts(video_path, scene_settings)
    return segments


def detect_scenes_with_counts(
    video_path: Path | str,
    scene_settings: SceneDetectionSettings,
) -> tuple[int, list[SceneSegment]]:
    """
    PySceneDetect (ContentDetector), перевод таймкодов в секунды, фильтр по длительности.

    Возвращает (число сцен до фильтра min_scene_seconds, отфильтрованный список).
    """
    path_str = str(Path(video_path))
    scene_list = detect(
        path_str,
        ContentDetector(threshold=scene_settings.threshold),
        show_progress=scene_settings.show_progress,
    )
    raw_bounds: list[tuple[float, float]] = []
    for start_tc, end_tc in scene_list:
        s = float(start_tc.get_seconds())
        e = float(end_tc.get_seconds())
        if e <= s:
            continue
        raw_bounds.append((s, e))
    raw_count = len(raw_bounds)
    eps = 1e-9
    min_len = scene_settings.min_scene_seconds
    filtered = [
        SceneSegment(s, e)
        for s, e in raw_bounds
        if (e - s) >= min_len - eps
    ]
    return raw_count, filtered


__all__ = ["SceneSegment", "detect_scenes", "detect_scenes_with_counts"]
