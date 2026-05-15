"""Обнаружение границ сцен: PySceneDetect или FFmpeg (только границы, без нарезки)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ffmpeg_utils import ffprobe_duration_seconds, run_ffmpeg_capture_stderr
from settings import SceneDetectionSettings

_PTS_TIME_RE = re.compile(r"pts_time:([0-9]+(?:\.[0-9]+)?)")
_SCENE_BACKENDS = frozenset({"pyscenedetect", "ffmpeg"})


@dataclass(frozen=True)
class SceneSegment:
    start_seconds: float
    end_seconds: float

    @property
    def duration_seconds(self) -> float:
        return max(0.0, self.end_seconds - self.start_seconds)


def _unknown_backend_message(backend: str) -> str:
    opts = ", ".join(sorted(_SCENE_BACKENDS))
    return f"Неизвестный scene_detection.backend: {backend!r}. Допустимо: {opts}."


def _filter_by_min_scene_seconds(
    raw_bounds: list[tuple[float, float]],
    min_scene_seconds: float,
) -> tuple[int, list[SceneSegment]]:
    raw_count = len(raw_bounds)
    eps = 1e-9
    filtered = [
        SceneSegment(s, e)
        for s, e in raw_bounds
        if (e - s) >= min_scene_seconds - eps
    ]
    return raw_count, filtered


def _dedupe_sorted_timestamps(times: list[float], *, eps: float = 1e-6) -> list[float]:
    if not times:
        return []
    out = [times[0]]
    for t in times[1:]:
        if t - out[-1] > eps:
            out.append(t)
    return out


def _segments_from_cut_timestamps(cuts: list[float], duration: float) -> list[tuple[float, float]]:
    """Сцены: 0 → cut₁, cut₁ → cut₂, …, last_cut → duration."""
    eps = 1e-9
    if duration <= eps:
        return []
    if not cuts:
        return [(0.0, duration)]
    bounds: list[tuple[float, float]] = []
    start = 0.0
    for t in cuts:
        if t <= start + eps or t >= duration - eps:
            continue
        bounds.append((start, t))
        start = t
    if start < duration - eps:
        bounds.append((start, duration))
    return bounds


def _parse_showinfo_pts_times(output: str, duration: float) -> list[float]:
    eps = 1e-9
    raw: list[float] = []
    for m in _PTS_TIME_RE.finditer(output):
        t = float(m.group(1))
        if t <= eps or t >= duration - eps:
            continue
        raw.append(t)
    raw.sort()
    return _dedupe_sorted_timestamps(raw)


def _detect_scenes_pyscenedetect(
    video_path: Path | str,
    scene_settings: SceneDetectionSettings,
) -> tuple[int, list[SceneSegment]]:
    from scenedetect import ContentDetector, detect

    from progress_ui import scenedetect_progress_compat

    path_str = str(Path(video_path))
    with scenedetect_progress_compat(enabled=scene_settings.show_progress):
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
    return _filter_by_min_scene_seconds(raw_bounds, scene_settings.min_scene_seconds)


def _detect_scenes_ffmpeg(
    video_path: Path,
    scene_settings: SceneDetectionSettings,
    *,
    ffmpeg_bin: Path,
    ffprobe_bin: Path,
    debug_lines: list[str] | None = None,
) -> tuple[int, list[SceneSegment], list[float]]:
    duration, dur_err = ffprobe_duration_seconds(ffprobe_bin, video_path)
    if duration is None or duration <= 0:
        raise RuntimeError(dur_err or "Не удалось получить длительность видео через ffprobe.")

    thr = scene_settings.ffmpeg_scene_threshold
    filter_expr = f"select='gt(scene,{thr})',showinfo"
    cmd = [
        str(ffmpeg_bin),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-i",
        str(video_path),
        "-filter:v",
        filter_expr,
        "-f",
        "null",
        "-",
    ]
    stderr_text, returncode = run_ffmpeg_capture_stderr(
        cmd,
        duration_sec=duration,
        show_progress=scene_settings.show_progress,
        progress_desc="[generate] Сцены: FFmpeg",
    )
    hay = stderr_text
    if returncode != 0 and not _PTS_TIME_RE.search(hay):
        tail = hay.strip().splitlines()
        msg = tail[-1][:500] if tail else f"ffmpeg код {returncode}"
        raise RuntimeError(msg)

    timestamps = _parse_showinfo_pts_times(hay, duration)
    raw_bounds = _segments_from_cut_timestamps(timestamps, duration)
    raw_count, filtered = _filter_by_min_scene_seconds(raw_bounds, scene_settings.min_scene_seconds)

    if debug_lines is not None:
        debug_lines.append("--- Scene detection (FFmpeg) ---")
        debug_lines.append("  scene backend = ffmpeg")
        debug_lines.append(f"  ffmpeg_scene_threshold = {thr}")
        debug_lines.append(f"  raw scene timestamps = {len(timestamps)}")
        debug_lines.append(f"  scenes before min_scene_seconds = {raw_count}")
        debug_lines.append(f"  scenes after min_scene_seconds = {len(filtered)}")
        if timestamps:
            preview = timestamps[:10]
            debug_lines.append(f"  first scene boundaries (cuts, s): {', '.join(f'{t:.3f}' for t in preview)}")
        else:
            debug_lines.append("  first scene boundaries (cuts, s): (none)")
        debug_lines.append(f"  ffmpeg command = {' '.join(cmd)}")
        debug_lines.append("")

    return raw_count, filtered, timestamps


def detect_scenes(video_path: Path | str, scene_settings: SceneDetectionSettings, **kwargs) -> list[SceneSegment]:
    """Границы сцен в секундах; сцены короче min_scene_seconds отбрасываются."""
    _, segments = detect_scenes_with_counts(video_path, scene_settings, **kwargs)
    return segments


def detect_scenes_with_counts(
    video_path: Path | str,
    scene_settings: SceneDetectionSettings,
    *,
    ffmpeg_bin: Path | None = None,
    ffprobe_bin: Path | None = None,
    debug_lines: list[str] | None = None,
) -> tuple[int, list[SceneSegment]]:
    """
    Обнаружение сцен выбранным backend; фильтр min_scene_seconds.

    Возвращает (число сцен до фильтра min_scene_seconds, отфильтрованный список).
  """
    backend = scene_settings.backend.strip().lower()
    if backend not in _SCENE_BACKENDS:
        raise ValueError(_unknown_backend_message(scene_settings.backend))

    path = Path(video_path)

    if backend == "pyscenedetect":
        try:
            return _detect_scenes_pyscenedetect(path, scene_settings)
        except ImportError as e:
            raise ImportError(
                "Для scene_detection.backend=pyscenedetect нужен пакет scenedetect."
            ) from e

    if ffmpeg_bin is None or ffprobe_bin is None:
        raise ValueError(
            "Для scene_detection.backend=ffmpeg нужны пути к ffmpeg и ffprobe "
            "(задайте [tools] ffmpeg_path / ffprobe_path в config.toml)."
        )
    raw_count, filtered, _ = _detect_scenes_ffmpeg(
        path,
        scene_settings,
        ffmpeg_bin=ffmpeg_bin,
        ffprobe_bin=ffprobe_bin,
        debug_lines=debug_lines,
    )
    return raw_count, filtered


__all__ = ["SceneSegment", "detect_scenes", "detect_scenes_with_counts"]
