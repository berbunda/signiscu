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


@dataclass
class SceneDetectionStats:
    raw_timestamps_count: int | None = None
    scenes_before_merge: int = 0
    scenes_after_short_merge: int = 0
    scenes_shorter_than_min: int = 0
    short_scenes_merged: int = 0
    avg_scene_duration: float | None = None
    median_scene_duration: float | None = None


def _unknown_backend_message(backend: str) -> str:
    opts = ", ".join(sorted(_SCENE_BACKENDS))
    return f"Неизвестный scene_detection.backend: {backend!r}. Допустимо: {opts}."


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


def _merge_short_scenes(
    segments: list[SceneSegment],
    min_scene_seconds: float,
    target_seconds: float,
) -> tuple[list[SceneSegment], int]:
    """Объединяет сцены короче min_scene_seconds с соседями (gap к target минимален)."""
    eps = 1e-9
    segs = [SceneSegment(s.start_seconds, s.end_seconds) for s in segments]
    merges = 0

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(segs):
            dur = segs[i].duration_seconds
            if dur >= min_scene_seconds - eps:
                i += 1
                continue
            if len(segs) <= 1:
                break

            left_ok = i > 0
            right_ok = i < len(segs) - 1
            if not left_ok and not right_ok:
                i += 1
                continue

            def _gap_to_target(combined: float) -> float:
                return abs(combined - target_seconds)

            best_side: str | None = None
            best_gap = float("inf")

            if left_ok:
                combined = segs[i - 1].duration_seconds + dur
                g = _gap_to_target(combined)
                if g < best_gap - eps:
                    best_gap = g
                    best_side = "left"
            if right_ok:
                combined = dur + segs[i + 1].duration_seconds
                g = _gap_to_target(combined)
                if g < best_gap - eps or (abs(g - best_gap) <= eps and best_side != "right"):
                    best_gap = g
                    best_side = "right"

            if best_side == "left":
                merged = SceneSegment(segs[i - 1].start_seconds, segs[i].end_seconds)
                segs[i - 1 : i + 1] = [merged]
                merges += 1
                changed = True
                i = max(0, i - 1)
            elif best_side == "right":
                merged = SceneSegment(segs[i].start_seconds, segs[i + 1].end_seconds)
                segs[i : i + 2] = [merged]
                merges += 1
                changed = True
            else:
                i += 1

    return segs, merges


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def _postprocess_scenes(
    raw_bounds: list[tuple[float, float]],
    scene_settings: SceneDetectionSettings,
    stats: SceneDetectionStats,
) -> list[SceneSegment]:
    eps = 1e-9
    segments = [
        SceneSegment(float(s), float(e))
        for s, e in raw_bounds
        if e - s > eps
    ]
    stats.scenes_before_merge = len(segments)
    stats.scenes_shorter_than_min = sum(
        1 for s in segments if s.duration_seconds < scene_settings.min_scene_seconds - eps
    )

    if scene_settings.merge_short_scenes:
        segments, merges = _merge_short_scenes(
            segments,
            scene_settings.min_scene_seconds,
            scene_settings.merge_short_scenes_target_seconds,
        )
        stats.short_scenes_merged = merges
    else:
        segments = [
            s
            for s in segments
            if s.duration_seconds >= scene_settings.min_scene_seconds - eps
        ]

    stats.scenes_after_short_merge = len(segments)
    durations = [s.duration_seconds for s in segments]
    if durations:
        stats.avg_scene_duration = sum(durations) / len(durations)
        stats.median_scene_duration = _median(durations)
    return segments


def _format_scene_detection_debug(
    scene_settings: SceneDetectionSettings,
    stats: SceneDetectionStats,
    *,
    backend: str,
    timestamps: list[float] | None = None,
    ffmpeg_cmd: list[str] | None = None,
) -> list[str]:
    lines: list[str] = []
    lines.append(f"--- Scene detection ({backend}) ---")
    lines.append(f"  scene backend = {backend}")
    if backend == "ffmpeg":
        lines.append(f"  ffmpeg_scene_threshold = {scene_settings.ffmpeg_scene_threshold}")
    if stats.raw_timestamps_count is not None:
        lines.append(f"  raw scene timestamps = {stats.raw_timestamps_count}")
    lines.append(f"  scenes before merge = {stats.scenes_before_merge}")
    lines.append(f"  scenes after short-scene merge = {stats.scenes_after_short_merge}")
    lines.append(f"  final candidates count (scenes) = {stats.scenes_after_short_merge}")
    if stats.avg_scene_duration is not None:
        lines.append(f"  average scene duration = {stats.avg_scene_duration:.3f}s")
    if stats.median_scene_duration is not None:
        lines.append(f"  median scene duration = {stats.median_scene_duration:.3f}s")
    lines.append(f"  scenes shorter than min_scene_seconds = {stats.scenes_shorter_than_min}")
    lines.append(f"  short scenes merged = {stats.short_scenes_merged}")
    if timestamps:
        preview = timestamps[:10]
        lines.append(
            f"  first scene boundaries (cuts, s): {', '.join(f'{t:.3f}' for t in preview)}"
        )
    elif timestamps is not None:
        lines.append("  first scene boundaries (cuts, s): (none)")
    if ffmpeg_cmd is not None:
        lines.append(f"  ffmpeg command = {' '.join(ffmpeg_cmd)}")
    lines.append("")
    return lines


def _detect_scenes_pyscenedetect(
    video_path: Path | str,
    scene_settings: SceneDetectionSettings,
) -> list[tuple[float, float]]:
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
    return raw_bounds


def _detect_scenes_ffmpeg(
    video_path: Path,
    scene_settings: SceneDetectionSettings,
    *,
    ffmpeg_bin: Path,
    ffprobe_bin: Path,
) -> tuple[list[tuple[float, float]], list[float], list[str]]:
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
    return raw_bounds, timestamps, cmd


def detect_scenes(video_path: Path | str, scene_settings: SceneDetectionSettings, **kwargs) -> list[SceneSegment]:
    """Границы сцен в секундах; короткие сцены объединяются или отбрасываются."""
    _, segments, _ = detect_scenes_with_counts(video_path, scene_settings, **kwargs)
    return segments


def detect_scenes_with_counts(
    video_path: Path | str,
    scene_settings: SceneDetectionSettings,
    *,
    ffmpeg_bin: Path | None = None,
    ffprobe_bin: Path | None = None,
    debug_lines: list[str] | None = None,
) -> tuple[int, list[SceneSegment], SceneDetectionStats]:
    """
    Обнаружение сцен выбранным backend; post-process (merge или filter коротких сцен).

    Возвращает (число сцен до post-process, финальный список, статистику).
    """
    backend = scene_settings.backend.strip().lower()
    if backend not in _SCENE_BACKENDS:
        raise ValueError(_unknown_backend_message(scene_settings.backend))

    path = Path(video_path)
    stats = SceneDetectionStats()
    timestamps: list[float] | None = None
    ffmpeg_cmd: list[str] | None = None

    if backend == "pyscenedetect":
        try:
            raw_bounds = _detect_scenes_pyscenedetect(path, scene_settings)
        except ImportError as e:
            raise ImportError(
                "Для scene_detection.backend=pyscenedetect нужен пакет scenedetect."
            ) from e
    else:
        if ffmpeg_bin is None or ffprobe_bin is None:
            raise ValueError(
                "Для scene_detection.backend=ffmpeg нужны пути к ffmpeg и ffprobe "
                "(задайте [tools] ffmpeg_path / ffprobe_path в config.toml)."
            )
        raw_bounds, timestamps, ffmpeg_cmd = _detect_scenes_ffmpeg(
            path,
            scene_settings,
            ffmpeg_bin=ffmpeg_bin,
            ffprobe_bin=ffprobe_bin,
        )
        stats.raw_timestamps_count = len(timestamps)

    raw_count = len(raw_bounds)
    segments = _postprocess_scenes(raw_bounds, scene_settings, stats)

    if debug_lines is not None:
        debug_lines.extend(
            _format_scene_detection_debug(
                scene_settings,
                stats,
                backend=backend,
                timestamps=timestamps,
                ffmpeg_cmd=ffmpeg_cmd,
            )
        )

    return raw_count, segments, stats


__all__ = [
    "SceneDetectionStats",
    "SceneSegment",
    "detect_scenes",
    "detect_scenes_with_counts",
]
