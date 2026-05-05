"""Запуск FFmpeg/FFmpeg и простые задачи звука/длительности."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class FFmpegMissingError(RuntimeError):
    """FFmpeg недоступен."""


def resolve_ffmpeg_path(executable: str) -> Path | None:
    exe = shutil.which(executable) if Path(executable).name == executable else executable
    if exe is None:
        p = Path(executable)
        if p.is_file():
            return p.resolve()
        return None
    return Path(exe)


def ffmpeg_available(executable: str) -> Path:
    p = resolve_ffmpeg_path(executable)
    if p is None:
        raise FFmpegMissingError(
            f"FFmpeg не найден ({executable}). Установите FFmpeg или задайте путь в JSON настроек."
        )
    return p.resolve()


def ffprobe_exe_for(ffmpeg_exe: Path, configured: str | None) -> Path:
    if configured and configured.strip():
        c = configured.strip()
        p = Path(c)
        if p.is_file():
            return p.resolve()
        wp = shutil.which(c)
        if wp:
            return Path(wp).resolve()
        raise FFmpegMissingError(f"ffprobe не найден ({c}).")

    name = ffmpeg_exe.name.lower()
    probe_name = "ffprobe.exe" if name.endswith(".exe") else "ffprobe"
    sibling = ffmpeg_exe.parent / probe_name
    if sibling.is_file():
        return sibling.resolve()
    wp = shutil.which("ffprobe")
    if wp:
        return Path(wp)
    raise FFmpegMissingError(
        "ffprobe не найден рядом с ffmpeg и не в PATH; задайте ffprobe_path в JSON настроек."
    )


_MEAN_VOL = re.compile(r"mean_volume:\s*([-+]?\d*\.?\d+)\s*dB", re.I)


def ffprobe_duration_seconds(ffprobe: Path, input_video: Path) -> tuple[float | None, str | None]:
    cmd = [
        str(ffprobe),
        "-hide_banner",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(input_video),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as e:
        return None, str(e)
    if proc.returncode != 0:
        return None, (proc.stderr or proc.stdout or "ffprobe error").strip()
    out = proc.stdout.strip()
    try:
        return float(out), None
    except ValueError:
        return None, f"Не удалось разобрать длительность: {out!r}"


def ffmpeg_volumedetect_mean_db(
    ffmpeg_bin: Path,
    input_video: Path,
    start_sec: float,
    duration_sec: float,
) -> tuple[float | None, str | None]:
    """Средний уровень (dB) по сегменту через фильтр volumedetect."""
    cmd = [
        str(ffmpeg_bin),
        "-hide_banner",
        "-nostdin",
        "-loglevel",
        "info",
        "-ss",
        f"{start_sec}",
        "-i",
        str(input_video),
        "-t",
        f"{duration_sec}",
        "-vn",
        "-af",
        "volumedetect",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as e:
        return None, str(e)
    hay = "\n".join(filter(None, [proc.stderr, proc.stdout]))
    m = _MEAN_VOL.search(hay)
    if proc.returncode != 0 and m is None:
        return None, (hay.strip() or f"ffmpeg код {proc.returncode}").splitlines()[-1][:500]
    if m is None:
        return None, "В выводе нет mean_volume (нет аудио или другой сбой)."
    try:
        return float(m.group(1)), None
    except ValueError:
        return None, f"Некорректное mean_volume: {m.group(1)!r}"


def cut_clip_copy(
    ffmpeg_bin: Path,
    input_video: Path,
    output_file: Path,
    start_tc: str,
    end_tc: str,
    overwrite: bool,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(ffmpeg_bin),
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    cmd.append("-y" if overwrite else "-n")
    cmd.extend(
        [
            "-ss",
            start_tc,
            "-to",
            end_tc,
            "-i",
            str(input_video),
            "-c",
            "copy",
            str(output_file),
        ]
    )
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


__all__ = [
    "FFmpegMissingError",
    "cut_clip_copy",
    "ffmpeg_available",
    "ffmpeg_volumedetect_mean_db",
    "ffprobe_duration_seconds",
    "ffprobe_exe_for",
    "resolve_ffmpeg_path",
]
