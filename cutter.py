"""Оркестрация нарезки: имена файлов, лог, ошибки по одному клипу."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ffmpeg_utils import FFmpegMissingError, cut_clip_copy, ffmpeg_available
from project_models import Project
from timecode import TimecodeError, resolve_range, seconds_to_ffmpeg_tc


@dataclass
class ClipProcessResult:
    index: int
    base_name_safe: str
    output_file: Path
    ok: bool
    message: str


def sanitize_clip_filename_part(name: str) -> str:
    """
    «Исправление» имени клипа для безопасного имени файла:
    убирает недопустимые символы, пробелы в подчёркивания.
    """
    trimmed = name.strip()
    trimmed = re.sub(r'[\s\\/:*?"<>|]+', "_", trimmed)
    trimmed = re.sub(r"_+", "_", trimmed).strip("_")
    return trimmed if trimmed else "clip"


def _append_log(log_path: Path, line: str, mirror: Path | None = None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = line.rstrip("\n") + "\n"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(payload)
    if mirror is not None:
        mirror.parent.mkdir(parents=True, exist_ok=True)
        with mirror.open("a", encoding="utf-8") as mf:
            mf.write(payload)


@dataclass(frozen=True)
class AppPaths:
    """Пути приложения после валидации."""

    ffmpeg: Path
    input_video: Path
    output_dir: Path


def validate_paths_and_ffmpeg(project: Project, ffmpeg_executable: str) -> AppPaths | str:
    if not project.input_video.is_file():
        return f"Входное видео не найдено: {project.input_video}"

    out = project.output_dir
    try:
        out.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        return f"Не удалось подготовить output_dir ({out}): {e}"

    if not out.is_dir():
        return f"output_dir не является каталогом: {out}"

    try:
        probe = out / ".signiscu_write_test"
        probe.write_text("")
        probe.unlink(missing_ok=True)
    except OSError as e:
        return f"Нет записи в output_dir ({out}): {e}"

    try:
        ff = ffmpeg_available(ffmpeg_executable)
    except FFmpegMissingError as e:
        return str(e)

    return AppPaths(ffmpeg=ff, input_video=project.input_video, output_dir=out)


def run_project(
    project: Project,
    ffmpeg_executable: str,
    overwrite_outputs: bool,
    mirror_log_path: Path | None = None,
) -> tuple[list[ClipProcessResult], str | None]:
    """
    Обработать все клипы; при ошибке одного — продолжить остальные.
    Возвращает результаты и сообщение блокирующей ошибки до цикла (если есть).
    """
    pre = validate_paths_and_ffmpeg(project, ffmpeg_executable)
    if isinstance(pre, str):
        return [], pre

    out_dir = pre.output_dir
    log_path = out_dir / "cut_log.txt"
    utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    _append_log(log_path, f"# session start {utc}", mirror_log_path)
    _append_log(
        log_path,
        f"input_video={project.input_video} output_dir={out_dir} ffmpeg={pre.ffmpeg}",
        mirror_log_path,
    )

    results: list[ClipProcessResult] = []
    idx_width = max(3, len(str(max(len(project.clips), 1))))

    for i, clip in enumerate(project.clips, start=1):
        safe_base = sanitize_clip_filename_part(clip.name)
        out_name = f"{i:0{idx_width}d}_{safe_base}.mp4"
        out_file = out_dir / out_name

        try:
            rng = resolve_range(clip.start, clip.end)
        except TimecodeError as e:
            msg = f"Клип #{i}: {e}"
            results.append(
                ClipProcessResult(
                    index=i,
                    base_name_safe=safe_base,
                    output_file=out_file,
                    ok=False,
                    message=msg,
                )
            )
            _append_log(log_path, f"[FAIL] #{i} {safe_base} {clip.start}->{clip.end} | {msg}", mirror_log_path)
            continue

        start_tc = seconds_to_ffmpeg_tc(rng.start_sec)
        end_tc = seconds_to_ffmpeg_tc(rng.end_sec)

        try:
            cp = cut_clip_copy(
                ffmpeg_bin=pre.ffmpeg,
                input_video=pre.input_video,
                output_file=out_file,
                start_tc=start_tc,
                end_tc=end_tc,
                overwrite=overwrite_outputs,
            )
        except OSError as e:
            msg = f"Клип #{i}: не удалось запустить FFmpeg: {e}"
            results.append(
                ClipProcessResult(
                    index=i,
                    base_name_safe=safe_base,
                    output_file=out_file,
                    ok=False,
                    message=msg,
                )
            )
            _append_log(log_path, f"[FAIL] #{i} {safe_base} {start_tc}->{end_tc} | {msg}", mirror_log_path)
            continue

        if cp.returncode == 0 and out_file.is_file():
            ok_msg = str(out_file)
            results.append(
                ClipProcessResult(
                    index=i,
                    base_name_safe=safe_base,
                    output_file=out_file,
                    ok=True,
                    message=ok_msg,
                )
            )
            _append_log(
                log_path,
                f"[OK] #{i} {safe_base} {start_tc}->{end_tc} -> {out_file.name}",
                mirror_log_path,
            )
        else:
            err = (cp.stderr or "").strip() or (cp.stdout or "").strip() or "(нет вывода)"
            msg = f"Клип #{i}: FFmpeg код {cp.returncode}: {err}"
            results.append(
                ClipProcessResult(
                    index=i,
                    base_name_safe=safe_base,
                    output_file=out_file,
                    ok=False,
                    message=msg,
                )
            )
            _append_log(log_path, f"[FAIL] #{i} {safe_base} {start_tc}->{end_tc} | {err}", mirror_log_path)

    return results, None


def print_summary(
    results: list[ClipProcessResult],
    output_dir: Path,
    block_msg: str | None,
) -> None:
    if block_msg:
        print(block_msg)

    ok = sum(1 for r in results if r.ok)
    fail = sum(1 for r in results if not r.ok)
    total = len(results)

    lines = []
    lines.append("")
    lines.append("--- Резюме ---")
    lines.append(f"Каталог вывода: {output_dir}")
    lines.append(f"Всего клипов в задаче: {total}")
    lines.append(f"Успешно: {ok}")
    lines.append(f"С ошибкой: {fail}")
    lines.append("")
    print("\n".join(lines))
