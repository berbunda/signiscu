"""Объединение флагов отладки (config.toml → project.toml → CLI) и вывод в stdout/файл."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from project_toml import ProjectToml
from settings import ToolSettings


@dataclass(frozen=True)
class EffectiveDebug:
    """Итоговый режим отладки после слияния источников."""

    enabled: bool
    """None — весь отладочный блок в stdout; иначе дозапись в файл."""
    log_path: Path | None


def merge_effective_debug(
    project: ProjectToml | None,
    cli_debug: bool | None,
    cli_debug_log: Path | None,
) -> EffectiveDebug:
    """
    Отладка только из project.toml и CLI: сначала project, затем --debug / --no-debug и --debug-log.
    """
    enabled = False
    log_path: Path | None = None

    if project is not None:
        if project.debug_enabled:
            enabled = True
        if project.debug_log_file is not None:
            log_path = project.debug_log_file

    if cli_debug is True:
        enabled = True
    elif cli_debug is False:
        enabled = False

    if cli_debug_log is not None:
        log_path = cli_debug_log.expanduser().resolve()

    return EffectiveDebug(enabled=enabled, log_path=log_path)


def emit_debug_lines(lines: list[str], *, log_path: Path | None) -> None:
    """Весь отладочный блок: в файл при заданном log_path, иначе в stdout."""
    text = "\n".join(lines)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
    else:
        print(text)


def format_tool_settings_lines(settings: ToolSettings, *, prefix: str = "  ") -> list[str]:
    sd = settings.scene_detection
    lines = [
        f"{prefix}ffmpeg_path = {settings.ffmpeg_path}",
        f"{prefix}ffprobe_path = {settings.ffprobe_path!r}",
        f"{prefix}overwrite = {settings.overwrite}",
        f"{prefix}window_seconds = {settings.window_seconds}",
        f"{prefix}threshold = {settings.threshold}",
        f"{prefix}min_audio_coverage_ratio = {settings.min_audio_coverage_ratio}",
        f"{prefix}min_peak_score = {settings.min_peak_score}",
        f"{prefix}merge_gap_seconds = {settings.merge_gap_seconds}",
        f"{prefix}padding_before_seconds = {settings.padding_before_seconds}",
        f"{prefix}padding_after_seconds = {settings.padding_after_seconds}",
        f"{prefix}min_clip_seconds = {settings.min_clip_seconds}",
        f"{prefix}max_clip_seconds = {settings.max_clip_seconds}",
        f"{prefix}[scene_detection] enabled = {sd.enabled}",
        f"{prefix}[scene_detection] backend = {sd.backend}",
        f"{prefix}[scene_detection] threshold = {sd.threshold}",
        f"{prefix}[scene_detection] min_scene_seconds = {sd.min_scene_seconds}",
        f"{prefix}[scene_detection] show_progress = {sd.show_progress}",
        f"{prefix}[scene_detection] max_scene_seconds = {sd.max_scene_seconds}",
        f"{prefix}[scene_detection] ffmpeg_scene_threshold = {sd.ffmpeg_scene_threshold}",
    ]
    lm = settings.motion_analysis
    lines.extend(
        [
            f"{prefix}[motion_analysis] backend = {lm.backend}",
            f"{prefix}[motion_analysis] enabled = {lm.enabled}",
            f"{prefix}[motion_analysis] sample_fps = {lm.sample_fps}",
            f"{prefix}[motion_analysis] resize_width = {lm.resize_width}",
            f"{prefix}[motion_analysis] residual_percentile = {lm.residual_percentile}",
            f"{prefix}[motion_analysis] min_motion_coverage_ratio = {lm.min_motion_coverage_ratio}",
            f"{prefix}[motion_analysis.optical_flow] motion_threshold = {lm.optical_flow.motion_threshold}",
            f"{prefix}[motion_analysis.optical_flow] min_motion_peak_score = {lm.optical_flow.min_motion_peak_score}",
            f"{prefix}[motion_analysis.mediapipe_pose] motion_threshold = {lm.mediapipe_pose.motion_threshold}",
            f"{prefix}[motion_analysis.mediapipe_pose] min_motion_peak_score = {lm.mediapipe_pose.min_motion_peak_score}",
            f"{prefix}[motion_analysis] active motion_threshold = {lm.motion_threshold} (backend={lm.backend})",
            f"{prefix}[motion_analysis] active min_motion_peak_score = {lm.min_motion_peak_score} (backend={lm.backend})",
            f"{prefix}[motion_analysis] affect_selection = {lm.affect_selection}",
            f"{prefix}[motion_analysis] affect_score = {lm.affect_score}",
            f"{prefix}[motion_analysis] weight_motion = {lm.weight_motion}",
            f"{prefix}[motion_analysis] mediapipe_pose_model = {lm.mediapipe_pose_model}",
            f"{prefix}[motion_analysis] mediapipe_min_detection_confidence = {lm.mediapipe_min_detection_confidence}",
            f"{prefix}[motion_analysis] mediapipe_min_tracking_confidence = {lm.mediapipe_min_tracking_confidence}",
            f"{prefix}[motion_analysis] mediapipe_visibility_threshold = {lm.mediapipe_visibility_threshold}",
        ]
    )
    return lines


def format_generate_pre_snapshot(
    settings: ToolSettings,
    *,
    video: Path,
    candidate_out: Path,
    clips_dir: Path,
    config_path: Path,
    config_explicit_cli: bool,
    project_path: Path | None,
) -> list[str]:
    """Снимок действующей конфигурации до начала анализа окон."""
    lines: list[str] = []
    lines.append("=== Снимок конфигурации (до анализа) ===")
    lines.append(f"входное_видео = {video.resolve()}")
    lines.append(f"файл_кандидата_вывода = {candidate_out.resolve()}")
    lines.append(f"каталог_клипов_в_json = {clips_dir.resolve()}")
    cfg_note = "да" if config_explicit_cli else "нет (взят путь по умолчанию рядом с приложением)"
    lines.append(f"файл_конфигурации = {config_path.resolve()} (явно_в_CLI: {cfg_note})")
    if project_path is not None:
        lines.append(f"файл_проекта = {project_path.resolve()}")
    else:
        lines.append("файл_проекта = (не использовался)")
    lines.append("действующие_настройки:")
    lines.extend(format_tool_settings_lines(settings))
    lines.append("переопределения_CLI_инструмента = (нет — только config.toml и project.toml)")
    lines.append("")
    return lines


def format_generate_post_metrics(
    settings: ToolSettings,
    *,
    video: Path,
    candidate_out: Path,
    config_path: Path,
    config_explicit_cli: bool,
    duration_sec: float | None,
    summary_analyzed_ok: int,
    summary_selected_windows: int,
    summary_generated: int,
    scene_pyscene_count: int | None = None,
    scene_after_min_count: int | None = None,
) -> list[str]:
    """Поля для журнала отладки после анализа."""
    lines: list[str] = []
    lines.append("=== Показатели после анализа ===")
    lines.append(f"входное_видео = {video.resolve()}")
    lines.append(f"файл_кандидата_вывода = {candidate_out.resolve()}")
    cfg_note = "да" if config_explicit_cli else "нет"
    lines.append(f"файл_конфигурации = {config_path.resolve()} (явно_в_CLI: {cfg_note})")
    lines.append("действующие_настройки:")
    lines.extend(format_tool_settings_lines(settings))
    dur = duration_sec if duration_sec is not None and duration_sec > 0 else None
    lines.append(f"общая_продолжительность_сек = {dur!s}")
    lines.append(f"размер_окна_сек = {settings.window_seconds}")
    lines.append(f"пороговое_значение = {settings.threshold}")
    lines.append(f"min_audio_coverage_ratio = {settings.min_audio_coverage_ratio}")
    lines.append(f"min_peak_score = {settings.min_peak_score}")
    lines.append(f"merge_gap_seconds = {settings.merge_gap_seconds}")
    lines.append(f"заполнение_перед_секундами = {settings.padding_before_seconds}")
    lines.append(f"заполнение_после_секунд = {settings.padding_after_seconds}")
    lines.append(f"минимальное_количество_секунд = {settings.min_clip_seconds}")
    lines.append(f"максимальное_количество_секунд = {settings.max_clip_seconds}")
    sd = settings.scene_detection
    lines.append(
        f"обнаружение_сцен = {'enabled' if sd.enabled else 'disabled'} "
        f"(backend={sd.backend}, threshold={sd.threshold}, min_scene_seconds={sd.min_scene_seconds})"
    )
    if scene_pyscene_count is not None:
        lines.append(f"сцен_до_min_scene_seconds = {scene_pyscene_count}")
    if scene_after_min_count is not None:
        lines.append(f"сцен_после_min_scene_seconds = {scene_after_min_count}")
    lines.append(f"количество_проанализированных_окон = {summary_analyzed_ok}")
    lines.append(f"выбранные_окна = {summary_selected_windows}")
    lines.append(f"сгенерированные_кандидаты = {summary_generated}")
    lines.append("")
    return lines


def format_cut_pre_snapshot(
    settings: ToolSettings,
    *,
    video: Path,
    candidate_in: Path,
    output_clips_dir: Path,
    config_path: Path,
    config_explicit_cli: bool,
    project_path: Path,
) -> list[str]:
    lines: list[str] = []
    lines.append("=== Снимок конфигурации (cut, до нарезки) ===")
    lines.append(f"входное_видео = {video.resolve()}")
    lines.append(f"файл_кандидата_входа = {candidate_in.resolve()}")
    lines.append(f"каталог_вывода_клипов = {output_clips_dir.resolve()}")
    cfg_note = "да" if config_explicit_cli else "нет (взят путь по умолчанию рядом с приложением)"
    lines.append(f"файл_конфигурации = {config_path.resolve()} (явно_в_CLI: {cfg_note})")
    lines.append(f"файл_проекта = {project_path.resolve()}")
    lines.append("действующие_настройки:")
    lines.extend(format_tool_settings_lines(settings))
    lines.append("")
    return lines


__all__ = [
    "EffectiveDebug",
    "emit_debug_lines",
    "format_cut_pre_snapshot",
    "format_generate_post_metrics",
    "format_generate_pre_snapshot",
    "format_tool_settings_lines",
    "merge_effective_debug",
]
