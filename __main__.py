#!/usr/bin/env python3
"""Точка входа: cut по project.toml + JSON-кандидату или generate (настройки из config.toml)."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from args import BuildCutArgs, CutArgs, GenerateArgs, ReportArgs, parse_args
from build_cut import run_build_cut
from candidate_loader import CandidateClipsError, load_candidate_clips_json
from candidate_summary_csv import write_candidate_summary_csv
from config_loader import ConfigLoadError, load_config_toml
from cutter import print_summary, run_project
from debug_runtime import emit_debug_lines, format_cut_pre_snapshot, merge_effective_debug
from ffmpeg_utils import (
    FFmpegMissingError,
    ffmpeg_available,
    ffprobe_exe_for,
    ffprobe_file_has_video_stream,
)
from generate_candidates import print_generate_summary, run_generate
from report import run_report
from project_models import Project
from project_toml import ProjectTomlError, load_project_toml


def _default_config_path() -> Path:
    return _APP_DIR / "config.toml"


def _default_candidate_json_path() -> Path:
    return _APP_DIR / "candidate.json"


def main() -> None:
    cli = parse_args(sys.argv[1:])
    config_path_used = cli.config_path if cli.config_path is not None else _default_config_path()
    if cli.config_path_explicit and not config_path_used.is_file():
        print(f"Файл настроек не найден: {config_path_used}", file=sys.stderr)
        sys.exit(1)
    if cli.project_path_explicit and cli.command in ("cut", "generate"):
        ptp = cli.payload.project_toml
        if ptp is not None and not ptp.is_file():
            print(f"Файл проекта не найден: {ptp}", file=sys.stderr)
            sys.exit(1)
    try:
        settings = load_config_toml(cli.config_path, _default_config_path())
    except ConfigLoadError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if cli.command == "build-cut":
        ba = cli.payload
        if not isinstance(ba, BuildCutArgs):
            raise TypeError("Внутренняя ошибка: ожидался BuildCutArgs.")
        if not ba.selection.is_file():
            print(f"Файл manifest не найден: {ba.selection}", file=sys.stderr)
            sys.exit(1)
        code, warnings = run_build_cut(ba.selection, ba.input_dir, ba.output_dir)
        for w in warnings:
            print(f"[build-cut] предупреждение: {w}", file=sys.stderr)
        print(f"[build-cut] Выходной каталог: {ba.output_dir.resolve()}")
        sys.exit(code)

    if cli.command == "report":
        ra = cli.payload
        if not isinstance(ra, ReportArgs):
            raise TypeError("Внутренняя ошибка: ожидался ReportArgs.")
        if not ra.input_dir.is_dir():
            print(f"Каталог --input-dir не найден: {ra.input_dir}", file=sys.stderr)
            sys.exit(1)
        try:
            out_path, warnings = run_report(ra.input_dir, ra.output)
        except Exception as e:
            print(f"[report] Ошибка: {e}", file=sys.stderr)
            sys.exit(1)
        for w in warnings:
            print(f"[report] предупреждение: {w}", file=sys.stderr)
        if not out_path.is_file():
            print("[report] Отчёт не создан.", file=sys.stderr)
            sys.exit(1)
        print(f"[report] Отчёт: {out_path}")
        sys.exit(0 if out_path.stat().st_size > 0 else 1)

    if cli.command == "generate":
        ga = cli.payload
        if not isinstance(ga, GenerateArgs):
            raise TypeError("Внутренняя ошибка: ожидался GenerateArgs.")

        pt = None
        project_path_used: Path | None = None
        if ga.project_toml is not None:
            project_path_used = ga.project_toml
            try:
                pt = load_project_toml(ga.project_toml)
            except ProjectTomlError as e:
                print(f"Ошибка project.toml: {e}", file=sys.stderr)
                sys.exit(1)
            except FileNotFoundError as e:
                print(str(e), file=sys.stderr)
                sys.exit(1)

        video_raw = ga.video or (pt.input_video if pt else None)
        candidate_base = (
            ga.candidate_json or (pt.output_candidate_file if pt else None) or _default_candidate_json_path()
        )
        candidate_base = candidate_base.expanduser().resolve()
        if ga.output_dir is not None:
            output_dir = ga.output_dir
        elif pt is not None and pt.output_clips_dir is not None:
            output_dir = pt.output_clips_dir
        else:
            output_dir = Path("out").expanduser().resolve()

        if video_raw is None:
            print("Входное видео не задано.", file=sys.stderr)
            sys.exit(1)

        video = Path(video_raw).expanduser().resolve()
        if not video.exists():
            print(f"Входной путь не найден: {video}", file=sys.stderr)
            sys.exit(1)

        if video.is_dir():
            sources = sorted(
                (p for p in video.iterdir() if p.is_file()),
                key=lambda p: p.name.casefold(),
            )
            batch = True
            if not sources:
                print("Каталог пуст: нечего анализировать.", file=sys.stderr)
                sys.exit(1)
        elif video.is_file():
            sources = [video]
            batch = False
        else:
            print("Ожидался видеофайл или каталог.", file=sys.stderr)
            sys.exit(1)

        try:
            ff_bin = ffmpeg_available(settings.ffmpeg_path)
            ffprobe_bin = ffprobe_exe_for(ff_bin, settings.ffprobe_path)
        except FFmpegMissingError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

        effective_debug = merge_effective_debug(pt, ga.debug, ga.debug_log)

        any_fatal = False
        processed_video_count = 0
        json_paths_for_csv: list[Path] = []
        for vpath in sources:
            if not ffprobe_file_has_video_stream(ffprobe_bin, vpath):
                print(f"{vpath.name} не является видеофайлом.")
                continue
            processed_video_count += 1
            candidate_out = (
                candidate_base.parent / f"candidate_clips_{processed_video_count:05d}.json"
                if batch
                else candidate_base
            )
            summary, fatal, wrote_json = run_generate(
                video=vpath,
                candidate_json=candidate_out,
                output_dir_display=output_dir,
                settings=settings,
                list_windows=ga.list_windows,
                effective_debug=effective_debug,
                config_path_used=config_path_used,
                config_path_explicit_cli=cli.config_path_explicit,
                project_path_used=project_path_used,
                metrics_settings=pt.metrics if pt is not None else None,
                mediapipe_pose_metrics=pt.mediapipe_pose_metrics if pt is not None else None,
                selection_mode=pt.selection.mode if pt is not None else "filtered",
                audio_analysis_enabled=pt.audio_analysis.enabled if pt is not None else True,
            )
            if wrote_json:
                json_paths_for_csv.append(candidate_out)
            if fatal:
                print(f"[generate] {fatal}")
                any_fatal = True
            print_generate_summary(summary)

        if batch and processed_video_count == 0:
            print("Ни один файл в каталоге не содержит видеопотока.", file=sys.stderr)
            sys.exit(1)

        if ga.csv:
            csv_path = write_candidate_summary_csv(candidate_base.parent, json_paths_for_csv)
            print(f"[generate] Сводный CSV: {csv_path}")

        sys.exit(1 if any_fatal else 0)

    ca = cli.payload
    if not isinstance(ca, CutArgs):
        raise TypeError("Внутренняя ошибка: ожидался CutArgs.")

    try:
        pt = load_project_toml(ca.project_toml)
    except ProjectTomlError as e:
        print(f"Ошибка project.toml: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    effective_cut = merge_effective_debug(pt, ca.debug, ca.debug_log)
    mirror = effective_cut.log_path if effective_cut.enabled and effective_cut.log_path is not None else None

    batch_dir: Path | None = None
    if ca.input_dir is not None:
        batch_dir = ca.input_dir
    elif pt.input_candidate_file is not None and pt.input_candidate_file.is_dir():
        batch_dir = pt.input_candidate_file

    if batch_dir is not None:
        in_dir = batch_dir
        if not in_dir.is_dir():
            print(
                f"Каталог с JSON-кандидатами не существует или не является каталогом: {in_dir}",
                file=sys.stderr,
            )
            sys.exit(1)
        batch_files = sorted(
            (p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() == ".json"),
            key=lambda p: p.name.casefold(),
        )
        if not batch_files:
            print(
                f"В каталоге не найдено ни одного файла *.json: {in_dir}",
                file=sys.stderr,
            )
            sys.exit(1)

        batches: list[tuple[Path, Project, Path]] = []
        any_json_error = False
        for cf in batch_files:
            try:
                base = load_candidate_clips_json(cf)
            except (CandidateClipsError, FileNotFoundError, OSError, UnicodeError) as e:
                any_json_error = True
                print(f"[cut] Ошибка JSON «{cf.name}»: {e}", file=sys.stderr)
                continue
            root_clips = pt.output_clips_dir if pt.output_clips_dir is not None else base.output_dir
            out_dir = root_clips / cf.stem
            batches.append((cf, base, out_dir))

        if not batches:
            print(
                "[cut] Ни один JSON из каталога не удалось загрузить; нарезка не выполнялась.",
                file=sys.stderr,
            )
            sys.exit(1)

        any_block_or_clip_fail = False
        for cf, base, out_dir in batches:
            proj_cut = replace(base, output_dir=out_dir)
            if effective_cut.enabled:
                snap = format_cut_pre_snapshot(
                    settings,
                    video=proj_cut.input_video,
                    candidate_in=cf,
                    output_clips_dir=out_dir,
                    config_path=config_path_used,
                    config_explicit_cli=cli.config_path_explicit,
                    project_path=ca.project_toml,
                )
                emit_debug_lines(snap, log_path=effective_cut.log_path)

            results, block = run_project(
                proj_cut,
                ffmpeg_executable=settings.ffmpeg_path,
                overwrite_outputs=settings.overwrite,
                mirror_log_path=mirror,
            )
            print_summary(results, proj_cut.output_dir, block)
            if block is not None:
                any_block_or_clip_fail = True
            if any(not r.ok for r in results):
                any_block_or_clip_fail = True

        sys.exit(1 if any_json_error or any_block_or_clip_fail else 0)

    cf = pt.input_candidate_file
    if cf is None:
        print("Не задан входной JSON кандидата в project.toml ([input].candidate_file).", file=sys.stderr)
        sys.exit(1)
    if not cf.is_file():
        print(
            f"Файл кандидата не найден ([input].candidate_file — путь к .json или каталог с .json): {cf}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        base = load_candidate_clips_json(cf)
    except CandidateClipsError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    out_dir = pt.output_clips_dir if pt.output_clips_dir is not None else base.output_dir
    vid = pt.input_video if pt.input_video is not None else base.input_video
    if vid is None:
        print(
            "Не задано входное видео: укажите [input] video в project.toml или input_video в JSON кандидата.",
            file=sys.stderr,
        )
        sys.exit(1)
    vid = Path(vid).expanduser().resolve()
    proj_cut = replace(base, input_video=vid, output_dir=out_dir)

    if effective_cut.enabled:
        snap = format_cut_pre_snapshot(
            settings,
            video=vid,
            candidate_in=cf,
            output_clips_dir=out_dir,
            config_path=config_path_used,
            config_explicit_cli=cli.config_path_explicit,
            project_path=ca.project_toml,
        )
        emit_debug_lines(snap, log_path=effective_cut.log_path)

    results, block = run_project(
        proj_cut,
        ffmpeg_executable=settings.ffmpeg_path,
        overwrite_outputs=settings.overwrite,
        mirror_log_path=mirror,
    )

    print_summary(results, proj_cut.output_dir, block)
    failed_units = sum(1 for r in results if not r.ok)
    sys.exit(1 if block is not None or failed_units else 0)


if __name__ == "__main__":
    main()
