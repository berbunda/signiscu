"""Описание и разбор аргументов командной строки."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CutArgs:
    project_toml: Path
    input_dir: Path | None
    debug: bool | None
    debug_log: Path | None


@dataclass(frozen=True)
class ReportArgs:
    input_dir: Path
    output: Path


@dataclass(frozen=True)
class GenerateArgs:
    video: Path | None
    candidate_json: Path | None
    output_dir: Path | None
    project_toml: Path | None
    list_windows: str
    csv: bool
    debug: bool | None
    debug_log: Path | None


@dataclass(frozen=True)
class BuildCutArgs:
    selection: Path
    input_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class ParsedCli:
    command: str
    payload: CutArgs | GenerateArgs | ReportArgs | BuildCutArgs
    config_path: Path | None
    config_path_explicit: bool
    project_path_explicit: bool


def parse_args(argv: list[str] | None = None) -> ParsedCli:
    """
    Подкоманды (общие флаги --config / --project — до имени подкоманды, как у argparse):
    - signiscu [--config …] [--project …] cut [--input-dir …] [--debug] …
    - signiscu [--config …] [--project …] generate [VIDEO] [-o …] …
    - signiscu report --input-dir PATH --output report.html
    - signiscu build-cut --selection PATH --input-dir PATH --output-dir PATH
    """
    parser = argparse.ArgumentParser(
        description="Нарезка по project.toml и JSON-кандидату или генерация кандидата (ffmpeg/ffprobe).",
        epilog=(
            "Флаги --config и --project задаются до подкоманды cut|generate, "
            "например: python -m signiscu --project D:/work/project.toml generate"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Путь к config.toml (по умолчанию — config.toml рядом с приложением)",
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Путь к project.toml при нестандартном расположении (по умолчанию — project.toml рядом с приложением, если есть)",
    )

    subs = parser.add_subparsers(dest="command", required=True)

    p_cut = subs.add_parser(
        "cut",
        help="Разрезать по project.toml и JSON-кандидату (видео — из project.toml и/или JSON).",
    )
    p_cut.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Отладка: по умолчанию из project.toml; --no-debug отключает явно.",
    )
    p_cut.add_argument(
        "--debug-log",
        type=Path,
        default=None,
        help="Файл журнала отладки (переопределяет debug.log_file в TOML).",
    )
    p_cut.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help='Каталог с любыми *.json кандидатов. Если задан — переопределяет [input].candidate_file; видео из каждого JSON (input_video).',
    )

    p_gen = subs.add_parser("generate", help="Сгенерировать JSON кандидата по окнам громкости.")
    p_gen.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=None,
        help="Входной видеофайл или каталог с файлами (или [input] video в project.toml)",
    )
    p_gen.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        dest="candidate_json",
        help="JSON кандидата одиночного режима (при каталоге — каталог этого файла; тогда имена candidate_clips_NNNNN.json)",
    )
    p_gen.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Каталог нарезки в JSON; по умолчанию out или [output] clips_dir",
    )
    p_gen.add_argument(
        "--list-windows",
        choices=("none", "all", "selected"),
        default="none",
        help="Вывод окон в stdout: none | all | selected.",
    )
    p_gen.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Отладка: по умолчанию из project.toml; --no-debug отключает явно.",
    )
    p_gen.add_argument(
        "--debug-log",
        type=Path,
        default=None,
        help="Файл журнала отладки (переопределяет log_file в TOML).",
    )
    p_gen.add_argument(
        "--csv",
        action="store_true",
        help="После generate записать сводный candidate_summary.csv (UTF-8, «;») рядом с JSON кандидатов.",
    )

    p_report = subs.add_parser("report", help="HTML-отчёт по JSON-кандидатам в каталоге (*.json).")
    p_report.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Каталог с JSON кандидатов (любые *.json, название файла произвольное)",
    )
    p_report.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Путь к выходному HTML-файлу отчёта",
    )

    p_build = subs.add_parser(
        "build-cut",
        help="Собрать cut JSON из selected_candidates.json (manifest HTML-отчёта).",
    )
    p_build.add_argument(
        "--selection",
        type=Path,
        required=True,
        help="Путь к selected_candidates.json",
    )
    p_build.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Каталог, где лежат JSON из manifest (имена как в candidate_file; обычно те же *.json, что и для report/cut)",
    )
    p_build.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Каталог для cut_<stem>.json",
    )

    ns = parser.parse_args(argv)
    cfg = ns.config
    cfg_explicit = cfg is not None
    proj_explicit = ns.project is not None

    if ns.command == "build-cut":
        return ParsedCli(
            command="build-cut",
            payload=BuildCutArgs(
                selection=Path(ns.selection).expanduser().resolve(),
                input_dir=Path(ns.input_dir).expanduser().resolve(),
                output_dir=Path(ns.output_dir).expanduser().resolve(),
            ),
            config_path=cfg,
            config_path_explicit=cfg_explicit,
            project_path_explicit=proj_explicit,
        )

    if ns.command == "report":
        return ParsedCli(
            command="report",
            payload=ReportArgs(
                input_dir=Path(ns.input_dir).expanduser().resolve(),
                output=Path(ns.output).expanduser().resolve(),
            ),
            config_path=cfg,
            config_path_explicit=cfg_explicit,
            project_path_explicit=proj_explicit,
        )

    if ns.command == "cut":
        proj = ns.project
        if proj is None:
            default_p = _APP_DIR / "project.toml"
            if default_p.is_file():
                proj = default_p
            else:
                parser.error(
                    "cut: укажите --project PATH перед подкомандой cut "
                    "(как --config: signiscu --project D:/prj/project.toml cut …) "
                    "или поместите project.toml в каталог приложения."
                )
        return ParsedCli(
            command="cut",
            payload=CutArgs(
                project_toml=Path(proj).expanduser().resolve(),
                input_dir=Path(ns.input_dir).expanduser().resolve() if ns.input_dir is not None else None,
                debug=ns.debug,
                debug_log=Path(ns.debug_log).expanduser().resolve() if ns.debug_log is not None else None,
            ),
            config_path=cfg,
            config_path_explicit=cfg_explicit,
            project_path_explicit=proj_explicit,
        )

    out_dir = Path(ns.output_dir).expanduser().resolve() if ns.output_dir is not None else None
    proj_g = ns.project
    if proj_g is not None:
        proj_g = Path(proj_g).expanduser().resolve()
    elif (_APP_DIR / "project.toml").is_file():
        proj_g = _APP_DIR / "project.toml"

    return ParsedCli(
        command="generate",
        payload=GenerateArgs(
            video=Path(ns.video).expanduser().resolve() if ns.video is not None else None,
            candidate_json=Path(ns.candidate_json).expanduser().resolve()
            if ns.candidate_json is not None
            else None,
            output_dir=out_dir,
            project_toml=proj_g,
            list_windows=str(ns.list_windows),
            csv=bool(ns.csv),
            debug=ns.debug,
            debug_log=Path(ns.debug_log).expanduser().resolve() if ns.debug_log is not None else None,
        ),
        config_path=cfg,
        config_path_explicit=cfg_explicit,
        project_path_explicit=proj_explicit,
    )


__all__ = ["BuildCutArgs", "CutArgs", "GenerateArgs", "ReportArgs", "ParsedCli", "parse_args"]
