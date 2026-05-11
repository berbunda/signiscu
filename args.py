"""Описание и разбор аргументов командной строки."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class CutArgs:
    video: Path
    project_toml: Path
    debug: bool | None
    debug_log: Path | None


@dataclass(frozen=True)
class GenerateArgs:
    video: Path | None
    candidate_json: Path | None
    output_dir: Path | None
    project_toml: Path | None
    list_windows: str
    debug: bool | None
    debug_log: Path | None


@dataclass(frozen=True)
class ParsedCli:
    command: str
    payload: CutArgs | GenerateArgs
    config_path: Path | None
    config_path_explicit: bool


def parse_args(argv: list[str] | None = None) -> ParsedCli:
    """
    Подкоманды:
    - cut VIDEO [--project project.toml] [--config config.toml] [--debug] [--debug-log PATH]
    - generate [VIDEO] [--project ...] [-o ...] ... (пути по умолчанию — рядом с __main__.py)
    """
    parser = argparse.ArgumentParser(
        description="Нарезка по project.toml и JSON-кандидату или генерация кандидата (ffmpeg/ffprobe).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Путь к config.toml (по умолчанию — config.toml рядом с приложением)",
    )

    subs = parser.add_subparsers(dest="command", required=True)

    p_cut = subs.add_parser("cut", help="Разрезать по project.toml и JSON-кандидату.")
    p_cut.add_argument(
        "video",
        type=Path,
        help="Путь к входному видео (переопределяет input_video из JSON кандидата)",
    )
    p_cut.add_argument(
        "--project",
        type=Path,
        default=None,
        help="Путь к project.toml (по умолчанию — project.toml рядом с приложением, если файл есть)",
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

    p_gen = subs.add_parser("generate", help="Сгенерировать JSON кандидата по окнам громкости.")
    p_gen.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=None,
        help="Входной видеофайл или каталог с файлами (или [input] video в project.toml)",
    )
    p_gen.add_argument(
        "--project",
        type=Path,
        default=None,
        help="project.toml (по умолчанию ищется рядом с приложением, если существует)",
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

    ns = parser.parse_args(argv)
    cfg = ns.config
    cfg_explicit = cfg is not None

    if ns.command == "cut":
        proj = ns.project
        if proj is None:
            default_p = _APP_DIR / "project.toml"
            if default_p.is_file():
                proj = default_p
            else:
                parser.error(
                    "cut: укажите --project или поместите project.toml в каталог приложения."
                )
        return ParsedCli(
            command="cut",
            payload=CutArgs(
                video=Path(ns.video).expanduser().resolve(),
                project_toml=Path(proj).expanduser().resolve(),
                debug=ns.debug,
                debug_log=Path(ns.debug_log).expanduser().resolve() if ns.debug_log is not None else None,
            ),
            config_path=cfg,
            config_path_explicit=cfg_explicit,
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
            debug=ns.debug,
            debug_log=Path(ns.debug_log).expanduser().resolve() if ns.debug_log is not None else None,
        ),
        config_path=cfg,
        config_path_explicit=cfg_explicit,
    )


__all__ = ["CutArgs", "GenerateArgs", "ParsedCli", "parse_args"]
