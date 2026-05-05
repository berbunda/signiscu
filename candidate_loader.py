"""Загрузка машинного JSON с клипами-кандидатами (generate → cut)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from project_models import ClipSpec, Project


class CandidateClipsError(ValueError):
    """Некорректная структура JSON кандидата."""


def _ensure_str(val: Any, ctx: str) -> str:
    if not isinstance(val, str):
        raise CandidateClipsError(f"{ctx}: ожидается строка, получено {type(val).__name__}.")
    return val


def _coerce_nonempty_path_text(val: Any, ctx: str) -> str:
    s = _ensure_str(val, ctx).strip()
    if not s:
        raise CandidateClipsError(f"{ctx}: путь не может быть пустым.")
    return s


def load_candidate_clips_json(path: Path) -> Project:
    """Прочитать JSON (поля input_video, output_dir, clips) в модель Project."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Файл кандидата не найден: {path}")

    raw_text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise CandidateClipsError(f"Некорректный JSON в {path}: {e}") from e

    if not isinstance(data, dict):
        raise CandidateClipsError("Корень JSON кандидата должен быть объектом.")

    in_vid = Path(_coerce_nonempty_path_text(data.get("input_video"), "Поле input_video"))
    out_dir_txt = _coerce_nonempty_path_text(data.get("output_dir"), "Поле output_dir")

    clips_raw = data.get("clips")
    if clips_raw is None:
        raise CandidateClipsError('Отсутствует поле "clips".')
    if not isinstance(clips_raw, list):
        raise CandidateClipsError('"clips" должен быть массивом.')

    specs: list[ClipSpec] = []
    for i, row in enumerate(clips_raw):
        if not isinstance(row, dict):
            raise CandidateClipsError(f"Клип #{i + 1}: ожидается объект.")
        specs.append(
            ClipSpec(
                name=_coerce_nonempty_path_text(row.get("name"), f"Клип #{i + 1} name"),
                start=_ensure_str(row.get("start"), f"Клип #{i + 1} start"),
                end=_ensure_str(row.get("end"), f"Клип #{i + 1} end"),
            )
        )

    return Project(
        input_video=in_vid.expanduser().resolve(),
        output_dir=Path(out_dir_txt).expanduser().resolve(),
        clips=tuple(specs),
    )


__all__ = ["CandidateClipsError", "load_candidate_clips_json"]
