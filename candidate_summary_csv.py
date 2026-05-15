"""Сводный CSV по кандидатам generate: UTF-8, разделитель «;»."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

# Метаданные строки (не из объекта clip в JSON).
_CSV_META_COLUMNS: tuple[str, ...] = ("source_video", "json_file")
_NULL = "null"


def _format_cell(val: Any) -> str:
    if val is None:
        return _NULL
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val != val:  # NaN
            return _NULL
        return str(val)
    if isinstance(val, (dict, list)):
        return _NULL
    return str(val)


def _load_json_clips(
    json_paths: list[Path],
) -> tuple[list[tuple[str, str, list[dict[str, Any]]]], tuple[str, ...]]:
    """Прочитать JSON; вернуть (source_video, json_name, clips) и порядок ключей clip."""
    entries: list[tuple[str, str, list[dict[str, Any]]]] = []
    clip_columns: list[str] = []
    seen_keys: set[str] = set()

    for jp in json_paths:
        try:
            raw = jp.read_text(encoding="utf-8")
            data = json.loads(raw)
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        src = data.get("input_video")
        source_video = _format_cell(src) if src is not None else _NULL
        clips_raw = data.get("clips")
        if not isinstance(clips_raw, list):
            continue
        clips: list[dict[str, Any]] = []
        for item in clips_raw:
            if isinstance(item, dict):
                clips.append(item)
                for key in item:
                    if key not in seen_keys:
                        seen_keys.add(key)
                        clip_columns.append(key)
        if clips:
            entries.append((source_video, jp.name, clips))

    return entries, tuple(clip_columns)


def write_candidate_summary_csv(
    out_dir: Path,
    json_paths: list[Path],
) -> Path:
    """Записать candidate_summary.csv в out_dir. Данные только из JSON (json_paths — в порядке generate)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "candidate_summary.csv"
    entries, clip_columns = _load_json_clips(json_paths)
    columns = _CSV_META_COLUMNS + clip_columns

    rows: list[list[str]] = []
    for source_video, json_name, clips in entries:
        for clip in clips:
            row = [source_video, json_name]
            for key in clip_columns:
                row.append(_format_cell(clip.get(key)))
            rows.append(row)

    with target.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter=";", quoting=csv.QUOTE_MINIMAL)
        w.writerow(list(columns))
        w.writerows(rows)

    return target


__all__ = ["write_candidate_summary_csv"]
