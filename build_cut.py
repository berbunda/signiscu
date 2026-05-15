"""Сборка стандартных cut JSON из manifest выбора (selected_candidates.json)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from cutter import sanitize_clip_filename_part
from timecode import TimecodeError, parse_timecode_seconds


def load_selection_manifest(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Разобрать selected_candidates.json → список {candidate_file, candidate_name}."""
    warnings: list[str] = []
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except OSError as e:
        return [], [f"Не удалось прочитать manifest: {e}"]
    except json.JSONDecodeError as e:
        return [], [f"Повреждённый JSON manifest: {e}"]
    if not isinstance(data, dict):
        return [], ["Корень manifest должен быть объектом."]
    sel = data.get("selected")
    if sel is None:
        return [], ["В manifest отсутствует ключ «selected»."]
    if not isinstance(sel, list):
        return [], ["Поле «selected» должно быть массивом."]
    out: list[dict[str, str]] = []
    for i, row in enumerate(sel):
        if not isinstance(row, dict):
            warnings.append(f"Элемент selected[{i}] не объект — пропуск.")
            continue
        cf = row.get("candidate_file")
        cn = row.get("candidate_name")
        if not isinstance(cf, str) or not cf.strip():
            warnings.append(f"Элемент selected[{i}]: нет candidate_file — пропуск.")
            continue
        if not isinstance(cn, str) or not cn.strip():
            warnings.append(f"Элемент selected[{i}]: нет candidate_name — пропуск.")
            continue
        out.append({"candidate_file": cf.strip(), "candidate_name": cn.strip()})
    return out, warnings


def run_build_cut(
    selection_path: Path,
    input_dir: Path,
    output_dir: Path,
) -> tuple[int, list[str]]:
    """
    Прочитать manifest, найти клипы в candidate JSON, сгруппировать по input_video,
    записать cut_<stem>.json в output_dir (поля input_video, при наличии — output_dir из кандидатов, clips).
    Возвращает (код выхода 0|1, предупреждения).
    """
    warnings: list[str] = []
    entries, mw = load_selection_manifest(selection_path.expanduser().resolve())
    warnings.extend(mw)
    if not entries:
        warnings.append("В manifest нет ни одной корректной записи.")
        return 1, warnings

    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        warnings.append(f"Каталог input-dir не найден: {input_dir}")
        return 1, warnings

    video_clips: dict[Path, list[dict[str, str]]] = {}
    video_output_dir: dict[Path, str | None] = {}
    seen_clip: set[tuple[Path, str]] = set()

    for row in entries:
        cf_name = row["candidate_file"]
        c_name = row["candidate_name"]
        jpath = (input_dir / cf_name).resolve()
        if not jpath.is_file():
            warnings.append(f"Файл не найден: {cf_name}")
            continue
        try:
            data = json.loads(jpath.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as e:
            warnings.append(f"{cf_name}: не удалось загрузить JSON ({e})")
            continue
        if not isinstance(data, dict):
            warnings.append(f"{cf_name}: корень JSON не объект")
            continue
        iv = data.get("input_video")
        if not isinstance(iv, str) or not iv.strip():
            warnings.append(f"{cf_name}: нет или пустое input_video")
            continue
        vid_path = Path(iv.strip()).expanduser().resolve()
        clips_raw = data.get("clips")
        if not isinstance(clips_raw, list):
            warnings.append(f"{cf_name}: поле clips отсутствует или не массив")
            continue
        found: dict[str, Any] | None = None
        for clip in clips_raw:
            if isinstance(clip, dict) and clip.get("name") == c_name:
                found = clip
                break
        if found is None:
            warnings.append(f"{cf_name}: кандидат «{c_name}» не найден")
            continue
        start = found.get("start")
        end = found.get("end")
        if not isinstance(start, str) or not isinstance(end, str):
            warnings.append(f"{cf_name}: у «{c_name}» нет строковых start/end")
            continue
        sk = (vid_path, c_name)
        if sk in seen_clip:
            warnings.append(f"{cf_name}: «{c_name}» уже выбран для этого видео — пропуск дубликата")
            continue
        seen_clip.add(sk)

        od_from_file: str | None = None
        odraw = data.get("output_dir")
        if isinstance(odraw, str) and odraw.strip():
            od_from_file = str(Path(odraw.strip()).expanduser().resolve())

        if vid_path not in video_output_dir:
            video_output_dir[vid_path] = od_from_file
        elif od_from_file is not None:
            cur_od = video_output_dir[vid_path]
            if cur_od is None:
                video_output_dir[vid_path] = od_from_file
            elif cur_od != od_from_file:
                warnings.append(
                    f"Для «{vid_path.name}» разные output_dir в кандидатах "
                    f"({cur_od} и {od_from_file} в {cf_name}); в cut JSON оставлен первый."
                )

        entry = {"name": c_name, "start": start.strip(), "end": end.strip()}
        video_clips.setdefault(vid_path, []).append(entry)

    if not video_clips:
        warnings.append("После обработки не осталось ни одного клипа для записи.")
        return 1, warnings

    fname_owner: dict[str, Path] = {}

    def sort_key_clip(c: dict[str, str]) -> float:
        try:
            return parse_timecode_seconds(c["start"])
        except TimecodeError:
            return 0.0

    for vid_path in sorted(video_clips.keys(), key=lambda p: str(p)):
        clips = sorted(video_clips[vid_path], key=sort_key_clip)
        stem = sanitize_clip_filename_part(vid_path.stem)
        fname = f"cut_{stem}.json"
        if fname in fname_owner and fname_owner[fname] != vid_path:
            h = hashlib.sha256(str(vid_path.resolve()).encode("utf-8")).hexdigest()[:8]
            fname = f"cut_{stem}_{h}.json"
            warnings.append(
                f"Коллизия имени cut JSON для stem «{stem}», файл для «{vid_path.name}»: {fname}"
            )
        fname_owner[fname] = vid_path

        od_out = video_output_dir.get(vid_path)
        payload: dict[str, Any] = {"input_video": str(vid_path)}
        if od_out is not None:
            payload["output_dir"] = od_out
        payload["clips"] = [{"name": c["name"], "start": c["start"], "end": c["end"]} for c in clips]
        out_path = output_dir / fname
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return 0, warnings


__all__ = ["load_selection_manifest", "run_build_cut"]
