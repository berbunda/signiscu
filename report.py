"""Single-file HTML-отчёт по JSON-кандидатам (любые *.json в каталоге)."""

from __future__ import annotations

import base64
import html
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from timecode import TimecodeError, parse_timecode_seconds

_THUMB_WIDTH_DEFAULT = 220
_MODAL_MAX_WIDTH = 960
_PREVIEW_COUNT = 6
# Порядок ячеек сетки 2×3: сначала колонка 1 (кадры 1–3), затем колонка 2 (4–6).
_PREVIEW_GRID_INDEX_ORDER = (0, 3, 1, 4, 2, 5)
_JSON_GLOB = "*.json"


@dataclass
class PreviewFrame:
    label: str
    thumb_uri: str
    full_uri: str


@dataclass
class ReportCandidate:
    candidate_name: str
    start_tc: str
    end_tc: str
    start_sec: float
    end_sec: float
    duration_sec: float | None
    metrics: dict[str, Any]
    previews: list[PreviewFrame] = field(default_factory=list)
    warning: str | None = None


@dataclass
class ReportGroup:
    video_name: str
    json_name: str
    source_video_abs: str
    video_path: Path | None
    candidates: list[ReportCandidate]
    warning: str | None = None


@dataclass
class ReportData:
    groups: list[ReportGroup]
    warnings: list[str]


def load_candidate_files(input_dir: Path) -> tuple[list[tuple[Path, dict[str, Any]]], list[str]]:
    """Найти и загрузить все *.json в каталоге; повреждённые пропускаются с warning."""
    input_dir = input_dir.expanduser().resolve()
    warnings: list[str] = []
    if not input_dir.is_dir():
        return [], [f"Каталог не найден: {input_dir}"]

    paths = sorted(input_dir.glob(_JSON_GLOB), key=lambda p: p.name.casefold())
    if not paths:
        return [], [f"В каталоге нет файлов {_JSON_GLOB}: {input_dir}"]

    loaded: list[tuple[Path, dict[str, Any]]] = []
    for jp in paths:
        try:
            raw = jp.read_text(encoding="utf-8")
            data = json.loads(raw)
        except OSError as e:
            warnings.append(f"{jp.name}: не удалось прочитать ({e})")
            continue
        except json.JSONDecodeError as e:
            warnings.append(f"{jp.name}: повреждённый JSON ({e})")
            continue
        if not isinstance(data, dict):
            warnings.append(f"{jp.name}: корень JSON должен быть объектом")
            continue
        loaded.append((jp, data))
    return loaded, warnings


def _clip_times(clip: dict[str, Any]) -> tuple[float, float, str, str] | None:
    start_raw = clip.get("start")
    end_raw = clip.get("end")
    if not isinstance(start_raw, str) or not isinstance(end_raw, str):
        return None
    try:
        start_sec = parse_timecode_seconds(start_raw)
        end_sec = parse_timecode_seconds(end_raw)
    except TimecodeError:
        return None
    if end_sec <= start_sec:
        return None
    return start_sec, end_sec, start_raw.strip(), end_raw.strip()


def _preview_sample_seconds(start_sec: float, end_sec: float) -> list[tuple[str, float]]:
    """Шесть моментов: #1 — start+10%, #6 — end−10%, #2–#5 — равные шаги между ними."""
    dur = end_sec - start_sec
    if dur <= 0:
        return []
    t_first = start_sec + 0.1 * dur
    t_last = end_sec - 0.1 * dur
    eps = max(1e-6, dur * 0.001)
    if t_last - t_first < eps:
        mid = 0.5 * (start_sec + end_sec)
        return [(f"{i}/{_PREVIEW_COUNT}", mid) for i in range(1, _PREVIEW_COUNT + 1)]
    stamps = [t_first + (t_last - t_first) * (i / 5.0) for i in range(_PREVIEW_COUNT)]
    out: list[tuple[str, float]] = []
    for i, t in enumerate(stamps, start=1):
        t = max(start_sec, min(end_sec, float(t)))
        out.append((f"{i}/{_PREVIEW_COUNT}", t))
    return out


def encode_image_base64(frame: Any, width: int = _THUMB_WIDTH_DEFAULT) -> str:
    """BGR-кадр OpenCV → data URI (WEBP, иначе JPEG). Уменьшение до width; без увеличения."""
    import cv2

    h, w = frame.shape[:2]
    if width > 0 and w > width:
        nh = max(1, int(h * (width / float(w))))
        frame = cv2.resize(frame, (width, nh), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".webp", frame, [int(cv2.IMWRITE_WEBP_QUALITY), 82])
    if ok:
        b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
        return f"data:image/webp;base64,{b64}"

    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise RuntimeError("Не удалось закодировать кадр в WEBP/JPEG")
    b64 = base64.standard_b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_preview_frames(
    video_path: Path,
    candidate: dict[str, Any],
    *,
    thumb_width: int = _THUMB_WIDTH_DEFAULT,
    modal_max_width: int = _MODAL_MAX_WIDTH,
) -> list[PreviewFrame]:
    """Шесть кадров по времени; миниатюра и отдельное изображение для модалки (до modal_max_width)."""
    import cv2

    times = _clip_times(candidate)
    if times is None:
        return []
    start_sec, end_sec, _, _ = times

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    out: list[PreviewFrame] = []
    try:
        for label, t in _preview_sample_seconds(start_sec, end_sec):
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            nw = int(frame.shape[1])
            full_target_w = min(modal_max_width, nw)
            try:
                thumb_uri = encode_image_base64(frame, thumb_width)
                full_uri = encode_image_base64(frame, full_target_w)
            except Exception:
                continue
            out.append(PreviewFrame(label=label, thumb_uri=thumb_uri, full_uri=full_uri))
    finally:
        cap.release()
    return out


def _metrics_from_clip(clip: dict[str, Any]) -> dict[str, Any]:
    skip = {"name"}
    return {k: v for k, v in clip.items() if k not in skip}


def _format_metric_value(val: Any) -> str:
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, float):
        if val != val:
            return "null"
        return str(val)
    if isinstance(val, (int, str)):
        return str(val)
    return str(val)


def _build_groups(
    loaded: list[tuple[Path, dict[str, Any]]],
    *,
    thumb_width: int,
) -> tuple[list[ReportGroup], list[str]]:
    warnings: list[str] = []
    groups: list[ReportGroup] = []

    for jp, data in loaded:
        json_name = jp.name
        in_raw = data.get("input_video")
        video_path: Path | None = None
        video_name = "(unknown)"
        source_abs = ""
        group_warn: str | None = None

        if isinstance(in_raw, str) and in_raw.strip():
            vp = Path(in_raw.strip()).expanduser()
            source_abs = str(vp.resolve())
            video_name = vp.name
            video_path = vp
            if not vp.is_file():
                group_warn = f"Видео не найдено: {vp}"
                warnings.append(f"{json_name}: {group_warn}")
                video_path = None
        else:
            group_warn = "В JSON нет поля input_video"
            warnings.append(f"{json_name}: {group_warn}")

        clips_raw = data.get("clips")
        if not isinstance(clips_raw, list):
            warnings.append(f"{json_name}: поле clips отсутствует или не массив")
            groups.append(
                ReportGroup(
                    video_name=video_name,
                    json_name=json_name,
                    source_video_abs=source_abs,
                    video_path=video_path,
                    candidates=[],
                    warning=group_warn,
                )
            )
            continue

        candidates: list[ReportCandidate] = []
        for i, clip in enumerate(clips_raw):
            if not isinstance(clip, dict):
                warnings.append(f"{json_name}: клип #{i + 1} не объект")
                continue
            cn_raw = clip.get("name")
            if not isinstance(cn_raw, str) or not cn_raw.strip():
                warnings.append(f"{json_name}: клип #{i + 1} — нет поля name")
                continue
            candidate_name = cn_raw.strip()
            times = _clip_times(clip)
            if times is None:
                warnings.append(f"{json_name}: клип #{i + 1} — неверные start/end")
                continue
            start_sec, end_sec, start_tc, end_tc = times
            dur_raw = clip.get("duration_sec")
            duration_sec: float | None
            if isinstance(dur_raw, (int, float)) and not isinstance(dur_raw, bool):
                duration_sec = float(dur_raw)
            else:
                duration_sec = end_sec - start_sec

            previews: list[PreviewFrame] = []
            cand_warn: str | None = None
            if video_path is not None:
                try:
                    previews = extract_preview_frames(video_path, clip, thumb_width=thumb_width)
                except Exception as e:
                    cand_warn = f"preview: {e}"
                    warnings.append(f"{json_name} [{start_tc}–{end_tc}]: {cand_warn}")
                if not previews and cand_warn is None:
                    cand_warn = "не удалось извлечь кадры"

            candidates.append(
                ReportCandidate(
                    candidate_name=candidate_name,
                    start_tc=start_tc,
                    end_tc=end_tc,
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=duration_sec,
                    metrics=_metrics_from_clip(clip),
                    previews=previews,
                    warning=cand_warn,
                )
            )

        candidates.sort(key=lambda c: c.start_sec)
        groups.append(
            ReportGroup(
                video_name=video_name,
                json_name=json_name,
                source_video_abs=source_abs,
                video_path=video_path,
                candidates=candidates,
                warning=group_warn,
            )
        )

    groups.sort(key=lambda g: (g.video_name.casefold(), g.json_name.casefold()))
    return groups, warnings


def _esc(text: str) -> str:
    return html.escape(text, quote=True)


def _render_metrics(metrics: dict[str, Any]) -> str:
    if not metrics:
        return '<p class="muted">нет метрик</p>'
    rows: list[str] = []
    for key, val in metrics.items():
        if key in ("start", "end"):
            continue
        cls = "metric-val mono"
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            cls += " score"
        rows.append(
            f'<div class="metric-row">'
            f'<span class="metric-key">{_esc(key)}</span>'
            f'<span class="{cls}">{_esc(_format_metric_value(val))}</span>'
            f"</div>"
        )
    return f'<div class="metrics">{"".join(rows)}</div>'


def _previews_grid_order(previews: list[PreviewFrame]) -> list[PreviewFrame]:
    """Сетка 2×3: строки (1,4), (2,5), (3,6) → порядок DOM: 1,4,2,5,3,6."""
    if len(previews) < _PREVIEW_COUNT:
        return list(previews)
    ordered: list[PreviewFrame] = []
    for i in _PREVIEW_GRID_INDEX_ORDER:
        if i < len(previews):
            ordered.append(previews[i])
    return ordered


def _render_previews(previews: list[PreviewFrame]) -> str:
    if not previews:
        return '<p class="muted no-preview">нет превью</p>'
    parts: list[str] = []
    for pf in _previews_grid_order(previews):
        thumb = _esc(pf.thumb_uri)
        full_u = _esc(pf.full_uri)
        label = _esc(pf.label)
        parts.append(
            f'<button type="button" class="thumb-btn" title="{label}" '
            f'data-full="{full_u}" aria-label="{label}">'
            f'<img class="thumb" src="{thumb}" alt="{label}" loading="lazy">'
            f'<span class="thumb-label">{label}</span>'
            f"</button>"
        )
    return f'<div class="previews">{"".join(parts)}</div>'


def _render_card(c: ReportCandidate, g: ReportGroup) -> str:
    dur = f"{c.duration_sec:.3f} с" if c.duration_sec is not None else "—"
    warn = ""
    if c.warning:
        warn = f'<p class="card-warn">{_esc(c.warning)}</p>'
    return (
        f'<article class="card" data-candidate-file="{_esc(g.json_name)}" '
        f'data-candidate-name="{_esc(c.candidate_name)}" '
        f'data-source-video="{_esc(g.source_video_abs)}" '
        f'data-start="{_esc(c.start_tc)}" data-end="{_esc(c.end_tc)}">'
        f'<label class="card-select-row">'
        f'<input type="checkbox" class="card-select">'
        f'<span class="card-select-label">Select</span>'
        f"</label>"
        f'<p class="card-source">{_esc(g.video_name)}</p>'
        f'<div class="card-times mono">'
        f"<span>{_esc(c.start_tc)}</span>"
        f'<span class="arrow">→</span>'
        f"<span>{_esc(c.end_tc)}</span>"
        f"</div>"
        f'<p class="duration mono">{_esc(dur)}</p>'
        f"{_render_previews(c.previews)}"
        f"{_render_metrics(c.metrics)}"
        f"{warn}"
        f"</article>"
    )


def _render_group(g: ReportGroup) -> str:
    n = len(g.candidates)
    warn = ""
    if g.warning:
        warn = f'<p class="group-warn">{_esc(g.warning)}</p>'
    cards = "".join(_render_card(c, g) for c in g.candidates)
    if not cards:
        cards = '<p class="muted">нет кандидатов</p>'
    return (
        f'<section class="group">'
        f"<h2>{_esc(g.video_name)} "
        f'<span class="json-name mono">{_esc(g.json_name)}</span> '
        f'<span class="count">({n})</span></h2>'
        f"{warn}"
        f'<div class="grid">{cards}</div>'
        f"</section>"
    )


_CSS = """
:root {
  --bg: #14171c;
  --surface: #1e232b;
  --card: #252b36;
  --border: #343c4a;
  --text: #e8eaed;
  --muted: #9aa3b2;
  --accent: #6eb5ff;
  --warn: #e8b86d;
  --mono: ui-monospace, "Cascadia Code", "Consolas", monospace;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.45;
}
header {
  padding: 1.25rem 1.5rem;
  border-bottom: 1px solid var(--border);
  background: var(--surface);
}
header h1 { margin: 0 0 0.25rem; font-size: 1.35rem; font-weight: 600; }
header p { margin: 0; color: var(--muted); font-size: 0.9rem; }
.report-toolbar {
  margin-top: 0.85rem;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.5rem 0.75rem;
}
.report-toolbar button {
  font: inherit;
  cursor: pointer;
  padding: 0.35rem 0.75rem;
  border-radius: 6px;
  border: 1px solid var(--border);
  background: var(--card);
  color: var(--text);
}
.report-toolbar button:hover {
  border-color: var(--accent);
  color: var(--accent);
}
.report-toolbar #btn-export-selection {
  background: #2a3f5c;
  border-color: #4a6a8c;
}
.selection-count {
  font-family: var(--mono);
  font-size: 0.85rem;
  color: var(--muted);
}
.warnings {
  margin: 1rem 1.5rem 0;
  padding: 0.75rem 1rem;
  background: #2a2418;
  border: 1px solid #4a3f28;
  border-radius: 8px;
  color: var(--warn);
  font-size: 0.85rem;
}
.warnings ul { margin: 0.25rem 0 0; padding-left: 1.2rem; }
main { padding: 1rem 1.5rem 2rem; max-width: 1600px; margin: 0 auto; }
.group { margin-bottom: 2rem; }
.group h2 {
  font-size: 1.1rem;
  font-weight: 600;
  margin: 0 0 0.75rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--border);
}
.json-name { color: var(--muted); font-weight: 400; font-size: 0.95em; }
.count { color: var(--accent); font-weight: 500; }
.group-warn, .card-warn { color: var(--warn); font-size: 0.8rem; margin: 0.35rem 0 0; }
.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 1rem;
}
@media (min-width: 900px) {
  .grid { grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); }
}
@media (min-width: 1400px) {
  .grid { grid-template-columns: repeat(4, 1fr); }
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 0.85rem;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.card:hover {
  border-color: #4a5568;
  box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.card-select-row {
  display: flex;
  align-items: center;
  gap: 0.4rem;
  margin: 0 0 0.5rem;
  cursor: pointer;
  font-size: 0.8rem;
  color: var(--muted);
}
.card-select-row input { cursor: pointer; }
.card-source {
  margin: 0 0 0.4rem;
  font-size: 0.82rem;
  font-weight: 600;
  color: var(--text);
  word-break: break-all;
}
.card-times {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
  font-size: 0.9rem;
}
.arrow { color: var(--muted); }
.duration { margin: 0.35rem 0 0.6rem; font-size: 0.8rem; color: var(--muted); }
.mono { font-family: var(--mono); }
.previews {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 0.4rem;
  margin-bottom: 0.65rem;
  justify-items: center;
}
.thumb-btn {
  padding: 0;
  border: 1px solid var(--border);
  border-radius: 6px;
  background: #1a1f27;
  cursor: pointer;
  overflow: hidden;
  position: relative;
  transition: transform 0.12s, border-color 0.12s;
  width: 100%;
  max-width: 220px;
}
.thumb-btn:hover {
  border-color: var(--accent);
  transform: scale(1.03);
}
.thumb {
  display: block;
  width: 100%;
  max-width: 220px;
  height: auto;
  vertical-align: middle;
}
.thumb-label {
  display: block;
  font-size: 0.65rem;
  color: var(--muted);
  padding: 0.15rem 0.25rem;
  text-align: center;
  font-family: var(--mono);
}
.metrics { font-size: 0.78rem; }
.metric-row {
  display: flex;
  justify-content: space-between;
  gap: 0.5rem;
  padding: 0.12rem 0;
  border-bottom: 1px solid rgba(255,255,255,0.04);
}
.metric-key { color: var(--muted); }
.metric-val.score { color: var(--accent); }
.muted { color: var(--muted); font-size: 0.85rem; }
.modal {
  position: fixed;
  inset: 0;
  z-index: 1000;
  background: rgba(0,0,0,0.85);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
  cursor: pointer;
}
.modal.hidden { display: none; }
.modal img {
  width: auto;
  height: auto;
  max-width: min(96vw, 960px);
  max-height: 92vh;
  border-radius: 8px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.5);
  cursor: default;
  object-fit: contain;
}
"""

_JS = """
(function () {
  function initModal() {
    var modal = document.getElementById('preview-modal');
    var modalImg = document.getElementById('preview-modal-img');
    if (!modal || !modalImg) return;
    function closeModal() {
      modal.classList.add('hidden');
      modalImg.removeAttribute('src');
    }
    function openModal(uri) {
      modalImg.src = uri;
      modal.classList.remove('hidden');
    }
    document.querySelectorAll('.thumb-btn').forEach(function (btn) {
      btn.addEventListener('click', function (e) {
        e.stopPropagation();
        var uri = btn.getAttribute('data-full');
        if (uri) openModal(uri);
      });
    });
    modal.addEventListener('click', function (e) {
      if (e.target === modal) closeModal();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeModal();
    });
  }

  function getCards() {
    return document.querySelectorAll('article.card');
  }

  function updateCount() {
    var n = document.querySelectorAll('article.card .card-select:checked').length;
    var el = document.getElementById('selection-count');
    if (el) el.textContent = 'Selected: ' + n;
  }

  function exportSelection() {
    var selected = [];
    getCards().forEach(function (card) {
      var cb = card.querySelector('.card-select');
      if (!cb || !cb.checked) return;
      var cf = card.getAttribute('data-candidate-file');
      var cn = card.getAttribute('data-candidate-name');
      if (!cf || !cn) return;
      selected.push({ candidate_file: cf, candidate_name: cn });
    });
    var blob = new Blob(
      [JSON.stringify({ selected: selected }, null, 2)],
      { type: 'application/json;charset=utf-8' }
    );
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'selected_candidates.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }

  document.getElementById('btn-select-all') && document.getElementById('btn-select-all').addEventListener('click', function () {
    document.querySelectorAll('article.card .card-select').forEach(function (cb) { cb.checked = true; });
    updateCount();
  });
  document.getElementById('btn-deselect-all') && document.getElementById('btn-deselect-all').addEventListener('click', function () {
    document.querySelectorAll('article.card .card-select').forEach(function (cb) { cb.checked = false; });
    updateCount();
  });
  document.getElementById('btn-export-selection') && document.getElementById('btn-export-selection').addEventListener('click', exportSelection);

  document.querySelectorAll('article.card .card-select').forEach(function (cb) {
    cb.addEventListener('change', updateCount);
  });

  initModal();
  updateCount();
})();
"""


def render_html_report(data: ReportData, output_path: Path) -> None:
    """Записать single-file HTML-отчёт."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warn_block = ""
    if data.warnings:
        items = "".join(f"<li>{_esc(w)}</li>" for w in data.warnings)
        warn_block = f'<div class="warnings"><strong>Предупреждения</strong><ul>{items}</ul></div>'

    groups_html = "".join(_render_group(g) for g in data.groups)
    if not groups_html:
        groups_html = '<p class="muted">Нет данных для отображения.</p>'

    total_candidates = sum(len(g.candidates) for g in data.groups)
    doc = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>signiscu report</title>
<style>{_CSS}</style>
</head>
<body>
<header>
  <h1>signiscu — отчёт по кандидатам</h1>
  <p>{_esc(str(len(data.groups)))} JSON · {_esc(str(total_candidates))} кандидатов</p>
  <div class="report-toolbar">
    <button type="button" id="btn-select-all">Select all</button>
    <button type="button" id="btn-deselect-all">Deselect all</button>
    <span id="selection-count" class="selection-count">Selected: 0</span>
    <button type="button" id="btn-export-selection">Export selection</button>
  </div>
</header>
{warn_block}
<main>
{groups_html}
</main>
<div id="preview-modal" class="modal hidden" role="dialog" aria-modal="true" aria-label="Превью кадра">
  <img id="preview-modal-img" src="" alt="preview">
</div>
<script>{_JS}</script>
</body>
</html>
"""
    output_path.write_text(doc, encoding="utf-8")


def build_report_data(
    input_dir: Path,
    *,
    thumb_width: int = _THUMB_WIDTH_DEFAULT,
) -> ReportData:
    loaded, load_warnings = load_candidate_files(input_dir)
    groups, build_warnings = _build_groups(loaded, thumb_width=thumb_width)
    return ReportData(groups=groups, warnings=load_warnings + build_warnings)


def run_report(input_dir: Path, output_path: Path, *, thumb_width: int = _THUMB_WIDTH_DEFAULT) -> tuple[Path, list[str]]:
    """Собрать и записать отчёт; вернуть путь и список предупреждений."""
    data = build_report_data(input_dir, thumb_width=thumb_width)
    if not data.groups and not data.warnings:
        data.warnings.append(f"Нет файлов {_JSON_GLOB} в {input_dir}")
    render_html_report(data, output_path)
    return output_path.resolve(), data.warnings


__all__ = [
    "PreviewFrame",
    "ReportData",
    "build_report_data",
    "encode_image_base64",
    "extract_preview_frames",
    "load_candidate_files",
    "render_html_report",
    "run_report",
]
