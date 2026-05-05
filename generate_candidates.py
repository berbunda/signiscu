"""Генерация JSON кандидата (candidate_clips) по окнам громкости или по сценам PySceneDetect + звук."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from cutter import sanitize_clip_filename_part
from ffmpeg_utils import (
    FFmpegMissingError,
    ffmpeg_available,
    ffmpeg_volumedetect_mean_db,
    ffprobe_duration_seconds,
    ffprobe_exe_for,
)
from debug_runtime import (
    EffectiveDebug,
    emit_debug_lines,
    format_generate_post_metrics,
    format_generate_pre_snapshot,
)
from scene_analysis import SceneSegment, detect_scenes_with_counts
from settings import ToolSettings
from timecode import seconds_to_display_span, seconds_to_ffmpeg_tc


def linear_raw_from_mean_db(mean_db: float) -> float:
    """Положительная величина громкости из mean_volume (дБ): амплитуда ~ 10**(dB/20)."""
    return float(math.pow(10.0, mean_db / 20.0))


@dataclass
class WindowRow:
    index: int
    t_start: float
    t_end: float
    mean_db: float | None  # mean_volume FFmpeg (дБ)
    raw_score: float | None  # линейная величина для деления на max_raw
    norm: float | None  # raw_score / max_raw_score
    analysis_error: str | None


@dataclass
class CandidateBuildSummary:
    total_windows: int
    analyzed_ok: int
    analyze_failed_windows: int
    selected_windows: int
    merged_segments: int
    generated_candidates: int
    skipped_short: int
    scene_detection_used: bool = False
    scenes_pyscenedetect: int | None = None
    scenes_after_min_duration: int | None = None
    scene_units_total: int | None = None


@dataclass
class SceneUnitEval:
    """Одна единица отбора: сцена целиком или фрагмент длинной сцены (окна window_seconds)."""

    index: int
    t_start: float
    t_end: float
    avg_audio_score: float
    max_audio_score: float
    coverage_ratio: float
    accepted: bool


def intersection_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def expand_scenes_to_units(
    segments: list[SceneSegment],
    max_scene_seconds: float,
    window_seconds: float,
) -> list[tuple[float, float]]:
    """
    Сцены из PySceneDetect; если сцена длиннее max_scene_seconds — режем на окна window_seconds.
    """
    eps = 1e-9
    out: list[tuple[float, float]] = []
    if max_scene_seconds <= eps or window_seconds <= eps:
        for seg in segments:
            out.append((float(seg.start_seconds), float(seg.end_seconds)))
        return out
    for seg in segments:
        a = float(seg.start_seconds)
        b = float(seg.end_seconds)
        dur = b - a
        if dur <= max_scene_seconds + eps:
            out.append((a, b))
            continue
        for rel_s, rel_e in iter_window_spans(dur, window_seconds):
            out.append((a + rel_s, a + rel_e))
    return out


def weighted_mean_norm_in_span(s: float, e: float, windows: list[WindowRow]) -> float:
    """Среднее norm по пересечениям, взвешенное длительностью пересечения."""
    num = 0.0
    den = 0.0
    for w in windows:
        if w.norm is None or not math.isfinite(w.norm):
            continue
        il = intersection_seconds(s, e, w.t_start, w.t_end)
        if il <= 1e-15:
            continue
        num += float(w.norm) * il
        den += il
    if den <= 1e-15:
        return 0.0
    return max(0.0, min(1.0, num / den))


def evaluate_scene_unit(
    index: int,
    us: float,
    ue: float,
    windows: list[WindowRow],
    threshold: float,
    min_audio_coverage_ratio: float,
    min_peak_score: float,
) -> SceneUnitEval:
    D = max(0.0, ue - us)
    eps = 1e-12
    if D <= eps:
        return SceneUnitEval(index, us, ue, 0.0, 0.0, 0.0, False)

    strong_cover = 0.0
    weighted_num = 0.0
    weighted_den = 0.0
    max_peak = 0.0

    for w in windows:
        if w.norm is None or not math.isfinite(w.norm):
            continue
        il = intersection_seconds(us, ue, w.t_start, w.t_end)
        if il <= eps:
            continue
        weighted_num += float(w.norm) * il
        weighted_den += il
        max_peak = max(max_peak, float(w.norm))
        if float(w.norm) >= threshold - eps:
            strong_cover += il

    coverage_ratio = strong_cover / D if D > eps else 0.0
    avg_audio = weighted_num / weighted_den if weighted_den > eps else 0.0
    accepted = coverage_ratio >= min_audio_coverage_ratio - eps or max_peak >= min_peak_score - eps
    return SceneUnitEval(
        index,
        us,
        ue,
        max(0.0, min(1.0, avg_audio)),
        max(0.0, min(1.0, max_peak)),
        max(0.0, min(1.0, coverage_ratio)),
        accepted,
    )


def merge_accepted_scene_spans(
    spans: list[tuple[float, float]],
    merge_gap_seconds: float,
    max_clip_seconds: float,
) -> list[tuple[float, float]]:
    """Склеивание только соседних по времени выбранных единиц с учётом gap и лимита max_clip_seconds."""
    eps = 1e-9
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: x[0])
    out: list[tuple[float, float]] = []
    cur_s, cur_e = spans[0]

    for s, e in spans[1:]:
        gap = s - cur_e
        merged_s, merged_e = cur_s, max(cur_e, e)
        if gap <= merge_gap_seconds + eps and (merged_e - merged_s) <= max_clip_seconds + eps:
            cur_e = merged_e
        else:
            out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    out.append((cur_s, cur_e))
    return out


def iter_window_spans(duration: float, window_sec: float) -> list[tuple[float, float]]:
    if duration <= 0 or window_sec <= 0:
        return []
    spans: list[tuple[float, float]] = []
    t = 0.0
    eps = 1e-9
    while t < duration - eps:
        seg_end = min(t + window_sec, duration)
        if seg_end - t < eps:
            break
        spans.append((t, seg_end))
        t = seg_end
    return spans


def normalize_window_scores(rows: list[WindowRow]) -> None:
    """
    После расчёта всех успешных окон: normalized_score = raw_score / max_raw_score.
    При max_raw_score == 0 (или ~0): все normalized_score безопасно 0.
    """
    positives: list[float] = []
    for r in rows:
        if r.raw_score is not None and math.isfinite(r.raw_score) and r.raw_score > 0.0:
            positives.append(r.raw_score)
    max_raw = max(positives) if positives else 0.0
    eps = 1e-15

    for r in rows:
        if r.raw_score is None:
            r.norm = None
            continue
        if max_raw <= eps:
            r.norm = 0.0
            continue
        r.norm = max(0.0, min(1.0, r.raw_score / max_raw))


def window_selected(r: WindowRow, threshold: float) -> bool:
    return (
        r.norm is not None
        and math.isfinite(r.norm)
        and r.norm >= threshold
    )


def _score_display(r: WindowRow) -> str:
    if r.norm is not None and math.isfinite(r.norm):
        s = f"{r.norm:.2f}"
        return s.rstrip("0").rstrip(".") if "." in s else s
    return "-"


def print_window_lines_stdout(rows: list[WindowRow], threshold: float, mode: str) -> None:
    """mode: none | all | selected — компактные строки window MM:SS–MM:SS score=…"""
    if mode == "none" or not rows:
        return
    n_out = 0
    for r in sorted(rows, key=lambda x: x.index):
        sel = window_selected(r, threshold)
        if mode == "selected" and not sel:
            continue
        span = seconds_to_display_span(r.t_start, r.t_end)
        score = _score_display(r)
        suffix = "  ← выбрано" if sel else ""
        print(f"window {span} score={score}{suffix}")
        n_out += 1
    if n_out:
        print("")


def format_scene_pipeline_debug_sections(
    scene_segments: list[SceneSegment],
    unit_evals: list[SceneUnitEval],
    merged_spans: list[tuple[float, float]],
    *,
    threshold: float,
    min_cov: float,
    min_peak: float,
) -> list[str]:
    lines: list[str] = []
    lines.append("--- Сцены (PySceneDetect, после min_scene_seconds) ---")
    if not scene_segments:
        lines.append("  (нет)")
    else:
        for i, seg in enumerate(scene_segments, start=1):
            d = seg.end_seconds - seg.start_seconds
            lines.append(
                f"  #{i} start={seg.start_seconds:.3f}s end={seg.end_seconds:.3f}s duration={d:.3f}s"
            )
    lines.append("")
    lines.append("--- Единицы отбора (сцена или фрагмент длинной сцены) ---")
    lines.append(
        f"  (threshold={threshold}, min_audio_coverage_ratio={min_cov}, min_peak_score={min_peak})"
    )
    if not unit_evals:
        lines.append("  (нет)")
    else:
        for ev in unit_evals:
            d = ev.t_end - ev.t_start
            status = "принято" if ev.accepted else "отклонено"
            lines.append(
                f"  #{ev.index} {seconds_to_display_span(ev.t_start, ev.t_end)} "
                f"start={ev.t_start:.3f}s end={ev.t_end:.3f}s duration={d:.3f}s | "
                f"avg_audio_score={ev.avg_audio_score:.4f} max_audio_score={ev.max_audio_score:.4f} "
                f"coverage_ratio={ev.coverage_ratio:.4f} | {status}"
            )
    lines.append("")
    lines.append("--- Слияние выбранных единиц (merge_gap_seconds, max_clip_seconds) ---")
    if not merged_spans:
        lines.append("  (нет)")
    else:
        for i, (s, e) in enumerate(merged_spans, start=1):
            d = e - s
            lines.append(
                f"  #{i} {seconds_to_display_span(s, e)} start={s:.3f}s end={e:.3f}s duration={d:.3f}s"
            )
    lines.append("")
    return lines


def format_generate_debug_lines(
    settings: ToolSettings,
    video: Path,
    duration: float | None,
    rows: list[WindowRow],
    threshold: float,
    raw_segments: list[tuple[float, float, list[float]]],
    merged_segments: list[tuple[float, float, list[float]]],
    clips_out: list[dict[str, object]],
    fatal: str | None,
    *,
    scene_mode: bool = False,
    scene_pyscene_count: int | None = None,
    scene_after_min_count: int | None = None,
    scene_debug_extra: list[str] | None = None,
) -> list[str]:
    lines: list[str] = []
    lines.append("")
    lines.append("--- Отладка: параметры ---")
    lines.append(f"  video: {video}")
    lines.append(f"  duration_sec: {duration}")
    sd = settings.scene_detection
    lines.append(f"  scene_detection: {'enabled' if sd.enabled else 'disabled'}")
    if scene_pyscene_count is not None:
        lines.append(f"  scenes_pyscenedetect_count: {scene_pyscene_count}")
    if scene_after_min_count is not None:
        lines.append(f"  scenes_after_min_scene_seconds: {scene_after_min_count}")
    lines.append(f"  window_seconds: {settings.window_seconds}")
    if scene_mode:
        lines.append(f"  min_audio_coverage_ratio: {settings.min_audio_coverage_ratio}")
        lines.append(f"  min_peak_score: {settings.min_peak_score}")
        lines.append(f"  max_scene_seconds: {settings.scene_detection.max_scene_seconds}")
    lines.append(f"  threshold: {threshold}")
    lines.append(f"  merge_gap_seconds: {settings.merge_gap_seconds}")
    lines.append(
        f"  padding_before/after: {settings.padding_before_seconds} / {settings.padding_after_seconds}"
    )
    lines.append(f"  min_clip / max_clip: {settings.min_clip_seconds} / {settings.max_clip_seconds}")
    if fatal:
        lines.append(f"  fatal: {fatal}")
    lines.append("")
    unit_title = "Аудио-окна (нормализованные)" if scene_mode else "Все окна анализа"
    lines.append(f"--- {unit_title} ---")
    for r in sorted(rows, key=lambda x: x.index):
        span = seconds_to_display_span(r.t_start, r.t_end)
        mark = " [выбрано]" if window_selected(r, threshold) else ""
        if r.mean_db is not None:
            nd = "" if r.norm is None else f"{r.norm:.6f}".rstrip("0").rstrip(".")
            raw_s = "" if r.raw_score is None else f"{r.raw_score:.6e}".rstrip("0").rstrip(".")
            lines.append(
                f"  #{r.index} {span}{mark} | mean_db={r.mean_db:.4f} dB | "
                f"raw={raw_s} | norm={nd}"
            )
        else:
            hint = r.analysis_error or "—"
            lines.append(f"  #{r.index} {span}{mark} | (нет mean_db) {hint}")
    lines.append("")
    if scene_debug_extra:
        lines.extend(scene_debug_extra)
    if not scene_mode:
        lines.append("--- Сегменты по порогу (contiguous) ---")
        if not raw_segments:
            lines.append("  (нет)")
        else:
            for i, (s, e, sc) in enumerate(raw_segments, start=1):
                lines.append(
                    f"  #{i} {seconds_to_display_span(s, e)} | scores({len(sc)}): "
                    f"min={min(sc):.4f} max={max(sc):.4f} mean={sum(sc)/len(sc):.4f}"
                )
        lines.append("")
        lines.append("--- После merge по gap ---")
        if not merged_segments:
            lines.append("  (нет)")
        else:
            for i, (s, e, sc) in enumerate(merged_segments, start=1):
                lines.append(
                    f"  #{i} {seconds_to_display_span(s, e)} | scores({len(sc)}): "
                    f"min={min(sc):.4f} max={max(sc):.4f} mean={sum(sc)/len(sc):.4f}"
                )
        lines.append("")
    lines.append(
        "--- Финальные кандидаты (JSON) ---" if scene_mode else "--- Клипы в JSON кандидате ---"
    )
    if not clips_out:
        lines.append("  (нет)")
    else:
        for c in clips_out:
            if scene_mode and "duration_sec" in c:
                ds = c.get("duration_sec")
                scv = c.get("normalized_score")
                lines.append(
                    f"  start={c.get('start')} end={c.get('end')} "
                    f"duration={float(ds):.3f}s score={scv}"
                )
            else:
                lines.append(f"  {c}")
    lines.append("")
    return lines


def build_window_scores(
    settings: ToolSettings,
    ffmpeg_bin: Path,
    ffprobe_bin: Path,
    video: Path,
    *,
    apply_normalize: bool = True,
) -> tuple[list[WindowRow], float | None, str | None]:
    dur, probe_err = ffprobe_duration_seconds(ffprobe_bin, video)
    rows: list[WindowRow] = []
    if dur is None or dur <= 0:
        return rows, dur, probe_err or "Некорректная длительность видео."

    spans = iter_window_spans(dur, settings.window_seconds)

    for i, (st, ed) in enumerate(spans):
        ln = max(0.0, ed - st)
        if ln <= 0:
            rows.append(WindowRow(i, st, ed, None, None, None, "Пустое окно."))
            continue
        mean_db, err = ffmpeg_volumedetect_mean_db(ffmpeg_bin, video, st, ln)
        if err is not None:
            rows.append(WindowRow(i, st, ed, None, None, None, err))
            continue
        rs = linear_raw_from_mean_db(mean_db)
        rows.append(WindowRow(i, st, ed, mean_db, rs, None, None))

    if apply_normalize:
        normalize_window_scores(rows)
    return rows, dur, None


def contiguous_selected_segments(
    rows: list[WindowRow],
    threshold: float,
) -> list[tuple[float, float, list[float]]]:
    """
    Сегменты по индексу окон: подряд идущие выбранные окна объединяются по времени.
    Возвращает (t0, t1, нормализованные оценки окон входа в сегмент).
    """
    out: list[tuple[float, float, list[float]]] = []
    run_start: float | None = None
    run_end: float | None = None
    run_scores: list[float] = []

    def flush() -> None:
        nonlocal run_start, run_end, run_scores
        if run_start is not None and run_end is not None and run_scores:
            out.append((run_start, run_end, run_scores.copy()))
        run_start = None
        run_end = None
        run_scores = []

    for r in sorted(rows, key=lambda x: x.index):
        sel = (
            r.norm is not None
            and math.isfinite(r.norm)
            and r.norm >= threshold
        )
        if not sel:
            flush()
            continue
        assert r.norm is not None
        if run_start is None:
            run_start = r.t_start
            run_end = r.t_end
            run_scores = [float(r.norm)]
        else:
            run_end = r.t_end
            run_scores.append(float(r.norm))
    flush()
    return out


def merge_by_gap(intervals: list[tuple[float, float, list[float]]], gap: float) -> list[tuple[float, float, list[float]]]:
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    cur_s, cur_e, cur_scores = intervals[0]
    merged: list[tuple[float, float, list[float]]] = []

    for s, e, sc in intervals[1:]:
        gap_here = s - cur_e
        if gap_here <= gap + 1e-9:
            cur_e = max(cur_e, e)
            cur_scores = cur_scores + sc
            continue
        merged.append((cur_s, cur_e, cur_scores))
        cur_s, cur_e, cur_scores = s, e, sc
    merged.append((cur_s, cur_e, cur_scores))
    return merged


def apply_padding(
    s: float,
    e: float,
    pad_before: float,
    pad_after: float,
    duration: float,
) -> tuple[float, float]:
    return max(0.0, s - pad_before), min(duration, e + pad_after)


def split_clip_bounds(
    s: float,
    e: float,
    max_sec: float,
    min_sec: float,
) -> list[tuple[float, float]]:
    """
    Разбиение [s,e): каждый кусок в [min_sec, max_sec], покрытие максимально без «дыр»;
    хвост короче min_sec отбрасывается (вызывающий считает это отдельно при необходимости).
    """
    eps = 1e-9
    out: list[tuple[float, float]] = []

    ln = max(0.0, e - s)
    if ln < min_sec - eps or max_sec < min_sec - eps:
        return out

    if ln <= max_sec + eps:
        return [(s, e)]

    cur = s
    while cur < e - eps:
        rem = max(0.0, e - cur)
        if rem < min_sec - eps:
            break
        if rem <= max_sec + eps:
            out.append((cur, e))
            break
        if rem - max_sec >= min_sec - eps:
            nxt = cur + max_sec
            out.append((cur, nxt))
            cur = nxt
        else:
            nxt = cur + (rem - min_sec)
            if nxt - cur < min_sec - eps:
                return []
            out.append((cur, nxt))
            cur = nxt

    return out


def build_candidate_clip_name(global_index: int, mean_norm_rounded_3: float) -> str:
    pct = max(0, min(999, int(round(mean_norm_rounded_3 * 1000))))
    base = sanitize_clip_filename_part(f"cand_{global_index:03d}_score{pct}")
    return base


def run_generate(
    video: Path,
    candidate_json: Path,
    output_dir_display: Path,
    settings: ToolSettings,
    *,
    list_windows: str = "none",
    effective_debug: EffectiveDebug,
    config_path_used: Path,
    config_path_explicit_cli: bool,
    project_path_used: Path | None,
) -> tuple[CandidateBuildSummary, str | None]:
    if not video.is_file():
        return CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0), f"Файл видео не найден: {video}"

    if effective_debug.enabled:
        pre = format_generate_pre_snapshot(
            settings,
            video=video,
            candidate_out=candidate_json,
            clips_dir=output_dir_display,
            config_path=config_path_used,
            config_explicit_cli=config_path_explicit_cli,
            project_path=project_path_used,
        )
        emit_debug_lines(pre, log_path=effective_debug.log_path)

    try:
        ff = ffmpeg_available(settings.ffmpeg_path)
        fp = ffprobe_exe_for(ff, settings.ffprobe_path)
    except FFmpegMissingError as e:
        return CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0), str(e)

    scene_mode = settings.scene_detection.enabled
    scene_pyscene_count: int | None = None
    scene_after_min_count: int | None = None
    scene_segments: list[SceneSegment] = []
    unit_evals: list[SceneUnitEval] = []
    scene_debug_extra: list[str] | None = None
    merged_raw: list[tuple[float, float]] = []

    if scene_mode:
        try:
            scene_pyscene_count, scene_segments = detect_scenes_with_counts(
                video, settings.scene_detection
            )
        except ImportError as e:
            return (
                CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0, scene_detection_used=True),
                f"Для scene_detection включён режим, но не установлен scenedetect: {e}",
            )
        except Exception as e:
            return (
                CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0, scene_detection_used=True),
                f"Ошибка PySceneDetect: {e}",
            )
        scene_after_min_count = len(scene_segments)
        units = expand_scenes_to_units(
            scene_segments,
            settings.scene_detection.max_scene_seconds,
            settings.window_seconds,
        )
        rows, duration, probe_msg = build_window_scores(
            settings, ff, fp, video, apply_normalize=False
        )
        normalize_window_scores(rows)
        unit_evals = [
            evaluate_scene_unit(
                i,
                us,
                ue,
                rows,
                settings.threshold,
                settings.min_audio_coverage_ratio,
                settings.min_peak_score,
            )
            for i, (us, ue) in enumerate(units)
        ]
        selected = sum(1 for e in unit_evals if e.accepted)
    else:
        rows, duration, probe_msg = build_window_scores(settings, ff, fp, video)
        selected = sum(
            1
            for r in rows
            if r.norm is not None and math.isfinite(r.norm) and r.norm >= settings.threshold
        )

    total_windows = len(rows)

    analyzed_ok = sum(1 for r in rows if r.mean_db is not None)
    analyze_failed = sum(1 for r in rows if r.mean_db is None)

    skipped_short = 0
    merged_segments = 0
    clips_out: list[dict[str, object]] = []
    fatal: str | None = None
    raw_segments: list[tuple[float, float, list[float]]] = []
    merged_segments_list: list[tuple[float, float, list[float]]] = []

    if scene_mode:
        if duration is None or duration <= 0:
            fatal = probe_msg
        else:
            accepted_spans = [(e.t_start, e.t_end) for e in unit_evals if e.accepted]
            merged_raw = merge_accepted_scene_spans(
                accepted_spans,
                settings.merge_gap_seconds,
                settings.max_clip_seconds,
            )
            merged_segments = len(merged_raw)
            gid = 0
            for ms, me in merged_raw:
                for cs, ce in split_clip_bounds(
                    ms,
                    me,
                    settings.max_clip_seconds,
                    settings.min_clip_seconds,
                ):
                    pad_s, pad_e = apply_padding(
                        cs,
                        ce,
                        settings.padding_before_seconds,
                        settings.padding_after_seconds,
                        duration,
                    )
                    if pad_e - pad_s < settings.min_clip_seconds - 1e-9:
                        skipped_short += 1
                        continue
                    gid += 1
                    agg = weighted_mean_norm_in_span(pad_s, pad_e, rows)
                    agg3 = round(agg, 3)
                    name = build_candidate_clip_name(gid, agg3)
                    clips_out.append(
                        {
                            "name": name,
                            "start": seconds_to_ffmpeg_tc(pad_s),
                            "end": seconds_to_ffmpeg_tc(pad_e),
                            "normalized_score": agg3,
                            "duration_sec": pad_e - pad_s,
                        }
                    )
        scene_debug_extra = format_scene_pipeline_debug_sections(
            scene_segments,
            unit_evals,
            merged_raw,
            threshold=settings.threshold,
            min_cov=settings.min_audio_coverage_ratio,
            min_peak=settings.min_peak_score,
        )
    elif duration is None or duration <= 0:
        merged_segments = 0
        fatal = probe_msg
    else:
        raw_segments = contiguous_selected_segments(rows, settings.threshold)
        merged_segments_list = merge_by_gap(raw_segments, settings.merge_gap_seconds)
        merged_segments = len(merged_segments_list)

        gid = 0
        for seg_s_raw, seg_e_raw, norms in merged_segments_list:
            if not norms:
                continue
            agg = sum(norms) / len(norms)

            pad_s, pad_e = apply_padding(
                seg_s_raw,
                seg_e_raw,
                settings.padding_before_seconds,
                settings.padding_after_seconds,
                duration,
            )

            slices = split_clip_bounds(pad_s, pad_e, settings.max_clip_seconds, settings.min_clip_seconds)
            for a, b in slices:
                if b - a < settings.min_clip_seconds - 1e-9:
                    skipped_short += 1
                    continue
                gid += 1
                agg3 = round(agg, 3)
                name = build_candidate_clip_name(gid, agg3)
                clips_out.append(
                    {
                        "name": name,
                        "start": seconds_to_ffmpeg_tc(a),
                        "end": seconds_to_ffmpeg_tc(b),
                        "normalized_score": agg3,
                    }
                )

    payload = {
        "input_video": str(video.resolve()),
        "output_dir": str(output_dir_display.resolve()),
        "clips": clips_out,
    }

    candidate_json.parent.mkdir(parents=True, exist_ok=True)
    candidate_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if list_windows in ("all", "selected"):
        print_window_lines_stdout(rows, settings.threshold, list_windows)

    if effective_debug.enabled:
        post = format_generate_post_metrics(
            settings,
            video=video,
            candidate_out=candidate_json,
            config_path=config_path_used,
            config_explicit_cli=config_path_explicit_cli,
            duration_sec=duration,
            summary_analyzed_ok=analyzed_ok,
            summary_selected_windows=selected,
            summary_generated=len(clips_out),
            scene_pyscene_count=scene_pyscene_count if scene_mode else None,
            scene_after_min_count=scene_after_min_count if scene_mode else None,
        )
        detail = format_generate_debug_lines(
            settings,
            video,
            duration,
            rows,
            settings.threshold,
            raw_segments,
            merged_segments_list,
            clips_out,
            fatal,
            scene_mode=scene_mode,
            scene_pyscene_count=scene_pyscene_count,
            scene_after_min_count=scene_after_min_count,
            scene_debug_extra=scene_debug_extra if scene_mode else None,
        )
        emit_debug_lines(post + detail, log_path=effective_debug.log_path)

    summary = CandidateBuildSummary(
        total_windows=total_windows,
        analyzed_ok=analyzed_ok,
        analyze_failed_windows=analyze_failed,
        selected_windows=selected,
        merged_segments=merged_segments,
        generated_candidates=len(clips_out),
        skipped_short=skipped_short,
        scene_detection_used=scene_mode,
        scenes_pyscenedetect=scene_pyscene_count,
        scenes_after_min_duration=scene_after_min_count,
        scene_units_total=len(unit_evals) if scene_mode else None,
    )
    return summary, fatal


def print_generate_summary(summary: CandidateBuildSummary) -> None:
    print("")
    print("--- Сводка generate ---")
    if summary.scene_detection_used:
        print(f"Обнаружение сцен (PySceneDetect): enabled")
        if summary.scenes_pyscenedetect is not None:
            print(f"  Сцен до фильтра min_scene_seconds: {summary.scenes_pyscenedetect}")
        if summary.scenes_after_min_duration is not None:
            print(f"  Сцен после фильтра min_scene_seconds: {summary.scenes_after_min_duration}")
        if summary.scene_units_total is not None:
            print(f"  Единиц отбора (сцена/фрагмент): {summary.scene_units_total}")
        print(f"Всего аудио-окон: {summary.total_windows}")
        print(f"Окон с успешным аудио-анализом: {summary.analyzed_ok}")
        print(f"Окон без аудио-анализа: {summary.analyze_failed_windows}")
        print(f"Выбрано единиц (coverage / peak): {summary.selected_windows}")
    else:
        print(f"Обнаружение сцен: disabled")
        print(f"Всего окон: {summary.total_windows}")
        print(f"Окон после анализа (успех): {summary.analyzed_ok}")
        print(f"Окон без анализа: {summary.analyze_failed_windows}")
        print(f"Выбрано окон по порогу: {summary.selected_windows}")
    print(f"Объединённых сегментов: {summary.merged_segments}")
    print(f"Сгенерировано клипов-кандидатов: {summary.generated_candidates}")
    if summary.skipped_short:
        print(f"Отсечено слишком коротких вставок: {summary.skipped_short}")
    print("")


__all__ = ["CandidateBuildSummary", "run_generate", "print_generate_summary"]
