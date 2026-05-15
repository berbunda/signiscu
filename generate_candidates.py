"""Генерация JSON кандидата (candidate_clips) по окнам громкости или по сценам PySceneDetect + звук."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, fields
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
from progress_ui import tqdm_labeled
from scene_analysis import SceneSegment, detect_scenes_with_counts
from motion_analysis import analyze_motion_raw_for_spans, normalize_motion_metrics
from project_toml import MetricsSettings
from settings import MotionAnalysisSettings, ToolSettings
from statistics import (
    collect_norms_in_span,
    peak_density,
    peak_duration_from_timed_windows,
    safe_median,
    safe_percentile,
    safe_stddev,
    shannon_entropy_histogram,
)
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
    audio_coverage_ratio: float
    avg_motion_score: float
    max_motion_score: float
    min_motion_score: float
    motion_coverage_ratio: float
    selected_reasons: tuple[str, ...]
    accepted: bool


def intersection_seconds(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def expand_scenes_to_units(
    segments: list[SceneSegment],
    max_scene_seconds: float,
    window_seconds: float,
) -> list[tuple[float, float]]:
    """
    Сцены из scene detection; если сцена длиннее max_scene_seconds — режем на окна window_seconds.
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


def audio_metrics_in_span(
    s: float,
    e: float,
    windows: list[WindowRow],
    threshold: float,
) -> tuple[float, float, float, float]:
    """
    Показатели аудио по интервалу [s,e]: min, max norm пересекающихся окон,
    средневзвешенное norm, доля времени где norm >= threshold.
    """
    D = max(0.0, e - s)
    eps = 1e-12
    if D <= eps:
        return 0.0, 0.0, 0.0, 0.0
    strong_cover = 0.0
    weighted_num = 0.0
    weighted_den = 0.0
    max_peak = 0.0
    min_peak: float | None = None
    for w in windows:
        if w.norm is None or not math.isfinite(w.norm):
            continue
        il = intersection_seconds(s, e, w.t_start, w.t_end)
        if il <= eps:
            continue
        n = float(w.norm)
        weighted_num += n * il
        weighted_den += il
        max_peak = max(max_peak, n)
        min_peak = n if min_peak is None else min(min_peak, n)
        if n >= threshold - eps:
            strong_cover += il
    coverage_ratio = strong_cover / D if D > eps else 0.0
    avg_audio = weighted_num / weighted_den if weighted_den > eps else 0.0
    min_audio = max(0.0, min(1.0, min_peak)) if min_peak is not None else 0.0
    return (
        min_audio,
        max(0.0, min(1.0, max_peak)),
        max(0.0, min(1.0, avg_audio)),
        max(0.0, min(1.0, coverage_ratio)),
    )


def weighted_mean_norm_in_span(s: float, e: float, windows: list[WindowRow]) -> float:
    """Среднее norm по пересечениям, взвешенное длительностью пересечения."""
    _, _, avg, _ = audio_metrics_in_span(s, e, windows, threshold=1.0)
    return avg


def evaluate_scene_unit(
    index: int,
    us: float,
    ue: float,
    windows: list[WindowRow],
    threshold: float,
    min_audio_coverage_ratio: float,
    min_peak_score: float,
    avg_motion_score: float,
    max_motion_score: float,
    min_motion_score: float,
    motion_coverage_ratio: float,
    min_motion_coverage_ratio: float,
    min_motion_peak_score: float,
    motion_for_selection: bool,
) -> SceneUnitEval:
    D = max(0.0, ue - us)
    eps = 1e-12
    if D <= eps:
        return SceneUnitEval(
            index,
            us,
            ue,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            tuple(),
            False,
        )

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
    reasons: list[str] = []
    if coverage_ratio >= min_audio_coverage_ratio - eps:
        reasons.append("audio_coverage")
    if max_peak >= min_peak_score - eps:
        reasons.append("audio_peak")
    if motion_for_selection and motion_coverage_ratio >= min_motion_coverage_ratio - eps:
        reasons.append("motion_coverage")
    if motion_for_selection and max_motion_score >= min_motion_peak_score - eps:
        reasons.append("motion_peak")
    accepted = bool(reasons)
    return SceneUnitEval(
        index,
        us,
        ue,
        max(0.0, min(1.0, avg_audio)),
        max(0.0, min(1.0, max_peak)),
        max(0.0, min(1.0, coverage_ratio)),
        max(0.0, min(1.0, avg_motion_score)),
        max(0.0, min(1.0, max_motion_score)),
        max(0.0, min(1.0, min_motion_score)),
        max(0.0, min(1.0, motion_coverage_ratio)),
        tuple(reasons),
        accepted,
    )


def clip_combined_normalized_score(
    avg_audio: float,
    avg_motion: float,
    motion_cfg: MotionAnalysisSettings,
) -> float:
    """Итоговый score для клипа: только аудио, либо смесь с motion при affect_score."""
    aa = max(0.0, min(1.0, avg_audio))
    if not motion_cfg.enabled or not motion_cfg.affect_score:
        return aa
    w = max(0.0, min(1.0, motion_cfg.weight_motion))
    al = max(0.0, min(1.0, avg_motion))
    return (1.0 - w) * aa + w * al


def _audio_triplets(rows: list[WindowRow]) -> list[tuple[float, float, float | None]]:
    return [(w.t_start, w.t_end, w.norm) for w in rows]


def audio_signal_in_clip(
    pad_s: float, pad_e: float, rows: list[WindowRow]
) -> tuple[list[float], list[tuple[float, float, float]]]:
    timed_full = collect_norms_in_span(pad_s, pad_e, _audio_triplets(rows))
    clipped: list[tuple[float, float, float]] = []
    for t0, t1, n in timed_full:
        il0 = max(pad_s, t0)
        il1 = min(pad_e, t1)
        if il1 > il0:
            clipped.append((il0, il1, n))
    return [n for _, _, n in clipped], clipped


def motion_signal_in_clip(
    pad_s: float,
    pad_e: float,
    timed_per_unit: list[list[tuple[float, float, float]]],
) -> tuple[list[float], list[tuple[float, float, float]]]:
    eps = 1e-12
    out: list[tuple[float, float, float]] = []
    for unit_segs in timed_per_unit:
        for t0, t1, n in unit_segs:
            il0 = max(pad_s, t0)
            il1 = min(pad_e, t1)
            if il1 - il0 > eps:
                out.append((il0, il1, n))
    out.sort(key=lambda x: x[0])
    return [x[2] for x in out], out


_METRIC_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in fields(MetricsSettings))


def metrics_enabled_and_skipped(m: MetricsSettings) -> tuple[list[str], list[str]]:
    enabled: list[str] = []
    skipped: list[str] = []
    for k in _METRIC_FIELD_NAMES:
        if getattr(m, k):
            enabled.append(k)
        else:
            skipped.append(k)
    return enabled, skipped


def build_clip_metrics_payload(
    m: MetricsSettings,
    *,
    duration_sec: float,
    audio_threshold: float,
    motion_threshold: float,
    audio_norms: list[float],
    audio_timed: list[tuple[float, float, float]],
    motion_norms: list[float],
    motion_timed: list[tuple[float, float, float]],
    motion_pipeline_active: bool,
) -> dict[str, float | None]:
    """Только включённые метрики; motion — только при активном motion pipeline."""
    out: dict[str, float | None] = {}

    if m.audio_median:
        v = safe_median(audio_norms)
        out["audio_median"] = None if v is None else round(v, 4)
    if m.audio_percentile:
        v = safe_percentile(audio_norms)
        out["audio_percentile"] = None if v is None else round(v, 4)
    if m.audio_stddev:
        v = safe_stddev(audio_norms)
        out["audio_stddev"] = None if v is None else round(v, 4)
    if m.audio_peak_density:
        v = peak_density(audio_norms, audio_threshold, duration_sec) if audio_norms else None
        out["audio_peak_density"] = None if v is None else round(v, 4)
    if m.audio_peak_duration:
        v = peak_duration_from_timed_windows(audio_timed, audio_threshold)
        out["audio_peak_duration"] = None if v is None else round(v, 4)
    if m.audio_entropy:
        v = shannon_entropy_histogram(audio_norms)
        out["audio_entropy"] = None if v is None else round(v, 4)

    motion_any = (
        m.motion_median
        or m.motion_percentile
        or m.motion_stddev
        or m.motion_peak_density
        or m.motion_peak_duration
        or m.motion_entropy
    )
    if motion_any and motion_pipeline_active:
        if m.motion_median:
            v = safe_median(motion_norms)
            out["motion_median"] = None if v is None else round(v, 4)
        if m.motion_percentile:
            v = safe_percentile(motion_norms)
            out["motion_percentile"] = None if v is None else round(v, 4)
        if m.motion_stddev:
            v = safe_stddev(motion_norms)
            out["motion_stddev"] = None if v is None else round(v, 4)
        if m.motion_peak_density:
            v = peak_density(motion_norms, motion_threshold, duration_sec) if motion_norms else None
            out["motion_peak_density"] = None if v is None else round(v, 4)
        if m.motion_peak_duration:
            v = peak_duration_from_timed_windows(motion_timed, motion_threshold)
            out["motion_peak_duration"] = None if v is None else round(v, 4)
        if m.motion_entropy:
            v = shannon_entropy_histogram(motion_norms)
            out["motion_entropy"] = None if v is None else round(v, 4)

    return out


def format_clip_metrics_debug_lines(
    clip_name: str,
    payload: dict[str, float | None],
) -> list[str]:
    parts = [f"{k}={payload[k]!r}" for k in sorted(payload.keys())]
    return [f"  metrics [{clip_name}]: " + ("; ".join(parts) if parts else "(none)")]


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
    motion_cfg: MotionAnalysisSettings,
) -> list[str]:
    lines: list[str] = []
    lines.append("--- Сцены (после min_scene_seconds) ---")
    if not scene_segments:
        lines.append("  (нет)")
    else:
        for i, seg in enumerate(scene_segments, start=1):
            d = seg.end_seconds - seg.start_seconds
            lines.append(
                f"  #{i} start={seg.start_seconds:.3f}s end={seg.end_seconds:.3f}s duration={d:.3f}s"
            )
    lines.append("")
    lines.append("--- [motion_analysis] (конфиг) ---")
    lm_sel = motion_cfg.enabled and motion_cfg.affect_selection
    lm_sc = motion_cfg.enabled and motion_cfg.affect_score
    lines.extend(
        [
            f"  enabled = {motion_cfg.enabled}",
            f"  sample_fps = {motion_cfg.sample_fps}",
            f"  resize_width = {motion_cfg.resize_width}",
            f"  residual_percentile = {motion_cfg.residual_percentile}",
            f"  motion_threshold = {motion_cfg.motion_threshold}",
            f"  min_motion_coverage_ratio = {motion_cfg.min_motion_coverage_ratio}",
            f"  min_motion_peak_score = {motion_cfg.min_motion_peak_score}",
            f"  affect_selection = {motion_cfg.affect_selection} (motion для отбора: {lm_sel})",
            f"  affect_score = {motion_cfg.affect_score} (motion в normalized_score: {lm_sc})",
            f"  weight_motion = {motion_cfg.weight_motion}",
        ]
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
            status = "selected" if ev.accepted else "rejected"
            base = (
                f"  #{ev.index} {seconds_to_display_span(ev.t_start, ev.t_end)} "
                f"start={ev.t_start:.3f}s end={ev.t_end:.3f}s duration={d:.3f}s | "
                f"avg_audio_score={ev.avg_audio_score:.4f} max_audio_score={ev.max_audio_score:.4f} "
                f"audio_coverage_ratio={ev.audio_coverage_ratio:.4f}"
            )
            if motion_cfg.enabled:
                base += (
                    f" | avg_motion_score={ev.avg_motion_score:.4f} "
                    f"max_motion_score={ev.max_motion_score:.4f} "
                    f"min_motion_score={ev.min_motion_score:.4f} "
                    f"motion_coverage_ratio={ev.motion_coverage_ratio:.4f}"
                )
            base += f" | {status} | reason={'/'.join(ev.selected_reasons) if ev.selected_reasons else '-'}"
            lines.append(base)
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
    lines.append(f"  scene_detection.backend: {sd.backend}")
    if scene_pyscene_count is not None:
        lines.append(f"  scenes_before_min_scene_seconds: {scene_pyscene_count}")
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
    progress_desc: str | None = None,
) -> tuple[list[WindowRow], float | None, str | None]:
    dur, probe_err = ffprobe_duration_seconds(ffprobe_bin, video)
    rows: list[WindowRow] = []
    if dur is None or dur <= 0:
        return rows, dur, probe_err or "Некорректная длительность видео."

    spans = iter_window_spans(dur, settings.window_seconds)
    pairs = list(enumerate(spans))
    iter_pairs = (
        tqdm_labeled(pairs, desc=progress_desc, unit="окно", total=len(pairs))
        if progress_desc
        else pairs
    )

    for i, (st, ed) in iter_pairs:
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


def build_candidate_clip_name(video_path: Path, global_index: int, score: float) -> str:
    """Имя кандидата: source_name-candNN_scoreMMM (score → int(round(score * 1000)))."""
    stem = sanitize_clip_filename_part(video_path.stem)
    sc = int(round(score * 1000)) if math.isfinite(score) else 0
    assembled = f"{stem}-cand{global_index:02d}_score{sc}"
    return sanitize_clip_filename_part(assembled)


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
    metrics_settings: MetricsSettings | None = None,
) -> tuple[CandidateBuildSummary, str | None, bool]:
    if not video.is_file():
        return CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0), f"Файл видео не найден: {video}", False

    m_cfg = metrics_settings if metrics_settings is not None else MetricsSettings()
    motion_cfg = settings.motion_analysis
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
        return CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0), str(e), False

    audio_progress_desc = "[generate] Аудио: FFmpeg volumedetect (окна)"

    scene_mode = settings.scene_detection.enabled
    scene_pyscene_count: int | None = None
    scene_after_min_count: int | None = None
    scene_segments: list[SceneSegment] = []
    unit_evals: list[SceneUnitEval] = []
    scene_debug_extra: list[str] | None = None
    merged_raw: list[tuple[float, float]] = []

    if scene_mode:
        scene_backend_debug: list[str] = []
        try:
            scene_pyscene_count, scene_segments = detect_scenes_with_counts(
                video,
                settings.scene_detection,
                ffmpeg_bin=ff,
                ffprobe_bin=fp,
                debug_lines=scene_backend_debug if effective_debug.enabled else None,
            )
        except ValueError as e:
            return (
                CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0, scene_detection_used=True),
                str(e),
                False,
            )
        except ImportError as e:
            return (
                CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0, scene_detection_used=True),
                f"Для scene_detection.backend=pyscenedetect нужен пакет scenedetect: {e}",
                False,
            )
        except RuntimeError as e:
            backend = settings.scene_detection.backend
            return (
                CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0, scene_detection_used=True),
                f"Ошибка scene detection ({backend}): {e}",
                False,
            )
        except Exception as e:
            backend = settings.scene_detection.backend
            return (
                CandidateBuildSummary(0, 0, 0, 0, 0, 0, 0, scene_detection_used=True),
                f"Ошибка scene detection ({backend}): {e}",
                False,
            )
        if effective_debug.enabled and scene_backend_debug:
            emit_debug_lines(scene_backend_debug, log_path=effective_debug.log_path)
        scene_after_min_count = len(scene_segments)
        units = expand_scenes_to_units(
            scene_segments,
            settings.scene_detection.max_scene_seconds,
            settings.window_seconds,
        )
        motion_avg_by_unit = [0.0 for _ in units]
        motion_max_by_unit = [0.0 for _ in units]
        motion_min_by_unit = [0.0 for _ in units]
        motion_cov_by_unit = [0.0 for _ in units]
        motion_timed_by_unit: list[list[tuple[float, float, float]]] = [[] for _ in units]
        if motion_cfg.enabled and units:
            motion_raw = analyze_motion_raw_for_spans(
                video,
                units,
                sample_fps=motion_cfg.sample_fps,
                resize_width=motion_cfg.resize_width,
                residual_percentile=motion_cfg.residual_percentile,
                progress_desc="[generate] Видео: OpenCV (optical flow, motion)",
            )
            motion_metrics, motion_timed_by_unit = normalize_motion_metrics(
                motion_raw, motion_cfg.motion_threshold
            )
            motion_avg_by_unit = [x.avg_motion_score for x in motion_metrics]
            motion_max_by_unit = [x.max_motion_score for x in motion_metrics]
            motion_min_by_unit = [x.min_motion_score for x in motion_metrics]
            motion_cov_by_unit = [x.motion_coverage_ratio for x in motion_metrics]
        motion_for_selection = motion_cfg.enabled and motion_cfg.affect_selection
        rows, duration, probe_msg = build_window_scores(
            settings,
            ff,
            fp,
            video,
            apply_normalize=False,
            progress_desc=audio_progress_desc,
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
                motion_avg_by_unit[i],
                motion_max_by_unit[i],
                motion_min_by_unit[i],
                motion_cov_by_unit[i],
                motion_cfg.min_motion_coverage_ratio,
                motion_cfg.min_motion_peak_score,
                motion_for_selection,
            )
            for i, (us, ue) in enumerate(units)
        ]
        selected = sum(1 for e in unit_evals if e.accepted)
    else:
        rows, duration, probe_msg = build_window_scores(
            settings, ff, fp, video, progress_desc=audio_progress_desc
        )
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
    metrics_debug_lines: list[str] = []
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
                    min_aud, max_aud, avg_aud, cov_aud = audio_metrics_in_span(
                        pad_s, pad_e, rows, settings.threshold
                    )
                    avg_m = 0.0
                    max_m_peak = 0.0
                    min_m_peak = float("inf")
                    m_cov_num = 0.0
                    m_cov_den = 0.0
                    m_ov_num = 0.0
                    m_ov_den = 0.0
                    if motion_cfg.enabled:
                        for ev in unit_evals:
                            il = intersection_seconds(pad_s, pad_e, ev.t_start, ev.t_end)
                            if il <= 1e-12:
                                continue
                            m_ov_num += ev.avg_motion_score * il
                            m_ov_den += il
                            max_m_peak = max(max_m_peak, ev.max_motion_score)
                            min_m_peak = min(min_m_peak, ev.min_motion_score)
                            m_cov_num += ev.motion_coverage_ratio * il
                            m_cov_den += il
                        if m_ov_den > 1e-12:
                            avg_m = m_ov_num / m_ov_den
                    m_cov_agg = m_cov_num / m_cov_den if m_cov_den > 1e-12 else 0.0
                    min_m_out = 0.0 if min_m_peak == float("inf") else min_m_peak
                    combined = clip_combined_normalized_score(avg_aud, avg_m, motion_cfg)
                    agg3 = round(combined, 3)
                    name = build_candidate_clip_name(video, gid, agg3)
                    clip_rec: dict[str, object] = {
                        "name": name,
                        "start": seconds_to_ffmpeg_tc(pad_s),
                        "end": seconds_to_ffmpeg_tc(pad_e),
                        "duration_sec": pad_e - pad_s,
                        "normalized_score": agg3,
                        "audio_score": round(avg_aud, 3),
                        "min_audio_score": round(min_aud, 3),
                        "avg_audio_score": round(avg_aud, 3),
                        "max_audio_score": round(max_aud, 3),
                        "audio_coverage_ratio": round(cov_aud, 3),
                        "motion_score": round(avg_m, 3),
                        "min_motion_score": round(min_m_out, 3),
                        "avg_motion_score": round(avg_m, 3),
                        "max_motion_score": round(max_m_peak, 3),
                        "motion_coverage_ratio": round(m_cov_agg, 3),
                    }
                    clip_dur = pad_e - pad_s
                    au_norms, au_timed = audio_signal_in_clip(pad_s, pad_e, rows)
                    mo_norms, mo_timed = motion_signal_in_clip(pad_s, pad_e, motion_timed_by_unit)
                    mp_active = motion_cfg.enabled and bool(units)
                    pl = build_clip_metrics_payload(
                        m_cfg,
                        duration_sec=clip_dur,
                        audio_threshold=settings.threshold,
                        motion_threshold=motion_cfg.motion_threshold,
                        audio_norms=au_norms,
                        audio_timed=au_timed,
                        motion_norms=mo_norms,
                        motion_timed=mo_timed,
                        motion_pipeline_active=mp_active,
                    )
                    clip_rec.update(pl)
                    if effective_debug.enabled and pl:
                        metrics_debug_lines.extend(format_clip_metrics_debug_lines(name, pl))
                    clips_out.append(clip_rec)
        scene_debug_extra = format_scene_pipeline_debug_sections(
            scene_segments,
            unit_evals,
            merged_raw,
            threshold=settings.threshold,
            min_cov=settings.min_audio_coverage_ratio,
            min_peak=settings.min_peak_score,
            motion_cfg=settings.motion_analysis,
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
                name = build_candidate_clip_name(video, gid, agg3)
                rec: dict[str, object] = {
                    "name": name,
                    "start": seconds_to_ffmpeg_tc(a),
                    "end": seconds_to_ffmpeg_tc(b),
                    "normalized_score": agg3,
                }
                clip_dur = b - a
                au_norms, au_timed = audio_signal_in_clip(a, b, rows)
                pl = build_clip_metrics_payload(
                    m_cfg,
                    duration_sec=clip_dur,
                    audio_threshold=settings.threshold,
                    motion_threshold=motion_cfg.motion_threshold,
                    audio_norms=au_norms,
                    audio_timed=au_timed,
                    motion_norms=[],
                    motion_timed=[],
                    motion_pipeline_active=False,
                )
                rec.update(pl)
                if effective_debug.enabled and pl:
                    metrics_debug_lines.extend(format_clip_metrics_debug_lines(name, pl))
                clips_out.append(rec)

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
        en, sk = metrics_enabled_and_skipped(m_cfg)
        metrics_header = [
            "",
            "--- Метрики [metrics] (project.toml) ---",
            f"  enabled: {', '.join(en) if en else '(none)'}",
            f"  skipped: {', '.join(sk) if sk else '(none)'}",
            "",
        ]
        emit_debug_lines(
            post + metrics_header + metrics_debug_lines + detail,
            log_path=effective_debug.log_path,
        )

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
    return summary, fatal, True


def print_generate_summary(summary: CandidateBuildSummary) -> None:
    print("")
    print("--- Сводка generate ---")
    if summary.scene_detection_used:
        print(f"Обнаружение сцен (PySceneDetect): enabled")
        if summary.scenes_pyscenedetect is not None:
            print(f"  Сцен до фильтра min_scene_seconds (raw): {summary.scenes_pyscenedetect}")
        if summary.scenes_after_min_duration is not None:
            print(f"  Сцен после фильтра min_scene_seconds: {summary.scenes_after_min_duration}")
        if summary.scene_units_total is not None:
            print(f"  Единиц отбора (сцена/фрагмент): {summary.scene_units_total}")
        print(f"Всего аудио-окон: {summary.total_windows}")
        print(f"Окон с успешным аудио-анализом: {summary.analyzed_ok}")
        print(f"Окон без аудио-анализа: {summary.analyze_failed_windows}")
        print(
            "Выбрано единиц (audio_coverage / audio_peak / "
            "motion_coverage / motion_peak): "
            f"{summary.selected_windows}"
        )
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
