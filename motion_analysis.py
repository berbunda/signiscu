"""Локальное движение (остаток после вычитания глобального сдвига) через dense optical flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from progress_ui import tqdm_labeled


@dataclass
class MotionSpanRaw:
    """Сырые пары кадров в интервале сцены: (t0, t1, raw_score)."""

    segments: list[tuple[float, float, float]]


@dataclass
class MotionSceneMetrics:
    avg_motion_score: float
    max_motion_score: float
    min_motion_score: float
    motion_coverage_ratio: float


def _prepare_gray(frame: np.ndarray, resize_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > resize_width > 0:
        new_h = max(1, int(h * (resize_width / float(w))))
        frame = cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _iter_sampled_frames(cap: cv2.VideoCapture, start_s: float, end_s: float, sample_fps: float):
    if sample_fps <= 0:
        return
    t = max(0.0, start_s)
    step = 1.0 / sample_fps
    while t < end_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        yield frame
        t += step


def _pair_residual_percentile(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    *,
    residual_percentile: float,
) -> float:
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    gx = float(np.median(flow[..., 0]))
    gy = float(np.median(flow[..., 1]))
    rdx = flow[..., 0] - gx
    rdy = flow[..., 1] - gy
    mag = np.sqrt(rdx * rdx + rdy * rdy)
    pct = float(np.percentile(mag, residual_percentile))
    return pct


def analyze_motion_raw_for_spans(
    video_path: Path,
    spans: list[tuple[float, float]],
    *,
    sample_fps: float,
    resize_width: int,
    residual_percentile: float,
    progress_desc: str | None = None,
) -> list[MotionSpanRaw]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео для motion: {video_path}")
    out: list[MotionSpanRaw] = []
    dt = 1.0 / sample_fps if sample_fps > 0 else 0.0
    try:
        span_iter = spans
        if progress_desc:
            span_iter = tqdm_labeled(spans, desc=progress_desc, unit="фрагмент", total=len(spans))
        for start_s, end_s in span_iter:
            if end_s <= start_s or dt <= 0:
                out.append(MotionSpanRaw(segments=[]))
                continue
            prev_gray: np.ndarray | None = None
            segs: list[tuple[float, float, float]] = []
            k = 0
            for frame in _iter_sampled_frames(cap, start_s, end_s, sample_fps):
                cur = _prepare_gray(frame, resize_width)
                if prev_gray is not None:
                    raw_v = _pair_residual_percentile(
                        prev_gray,
                        cur,
                        residual_percentile=residual_percentile,
                    )
                    t0 = start_s + k * dt
                    t1 = start_s + (k + 1) * dt
                    segs.append((t0, t1, raw_v))
                    k += 1
                prev_gray = cur
            if not segs:
                out.append(MotionSpanRaw(segments=[]))
                continue
            out.append(MotionSpanRaw(segments=segs))
    finally:
        cap.release()
    return out


def normalize_motion_metrics(
    raw_items: list[MotionSpanRaw],
    motion_threshold: float,
) -> tuple[list[MotionSceneMetrics], list[list[tuple[float, float, float]]]]:
    """
    Глобальная нормализация raw / max_raw по всем spans; метрики и (t0,t1,norm) на span.
    """
    all_raw = [v for r in raw_items for _, _, v in r.segments]
    max_raw = max(all_raw, default=0.0)
    timed_norm_out: list[list[tuple[float, float, float]]] = []

    if max_raw <= 0.0:
        for r in raw_items:
            timed_norm_out.append([(t0, t1, 0.0) for t0, t1, _ in r.segments])
        return [MotionSceneMetrics(0.0, 0.0, 0.0, 0.0) for _ in raw_items], timed_norm_out

    for r in raw_items:
        norms: list[tuple[float, float, float]] = []
        for t0, t1, raw_v in r.segments:
            n = max(0.0, min(1.0, raw_v / max_raw))
            norms.append((t0, t1, n))
        timed_norm_out.append(norms)

    out_metrics: list[MotionSceneMetrics] = []
    for norms in timed_norm_out:
        pairs = [n for _, _, n in norms]
        if not pairs:
            out_metrics.append(MotionSceneMetrics(0.0, 0.0, 0.0, 0.0))
            continue
        avg_n = sum(pairs) / len(pairs)
        max_n = max(pairs)
        min_n = min(pairs)
        over = sum(1 for v in pairs if v >= motion_threshold)
        cov = over / len(pairs)
        out_metrics.append(MotionSceneMetrics(avg_n, max_n, min_n, cov))
    return out_metrics, timed_norm_out


__all__ = [
    "MotionSceneMetrics",
    "MotionSpanRaw",
    "analyze_motion_raw_for_spans",
    "normalize_motion_metrics",
]
