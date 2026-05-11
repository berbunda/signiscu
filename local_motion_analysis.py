"""Локальное движение (остаток после вычитания глобального сдвига) через dense optical flow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from progress_ui import tqdm_labeled


@dataclass
class LocalMotionSceneRaw:
    raw_pair_values: list[float]


@dataclass
class LocalMotionSceneMetrics:
    avg_local_motion_score: float
    max_local_motion_score: float
    min_local_motion_score: float
    local_motion_coverage_ratio: float


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


def analyze_local_motion_raw_for_spans(
    video_path: Path,
    spans: list[tuple[float, float]],
    *,
    sample_fps: float,
    resize_width: int,
    residual_percentile: float,
    progress_desc: str | None = None,
) -> list[LocalMotionSceneRaw]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео для local motion: {video_path}")
    out: list[LocalMotionSceneRaw] = []
    try:
        span_iter = spans
        if progress_desc:
            span_iter = tqdm_labeled(spans, desc=progress_desc, unit="фрагмент", total=len(spans))
        for start_s, end_s in span_iter:
            if end_s <= start_s:
                out.append(LocalMotionSceneRaw(raw_pair_values=[]))
                continue
            prev_gray: np.ndarray | None = None
            values: list[float] = []
            for frame in _iter_sampled_frames(cap, start_s, end_s, sample_fps):
                cur = _prepare_gray(frame, resize_width)
                if prev_gray is not None:
                    values.append(
                        _pair_residual_percentile(
                            prev_gray,
                            cur,
                            residual_percentile=residual_percentile,
                        )
                    )
                prev_gray = cur
            if not values:
                out.append(LocalMotionSceneRaw(raw_pair_values=[]))
                continue
            out.append(LocalMotionSceneRaw(raw_pair_values=values))
    finally:
        cap.release()
    return out


def normalize_local_motion_metrics(
    raw_items: list[LocalMotionSceneRaw],
    local_motion_threshold: float,
) -> list[LocalMotionSceneMetrics]:
    all_values = [v for r in raw_items for v in r.raw_pair_values]
    max_raw = max(all_values, default=0.0)
    if max_raw <= 0.0:
        return [LocalMotionSceneMetrics(0.0, 0.0, 0.0, 0.0) for _ in raw_items]

    per_scene: list[list[float]] = []
    for r in raw_items:
        if not r.raw_pair_values:
            per_scene.append([])
            continue
        normalized = [max(0.0, min(1.0, v / max_raw)) for v in r.raw_pair_values]
        per_scene.append(normalized)

    out: list[LocalMotionSceneMetrics] = []
    for pairs in per_scene:
        if not pairs:
            out.append(LocalMotionSceneMetrics(0.0, 0.0, 0.0, 0.0))
            continue
        avg_n = sum(pairs) / len(pairs)
        max_n = max(pairs)
        min_n = min(pairs)
        over = sum(1 for v in pairs if v >= local_motion_threshold)
        cov = over / len(pairs)
        out.append(LocalMotionSceneMetrics(avg_n, max_n, min_n, cov))
    return out


__all__ = [
    "LocalMotionSceneMetrics",
    "LocalMotionSceneRaw",
    "analyze_local_motion_raw_for_spans",
    "normalize_local_motion_metrics",
]
