"""Метрики по нормализованным per-window сигналам (0..1).

Не путать со стандартным пакетом statistics: в этом проекте при импорте из
каталога приложения используется данный модуль (см. sys.path в __main__.py).
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

# Фиксированные константы (позже могут стать настройками).
DEFAULT_PERCENTILE = 90.0
ENTROPY_BIN_COUNT = 25


def safe_median(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    s = sorted(float(x) for x in values if math.isfinite(x))
    if len(s) < 2:
        return None
    n = len(s)
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def safe_percentile(values: Sequence[float], percentile: float = DEFAULT_PERCENTILE) -> float | None:
    if len(values) < 2:
        return None
    s = sorted(float(x) for x in values if math.isfinite(x))
    if len(s) < 2:
        return None
    if not (0.0 < percentile < 100.0):
        return None
    pos = (len(s) - 1) * (percentile / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return s[lo]
    w = pos - lo
    return s[lo] * (1.0 - w) + s[hi] * w


def safe_stddev(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    xs = [float(x) for x in values if math.isfinite(x)]
    if len(xs) < 2:
        return None
    mean = sum(xs) / len(xs)
    var = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    if var < 0.0:
        return None
    r = math.sqrt(var)
    return r if math.isfinite(r) else None


def peak_density(
    values: Sequence[float],
    threshold: float,
    duration_sec: float,
) -> float | None:
    if duration_sec <= 0.0 or not math.isfinite(duration_sec):
        return None
    peak_count = sum(
        1
        for v in values
        if math.isfinite(v) and float(v) >= threshold
    )
    return peak_count / duration_sec


def peak_duration_from_timed_windows(
    timed: Iterable[tuple[float, float, float]],
    threshold: float,
) -> float | None:
    """
    Средняя длительность непрерывных активных сегментов.
    timed: (t0, t1, score) интервалы в порядке времени; разрыв между активными
    интервалами завершает текущий peak-segment.
    """
    items = [
        (float(t0), float(t1), float(sc))
        for t0, t1, sc in timed
        if math.isfinite(t0) and math.isfinite(t1) and t1 > t0 and math.isfinite(sc)
    ]
    if not items:
        return None
    items.sort(key=lambda x: x[0])
    eps = 1e-9
    durations: list[float] = []
    run_len = 0.0
    run_end: float | None = None

    for t0, t1, sc in items:
        seg_d = max(0.0, t1 - t0)
        active = sc >= threshold - eps
        if active:
            if run_end is None:
                run_len = seg_d
                run_end = t1
            elif t0 <= run_end + eps:
                run_len += seg_d
                run_end = max(run_end, t1)
            else:
                if run_len > eps:
                    durations.append(run_len)
                run_len = seg_d
                run_end = t1
        else:
            if run_end is not None and run_len > eps:
                durations.append(run_len)
            run_end = None
            run_len = 0.0
    if run_end is not None and run_len > eps:
        durations.append(run_len)
    if not durations:
        return None
    return sum(durations) / len(durations)


def shannon_entropy_histogram(
    values: Sequence[float],
    *,
    num_bins: int = ENTROPY_BIN_COUNT,
) -> float | None:
    if len(values) < 2:
        return None
    if num_bins < 1:
        return None
    xs = [float(x) for x in values if math.isfinite(x)]
    if len(xs) < 2:
        return None
    counts = [0] * num_bins
    width = 1.0 / num_bins
    for x in xs:
        x = max(0.0, min(1.0, x))
        if x >= 1.0 - 1e-15:
            idx = num_bins - 1
        else:
            idx = int(x / width)
            idx = max(0, min(num_bins - 1, idx))
        counts[idx] += 1
    n = float(len(xs))
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / n
        ent -= p * math.log(p)
    return ent if math.isfinite(ent) else None


def collect_norms_in_span(
    s: float,
    e: float,
    windows: list[tuple[float, float, float | None]],
) -> list[tuple[float, float, float]]:
    """
    Окна (t_start, t_end, norm); norm может быть None — пропуск.
    Возвращает список (t0, t1, norm) для окон с ненулевым пересечением [s,e].
    """
    eps = 1e-12
    out: list[tuple[float, float, float]] = []
    for t0, t1, norm in windows:
        if norm is None or not math.isfinite(norm):
            continue
        il0 = max(s, t0)
        il1 = min(e, t1)
        if il1 - il0 <= eps:
            continue
        out.append((t0, t1, float(norm)))
    out.sort(key=lambda x: x[0])
    return out


__all__ = [
    "DEFAULT_PERCENTILE",
    "ENTROPY_BIN_COUNT",
    "collect_norms_in_span",
    "peak_density",
    "peak_duration_from_timed_windows",
    "safe_median",
    "safe_percentile",
    "safe_stddev",
    "shannon_entropy_histogram",
]
