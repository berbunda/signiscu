"""Разбор временных кодов для FFmpeg."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


class TimecodeError(ValueError):
    """Неверный или не поддерживаемый формат времени."""


_WS = r"\s*"
# ЧЧ:ММ:СС[.дробь] — дробь в последнем поле — десятичные доли секунды (как у FFmpeg)
_RE_HMS = re.compile(
    rf"^{_WS}(\d+)\s*:\s*(\d{{1,2}})\s*:\s*(\d{{1,2}})(?:\.(\d+))?\s*$"
)
# ММ:СС[.дробь]
_RE_MS = re.compile(rf"^{_WS}(\d{{1,2}})\s*:\s*(\d{{1,2}})(?:\.(\d+))?\s*$")
# СС[.дробь]
_RE_S = re.compile(rf"^{_WS}(\d+)(?:\.(\d+))?\s*$")


def _tail_as_seconds(sec: str, frac: str | None) -> float:
    """Секундная часть с необязательной десятичной дробью: 03.5 → 3.5 с."""
    if frac is None:
        return float(sec)
    return float(f"{int(sec)}.{frac}")


def parse_timecode_seconds(text: str) -> float:
    """
    Форматы:
    - ЧЧ:ММ:СС и ЧЧ:ММ:СС.… (после точки — десятичные доли секунды последнего поля);
    - ММ:СС, ММ:СС.…;
    - СС, СС.… (число секунд с десятичной частью).
    """
    s = text.strip()
    if not s:
        raise TimecodeError("Пустая строка времени.")

    if (m := _RE_HMS.match(s)) is not None:
        h, mi = int(m.group(1)), int(m.group(2))
        tail = _tail_as_seconds(m.group(3), m.group(4))
        return h * 3600 + mi * 60 + tail

    if (m := _RE_MS.match(s)) is not None:
        mi = int(m.group(1))
        tail = _tail_as_seconds(m.group(2), m.group(3))
        return mi * 60 + tail

    if (m := _RE_S.match(s)) is not None:
        return _tail_as_seconds(m.group(1), m.group(2))

    raise TimecodeError(f"Не удалось распознать время: {text!r}")


def seconds_to_display_span(start_sec: float, end_sec: float) -> str:
    """
    Компактная подпись интервала для stdout (например 00:10–00:15).
    Используется тире «–» (en dash), как в пользовательском формате.
    """
    if (
        start_sec < 0
        or end_sec < 0
        or not math.isfinite(start_sec)
        or not math.isfinite(end_sec)
    ):
        return "??:??–??:??"
    return f"{_seconds_to_display_clock(start_sec)}–{_seconds_to_display_clock(end_sec)}"


def _seconds_to_display_clock(total: float) -> str:
    """ММ:СС или ЧЧ:ММ:СС при длительности ≥ 1 ч; дробная часть секунды только если нужна."""
    ms_total = int(round(total * 1000))
    if ms_total < 0:
        ms_total = 0
    ss, frac_ms = divmod(ms_total, 1000)
    mm, s_int = divmod(ss, 60)
    hh, m_int = divmod(mm, 60)
    frac = "" if frac_ms == 0 else f".{frac_ms:03d}".rstrip("0").rstrip(".")
    if hh > 0:
        return f"{hh:02d}:{m_int:02d}:{s_int:02d}{frac}"
    return f"{m_int:02d}:{s_int:02d}{frac}"


def seconds_to_ffmpeg_tc(total: float) -> str:
    """Формат ЧЧ:ММ:СС.mmm для передачи в -ss/-to."""
    if total < 0 or not math.isfinite(total):
        raise TimecodeError("Время должно быть неотрицательным конечным числом секунд.")
    ms_total = int(round(total * 1000))
    ss, frac_ms = divmod(ms_total, 1000)
    mm, s_int = divmod(ss, 60)
    hh, m_int = divmod(mm, 60)
    return f"{hh:02d}:{m_int:02d}:{s_int:02d}.{frac_ms:03d}"


@dataclass(frozen=True)
class TimeRange:
    start_sec: float
    end_sec: float


def resolve_range(start_raw: str, end_raw: str) -> TimeRange:
    start = parse_timecode_seconds(start_raw)
    end = parse_timecode_seconds(end_raw)
    if end <= start:
        raise TimecodeError(f"Конец ({end_raw!r}) должен быть после начала ({start_raw!r}).")
    return TimeRange(start_sec=start, end_sec=end)


__all__ = [
    "TimeRange",
    "TimecodeError",
    "parse_timecode_seconds",
    "resolve_range",
    "seconds_to_display_span",
    "seconds_to_ffmpeg_tc",
]
