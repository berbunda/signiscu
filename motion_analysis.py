"""Motion analysis: optical flow (scene residual) or MediaPipe Pose (human body)."""

from __future__ import annotations

import math
import time
import urllib.request
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np

from progress_ui import tqdm_labeled
from project_toml import MediapipePoseMetricsSettings

_MOTION_BACKENDS = frozenset({"optical_flow", "mediapipe_pose"})
_MEDIAPIPE_UNAVAILABLE_MSG = "MediaPipe backend unavailable. Install mediapipe package."

# MediaPipe Pose landmark indices (33-landmark model).
# https://developers.google.com/mediapipe/solutions/vision/pose_landmarker
_LM_NOSE = 0
_LM_LEFT_EYE_INNER = 1
_LM_LEFT_EYE = 2
_LM_LEFT_EYE_OUTER = 3
_LM_RIGHT_EYE_INNER = 4
_LM_RIGHT_EYE = 5
_LM_RIGHT_EYE_OUTER = 6
_LM_LEFT_EAR = 7
_LM_RIGHT_EAR = 8
_LM_MOUTH_LEFT = 9
_LM_MOUTH_RIGHT = 10
_LM_LEFT_SHOULDER = 11
_LM_RIGHT_SHOULDER = 12
_LM_LEFT_ELBOW = 13
_LM_RIGHT_ELBOW = 14
_LM_LEFT_WRIST = 15
_LM_RIGHT_WRIST = 16
_LM_LEFT_HIP = 23
_LM_RIGHT_HIP = 24
_LM_LEFT_KNEE = 25
_LM_RIGHT_KNEE = 26
_LM_LEFT_ANKLE = 27
_LM_RIGHT_ANKLE = 28
_LM_LEFT_HEEL = 29
_LM_RIGHT_HEEL = 30
_LM_LEFT_FOOT_INDEX = 31
_LM_RIGHT_FOOT_INDEX = 32

# Именованные части тела → индексы (только то, что есть в модели).
_LM_PART_SHOULDERS = (_LM_LEFT_SHOULDER, _LM_RIGHT_SHOULDER)
_LM_PART_ELBOWS = (_LM_LEFT_ELBOW, _LM_RIGHT_ELBOW)
_LM_PART_WRISTS = (_LM_LEFT_WRIST, _LM_RIGHT_WRIST)
_LM_PART_HIPS = (_LM_LEFT_HIP, _LM_RIGHT_HIP)
_LM_PART_KNEES = (_LM_LEFT_KNEE, _LM_RIGHT_KNEE)
_LM_PART_ANKLES = (_LM_LEFT_ANKLE, _LM_RIGHT_ANKLE)
_LM_PART_FEET = (_LM_LEFT_HEEL, _LM_RIGHT_HEEL, _LM_LEFT_FOOT_INDEX, _LM_RIGHT_FOOT_INDEX)
_LM_PART_LIPS = (_LM_MOUTH_LEFT, _LM_MOUTH_RIGHT)
_LM_PART_HEAD = (
    _LM_NOSE,
    _LM_LEFT_EYE_INNER,
    _LM_LEFT_EYE,
    _LM_LEFT_EYE_OUTER,
    _LM_RIGHT_EYE_INNER,
    _LM_RIGHT_EYE,
    _LM_RIGHT_EYE_OUTER,
    _LM_LEFT_EAR,
    _LM_RIGHT_EAR,
)


def _lm_indices_union(*parts: tuple[int, ...]) -> tuple[int, ...]:
    seen: set[int] = set()
    out: list[int] = []
    for part in parts:
        for i in part:
            if i not in seen:
                seen.add(i)
                out.append(i)
    return tuple(out)


# Landmark groups (MediaPipe Pose 33):
#   Upper body — shoulders, elbows, wrists
#   Lower body — hips, knees, ankles, feet (heel, foot index)
#   Torso — hips
#   Head — lips (mouth), head (nose, eyes, ears)
_LM_GROUP_UPPER = _lm_indices_union(_LM_PART_SHOULDERS, _LM_PART_ELBOWS, _LM_PART_WRISTS)
_LM_GROUP_LOWER = _lm_indices_union(
    _LM_PART_HIPS, _LM_PART_KNEES, _LM_PART_ANKLES, _LM_PART_FEET
)
_LM_GROUP_TORSO = _LM_PART_HIPS
_LM_GROUP_HEAD = _lm_indices_union(_LM_PART_LIPS, _LM_PART_HEAD)

_MIN_VISIBLE_LANDMARKS_POSE = 4
_TORSO_SCALE_MIN = 1e-6


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


@dataclass
class MediapipePoseSpanMetrics:
    pose_detection_ratio: float | None = None
    avg_visible_landmarks: float | None = None
    pose_visibility_score: float | None = None
    landmark_dropout_ratio: float | None = None
    pose_tracking_stability: float | None = None
    upper_body_motion_score: float | None = None
    lower_body_motion_score: float | None = None
    torso_motion_score: float | None = None
    head_motion_score: float | None = None
    pose_motion_direction_variance: float | None = None
    pose_motion_periodicity: float | None = None


@dataclass
class MotionBackendDebugStats:
    backend: str
    pose_model: str
    frames_analyzed: int
    poses_detected: int
    avg_visible_landmarks: float
    avg_pose_displacement: float
    elapsed_sec: float
    pose_detection_ratio: float = 0.0
    pose_visibility_score: float = 0.0
    landmark_dropout_count: int = 0
    torso_normalization_warnings: int = 0


def _unknown_motion_backend_message(backend: str) -> str:
    opts = ", ".join(sorted(_MOTION_BACKENDS))
    return f"Неизвестный motion backend {backend!r}; допустимо: {opts}."


def _resize_frame(frame: np.ndarray, resize_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > resize_width > 0:
        new_h = max(1, int(h * (resize_width / float(w))))
        return cv2.resize(frame, (resize_width, new_h), interpolation=cv2.INTER_AREA)
    return frame


def _prepare_gray(frame: np.ndarray, resize_width: int) -> np.ndarray:
    frame = _resize_frame(frame, resize_width)
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
    return float(np.percentile(mag, residual_percentile))


def _analyze_optical_flow(
    video_path: Path,
    spans: list[tuple[float, float]],
    *,
    sample_fps: float,
    resize_width: int,
    residual_percentile: float,
    progress_desc: str | None,
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
            out.append(MotionSpanRaw(segments=segs))
    finally:
        cap.release()
    return out


_POSE_MODEL_VARIANTS = frozenset({"lite", "full", "heavy"})
_POSE_MODEL_BASE_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
)


def _pose_model_assets(variant: str) -> tuple[str, str]:
    name = variant.strip().lower()
    if name not in _POSE_MODEL_VARIANTS:
        opts = ", ".join(sorted(_POSE_MODEL_VARIANTS))
        raise ValueError(
            f"Неизвестная MediaPipe Pose модель {variant!r}; допустимо: {opts}."
        )
    file_name = f"pose_landmarker_{name}.task"
    url = f"{_POSE_MODEL_BASE_URL}pose_landmarker_{name}/float16/1/{file_name}"
    return file_name, url


def _import_mediapipe():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
    except ImportError as e:
        raise RuntimeError(_MEDIAPIPE_UNAVAILABLE_MSG) from e
    return mp, mp_tasks, mp_vision


def _pose_model_cache_path(model_file_name: str) -> Path:
    cache_dir = Path.home() / ".cache" / "signiscu"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / model_file_name


def _ensure_pose_model(variant: str) -> Path:
    model_file_name, model_url = _pose_model_assets(variant)
    model_path = _pose_model_cache_path(model_file_name)
    if model_path.is_file():
        return model_path
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except OSError as e:
        raise RuntimeError(
            f"MediaPipe backend unavailable. Failed to download pose model "
            f"({variant!r}): {e}"
        ) from e
    return model_path


def _lm_visible(lm, visibility_threshold: float) -> bool:
    return float(lm.visibility) >= visibility_threshold


def _lm_xy(lm) -> tuple[float, float]:
    return float(lm.x), float(lm.y)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class _TorsoScaleResult:
    scale: float
    used_frame_fallback: bool


def _torso_scale(
    landmarks: list,
    visibility_threshold: float,
    *,
    frame_fallback_scale: float,
) -> _TorsoScaleResult:
    """Среднее: плечи, бёдра, центр плеч — центр бёдер; иначе fallback на размер кадра."""
    scales: list[float] = []
    if (
        _LM_LEFT_SHOULDER < len(landmarks)
        and _LM_RIGHT_SHOULDER < len(landmarks)
    ):
        a, b = landmarks[_LM_LEFT_SHOULDER], landmarks[_LM_RIGHT_SHOULDER]
        if _lm_visible(a, visibility_threshold) and _lm_visible(b, visibility_threshold):
            scales.append(_dist(_lm_xy(a), _lm_xy(b)))
    if _LM_LEFT_HIP < len(landmarks) and _LM_RIGHT_HIP < len(landmarks):
        a, b = landmarks[_LM_LEFT_HIP], landmarks[_LM_RIGHT_HIP]
        if _lm_visible(a, visibility_threshold) and _lm_visible(b, visibility_threshold):
            scales.append(_dist(_lm_xy(a), _lm_xy(b)))
    shoulder_pts: list[tuple[float, float]] = []
    hip_pts: list[tuple[float, float]] = []
    for idx in (_LM_LEFT_SHOULDER, _LM_RIGHT_SHOULDER):
        if idx < len(landmarks) and _lm_visible(landmarks[idx], visibility_threshold):
            shoulder_pts.append(_lm_xy(landmarks[idx]))
    for idx in (_LM_LEFT_HIP, _LM_RIGHT_HIP):
        if idx < len(landmarks) and _lm_visible(landmarks[idx], visibility_threshold):
            hip_pts.append(_lm_xy(landmarks[idx]))
    if shoulder_pts and hip_pts:
        sc = (
            sum(p[0] for p in shoulder_pts) / len(shoulder_pts),
            sum(p[1] for p in shoulder_pts) / len(shoulder_pts),
        )
        hc = (
            sum(p[0] for p in hip_pts) / len(hip_pts),
            sum(p[1] for p in hip_pts) / len(hip_pts),
        )
        scales.append(_dist(sc, hc))
    if scales:
        return _TorsoScaleResult(sum(scales) / len(scales), False)
    fb = max(_TORSO_SCALE_MIN, frame_fallback_scale)
    return _TorsoScaleResult(fb, True)


def _frame_fallback_scale(_frame_w: int, _frame_h: int) -> float:
    """Нормализованный масштаб кадра (landmarks в [0,1]); единичный охват по большей стороне."""
    return 1.0


def _mean_visibility(landmarks: list, visibility_threshold: float) -> float | None:
    vis = [float(lm.visibility) for lm in landmarks if _lm_visible(lm, visibility_threshold)]
    if not vis:
        return None
    return sum(vis) / len(vis)


def _group_pair_displacement(
    prev_landmarks: list,
    cur_landmarks: list,
    indices: tuple[int, ...],
    *,
    visibility_threshold: float,
    torso_scale: float,
) -> float | None:
    displacements: list[float] = []
    for i in indices:
        if i >= len(prev_landmarks) or i >= len(cur_landmarks):
            continue
        p, c = prev_landmarks[i], cur_landmarks[i]
        if _lm_visible(p, visibility_threshold) and _lm_visible(c, visibility_threshold):
            displacements.append(_dist(_lm_xy(p), _lm_xy(c)))
    if not displacements:
        return None
    scale = max(torso_scale, _TORSO_SCALE_MIN)
    return (sum(displacements) / len(displacements)) / scale


def _pair_tracking_jitter(
    prev_landmarks: list,
    cur_landmarks: list,
    *,
    visibility_threshold: float,
    torso_scale: float,
) -> float | None:
    mean_disp, n = _pair_pose_displacement(
        prev_landmarks, cur_landmarks, visibility_threshold=visibility_threshold
    )
    if n <= 0:
        return None
    scale = max(torso_scale, _TORSO_SCALE_MIN)
    return mean_disp / scale


def _collect_motion_angles(
    prev_landmarks: list,
    cur_landmarks: list,
    *,
    visibility_threshold: float,
) -> list[float]:
    angles: list[float] = []
    n = min(len(prev_landmarks), len(cur_landmarks))
    for i in range(n):
        p, c = prev_landmarks[i], cur_landmarks[i]
        if _lm_visible(p, visibility_threshold) and _lm_visible(c, visibility_threshold):
            px, py = _lm_xy(p)
            cx, cy = _lm_xy(c)
            dx, dy = cx - px, cy - py
            if abs(dx) > 1e-12 or abs(dy) > 1e-12:
                angles.append(math.atan2(dy, dx))
    return angles


def _direction_variance_score(angles: list[float]) -> float | None:
    if len(angles) < 2:
        return None
    c = sum(math.cos(a) for a in angles)
    s = sum(math.sin(a) for a in angles)
    r = math.hypot(c, s) / len(angles)
    return max(0.0, min(1.0, 1.0 - r))


def _motion_periodicity_score(magnitudes: list[float]) -> float | None:
    if len(magnitudes) < 4:
        return None
    xs = [float(x) for x in magnitudes]
    mean = sum(xs) / len(xs)
    centered = [x - mean for x in xs]
    var0 = sum(x * x for x in centered)
    if var0 <= 1e-15:
        return None
    max_corr = 0.0
    max_lag = min(len(centered) // 2, 12)
    for lag in range(1, max_lag + 1):
        num = sum(centered[i] * centered[i + lag] for i in range(len(centered) - lag))
        corr = num / var0
        if corr > max_corr:
            max_corr = corr
    if max_corr <= 0.0:
        return None
    return max(0.0, min(1.0, max_corr))


@dataclass
class _SpanPoseAccumulator:
    total_frames: int = 0
    detected_frames: int = 0
    visible_landmark_sum: float = 0.0
    visibility_sum: float = 0.0
    visibility_count: int = 0
    dropout_frames: int = 0
    tracking_jitters: list[float] = None  # type: ignore[assignment]
    upper_motions: list[float] = None  # type: ignore[assignment]
    lower_motions: list[float] = None  # type: ignore[assignment]
    torso_motions: list[float] = None  # type: ignore[assignment]
    head_motions: list[float] = None  # type: ignore[assignment]
    motion_angles: list[float] = None  # type: ignore[assignment]
    motion_magnitudes: list[float] = None  # type: ignore[assignment]
    torso_fallback_warnings: int = 0

    def __post_init__(self) -> None:
        if self.tracking_jitters is None:
            self.tracking_jitters = []
        if self.upper_motions is None:
            self.upper_motions = []
        if self.lower_motions is None:
            self.lower_motions = []
        if self.torso_motions is None:
            self.torso_motions = []
        if self.head_motions is None:
            self.head_motions = []
        if self.motion_angles is None:
            self.motion_angles = []
        if self.motion_magnitudes is None:
            self.motion_magnitudes = []


def _finalize_span_pose_metrics(acc: _SpanPoseAccumulator) -> MediapipePoseSpanMetrics:
    total = acc.total_frames
    if total <= 0:
        return MediapipePoseSpanMetrics()

    det_ratio = acc.detected_frames / total
    avg_vis_lm = acc.visible_landmark_sum / total
    vis_score = (
        acc.visibility_sum / acc.visibility_count if acc.visibility_count > 0 else None
    )
    dropout_ratio = acc.dropout_frames / total

    stability: float | None = None
    if acc.tracking_jitters:
        mean_jitter = sum(acc.tracking_jitters) / len(acc.tracking_jitters)
        stability = max(0.0, min(1.0, 1.0 - mean_jitter))

    upper: float | None = None
    if acc.upper_motions:
        upper = sum(acc.upper_motions) / len(acc.upper_motions)

    lower: float | None = None
    if acc.lower_motions:
        lower = sum(acc.lower_motions) / len(acc.lower_motions)

    torso: float | None = None
    if acc.torso_motions:
        torso = sum(acc.torso_motions) / len(acc.torso_motions)

    head: float | None = None
    if acc.head_motions:
        head = sum(acc.head_motions) / len(acc.head_motions)

    dir_var = _direction_variance_score(acc.motion_angles)
    periodicity = _motion_periodicity_score(acc.motion_magnitudes)

    return MediapipePoseSpanMetrics(
        pose_detection_ratio=det_ratio,
        avg_visible_landmarks=avg_vis_lm,
        pose_visibility_score=vis_score,
        landmark_dropout_ratio=dropout_ratio,
        pose_tracking_stability=stability,
        upper_body_motion_score=upper,
        lower_body_motion_score=lower,
        torso_motion_score=torso,
        head_motion_score=head,
        pose_motion_direction_variance=dir_var,
        pose_motion_periodicity=periodicity,
    )


_POSE_METRIC_FIELD_NAMES: tuple[str, ...] = tuple(
    f.name for f in fields(MediapipePoseSpanMetrics)
)


def build_mediapipe_pose_json_fields(
    metrics: MediapipePoseSpanMetrics | None,
    cfg: MediapipePoseMetricsSettings,
) -> dict[str, float | None]:
    if metrics is None or not cfg.enabled:
        return {}
    out: dict[str, float | None] = {}
    for name in _POSE_METRIC_FIELD_NAMES:
        if not getattr(cfg, name, False):
            continue
        val = getattr(metrics, name)
        if val is None:
            out[name] = None
        elif isinstance(val, float) and math.isfinite(val):
            out[name] = round(val, 4)
        else:
            out[name] = None
    return out


def aggregate_pose_span_metrics(
    items: list[tuple[MediapipePoseSpanMetrics, float]],
) -> MediapipePoseSpanMetrics | None:
    """Взвешенное среднее по длительности пересечения (weight > 0)."""
    if not items:
        return None
    w_sum = sum(w for _, w in items if w > 0)
    if w_sum <= 0:
        return None
    out = MediapipePoseSpanMetrics()
    for fname in _POSE_METRIC_FIELD_NAMES:
        num = 0.0
        den = 0.0
        for m, w in items:
            if w <= 0:
                continue
            v = getattr(m, fname)
            if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                continue
            num += float(v) * w
            den += w
        setattr(out, fname, (num / den) if den > 0 else None)
    return out


def _visible_landmark_count(landmarks: list, visibility_threshold: float) -> int:
    return sum(1 for lm in landmarks if _lm_visible(lm, visibility_threshold))


def _pair_pose_displacement(
    prev_landmarks: list,
    cur_landmarks: list,
    *,
    visibility_threshold: float,
) -> tuple[float, int]:
    n = min(len(prev_landmarks), len(cur_landmarks))
    displacements: list[float] = []
    for i in range(n):
        p, c = prev_landmarks[i], cur_landmarks[i]
        if _lm_visible(p, visibility_threshold) and _lm_visible(c, visibility_threshold):
            displacements.append(_dist(_lm_xy(p), _lm_xy(c)))
    if not displacements:
        return 0.0, 0
    return sum(displacements) / len(displacements), len(displacements)


def _analyze_mediapipe_pose(
    video_path: Path,
    spans: list[tuple[float, float]],
    *,
    sample_fps: float,
    resize_width: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
    visibility_threshold: float,
    pose_model: str,
    progress_desc: str | None,
    debug_stats: MotionBackendDebugStats | None,
    collect_pose_metrics: bool = False,
    pose_metrics_out: list[MediapipePoseSpanMetrics] | None = None,
) -> list[MotionSpanRaw]:
    mp, mp_tasks, mp_vision = _import_mediapipe()
    model_variant = pose_model.strip().lower()
    model_path = _ensure_pose_model(model_variant)
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть видео для motion: {video_path}")

    t0_run = time.perf_counter()
    frames_analyzed = 0
    poses_detected = 0
    visible_landmark_total = 0
    displacement_total = 0.0
    displacement_pairs = 0
    global_dropout = 0
    global_visibility_sum = 0.0
    global_visibility_count = 0
    global_torso_warnings = 0

    out: list[MotionSpanRaw] = []
    dt = 1.0 / sample_fps if sample_fps > 0 else 0.0

    try:
        span_iter = spans
        if progress_desc:
            span_iter = tqdm_labeled(spans, desc=progress_desc, unit="фрагмент", total=len(spans))
        for start_s, end_s in span_iter:
            if end_s <= start_s or dt <= 0:
                out.append(MotionSpanRaw(segments=[]))
                if pose_metrics_out is not None:
                    pose_metrics_out.append(MediapipePoseSpanMetrics())
                continue
            prev_pose: list | None = None
            segs: list[tuple[float, float, float]] = []
            k = 0
            frame_idx = 0
            have_prev_frame = False
            span_acc = _SpanPoseAccumulator() if collect_pose_metrics else None
            fh, fw = 0, 0
            landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            try:
                for frame in _iter_sampled_frames(cap, start_s, end_s, sample_fps):
                    frames_analyzed += 1
                    resized = _resize_frame(frame, resize_width)
                    fh, fw = resized.shape[:2]
                    frame_fb = _frame_fallback_scale(fw, fh)
                    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                    timestamp_ms = int((start_s + frame_idx * dt) * 1000.0)
                    frame_idx += 1
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)
                    cur_pose = result.pose_landmarks[0] if result.pose_landmarks else None

                    if span_acc is not None:
                        span_acc.total_frames += 1
                        vis_n = 0
                        if cur_pose is not None:
                            span_acc.detected_frames += 1
                            vis_n = _visible_landmark_count(cur_pose, visibility_threshold)
                            span_acc.visible_landmark_sum += float(vis_n)
                            mv = _mean_visibility(cur_pose, visibility_threshold)
                            if mv is not None:
                                span_acc.visibility_sum += mv
                                span_acc.visibility_count += 1
                                global_visibility_sum += mv
                                global_visibility_count += 1
                        if cur_pose is None or vis_n < _MIN_VISIBLE_LANDMARKS_POSE:
                            span_acc.dropout_frames += 1
                            global_dropout += 1

                    if cur_pose is not None:
                        poses_detected += 1
                        visible_landmark_total += _visible_landmark_count(
                            cur_pose, visibility_threshold
                        )

                    torso_res = (
                        _torso_scale(
                            cur_pose,
                            visibility_threshold,
                            frame_fallback_scale=frame_fb,
                        )
                        if cur_pose is not None
                        else _TorsoScaleResult(frame_fb, True)
                    )
                    if torso_res.used_frame_fallback and cur_pose is not None:
                        if span_acc is not None:
                            span_acc.torso_fallback_warnings += 1
                        global_torso_warnings += 1

                    if have_prev_frame:
                        raw_v = 0.0
                        if prev_pose is not None and cur_pose is not None:
                            mean_disp, _ = _pair_pose_displacement(
                                prev_pose,
                                cur_pose,
                                visibility_threshold=visibility_threshold,
                            )
                            scale = torso_res.scale
                            raw_v = mean_disp / max(scale, _TORSO_SCALE_MIN)
                            displacement_total += mean_disp
                            displacement_pairs += 1

                            if span_acc is not None:
                                prev_torso = _torso_scale(
                                    prev_pose,
                                    visibility_threshold,
                                    frame_fallback_scale=frame_fb,
                                )
                                ts = max(prev_torso.scale, torso_res.scale, _TORSO_SCALE_MIN)
                                jitter = _pair_tracking_jitter(
                                    prev_pose,
                                    cur_pose,
                                    visibility_threshold=visibility_threshold,
                                    torso_scale=ts,
                                )
                                if jitter is not None:
                                    span_acc.tracking_jitters.append(jitter)
                                span_acc.motion_magnitudes.append(raw_v)
                                span_acc.motion_angles.extend(
                                    _collect_motion_angles(
                                        prev_pose,
                                        cur_pose,
                                        visibility_threshold=visibility_threshold,
                                    )
                                )
                                for idxs, bucket in (
                                    (_LM_GROUP_UPPER, span_acc.upper_motions),
                                    (_LM_GROUP_LOWER, span_acc.lower_motions),
                                    (_LM_GROUP_TORSO, span_acc.torso_motions),
                                    (_LM_GROUP_HEAD, span_acc.head_motions),
                                ):
                                    gd = _group_pair_displacement(
                                        prev_pose,
                                        cur_pose,
                                        idxs,
                                        visibility_threshold=visibility_threshold,
                                        torso_scale=ts,
                                    )
                                    if gd is not None:
                                        bucket.append(gd)

                        seg_t0 = start_s + k * dt
                        seg_t1 = start_s + (k + 1) * dt
                        segs.append((seg_t0, seg_t1, raw_v))
                        k += 1
                    have_prev_frame = True
                    prev_pose = cur_pose
            finally:
                landmarker.close()
            out.append(MotionSpanRaw(segments=segs))
            if pose_metrics_out is not None and span_acc is not None:
                pose_metrics_out.append(_finalize_span_pose_metrics(span_acc))
    finally:
        cap.release()

    if debug_stats is not None:
        avg_visible = (
            visible_landmark_total / poses_detected if poses_detected > 0 else 0.0
        )
        avg_disp = (
            displacement_total / displacement_pairs if displacement_pairs > 0 else 0.0
        )
        debug_stats.backend = "mediapipe_pose"
        debug_stats.pose_model = model_variant
        debug_stats.frames_analyzed = frames_analyzed
        debug_stats.poses_detected = poses_detected
        debug_stats.avg_visible_landmarks = avg_visible
        debug_stats.avg_pose_displacement = avg_disp
        debug_stats.elapsed_sec = time.perf_counter() - t0_run
        debug_stats.pose_detection_ratio = (
            poses_detected / frames_analyzed if frames_analyzed > 0 else 0.0
        )
        debug_stats.pose_visibility_score = (
            global_visibility_sum / global_visibility_count
            if global_visibility_count > 0
            else 0.0
        )
        debug_stats.landmark_dropout_count = global_dropout
        debug_stats.torso_normalization_warnings = global_torso_warnings

    return out


def format_motion_backend_debug_lines(stats: MotionBackendDebugStats) -> list[str]:
    lines = [
        "--- motion backend (debug) ---",
        f"  MediaPipe backend enabled = {stats.backend == 'mediapipe_pose'}",
        f"  backend = {stats.backend}",
        f"  pose model = {stats.pose_model}",
        f"  pose frames analyzed = {stats.frames_analyzed}",
        f"  detected pose frames = {stats.poses_detected}",
        f"  pose_detection_ratio = {stats.pose_detection_ratio:.4f}",
        f"  avg_visible_landmarks = {stats.avg_visible_landmarks:.2f}",
        f"  pose_visibility_score = {stats.pose_visibility_score:.4f}",
        f"  dropout count = {stats.landmark_dropout_count}",
        f"  average pose displacement = {stats.avg_pose_displacement:.6f}",
        f"  elapsed time = {stats.elapsed_sec:.3f}s",
    ]
    if stats.torso_normalization_warnings > 0:
        lines.append(
            f"  warning: invalid torso normalization (frame fallback) "
            f"x{stats.torso_normalization_warnings}"
        )
    return lines


def analyze_motion_raw_for_spans(
    video_path: Path,
    spans: list[tuple[float, float]],
    *,
    backend: str = "optical_flow",
    sample_fps: float,
    resize_width: int,
    residual_percentile: float,
    mediapipe_min_detection_confidence: float = 0.5,
    mediapipe_min_tracking_confidence: float = 0.5,
    mediapipe_visibility_threshold: float = 0.5,
    mediapipe_pose_model: str = "lite",
    progress_desc: str | None = None,
    debug_stats: MotionBackendDebugStats | None = None,
    mediapipe_pose_metrics: MediapipePoseMetricsSettings | None = None,
    pose_metrics_out: list[MediapipePoseSpanMetrics] | None = None,
) -> list[MotionSpanRaw]:
    name = backend.strip().lower()
    if name not in _MOTION_BACKENDS:
        raise ValueError(_unknown_motion_backend_message(backend))

    if name == "optical_flow":
        return _analyze_optical_flow(
            video_path,
            spans,
            sample_fps=sample_fps,
            resize_width=resize_width,
            residual_percentile=residual_percentile,
            progress_desc=progress_desc,
        )

    collect_pose = (
        mediapipe_pose_metrics is not None
        and mediapipe_pose_metrics.enabled
        and pose_metrics_out is not None
    )
    return _analyze_mediapipe_pose(
        video_path,
        spans,
        sample_fps=sample_fps,
        resize_width=resize_width,
        min_detection_confidence=mediapipe_min_detection_confidence,
        min_tracking_confidence=mediapipe_min_tracking_confidence,
        visibility_threshold=mediapipe_visibility_threshold,
        pose_model=mediapipe_pose_model,
        progress_desc=progress_desc,
        debug_stats=debug_stats,
        collect_pose_metrics=collect_pose,
        pose_metrics_out=pose_metrics_out,
    )


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
    "MediapipePoseSpanMetrics",
    "MotionBackendDebugStats",
    "MotionSceneMetrics",
    "MotionSpanRaw",
    "aggregate_pose_span_metrics",
    "analyze_motion_raw_for_spans",
    "build_mediapipe_pose_json_fields",
    "format_motion_backend_debug_lines",
    "normalize_motion_metrics",
]
