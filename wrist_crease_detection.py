"""以 MediaPipe 幾何定位 + 灰階 intensity profile 尋找掌側腕橫紋，不包含 PC7 辨識。

流程：MediaPipe landmark 幾何定位 -> wrist ROI -> 沿 hand_axis 的多條掃描線
grayscale intensity profile -> local dark valley candidate -> outlier 移除
-> curve fitting -> temporal smoothing。

這是幾何／灰階啟發式偵測，無法自行確認掌側或排除所有背景暗紋。
請將掌側朝向鏡頭，並傳入尚未畫上骨架與文字的 BGR 影像。
"""

from collections import deque
from collections.abc import Mapping
import math

import cv2
import numpy as np


# ROI / 搜尋範圍幾何（單位：palm_width = |L17 - L5|），沿用原本的腕橫紋搜尋帶。
AXIAL_SEARCH_MIN_RATIO = -0.15
AXIAL_SEARCH_MAX_RATIO = 0.65
ROI_HALF_WIDTH_RATIO = 0.65

# 橫向 scan line 取樣。
NUM_SCAN_LINES = 19  # 15~25 條，沿 transverse 分布
SCAN_LATERAL_RATIO = ROI_HALF_WIDTH_RATIO * 0.85  # 略窄於 ROI 邊界，避免貼邊取樣
AXIAL_SAMPLES_PER_UNIT = 40  # 每 1 個 palm_width 沿 hand_axis 的取樣點數

# Local dark valley 判斷（比前後鄰近區域暗，且有足夠對比）。
VALLEY_NEIGHBOR_RATIO = 0.05  # 左右鄰近比對區的寬度（palm_width 比例）
VALLEY_GAP_RATIO = 0.02       # valley 與鄰近比對區之間的留白，避免包含谷底本身
MIN_VALLEY_CONTRAST = 6.0     # 灰階差最少需達到多少才算「夠暗」
DARKNESS_FULL_SCALE = 25.0    # 灰階差達到此值即視為滿分對比

# Candidate 一致性與 curve fitting。
MIN_CANDIDATE_RATIO = 0.4     # 至少要有這個比例的 scan line 找到 valley
MIN_INLIER_COUNT = 6
MIN_INLIER_RATIO = 0.55       # outlier 移除後至少要留下的候選比例
OUTLIER_RESIDUAL_K = 3.0      # 以 MAD 為單位的離群門檻
MIN_LATERAL_SPAN_RATIO = 0.5  # inlier 需覆蓋的最小橫向範圍（相對掃描總寬度）
FIT_RMS_BAD = 0.15            # 殘差 RMS（palm_width 比例）達到此值視為信心 0

MIN_CONFIDENCE = 0.5

# Temporal smoothing：在「以 hand_axis 為基準的局部座標」中平滑，
# 因此手部平移／旋轉不會被誤判成 crease 跳動，只有形狀本身的抖動會被平滑。
SMOOTHING_HISTORY = 8             # 5~10 幀
SMOOTHING_ALPHA = 0.35            # EMA 權重（新結果佔比）
SMOOTHING_OUTLIER_RATIO = 0.12    # 與近期中位數的最大差異超過此值（palm_width）視為離群
SMOOTHING_FORCE_ACCEPT_STREAK = 3  # 連續被拒絕超過此次數，強制接受（避免真實變化卡死）

CURVE_GRID = np.linspace(-SCAN_LATERAL_RATIO, SCAN_LATERAL_RATIO, 21)


def _extract_geometry(frame, hand_landmarks):
    """回傳 (geometry, reason)；geometry 內含 wrist/index/pinky/palm_center/scale/axis/transverse。"""
    height, width = frame.shape[:2]
    try:
        if isinstance(hand_landmarks, Mapping):
            wrist = np.asarray(hand_landmarks[0], dtype=float)
            index = np.asarray(hand_landmarks[5], dtype=float)
            pinky = np.asarray(hand_landmarks[17], dtype=float)
        else:
            points = getattr(hand_landmarks, "landmark", hand_landmarks)
            wrist = np.array([points[0].x * width, points[0].y * height])
            index = np.array([points[5].x * width, points[5].y * height])
            pinky = np.array([points[17].x * width, points[17].y * height])
        if any(point.shape != (2,) for point in (wrist, index, pinky)):
            return None, "Invalid or missing landmarks"
        if not np.isfinite([wrist, index, pinky]).all():
            return None, "Invalid or missing landmarks"
    except (AttributeError, KeyError, IndexError, TypeError, ValueError):
        return None, "Invalid or missing landmarks"

    palm_center = (index + pinky) / 2
    scale = float(np.linalg.norm(pinky - index))  # palm_width: L5-L17
    axis_vector = palm_center - wrist
    axis_length = float(np.linalg.norm(axis_vector))
    if scale < 8 or axis_length < 8 or not (0 <= wrist[0] < width and 0 <= wrist[1] < height):
        return None, "Degenerate hand geometry"
    axis = axis_vector / axis_length
    transverse = np.array([-axis[1], axis[0]])  # perpendicular(hand_axis)，局部座標系統的橫向軸
    return {
        "wrist": wrist, "index": index, "pinky": pinky, "palm_center": palm_center,
        "scale": scale, "axis": axis, "transverse": transverse, "hand_axis": axis_vector,
    }, None


def _build_roi(geometry, width, height):
    """以 hand_axis/transverse 為基準建立 wrist ROI（axis-aligned bbox + 真正旋轉的多邊形）。"""
    wrist, axis, transverse, scale = (geometry[k] for k in ("wrist", "axis", "transverse", "scale"))
    half_width = ROI_HALF_WIDTH_RATIO * scale
    half_depth = (AXIAL_SEARCH_MAX_RATIO - AXIAL_SEARCH_MIN_RATIO) * scale / 2
    roi_center = wrist + axis * (AXIAL_SEARCH_MIN_RATIO + AXIAL_SEARCH_MAX_RATIO) * scale / 2
    extent = np.abs(transverse) * half_width + np.abs(axis) * half_depth
    x1, y1 = np.maximum(0, np.floor(roi_center - extent)).astype(int)
    x2, y2 = np.minimum([width, height], np.ceil(roi_center + extent) + 1).astype(int)
    polygon = np.rint([
        roi_center + a * half_width * transverse + b * half_depth * axis
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ]).astype(np.int32)
    return (int(x1), int(y1), int(x2), int(y2)), polygon


def _ratio_to_pixel(geometry, lateral_ratio, axial_ratio):
    wrist, axis, transverse, scale = (geometry[k] for k in ("wrist", "axis", "transverse", "scale"))
    return wrist + axial_ratio * scale * axis + lateral_ratio * scale * transverse


def _sample_profile(enhanced_f, origin, geometry, lateral_ratio, axial_ratios):
    """沿 hand_axis 方向、固定 lateral_ratio，取 ROI 局部座標的灰階 intensity profile。"""
    scale = geometry["scale"]
    world_points = (geometry["wrist"] + axial_ratios[:, None] * scale * geometry["axis"]
                    + lateral_ratio * scale * geometry["transverse"])
    local = world_points - origin
    x = local[:, 0].astype(np.float32).reshape(-1, 1)
    y = local[:, 1].astype(np.float32).reshape(-1, 1)
    h, w = enhanced_f.shape
    valid = ((x[:, 0] >= 0) & (x[:, 0] <= w - 1) & (y[:, 0] >= 0) & (y[:, 0] <= h - 1))
    values = cv2.remap(enhanced_f, x, y, cv2.INTER_LINEAR).reshape(-1)
    return values, valid, world_points


def _find_dark_valley(profile, valid, axial_ratios):
    """在單條 profile 中找出唯一一個主要的 local dark valley candidate（若有）。

    要求：比左右鄰近區域暗（有足夠 local contrast），且鄰近區域樣本足夠有效。
    多個符合條件的 local minima 時，取對比度最高者，確保每條 scan line 最多一點。
    """
    n = len(profile)
    if n < 5:
        return None
    step = (axial_ratios[-1] - axial_ratios[0]) / (n - 1)
    if step <= 0:
        return None
    neighbor_n = max(1, round(VALLEY_NEIGHBOR_RATIO / step))
    gap_n = max(0, round(VALLEY_GAP_RATIO / step))
    smoothed = np.convolve(profile, np.ones(3) / 3, mode="same")
    margin = gap_n + neighbor_n
    best = None
    for i in range(margin, n - margin):
        if not valid[i] or smoothed[i] > smoothed[i - 1] or smoothed[i] > smoothed[i + 1]:
            continue
        left = smoothed[i - margin:i - gap_n] if gap_n else smoothed[i - margin:i]
        right = smoothed[i + gap_n + 1:i + margin + 1] if gap_n else smoothed[i + 1:i + margin + 1]
        left_valid = valid[i - margin:i - gap_n] if gap_n else valid[i - margin:i]
        right_valid = valid[i + gap_n + 1:i + margin + 1] if gap_n else valid[i + 1:i + margin + 1]
        if left_valid.sum() == 0 or right_valid.sum() == 0:
            continue
        surrounding = (left[left_valid].mean() + right[right_valid].mean()) / 2
        contrast = float(surrounding - smoothed[i])
        if contrast < MIN_VALLEY_CONTRAST:
            continue
        if best is None or contrast > best["contrast"]:
            best = {"index": i, "contrast": contrast, "axial_ratio": float(axial_ratios[i])}
    return best


def _fit_curve(candidates):
    """對 candidate points 做 outlier 移除與 polynomial curve fitting。

    回傳 dict（含 inlier mask、係數、殘差 RMS）或 None（fit 失敗，如點數不足或共線）。
    """
    lateral = np.array([c["lateral_ratio"] for c in candidates])
    axial = np.array([c["axial_ratio"] for c in candidates])
    weights = np.array([c["darkness_score"] for c in candidates]) + 0.1
    try:
        degree = 2 if len(candidates) >= 8 else 1
        coeffs = np.polyfit(lateral, axial, degree, w=weights)
    except (np.linalg.LinAlgError, ValueError):
        return None
    residuals = axial - np.polyval(coeffs, lateral)
    mad = float(np.median(np.abs(residuals - np.median(residuals))))
    mad_std = mad * 1.4826 if mad > 1e-6 else float(np.std(residuals)) + 1e-6
    threshold = max(0.03, OUTLIER_RESIDUAL_K * mad_std)
    inlier_mask = np.abs(residuals) <= threshold
    if inlier_mask.sum() < 3:
        return {"coeffs": coeffs, "inlier_mask": inlier_mask, "rms": float("inf"), "lateral_span": 0.0}
    degree = 2 if inlier_mask.sum() >= 8 else 1
    try:
        coeffs2 = np.polyfit(lateral[inlier_mask], axial[inlier_mask], degree, w=weights[inlier_mask])
    except (np.linalg.LinAlgError, ValueError):
        return None
    residuals2 = axial[inlier_mask] - np.polyval(coeffs2, lateral[inlier_mask])
    rms = float(np.sqrt(np.mean(residuals2 ** 2)))
    lateral_span = float(lateral[inlier_mask].max() - lateral[inlier_mask].min())
    return {"coeffs": coeffs2, "inlier_mask": inlier_mask, "rms": rms, "lateral_span": lateral_span}


class _WristCreaseTracker:
    """保存單一手（track_id）最近幾幀的 curve，做 outlier 拒絕 + EMA 平滑。"""

    def __init__(self):
        self.history = deque(maxlen=SMOOTHING_HISTORY)
        self.ema = None
        self.reject_streak = 0

    def update(self, raw_axial_grid):
        """raw_axial_grid: 沿 CURVE_GRID 取樣的 axial_ratio 陣列（局部座標，與手部平移/旋轉/縮放無關）。

        回傳 (smoothed_axial_grid, accepted)。差異過大且非連續多次時，拒絕本幀、沿用前一次結果。
        """
        if self.history:
            reference = np.median(np.stack(self.history), axis=0)
            deviation = float(np.max(np.abs(raw_axial_grid - reference)))
            if deviation > SMOOTHING_OUTLIER_RATIO and self.reject_streak < SMOOTHING_FORCE_ACCEPT_STREAK:
                self.reject_streak += 1
                return self.ema, False
        self.reject_streak = 0
        self.history.append(raw_axial_grid)
        self.ema = raw_axial_grid if self.ema is None else (
            SMOOTHING_ALPHA * raw_axial_grid + (1 - SMOOTHING_ALPHA) * self.ema)
        return self.ema, True


_TRACKERS = {}


def _get_tracker(track_id):
    tracker = _TRACKERS.get(track_id)
    if tracker is None:
        tracker = _WristCreaseTracker()
        _TRACKERS[track_id] = tracker
    return tracker


def reset_wrist_crease_tracking(track_id=None):
    """清除 temporal smoothing 歷史；track_id 為 None 時清除全部（例如手離開畫面）。"""
    if track_id is None:
        _TRACKERS.clear()
    else:
        _TRACKERS.pop(track_id, None)


def detect_wrist_crease(frame, hand_landmarks, *, min_confidence=MIN_CONFIDENCE, track_id=0):
    """回傳腕橫紋偵測結果（座標皆為原圖像素）。

    frame：OpenCV BGR uint8 影像（不會被修改）。
    hand_landmarks：MediaPipe NormalizedLandmarkList、normalized landmarks
    序列，或既有的 {landmark_id: (u, v)} 像素座標字典。
    track_id：區分不同手（例如 hand_index），讓各自的 temporal smoothing 互不干擾。

    流程：MediaPipe 幾何定位 -> wrist ROI -> 多條 scan line 的 grayscale profile
    -> local dark valley candidate -> outlier 移除 -> curve fitting -> temporal smoothing。
    候選點太少或一致性不足時，不強制產生結果，line/curve 為 None。
    confidence（0~1）綜合 inlier 比例、valley 對比度與 fit 殘差，不是醫學或模型信心值。
    """
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be in [0, 1]")
    result = {
        "roi": None, "roi_polygon": None, "geometry": None, "line": None, "curve": None,
        "curve_raw": None, "score": None, "confidence": None, "scan_lines": [], "candidates": [],
        "hough_debug_lines": [], "min_confidence": min_confidence, "track_id": track_id,
        "smoothing": None, "reason": "Invalid frame",
    }
    if frame is None or frame.size == 0:
        return result
    height, width = frame.shape[:2]

    geometry, reason = _extract_geometry(frame, hand_landmarks)
    if geometry is None:
        result["reason"] = reason
        return result
    result["geometry"] = geometry

    roi, polygon = _build_roi(geometry, width, height)
    result["roi"], result["roi_polygon"] = roi, polygon
    x1, y1, x2, y2 = roi
    if x2 - x1 < 8 or y2 - y1 < 8:
        result["reason"] = "ROI too small"
        return result

    origin = np.array([x1, y1], dtype=float)
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    enhanced = cv2.GaussianBlur(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray), (3, 3), 0)
    enhanced_f = enhanced.astype(np.float32)

    scale = geometry["scale"]
    axial_len = (AXIAL_SEARCH_MAX_RATIO - AXIAL_SEARCH_MIN_RATIO)
    num_axial_samples = max(12, int(round(axial_len * AXIAL_SAMPLES_PER_UNIT)))
    axial_ratios = np.linspace(AXIAL_SEARCH_MIN_RATIO, AXIAL_SEARCH_MAX_RATIO, num_axial_samples)
    lateral_ratios = np.linspace(-SCAN_LATERAL_RATIO, SCAN_LATERAL_RATIO, NUM_SCAN_LINES)

    candidates = []
    for lateral_ratio in lateral_ratios:
        profile, valid, world_points = _sample_profile(enhanced_f, origin, geometry, lateral_ratio, axial_ratios)
        record = {
            "lateral_ratio": float(lateral_ratio),
            "start": tuple(np.rint(world_points[0]).astype(int)),
            "end": tuple(np.rint(world_points[-1]).astype(int)),
            "valley": None,
        }
        valley = _find_dark_valley(profile, valid, axial_ratios)
        if valley is not None:
            point = _ratio_to_pixel(geometry, lateral_ratio, valley["axial_ratio"])
            candidate = {
                "lateral_ratio": float(lateral_ratio), "axial_ratio": valley["axial_ratio"],
                "contrast": valley["contrast"],
                "darkness_score": float(np.clip(valley["contrast"] / DARKNESS_FULL_SCALE, 0, 1)),
                "point": tuple(np.rint(point).astype(int)), "inlier": False,
            }
            record["valley"] = candidate
            candidates.append(candidate)
        result["scan_lines"].append(record)
    result["candidates"] = candidates

    min_candidates = max(5, round(MIN_CANDIDATE_RATIO * NUM_SCAN_LINES))
    if len(candidates) < min_candidates:
        result["reason"] = "Too few valley candidates"
        return result

    fit = _fit_curve(candidates)
    if fit is None:
        result["reason"] = "Curve fit failed"
        return result
    inlier_mask = fit["inlier_mask"]
    for candidate, is_inlier in zip(candidates, inlier_mask):
        candidate["inlier"] = bool(is_inlier)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < MIN_INLIER_COUNT or inlier_count / len(candidates) < MIN_INLIER_RATIO:
        result["reason"] = "Low consistency after outlier removal"
        return result
    if fit["lateral_span"] < MIN_LATERAL_SPAN_RATIO * (2 * SCAN_LATERAL_RATIO):
        result["reason"] = "Insufficient lateral coverage"
        return result

    axial_grid = np.polyval(fit["coeffs"], CURVE_GRID)
    result["curve_raw"] = [tuple(np.rint(_ratio_to_pixel(geometry, lat, ax)).astype(int))
                           for lat, ax in zip(CURVE_GRID, axial_grid)]

    avg_darkness = float(np.mean([c["darkness_score"] for c, keep in zip(candidates, inlier_mask) if keep]))
    inlier_ratio_score = min(inlier_count / NUM_SCAN_LINES, 1.0)
    residual_score = float(np.clip(1 - fit["rms"] / FIT_RMS_BAD, 0, 1))
    confidence = float(np.clip(0.4 * inlier_ratio_score + 0.35 * avg_darkness + 0.25 * residual_score, 0, 1))
    result["confidence"] = result["score"] = confidence
    if confidence < min_confidence:
        result["reason"] = "Confidence below threshold"
        return result

    smoothed_grid, accepted = _get_tracker(track_id).update(axial_grid)
    result["smoothing"] = {"accepted": accepted, "reject_streak": _get_tracker(track_id).reject_streak}
    curve = [tuple(np.rint(_ratio_to_pixel(geometry, lat, ax)).astype(int))
             for lat, ax in zip(CURVE_GRID, smoothed_grid)]
    result["curve"] = result["line"] = curve
    result["reason"] = None
    return result


def _hough_debug_lines(frame, geometry, roi):
    """僅供 debug 參考／fallback 用途，不參與主要偵測決策。"""
    x1, y1, x2, y2 = roi
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    edges = cv2.Canny(cv2.GaussianBlur(enhanced, (5, 5), 0), 40, 100)
    scale = geometry["scale"]
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=max(10, round(scale * 0.12)),
        minLineLength=max(1, math.floor(scale * 0.30)), maxLineGap=max(3, round(scale * 0.08)),
    )
    if lines is None:
        return []
    return [((int(a) + x1, int(b) + y1), (int(c) + x1, int(d) + y1)) for a, b, c, d in lines.reshape(-1, 4)]


def _draw_geometry(frame, result, origin=(0, 0), factor=1.0):
    """主畫面與 debug 面板共用同一份幾何，所有位置均由原圖轉換。"""
    geometry = result.get("geometry")
    if geometry is None:
        return

    def pixel(point):
        return tuple(np.rint((np.asarray(point) - origin) * factor).astype(int))

    wrist, axis = geometry["wrist"], geometry["hand_axis"]
    cv2.arrowedLine(frame, pixel(wrist), pixel(wrist + axis), (255, 255, 0), 2, cv2.LINE_AA)
    cv2.circle(frame, pixel(wrist), 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.putText(frame, "L0", pixel(wrist), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "hand_axis", pixel(wrist + axis * 0.45), cv2.FONT_HERSHEY_SIMPLEX,
                .4, (255, 255, 0), 1, cv2.LINE_AA)


def draw_wrist_crease(frame, result, label_position=None, debug=True):
    """畫出 ROI、hand_axis、scan lines、candidate points 與最終（平滑後）crease curve。"""
    roi, curve = result["roi"], result["curve"]
    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)
    if debug:
        _draw_geometry(frame, result)
        if result.get("roi_polygon") is not None:
            cv2.polylines(frame, [result["roi_polygon"]], True, (0, 255, 0), 1)
        for record in result.get("scan_lines", []):
            cv2.line(frame, record["start"], record["end"], (120, 90, 0), 1, cv2.LINE_AA)
        for candidate in result.get("candidates", []):
            color = (0, 255, 255) if candidate["inlier"] else (0, 0, 150)
            cv2.circle(frame, candidate["point"], 3, color, -1, cv2.LINE_AA)
        if result.get("curve_raw"):
            cv2.polylines(frame, [np.array(result["curve_raw"])], False, (255, 128, 0), 1, cv2.LINE_AA)
    if curve is not None:
        cv2.polylines(frame, [np.array(curve)], False, (0, 0, 255), 3, cv2.LINE_AA)
    label = "Wrist Crease" if curve is not None else "Wrist Crease: Not Found"
    if debug and result.get("reason"):
        label += " | " + result["reason"]
    if debug and result.get("confidence") is not None:
        label += f" | confidence={result['confidence']:.2f}"
    position = label_position if label_position is not None else (10, frame.shape[0] - 15)
    text_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0][0]
    font_scale = 0.6 * min(1.0, max(1, frame.shape[1] - position[0] - 5) / max(1, text_width))
    for color, thickness in (((0, 0, 0), 3), ((0, 0, 255), 1)):
        cv2.putText(frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness, cv2.LINE_AA)
    return frame


def print_wrist_crease_debug(result, prefix=""):
    candidates = result["candidates"]
    inliers = sum(1 for c in candidates if c["inlier"])
    outliers = len(candidates) - inliers
    confidence = "n/a" if result["confidence"] is None else f"{result['confidence']:.3f}"
    smoothing = result.get("smoothing")
    smoothing_text = "n/a" if smoothing is None else (
        f"accepted={smoothing['accepted']} reject_streak={smoothing['reject_streak']}")
    print(f"{prefix} scan_lines={len(result['scan_lines'])} candidates={len(candidates)}"
          f" inliers={inliers} outliers={outliers} confidence={confidence}"
          f" smoothing=({smoothing_text}) | {result['reason'] or 'Selected'}"
          f" | threshold={result['min_confidence']:.3f}", flush=True)


def render_wrist_crease_debug(frame, result, page=0):
    """純繪圖、不開視窗；顯示 ROI/hand_axis/scan lines/candidates/curve 各面板，
    以及 candidate 對比度分數表（每頁 10 筆，page 可切換）。
    """
    canvas = np.zeros((730, 1080, 3), dtype=np.uint8)

    def text_at(text, position, color=(255, 255, 255)):
        cv2.putText(canvas, text, position, cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)

    roi = result["roi"]
    inliers = [c for c in result["candidates"] if c["inlier"]]
    outliers = [c for c in result["candidates"] if not c["inlier"]]
    panels = [
        ("scan_lines", [{"line": (r["start"], r["end"])} for r in result["scan_lines"]], (120, 90, 0)),
        ("valley candidates", [{"line": (c["point"], c["point"])} for c in result["candidates"]], (0, 255, 255)),
        ("inliers", [{"line": (c["point"], c["point"])} for c in inliers], (0, 255, 0)),
        ("rejected outliers", [{"line": (c["point"], c["point"])} for c in outliers], (0, 0, 150)),
        ("raw fitted curve", [{"line": pair} for pair in zip(
            (result["curve_raw"] or [])[:-1], (result["curve_raw"] or [])[1:])], (255, 128, 0)),
        ("smoothed curve", [{"line": pair} for pair in zip(
            (result["curve"] or [])[:-1], (result["curve"] or [])[1:])], (0, 0, 255)),
    ]
    for i, (name, records, color) in enumerate(panels):
        left, top = (i % 3) * 360, (i // 3) * 205
        text_at(f"{name}: {len(records)}", (left + 8, top + 20), color)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2].copy()
        factor = min(350 / crop.shape[1], 175 / crop.shape[0])
        crop = cv2.resize(crop, None, fx=factor, fy=factor)
        _draw_geometry(crop, result, origin=np.array([x1, y1]), factor=factor)
        for record in records:
            points = [tuple(np.rint((np.array(p) - [x1, y1]) * factor).astype(int)) for p in record["line"]]
            if points[0] == points[1]:
                cv2.circle(crop, points[0], 3, color, -1, cv2.LINE_AA)
            else:
                cv2.line(crop, *points, color, 2, cv2.LINE_AA)
        canvas[top + 26:top + 26 + crop.shape[0], left:left + crop.shape[1]] = crop

    confidence = "n/a" if result["confidence"] is None else f"{result['confidence']:.3f}"
    text_at(f"{result['reason'] or 'Wrist Crease selected'} | confidence={confidence}"
            f" | threshold={result['min_confidence']:.3f}", (8, 432))
    candidates = result["candidates"]
    pages = max(1, math.ceil(len(candidates) / 10))
    page %= pages
    text_at(f"Candidate scores: page {page + 1}/{pages} (press [ / ] to change)", (8, 455))
    text_at("lateral_ratio     axial_ratio     contrast     darkness_score     inlier", (8, 479))
    for row, candidate in enumerate(candidates[page * 10:(page + 1) * 10]):
        color = (0, 255, 0) if candidate["inlier"] else (0, 0, 150)
        for x, key in zip((8, 190, 370, 520, 720),
                          ("lateral_ratio", "axial_ratio", "contrast", "darkness_score", "inlier")):
            value = candidate[key]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            text_at(text, (x, 502 + row * 22), color)
    return canvas
