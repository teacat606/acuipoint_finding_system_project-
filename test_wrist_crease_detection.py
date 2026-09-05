"""無相機的幾何、灰階 profile、curve fitting 與 temporal smoothing 回歸測試。"""
import unittest

import cv2
import numpy as np

import wrist_crease_detection as detector


def _draw_band(frame, geometry, axial_ratio, color=(95, 95, 95), half_thickness=3, lateral_scale=1.15):
    """畫一條垂直於 hand_axis（沿 transverse 方向）的深色band，模擬腕橫紋。"""
    wrist, axis, transverse, scale = (geometry[k] for k in ("wrist", "axis", "transverse", "scale"))
    center = wrist + axis * axial_ratio * scale
    half_len = detector.SCAN_LATERAL_RATIO * scale * lateral_scale
    corners = np.array([
        center + a * half_len * transverse + b * half_thickness * axis
        for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))
    ])
    cv2.fillPoly(frame, [np.rint(corners).astype(np.int32)], color)


class WristCreaseTests(unittest.TestCase):
    def setUp(self):
        detector.reset_wrist_crease_tracking()
        self.frame = np.full((300, 400, 3), 180, np.uint8)
        self.landmarks = {0: (200, 200), 5: (140, 100), 17: (260, 100)}
        self.geometry, _ = detector._extract_geometry(self.frame, self.landmarks)

    def tearDown(self):
        detector.reset_wrist_crease_tracking()

    def test_blank_and_invalid_landmarks(self):
        for landmarks in (self.landmarks, None, {}, {0: (0, 0), 5: (1, 1), 17: (1, 1)}):
            result = detector.detect_wrist_crease(self.frame, landmarks, track_id="invalid")
            self.assertIsNone(result["curve"])
            self.assertIsNotNone(result["reason"])

    def test_normal_dark_crease_detected_and_input_unchanged(self):
        _draw_band(self.frame, self.geometry, axial_ratio=0.15)
        original = self.frame.copy()
        result = detector.detect_wrist_crease(self.frame, self.landmarks, track_id="normal")
        self.assertIsNotNone(result["curve"])
        self.assertIsNone(result["reason"])
        self.assertGreaterEqual(result["confidence"], detector.MIN_CONFIDENCE)
        self.assertGreaterEqual(sum(c["inlier"] for c in result["candidates"]), detector.MIN_INLIER_COUNT)
        np.testing.assert_array_equal(original, self.frame)
        # debug 繪圖不應修改輸入影像。
        debug_frame = detector.render_wrist_crease_debug(self.frame, result)
        self.assertEqual(debug_frame.shape, (730, 1080, 3))
        np.testing.assert_array_equal(original, self.frame)
        detector.draw_wrist_crease(self.frame.copy(), result)

    def test_no_crease_returns_not_found(self):
        # 完全平坦的手腕影像：沒有任何 local dark valley。
        result = detector.detect_wrist_crease(self.frame, self.landmarks, track_id="blank")
        self.assertIsNone(result["curve"])
        self.assertEqual(result["reason"], "Too few valley candidates")

    def test_hand_rotation_still_detected(self):
        for rotation in (0, np.pi / 6, np.pi / 2, np.pi):
            matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                               [np.sin(rotation), np.cos(rotation)]])
            wrist = np.array([400., 400.])
            landmarks = {0: wrist, 5: wrist + matrix @ np.array([-50., -150.]),
                        17: wrist + matrix @ np.array([50., -50.])}
            frame = np.full((800, 800, 3), 180, np.uint8)
            geometry, reason = detector._extract_geometry(frame, landmarks)
            self.assertIsNone(reason)
            _draw_band(frame, geometry, axial_ratio=0.2)
            result = detector.detect_wrist_crease(frame, landmarks, track_id=f"rot-{rotation}")
            self.assertIsNotNone(result["curve"], f"rotation={rotation} should still detect crease")
            self.assertGreaterEqual(result["confidence"], detector.MIN_CONFIDENCE)

    def test_too_few_candidates_returns_not_found(self):
        # 只有一小塊暗區，寬度小於 scan line 間距，能碰到的 scan line 數不足門檻。
        cv2.rectangle(self.frame, (195, 180), (205, 190), (90, 90, 90), -1)
        result = detector.detect_wrist_crease(self.frame, self.landmarks, track_id="sparse")
        self.assertIsNone(result["curve"])
        self.assertEqual(result["reason"], "Too few valley candidates")
        min_candidates = max(5, round(detector.MIN_CANDIDATE_RATIO * detector.NUM_SCAN_LINES))
        self.assertLess(len(result["candidates"]), min_candidates)

    def test_sudden_single_frame_bad_candidate_does_not_jump(self):
        track_id = "jump"
        stable_frame = self.frame.copy()
        _draw_band(stable_frame, self.geometry, axial_ratio=0.15)
        for _ in range(4):
            result = detector.detect_wrist_crease(stable_frame, self.landmarks, track_id=track_id)
            self.assertIsNotNone(result["curve"])
            self.assertTrue(result["smoothing"]["accepted"])
        stable_curve = result["curve"]

        bad_frame = self.frame.copy()
        _draw_band(bad_frame, self.geometry, axial_ratio=0.55)
        bad_result = detector.detect_wrist_crease(bad_frame, self.landmarks, track_id=track_id)
        self.assertFalse(bad_result["smoothing"]["accepted"])
        self.assertEqual(bad_result["curve"], stable_curve)

        # 之後恢復原本位置，應維持穩定（不受單幀離群值影響）。
        recovered = detector.detect_wrist_crease(stable_frame, self.landmarks, track_id=track_id)
        self.assertTrue(recovered["smoothing"]["accepted"])

    def test_temporal_smoothing_tracker_blends_and_rejects_outliers(self):
        tracker = detector._WristCreaseTracker()
        grid_a = np.full(len(detector.CURVE_GRID), 0.2)
        smoothed_a, accepted_a = tracker.update(grid_a)
        self.assertTrue(accepted_a)
        np.testing.assert_array_almost_equal(smoothed_a, grid_a)

        grid_b = np.full_like(grid_a, 0.22)  # 小幅變化，應被接受並做 EMA 平滑。
        smoothed_b, accepted_b = tracker.update(grid_b)
        self.assertTrue(accepted_b)
        expected_b = detector.SMOOTHING_ALPHA * grid_b + (1 - detector.SMOOTHING_ALPHA) * grid_a
        np.testing.assert_array_almost_equal(smoothed_b, expected_b)
        self.assertFalse(np.allclose(smoothed_b, grid_b))

        grid_c = np.full_like(grid_a, 0.6)  # 大幅跳動，應視為離群並拒絕。
        smoothed_c, accepted_c = tracker.update(grid_c)
        self.assertFalse(accepted_c)
        np.testing.assert_array_almost_equal(smoothed_c, smoothed_b)

        last_accepted = False
        for _ in range(detector.SMOOTHING_FORCE_ACCEPT_STREAK):
            _, last_accepted = tracker.update(grid_c)
        self.assertTrue(last_accepted)  # 連續離群到一定次數後應強制接受真實變化。

    def test_find_dark_valley_requires_local_contrast(self):
        axial_ratios = np.linspace(-0.15, 0.65, 40)
        flat = np.full_like(axial_ratios, 150.0)
        valid = np.ones_like(axial_ratios, dtype=bool)
        self.assertIsNone(detector._find_dark_valley(flat, valid, axial_ratios))

        profile = flat.copy()
        profile[18:22] = 110.0  # 明顯的 local dark valley。
        valley = detector._find_dark_valley(profile, valid, axial_ratios)
        self.assertIsNotNone(valley)
        self.assertGreaterEqual(valley["contrast"], detector.MIN_VALLEY_CONTRAST)

        shallow = flat.copy()
        shallow[18:22] = 147.0  # 對比不足，不應視為有效 valley。
        self.assertIsNone(detector._find_dark_valley(shallow, valid, axial_ratios))


if __name__ == "__main__":
    unittest.main()
