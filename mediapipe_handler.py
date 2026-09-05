"""使用 MediaPipe Hands 畫出手部 landmarks 與連接線。

安裝相容的 Hands API：python -m pip install mediapipe==0.10.21
"""

import cv2
import mediapipe as mp


class MediaPipeHandler:
    def __init__(self):
        if not hasattr(mp, "solutions"):
            raise RuntimeError(
                "此 MediaPipe 版本不提供 Hands API，"
                "請執行：python -m pip install mediapipe==0.10.21"
            )
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, frame):
        """接收 BGR 影像，在原圖畫上偵測結果並回傳。"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
        return frame

    def close(self):
        self.hands.close()
