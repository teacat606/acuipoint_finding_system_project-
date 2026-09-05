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
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS
                )
                # 每隻手各自使用原本的 0～20 ID；左上角為原點，u 向右、v 向下。
                joint_coordinates = {}
                for joint_id, landmark in enumerate(landmarks.landmark):
                    u = int(landmark.x * w)
                    v = int(landmark.y * h)
                    joint_coordinates[joint_id] = (u, v)

                for joint_id, (u, v) in joint_coordinates.items():
                    label = f"{joint_id}:({u},{v})"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (text_width, text_height), baseline = cv2.getTextSize(
                        label, font, 0.4, 1
                    )
                    # 只調整文字位置以避免超出畫面，不更改實際座標。
                    text_x = max(0, min(u + 6, w - text_width - 1))
                    text_y = max(text_height, min(v - 6, h - baseline - 1))
                    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                        cv2.putText(
                            frame, label, (text_x, text_y), font, 0.4,
                            color, thickness, cv2.LINE_AA
                        )

                print(f"Wrist L0: {joint_coordinates[0]}", flush=True)
        return frame

    def close(self):
        self.hands.close()
