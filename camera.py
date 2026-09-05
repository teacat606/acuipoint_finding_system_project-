"""USB Camera 的開啟、影像讀取與釋放。"""

import cv2


class Camera:
    def __init__(self, index=0, width=1280, height=720):
        self.capture = cv2.VideoCapture(index)
        try:
            if not self.capture.isOpened():
                raise RuntimeError(
                    f"無法開啟相機（索引 {index}），請檢查 USB 連線及相機權限。"
                )
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        except Exception:
            self.release()
            raise

    def read(self):
        """取得一幀 BGR 影像；讀取失敗時拋出錯誤。"""
        success, frame = self.capture.read()
        if not success or frame is None:
            raise RuntimeError("無法讀取相機影像，請檢查相機是否斷線。")
        return frame

    def release(self):
        self.capture.release()
