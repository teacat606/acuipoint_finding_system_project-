"""USB Camera 手部 landmarks 即時預覽：在影像視窗按 q 結束。"""

import sys

import cv2

from camera import Camera


CAMERA_INDEX = 0  # 多個相機時，可以改成 1 或 2。
WINDOW_NAME = "MediaPipe Hands - Press q to quit"


def main():
    camera = None
    handler = None
    try:
        from mediapipe_handler import MediaPipeHandler

        camera = Camera(index=CAMERA_INDEX)
        handler = MediaPipeHandler()
        print("請選取影像視窗，按 q 結束。")

        while True:
            frame = camera.read()
            frame = handler.process(frame)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        return 0
    except ImportError as error:
        print(f"無法載入套件：{error}")
        print("請執行：python -m pip install mediapipe==0.10.21")
        return 1
    except KeyboardInterrupt:
        print("\n已停止預覽。")
        return 0
    except (RuntimeError, cv2.error) as error:
        print(f"錯誤：{error}")
        return 1
    finally:
        try:
            if camera is not None:
                camera.release()
        finally:
            try:
                if handler is not None:
                    handler.close()
            finally:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
