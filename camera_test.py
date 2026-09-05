"""USB Camera 即時預覽：在影像視窗按 q 結束。"""

import sys

try:
    import cv2
except ImportError:
    print("錯誤：無法匯入 OpenCV，請執行：python3 -m pip install opencv-python")
    sys.exit(1)


# 多個相機時，可以嘗試把 CAMERA_INDEX 改成 1 或 2。
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
WINDOW_NAME = "USB Camera - Press q to quit"


def main():
    """開啟相機、逐幀顯示，並在結束時釋放資源。"""
    camera = cv2.VideoCapture(CAMERA_INDEX)

    try:
        if not camera.isOpened():
            print(
                f"錯誤：無法開啟相機（索引 {CAMERA_INDEX}）。"
                "請檢查 USB 連線、相機權限、是否被其他程式占用，"
                "或嘗試修改 CAMERA_INDEX。"
            )
            if sys.platform == "darwin":
                print(
                    "macOS：若出現相機存取提示，請選擇允許。"
                    "也可到「系統設定 → 隱私權與安全性 → 相機」，"
                    "允許啟動此程式的應用程式存取相機，再重新執行。"
                )
            return 1

        # 這是向相機要求的解析度；實際支援程度取決於相機與驅動程式。
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print("相機已開啟。請選取影像視窗，按 q 結束。")
        first_frame = True

        while True:
            success, frame = camera.read()
            if not success or frame is None:
                print("錯誤：無法讀取相機影像，請檢查相機是否斷線或被其他程式占用。")
                return 1

            if first_frame:
                height, width = frame.shape[:2]
                print(f"實際影像解析度：{width}×{height}")
                if (width, height) != (FRAME_WIDTH, FRAME_HEIGHT):
                    print("提醒：相機未採用 1280×720，將顯示實際取得的影像。")
                first_frame = False

            # 後續影像處理可以加在這裡，直接使用當前這一幀 frame。
            cv2.imshow(WINDOW_NAME, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        return 0
    except KeyboardInterrupt:
        print("\n已停止相機預覽。")
        return 0
    except cv2.error as error:
        print(f"錯誤：OpenCV 無法擷取或顯示影像：{error}")
        return 1
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
