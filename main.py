"""USB Camera 手部 landmarks 即時預覽：在影像視窗按 q 結束。"""

import sys

import cv2

from camera import Camera
from hand_direction import calculate_hand_direction


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
        frame_count = 0

        while True:
            frame = camera.read()
            frame = handler.process(frame)
            frame_count += 1
            for hand_index, joint_coordinates in enumerate(
                handler.joint_coordinates_per_hand
            ):
                direction = calculate_hand_direction(joint_coordinates)
                if direction is None:
                    continue

                cv2.arrowedLine(
                    frame, direction["wrist"], direction["middle_mcp"],
                    (0, 255, 255), 3, cv2.LINE_AA, tipLength=0.2,
                )
                ux, uy = direction["unit_vector"]
                labels = (
                    f"Hand angle: {direction['angle']:.1f} deg",
                    f"Direction: ({ux:.3f}, {uy:.3f})",
                )
                for line_index, label in enumerate(labels):
                    position = (10, 30 + hand_index * 65 + line_index * 25)
                    for color, thickness in (((0, 0, 0), 3), ((255, 255, 255), 1)):
                        cv2.putText(
                            frame, label, position, cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, color, thickness, cv2.LINE_AA,
                        )

                if frame_count % 30 == 0:
                    print(
                        f"Hand {hand_index + 1}\n"
                        f"Wrist L0: {direction['wrist']}\n"
                        f"Middle MCP L9: {direction['middle_mcp']}\n"
                        f"Direction vector: {direction['direction_vector']}\n"
                        f"Unit vector: {direction['unit_vector']}\n"
                        f"Hand angle: {direction['angle']:.1f} deg",
                        flush=True,
                    )
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
