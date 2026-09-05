"""根據影像像素座標計算 L0（手腕）指向 L9（中指 MCP）的方向。"""

import math


def calculate_hand_direction(joint_coordinates):
    """回傳方向資料；缺少 L0/L9 或兩點重合時回傳 None。

    direction_vector 與 unit_vector 使用影像座標（向右、向下為正）。
    angle 使用數學方向：右 0°、上 90°、左 ±180°、下 -90°。
    """
    if 0 not in joint_coordinates or 9 not in joint_coordinates:
        return None

    wrist = joint_coordinates[0]
    middle_mcp = joint_coordinates[9]
    dx = middle_mcp[0] - wrist[0]
    dy = middle_mcp[1] - wrist[1]
    hand_length = math.hypot(dx, dy)
    if hand_length == 0:
        return None

    return {
        "wrist": wrist,
        "middle_mcp": middle_mcp,
        "direction_vector": (dx, dy),
        "unit_vector": (dx / hand_length, dy / hand_length),
        "length": hand_length,
        "angle": math.degrees(math.atan2(-dy, dx)),
    }
