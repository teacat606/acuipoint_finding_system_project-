import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
import re

# 1. 讀取 data.csv

DATA_FILE = "data.csv"


if not os.path.exists(DATA_FILE):

    print("找不到 data.csv")

    print(
        "請先執行 receive_arduino.py"
    )

    raise SystemExit


df = pd.read_csv(DATA_FILE)


if df.empty:

    print("data.csv 沒有資料。")

    raise SystemExit


# 2. 取得受測者資訊

surname = str(
    df["surname"].iloc[0]
)

age = str(
    df["age"].iloc[0]
)

position = str(
    df["position"].iloc[0]
)

note = str(
    df["note"].iloc[0]
)


print("================================")
print("         Data Analysis")
print("================================")

print()

print(f"Surname  : {surname}")
print(f"Age      : {age}")
print(f"Position : {position}")
print(f"Note     : {note}")


# 3. 日期

today = datetime.now().strftime(
    "%Y_%m_%d"
)



# 4. Windows 安全檔名  最操蛋的東西


def clean_filename(text):

    text = str(text)

    return re.sub(
        r'[<>:"/\\|?*]',
        "_",
        text
    ).strip()


surname_safe = clean_filename(
    surname
)

age_safe = clean_filename(
    age
)

position_safe = clean_filename(
    position
)


# 5. results 資料夾

OUTPUT_FOLDER = "results"


os.makedirs(
    OUTPUT_FOLDER,
    exist_ok=True
)


# 6. 自動編號.  第二操蛋

counter = 1


while True:

    filename = (
        f"{today}_"
        f"{surname_safe}_"
        f"{age_safe}_"
        f"{position_safe}_"
        f"{counter:02d}"
    )


    csv_path = os.path.join(
        OUTPUT_FOLDER,
        filename + ".csv"
    )


    png_path = os.path.join(
        OUTPUT_FOLDER,
        filename + ".png"
    )


    if (
        not os.path.exists(csv_path)
        and
        not os.path.exists(png_path)
    ):

        break


    counter += 1


# 7. 儲存正式 CSV

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)


# 8. 顯示 12 個 Point


print()
print("Point Data")
print("--------------------------------")

print(
    df[
        [
            "point",
            "A0",
            "A1",
            "A2",
            "A3"
        ]
    ]
)


# 9. 畫 A0 ~ A3 四條折線圖

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    df["point"],
    df["A0"],
    marker="o",
    label="A0 (Up)"
)


plt.plot(
    df["point"],
    df["A1"],
    marker="o",
    label="A1 (Right)"
)


plt.plot(
    df["point"],
    df["A2"],
    marker="o",
    label="A2 (Down)"
)


plt.plot(
    df["point"],
    df["A3"],
    marker="o",
    label="A3 (Left)"
)


# 10. X 軸設定為 Point 1 ~ 12

plt.xticks(
    df["point"]
)


plt.xlabel(
    "Point"
)

plt.ylabel(
    "Average Voltage (V)"
)


plt.title(
    "Four-Quadrant Sensor Voltage\n"
    f"Subject: {surname} | "
    f"Age: {age} | "
    f"Position: {position}"
)


plt.legend()

plt.grid()

plt.tight_layout()


# ==========================================
# 11. 儲存圖表
# ==========================================

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)


# ==========================================
# 12. 顯示圖
# ==========================================

plt.show()


# ==========================================
# 13. 完成
# ==========================================

print()
print("================================")
print("       Analysis Completed")
print("================================")

print()

print(
    f"Measurement number: {counter:02d}"
)

print()

print("Saved files:")

print(csv_path)

print(png_path)