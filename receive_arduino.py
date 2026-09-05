import serial
import csv
import time


# 基本設定

PORT = "COM4"

BAUD_RATE = 9600

NUM_POINTS = 12

START_COUNTDOWN = 5

POINT_COUNTDOWN = 3


# 輸入受測者資料

print("================================")
print("   Acupoint Measurement System")
print("================================")

print()

surname = input("請輸入姓氏：").strip()

age = input("請輸入年齡：").strip()

position = input("請輸入測量位置：").strip()

note = input("請輸入備註：").strip()


print()

print("================================")
print("受測者資料")
print("================================")

print(f"姓氏     ：{surname}")
print(f"年齡     ：{age}")
print(f"測量位置 ：{position}")
print(f"備註     ：{note}")

print("================================")



# 連接 Arduino

print()
print("正在連接 Arduino...")


ser = serial.Serial(
    PORT,
    BAUD_RATE,
    timeout=5
)

time.sleep(2)

ser.reset_input_buffer()

print("Arduino 已連接。")


# 整體開始前倒數5秒

print()
print("請將感測器放置於 Point 1。")
print()


for i in range(
    START_COUNTDOWN,
    0,
    -1
):

    print(
        f"測量將在 {i} 秒後開始..."
    )

    time.sleep(1)


# START

ser.write(
    b"START\n"
)


# ==========================================
# 等 Arduino READY
# ==========================================

while True:

    line = (
        ser.readline()
        .decode(
            "utf-8",
            errors="ignore"
        )
        .strip()
    )

    if line == "READY":
        break


print()
print("Arduino Ready!")
print()


# ==========================================
# 建立 data.csv
# ==========================================

with open(
    "data.csv",
    "w",
    newline="",
    encoding="utf-8-sig"
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "surname",
        "age",
        "position",
        "note",
        "point",
        "A0",
        "A1",
        "A2",
        "A3"
    ])


    successful_points = 0


    # ======================================
    # 12 個 Point
    # ======================================

    for point in range(
        1,
        NUM_POINTS + 1
    ):

        print()
        print("================================")
        print(f"Point {point}/{NUM_POINTS}")
        print("================================")


        if point > 1:

            print(
                "請將感測器移動 0.5 cm 到下一個 Point。"
            )

            print()


    
        # Point 測量前倒數
    

        for i in range(
            POINT_COUNTDOWN,
            0,
            -1
        ):

            print(
                f"{i}..."
            )

            time.sleep(1)


        print("開始測量！")


    
        # 告訴 Arduino 開始量這個 Point
    

        ser.write(
            b"MEASURE\n"
        )


        # Arduino 約 1 秒內量 10 次並平均
        line = (
            ser.readline()
            .decode(
                "utf-8",
                errors="ignore"
            )
            .strip()
        )


        data = line.split(",")


        if len(data) == 4:

            try:

                A0 = float(data[0])

                A1 = float(data[1])

                A2 = float(data[2])

                A3 = float(data[3])


                
                # 寫入 CSV
            

                writer.writerow([
                    surname,
                    age,
                    position,
                    note,
                    point,
                    A0,
                    A1,
                    A2,
                    A3
                ])


                f.flush()

                successful_points += 1


            
                # 顯示結果
            

                print()

                print(
                    f"Point {point:02d} 完成 | "
                    f"A0={A0:.4f} V | "
                    f"A1={A1:.4f} V | "
                    f"A2={A2:.4f} V | "
                    f"A3={A3:.4f} V"
                )


            except ValueError:

                print(
                    "資料轉換錯誤：",
                    line
                )


        else:

            print(
                "收到非預期資料：",
                line
            )


# STOP

ser.write(
    b"STOP\n"
)


while True:

    line = (
        ser.readline()
        .decode(
            "utf-8",
            errors="ignore"
        )
        .strip()
    )

    if line == "DONE":
        break


ser.close()


# 完成

print()
print("================================")
print("          測量完成")
print("================================")

print(
    f"成功取得 {successful_points}/{NUM_POINTS} 個 Point"
)

print()

print("資料已儲存至：")

print("data.csv")

print()

print(
    "接下來執行 analysis.py"
)