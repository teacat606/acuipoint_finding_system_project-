import pandas as pd
import os
import glob



# 1. 找 results 裡最新的 CSV


RESULTS_FOLDER = "results"

csv_files = glob.glob(
    os.path.join(RESULTS_FOLDER, "*.csv")
)

if not csv_files:
    print("results 資料夾內沒有 CSV 檔案。")
    print("請先執行 analysis.py。")
    raise SystemExit


latest_file = max(
    csv_files,
    key=os.path.getmtime
)


print("================================")
print("    Information Analysis")
print("================================")

print()
print("分析檔案：")
print(latest_file)



# 2. 讀取資料


df = pd.read_csv(latest_file)


if df.empty:
    print("CSV 沒有資料。")
    raise SystemExit



# 3. 受測者資訊


surname = str(df["surname"].iloc[0])
age = str(df["age"].iloc[0])
position = str(df["position"].iloc[0])
note = str(df["note"].iloc[0])


print()
print("Subject Information")
print("--------------------------------")
print(f"Surname  : {surname}")
print(f"Age      : {age}")
print(f"Position : {position}")
print(f"Note     : {note}")
print("--------------------------------")



# 4. 感測器欄位


sensor_columns = [
    "A0",
    "A1",
    "A2",
    "A3"
]


# 5. 基本統計

print()
print("================================")
print("Basic Statistics")
print("================================")


for sensor in sensor_columns:

    data = df[sensor]

    mean = data.mean()

    median = data.median()

    std = data.std()

    q1 = data.quantile(0.25)

    q3 = data.quantile(0.75)

    iqr = q3 - q1


    # 25% ~ 75% 範圍
    middle_data = data[
        (data >= q1)
        &
        (data <= q3)
    ]


    middle_mean = middle_data.mean()


    # CV
    if mean != 0:

        cv = (
            std / mean
        ) * 100

    else:

        cv = 0


    print()

    print(f"{sensor}")
    print("--------------------------------")

    print(
        f"Mean           : "
        f"{mean:.4f} V"
    )

    print(
        f"Median         : "
        f"{median:.4f} V"
    )

    print(
        f"STD            : "
        f"{std:.4f} V"
    )

    print(
        f"Q1 (25%)       : "
        f"{q1:.4f} V"
    )

    print(
        f"Q3 (75%)       : "
        f"{q3:.4f} V"
    )

    print(
        f"IQR            : "
        f"{iqr:.4f} V"
    )

    print(
        f"25-75% Mean    : "
        f"{middle_mean:.4f} V"
    )

    print(
        f"CV             : "
        f"{cv:.2f}%"
    )


# ==========================================
# 6. Total
# ==========================================

df["Total"] = (
    df["A0"]
    + df["A1"]
    + df["A2"]
    + df["A3"]
)


print()
print("================================")
print("Total Signal")
print("================================")

print(
    f"Mean Total : "
    f"{df['Total'].mean():.4f} V"
)

print(
    f"STD Total  : "
    f"{df['Total'].std():.4f} V"
)


# ==========================================
# 7. X / Y
#
#       A0
#
# A3         A1
#
#       A2
#
# ==========================================

df["X"] = (
    df["A1"]
    - df["A3"]
) / df["Total"]


df["Y"] = (
    df["A0"]
    - df["A2"]
) / df["Total"]


print()
print("================================")
print("Position Analysis")
print("================================")


print(
    f"Mean X : "
    f"{df['X'].mean():.4f}"
)

print(
    f"Mean Y : "
    f"{df['Y'].mean():.4f}"
)


# ==========================================
# 8. 簡單穩定度判斷
# ==========================================

print()
print("================================")
print("Signal Stability")
print("================================")


for sensor in sensor_columns:

    mean = df[sensor].mean()

    std = df[sensor].std()


    if mean != 0:

        cv = (
            std / mean
        ) * 100

    else:

        cv = 0


    if cv < 5:

        status = "Very Stable"

    elif cv < 10:

        status = "Stable"

    elif cv < 20:

        status = "Moderate"

    else:

        status = "Unstable"


    print(
        f"{sensor}: "
        f"CV={cv:.2f}% "
        f"→ {status}"
    )


# ==========================================
# 9. 完成
# ==========================================

print()
print("================================")
print("Information Analysis Completed")
print("================================")