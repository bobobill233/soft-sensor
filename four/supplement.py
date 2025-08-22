import os
import pandas as pd

# 字典映射文件夹名到相应的数字
folder_mapping = {
    "filterdata2-300-1.4": (300, 1.4),
    "filterdata2-320-1.4": (320, 1.4),
    "filterdata2-340-1.4": (340, 1.4),
    "filterdata2-360-1.4": (360, 1.4),
    "filterdata2-380-1.4": (380, 1.4),
    "filterdata2-400-1.4": (400, 1.4),
    "filterdata2-440-1.4": (440, 1.4),
    "filterdata2-480-1.4": (480, 1.4),
    "filterdata2-520-1.4": (520, 1.4),
    "filterdata2-560-1.4": (560, 1.4),
    "filterdata2-600-1.4": (600, 1.4),
    "filterdata2-640-1.4": (640, 1.4),
    "filterdata2-680-1.4": (680, 1.4),
    # 可以继续添加更多文件夹和对应的数字
}

# 基于文件夹路径更新 CSV 文件
base_path = r"D:\flow_idea3\allfilterdata"

# 遍历文件夹和相应的数字
for folder_name, (num1, num2) in folder_mapping.items():
    folder_path = os.path.join(base_path, folder_name)

    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 遍历文件夹中的所有 CSV 文件
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".csv"):
                file_path = os.path.join(folder_path, file_name)

                # 读取 CSV 文件
                df = pd.read_csv(file_path)

                # 将相应的数字放到第二行和第三行的第一列
                df.iloc[1, 0] = num1
                df.iloc[2, 0] = num2

                # 保存更新后的文件
                df.to_csv(file_path, index=False)

        print(f"文件夹 {folder_name} 中的所有文件已更新")
    else:
        print(f"文件夹 {folder_name} 不存在")

print("所有文件处理完毕！")
