import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import time


def main(file_path):
    print(f"========{file_path}========")
    df = pd.read_csv(file_path, sep=',', header=None)
    df.iloc[2] = pd.to_numeric(df.iloc[2], errors='coerce')  # 将数据点转换为数值类型
    data_points = df.iloc[2].dropna().to_list()

    # 计算连续数据点之间的差值
    differences = np.diff(data_points)
    # 定义阈值
    threshold = 5  # 这个值可以根据数据的具体情况进行调整
    # 找出差值大于阈值的点的索引
    spikes = np.where(np.abs(differences) > threshold)[0]
    # 由于差分操作减少了一个元素，我们需要考虑数据点的下一个点
    # 因此，我们同时移除 i 和 i+1
    spikes = np.concatenate((spikes, spikes + 1))

    df_cleaned = df.drop(spikes, axis=1)

    # 针对一开始的数据点，由于它必须得是从0开始的，但往往会突变到一个很高或者很低的值，然后从这个值开始逐渐增加，这样的话异常点就是0，于是我只需要将clean过后的点整体向x轴平移即可完成这个事情
    df_cleaned.iloc[2] = df_cleaned.iloc[2] - df_cleaned.iloc[2, 0]
    df_cleaned.iloc[2] = df_cleaned.iloc[2].map(lambda x: np.round(x, 2))
    cleaned_data_points = df_cleaned.iloc[2].dropna().tolist()  # 更新清洗后的数据点

    # 仅保留前8192列数据
    df_cleaned = df_cleaned.iloc[:, :8192]

    # # 设置画布和子图
    # plt.figure(figsize=(12, 4))
    # # 左边子图：原始数据
    # plt.subplot(1, 2, 1)
    # plt.plot(range(len(data_points)), data_points, marker='o', markersize=0.1, color='r')
    # plt.title('Original Data')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    # # 右边子图：清洗后的数据
    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(cleaned_data_points)), cleaned_data_points, marker='o', markersize=0.1, color='b')
    # plt.title(f'Cleaned Data (Threshold: {threshold})')
    # plt.xlabel('Index')
    # plt.ylabel('Value')
    #
    # # 显示图形
    # plt.tight_layout()
    # plt.show()
    #
    # # 清除前一个图形的输出
    # clear_output(wait=True)

    return df_cleaned


directory = r'D:\flow_idea2\flowdatanew'
output_dir = r'D:\flow_idea2\flowdata_cleanednew'
os.makedirs(output_dir, exist_ok=True)


def annomy_test(directory):
    csv_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                cleaned_df = main(file_path)
                # 创建与原始文件相同目录结构的路径
                relative_path = os.path.relpath(root, directory)
                output_path = os.path.join(output_dir, relative_path)
                # 如果目标目录不存在，则创建它
                os.makedirs(output_path, exist_ok=True)
                # 保存清洗后的 DataFrame
                cleaned_df.to_csv(os.path.join(output_path, file), index=False)
                # time.sleep(1)

    return csv_files

annomy_test(directory)
