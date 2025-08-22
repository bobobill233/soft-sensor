import os
import pandas as pd
import numpy as np
from glob import glob

# 设置文件夹路径
folder_path = r"D:\flow_idea3\allAddclean\Addclean2-680-1.4" # 替换为您的文件夹路径
output_folder = r"D:\flow_idea3\allfilterdata\filterdata2-680-1.4"  # 保存路径

# 如果目标文件夹不存在，则创建该文件夹
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 遍历所有子文件夹中的CSV文件
csv_files = glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)


# 卡尔曼滤波实现
def kalman_filter(data):
    n = len(data)
    filtered_values = np.zeros(n)
    estimated_state = data[0]  # 初始估计值
    estimate_covariance = 1.0  # 初始协方差
    process_variance = 1e-5  # 小的过程噪声
    measurement_variance = 0.001 # 测量噪声

    for t in range(n):
        # 预测步骤
        predicted_state = estimated_state
        predicted_covariance = estimate_covariance + process_variance

        # 更新步骤（根据测量进行修正）
        kalman_gain = predicted_covariance / (predicted_covariance + measurement_variance)
        estimated_state = predicted_state + kalman_gain * (data[t] - predicted_state)
        estimate_covariance = (1 - kalman_gain) * predicted_covariance

        # 存储滤波后的值
        filtered_values[t] = estimated_state

    return filtered_values


# 遍历每个CSV文件，进行卡尔曼滤波处理
for csv_file in csv_files:
    # 加载CSV文件
    data = pd.read_csv(csv_file)

    # 提取第一行数据进行卡尔曼滤波处理
    values_first_row = data.iloc[0].values.astype(float)
    filtered_first_row = kalman_filter(values_first_row)

    # 提取四行数据（假设是索引为6到10的行）
    rows_to_add = data.iloc[6:11]

    # 创建新的DataFrame，第一行是卡尔曼滤波后的数据
    new_data = pd.DataFrame([filtered_first_row], columns=data.columns)

    # 将四行数据追加到新的DataFrame中（从第2行开始）
    new_data = pd.concat([new_data, rows_to_add], ignore_index=True)

    # 设置保存路径
    output_path = os.path.join(output_folder, 'filtered_' + os.path.basename(csv_file))

    # 保存新的CSV文件
    new_data.to_csv(output_path, index=False)

    print(f"已处理并保存：{output_path}")
