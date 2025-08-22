# 导入必要的库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ===========================
# 1. 定义滑动平均函数
# ===========================
def moving_average(data, window_size):
    """对数据进行滑动平均平滑处理"""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


# ===========================
# 2. 定义主处理函数
# ===========================
def process_csv(file_path, output_folder, skipped_files):
    """处理单个 CSV 文件，添加转折点数据并保存更新后的文件"""

    # 读取 CSV 文件
    new_df = pd.read_csv(file_path)

    # 检查数据行数是否足够（至少有 1027 行才能进行分析）
    if len(new_df) < 1027:
        print(f"⚠️ 文件 {file_path} 数据行数不足 1027 行，跳过处理。")
        skipped_files.append(file_path)  # 记录跳过的文件
        return

    # 提取第 4 行到第 1027 行的数据（索引 3 到 1026）
    new_data_array = new_df.iloc[3:1027, 0].to_numpy()

    # ===========================
    # 3. 滑动窗口平滑数据
    # ===========================
    window_size_new = 40  # 滑动窗口大小
    smoothed_data_new = moving_average(new_data_array, window_size_new)

    # 计算一阶导数
    smoothed_gradient_new = np.gradient(smoothed_data_new)

    # ===========================
    # 4. 定义搜索范围
    # ===========================
    rise_range_start, rise_range_end = 50, 250  # 上升区到平稳区
    fall_range_start, fall_range_end = 500, 700  # 平稳区到下降区

    # ===========================
    # 5. 寻找转折点
    # ===========================
    # 上升到平稳区的转折点：梯度接近 0 的点
    rise_to_stable_range = smoothed_gradient_new[rise_range_start:rise_range_end]
    rise_to_stable_index_new = np.argmin(np.abs(rise_to_stable_range)) + rise_range_start

    # 平稳到下降区的转折点：梯度接近 0 的点
    stable_to_fall_range = smoothed_gradient_new[fall_range_start:fall_range_end]
    stable_to_fall_index_new = np.argmin(np.abs(stable_to_fall_range)) + fall_range_start

    # ===========================
    # 6. 提取转折点的值（来自原始数据）
    # ===========================
    rise_to_stable_value_new = round(new_data_array[rise_to_stable_index_new], 4)
    stable_to_fall_value_new = round(new_data_array[stable_to_fall_index_new], 4)

    print(f"✅ {os.path.basename(file_path)} 转折点检测完成：")
    print(f" - 上升区到平稳区的转折点索引：{rise_to_stable_index_new}, 对应值：{rise_to_stable_value_new}")
    print(f" - 平稳区到下降区的转折点索引：{stable_to_fall_index_new}, 对应值：{stable_to_fall_value_new}")

    # ===========================
    # 7. 创建新行并插入到原始数据
    # ===========================
    # 创建新行数据
    new_rows = pd.DataFrame({new_df.columns[0]: [rise_to_stable_value_new, stable_to_fall_value_new]})

    # 将原始数据分成前 3 行、插入新数据、再拼接剩余数据
    new_df_final = pd.concat([new_df.iloc[:3], new_rows, new_df.iloc[3:]]).reset_index(drop=True)

    # ===========================
    # 8. 创建输出路径并保存更新后的数据
    # ===========================
    # 构造目标输出路径，保持原始子文件夹结构
    output_file_path = os.path.join(output_folder, os.path.relpath(file_path, input_folder))
    output_file_path = output_file_path.replace('.csv', '_final.csv')

    # 确保目标文件夹存在
    output_dir = os.path.dirname(output_file_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 保存更新后的数据
    new_df_final.to_csv(output_file_path, index=False)
    print(f"✅ 文件已保存为：{output_file_path}")


# ===========================
# 9. 递归遍历目标文件夹并处理所有 CSV 文件
# ===========================
def process_all_csv_recursive(input_folder, output_folder_root):
    """递归遍历指定目录及其子文件夹下的所有 CSV 文件并进行处理"""

    # 存储跳过的文件列表
    skipped_files = []

    # 遍历主文件夹及其所有子文件夹
    for root, _, files in os.walk(input_folder):
        for file_name in files:
            if file_name.endswith('.csv'):
                # 构造完整的文件路径
                file_path = os.path.join(root, file_name)

                # 处理 CSV 文件
                process_csv(file_path, output_folder_root, skipped_files)

    # ===========================
    # 10. 输出跳过的文件列表
    # ===========================
    if skipped_files:
        print("\n⚠️ 以下文件由于数据行数不足 1027 行已被跳过：")
        for skipped_file in skipped_files:
            print(f" - {skipped_file}")
    else:
        print("\n✅ 没有跳过任何文件。")


# ===========================
# 11. 定义输入和输出文件夹路径
# ===========================
input_folder = r"D:\flow_idea3\downsampling1024"  # 输入文件夹路径
output_folder_root = r"D:\flow_idea3\downsampling_ipoint1024"  # 结果保存路径

# 运行批量处理
process_all_csv_recursive(input_folder, output_folder_root)
