import os
import pandas as pd
import numpy as np
from glob import glob

# 输入输出文件夹路径
input_folder = r"D:\flow_idea3\allfilterdata\filterdata2-640-1.4"
output_folder_8192 = r"D:\flow_idea3\downsampling8192\downsampling8192-2-640-1.4"
output_folder_4096 = r"D:\flow_idea3\downsampling4096\downsampling4096-2-640-1.4"
output_folder_2048 = r"D:\flow_idea3\downsampling2048\downsampling2048-2-640-1.4"
output_folder_1024 = r"D:\flow_idea3\downsampling1024\downsampling1024-2-640-1.4"

# 确保输出文件夹存在
os.makedirs(output_folder_8192, exist_ok=True)
os.makedirs(output_folder_4096, exist_ok=True)
os.makedirs(output_folder_2048, exist_ok=True)
os.makedirs(output_folder_1024, exist_ok=True)

# 获取所有CSV文件路径
csv_files = glob(os.path.join(input_folder, '**', '*.csv'), recursive=True)

# 设置窗口大小为2，每两个数据点下采样
window_size = 2

# 处理每个文件
for csv_file in csv_files:
    # 读取CSV文件
    data_new = pd.read_csv(csv_file)

    # 提取第一行数据进行最大值下采样
    first_row_new = data_new.iloc[0]
    first_row_new1 = data_new.iloc[0].to_list()
    # 对第一行数据进行最大值下采样（4096个数据）
    max_sampled_first_row = [max(first_row_new[i:i+window_size]) for i in range(0, len(first_row_new), window_size)]
    max_sampled_first_row = max_sampled_first_row[:4096]  # 截取至4096个数据点

    # 对下采样后的4096个数据进行最大值下采样至2048个数据点
    max_sampled_first_row_2048 = [max(max_sampled_first_row[i:i+window_size]) for i in range(0, len(max_sampled_first_row), window_size)]
    max_sampled_first_row_2048 = max_sampled_first_row_2048[:2048]  # 截取至2048个数据点

    # 对下采样后的2048个数据进行最大值下采样至1024个数据点
    max_sampled_first_row_1024 = [max(max_sampled_first_row_2048[i:i+window_size]) for i in range(0, len(max_sampled_first_row_2048), window_size)]
    max_sampled_first_row_1024 = max_sampled_first_row_1024[:1024]  # 截取至1024个数据点

    # 提取第二行，第五行和第六行的数据（每行只取一个数据点）
    second_row_value = data_new.iloc[1].values[0]  # 取第二行的第一个数据点
    fifth_row_value = data_new.iloc[4].values[0]   # 取第五行的第一个数据点
    sixth_row_value = data_new.iloc[5].values[0]   # 取第六行的第一个数据点

    # 提取第四行的数据作为label
    label_value = data_new.iloc[3].values[0]  # 取第四行的第一个数据点作为label

    # 将这些数据插入到下采样数据的前3行
    final_data_8192 = [second_row_value, fifth_row_value, sixth_row_value] + first_row_new1 + [label_value]
    final_data_8192 = final_data_8192[:8196]

    # 第一种情况：将三个数据与4096个数据结合
    final_data_4096 = [second_row_value, fifth_row_value, sixth_row_value] + max_sampled_first_row + [label_value]
    final_data_4096 = final_data_4096[:4100]  # 确保总数据点数为4099（3行数据 + 4096个下采样数据点）

    # 第二种情况：将三个数据与2048个数据结合
    final_data_2048 = [second_row_value, fifth_row_value, sixth_row_value] + max_sampled_first_row_2048 + [label_value]
    final_data_2048 = final_data_2048[:2052]  # 确保总数据点数为2051（3行数据 + 2048个下采样数据点）

    # 第三种情况：将三个数据与1024个数据结合
    final_data_1024 = [second_row_value, fifth_row_value, sixth_row_value] + max_sampled_first_row_1024 + [label_value]
    final_data_1024 = final_data_1024[:1028]  # 确保总数据点数为1027（3行数据 + 1024个下采样数据点）

    # 创建最终的DataFrame并保存为CSV文件
    final_data_df_8192 = pd.DataFrame({'Final_Max_Sampled_8192': final_data_8192})
    final_data_df_4096 = pd.DataFrame({'Final_Max_Sampled_4096': final_data_4096})
    final_data_df_2048 = pd.DataFrame({'Final_Max_Sampled_2048': final_data_2048})
    final_data_df_1024 = pd.DataFrame({'Final_Max_Sampled_1024': final_data_1024})
    # final_data_df_1024 = pd.DataFrame({'Final_Max_Sampled_1024': final_data_1024, 'Label': [label_value] + [np.nan] * (len(final_data_1024)-1)})

    # 设置输出文件路径
    output_file_8192 = os.path.join(output_folder_8192, os.path.basename(csv_file))
    output_file_4096 = os.path.join(output_folder_4096, os.path.basename(csv_file))
    output_file_2048 = os.path.join(output_folder_2048, os.path.basename(csv_file))
    output_file_1024 = os.path.join(output_folder_1024, os.path.basename(csv_file))

    # 保存修改后的数据为新的CSV文件
    final_data_df_8192.to_csv(output_file_8192, index=False)
    final_data_df_4096.to_csv(output_file_4096, index=False)
    final_data_df_2048.to_csv(output_file_2048, index=False)
    final_data_df_1024.to_csv(output_file_1024, index=False)

    print(f"处理并保存文件：{output_file_8192} 和 {output_file_4096} 和 {output_file_2048} 和 {output_file_1024}")

print("所有文件处理完成。")
