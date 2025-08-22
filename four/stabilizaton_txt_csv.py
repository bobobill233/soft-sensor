import os
import re
import pandas as pd

# 设置文件夹路径
folder_path = r'G:\PYCHARM\newflow\four\test_log\1'  # txt 文件所在的文件夹
output_folder = r'G:\PYCHARM\newflow\four\test_log_csv'  # 保存 CSV 文件的文件夹

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有 txt 文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # 只处理 .txt 文件
        file_path = os.path.join(folder_path, filename)

        # 初始化存储数据的列表
        overall_test_mae = []
        test_accuracy = []

        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                # 提取Overall Test MAE后的数据
                if "Overall Test MAE:" in line:
                    mae_value = re.search(r'Overall Test MAE: ([0-9\.]+)', line)
                    if mae_value:
                        overall_test_mae.append(float(mae_value.group(1)))

                # 提取Test Accuracy后的数据
                if "Test Accuracy (within threshold of 5.0):" in line:
                    acc_value = re.search(r'Test Accuracy \(within threshold of 5.0\): ([0-9\.]+)', line)
                    if acc_value:
                        test_accuracy.append(float(acc_value.group(1)))

        # 将提取的数据转换为 DataFrame
        data = {
            'Overall Test MAE': overall_test_mae,
            'Test Accuracy (within threshold of 5.0)': test_accuracy
        }
        df = pd.DataFrame(data)

        # 保存为CSV文件，文件名与txt文件名一致
        csv_file_name = filename.replace('.txt', '.csv')  # 将 txt 替换为 csv
        csv_file_path = os.path.join(output_folder, csv_file_name)
        df.to_csv(csv_file_path, index=False)
        print(f"数据已保存为 CSV 文件: {csv_file_path}")
