import os
import pandas as pd

# 设置文件路径
folder_path = r'G:\PYCHARM\newflow\four\log_7g'  # 文件夹路径
output_folder = r'G:\PYCHARM\newflow\four\log_7g_csv'  # 保存CSV文件的路径

# 确保输出目录存在
os.makedirs(output_folder, exist_ok=True)

# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):  # 只处理 .txt 文件
        txt_file_path = os.path.join(folder_path, filename)

        # 提取文件名用于保存CSV文件
        base_name = filename.replace('.txt', '.csv')
        csv_file_path = os.path.join(output_folder, base_name)

        # 初始化空列表来存储数据
        data = []

        # 打开txt文件并解析每一行
        with open(txt_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split(', ')
                if len(parts) < 7:
                    print(f"Skipping malformed line: {line}")
                    continue  # 如果这一行的长度不符合预期，则跳过

                try:
                    # 提取各个字段的数据
                    epoch = parts[0].split(' ')[1]
                    train_loss = float(parts[1].split(': ')[1])
                    val_loss = float(parts[2].split(': ')[1])
                    train_mae = float(parts[3].split(': ')[1])
                    val_mae = float(parts[4].split(': ')[1])
                    train_acc = float(parts[5].split(': ')[1])
                    val_acc = float(parts[6].split(': ')[1].strip('.'))  # 去掉结尾的多余句号，但保留小数点

                    # 将数据存入列表
                    data.append([epoch, train_loss, val_loss, train_mae, val_mae, train_acc, val_acc])
                except (IndexError, ValueError) as e:
                    print(f"Error processing line: {line}, error: {e}")
                    continue  # 如果遇到错误，跳过这一行

        # 创建数据框
        columns = ['Epoch', 'Train Loss', 'Validation Loss', 'Train MAE', 'Validation MAE', 'Train Accuracy',
                   'Validation Accuracy']
        df = pd.DataFrame(data, columns=columns)

        # 将数据框保存为CSV文件
        df.to_csv(csv_file_path, index=False)

        print(f"数据已保存为CSV文件：{csv_file_path}")
