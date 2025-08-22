import os
import shutil


def move_csv_files(source_dirs, dest_dir):
    # 创建目标文件夹
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # 遍历所有源文件夹
    for source_dir in source_dirs:
        # 检查源文件夹是否存在
        if os.path.exists(source_dir):
            # 遍历源文件夹中的所有文件
            for root, _, files in os.walk(source_dir):
                for file in files:
                    if file.endswith('.csv'):
                        # 构建源文件路径和目标文件路径
                        source_file_path = os.path.join(root, file)
                        dest_file_path = os.path.join(dest_dir, file)

                        # 移动文件
                        shutil.move(source_file_path, dest_file_path)
                        print(f"Moved: {source_file_path} to {dest_file_path}")


# 指定源文件夹和目标文件夹
source_dirs = [r'D:\flow_idea2\water9-22']
dest_dir = r'D:\flow_idea2\flowdatanew'

# 调用函数移动文件
move_csv_files(source_dirs, dest_dir)
