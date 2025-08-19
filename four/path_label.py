import os
import pandas as pd


def save_image_paths_to_csv(gasf_folder, mtf_folder, csv_file):
    # 获取gasf文件夹中的图片路径列表
    gasf_paths = [os.path.join(gasf_folder, filename) for filename in os.listdir(gasf_folder) if
                  filename.endswith('.png')]
    # 获取mtf文件夹中的图片路径列表
    mtf_paths = [os.path.join(mtf_folder, filename) for filename in os.listdir(mtf_folder) if filename.endswith('.png')]

    # 提取gasf图片名中的数字作为标签
    labels = [os.path.splitext(filename)[0] for filename in os.listdir(gasf_folder) if filename.endswith('.png')]

    # 创建DataFrame
    df = pd.DataFrame({'gasf_path': gasf_paths, 'mtf_path': mtf_paths, 'label': labels})

    # 将DataFrame保存到CSV文件
    df.to_csv(csv_file, index=False)
    print("CSV文件保存完成！")


# 使用示例
gasf_folder = r'D:\flow_idea2\picture\gasfnew'
mtf_folder = r'D:\flow_idea2\picture\mtfnew'
csv_file = r'D:\flow_idea2\picture_pathsnew.csv'
save_image_paths_to_csv(gasf_folder, mtf_folder, csv_file)