import os
import pandas as pd
from pyts.image import GramianAngularField, MarkovTransitionField
import matplotlib
import matplotlib.pyplot as plt
from multiprocessing import Pool

# 设置全局字体为 Times New Roman 且字号为 14
matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 14

def process_file(args):
    file_path, gasf_output_folder, mtf_output_folder, gadf_output_folder = args

    # 读取CSV文件
    df = pd.read_csv(file_path, header=None)

    # 提取时间序列数据
    time_series = df.iloc[0].dropna().astype(float).values

    # 初始化Gasf、Mtf 和 Gadf 对象
    gasf = GramianAngularField(image_size=len(time_series), method='summation')
    mtf = MarkovTransitionField(image_size=len(time_series))
    gadf = GramianAngularField(image_size=len(time_series), method='difference')

    # 转换成 Gasf、Mtf、Gadf 图像
    gasf_image = gasf.fit_transform([time_series])
    mtf_image = mtf.fit_transform([time_series])
    gadf_image = gadf.fit_transform([time_series])

    # 构建输出文件路径
    base_file_name = os.path.splitext(os.path.basename(file_path))[0]
    gasf_file_path = os.path.join(gasf_output_folder, base_file_name + '.png')
    mtf_file_path = os.path.join(mtf_output_folder, base_file_name + '.png')
    gadf_file_path = os.path.join(gadf_output_folder, base_file_name + '.png')

    # 设置颜色映射
    cmap = 'viridis'

    # 保存 Gasf 图像
    plt.figure()
    plt.imshow(gasf_image[0], cmap=cmap)
    cbar = plt.colorbar()  # 添加颜色条
    cbar.ax.tick_params(labelsize=14)  # 确保颜色条刻度字号为14
    plt.savefig(gasf_file_path)
    plt.close()

    # 保存 Mtf 图像
    plt.figure()
    plt.imshow(mtf_image[0], cmap=cmap)
    cbar = plt.colorbar()  # 添加颜色条
    cbar.ax.tick_params(labelsize=14)  # 确保颜色条刻度字号为14
    plt.savefig(mtf_file_path)
    plt.close()

    # 保存 Gadf 图像
    plt.figure()
    plt.imshow(gadf_image[0], cmap=cmap)
    cbar = plt.colorbar()  # 添加颜色条
    cbar.ax.tick_params(labelsize=14)  # 确保颜色条刻度字号为14
    plt.savefig(gadf_file_path)
    plt.close()


def csv_to_images(csv_folder, gasf_output_folder, mtf_output_folder, gadf_output_folder):
    # 如果输出文件夹不存在，则创建
    os.makedirs(gasf_output_folder, exist_ok=True)
    os.makedirs(mtf_output_folder, exist_ok=True)
    os.makedirs(gadf_output_folder, exist_ok=True)

    # 获取CSV文件夹中的文件列表
    files = [os.path.join(csv_folder, file_name)
             for file_name in os.listdir(csv_folder) if file_name.endswith('.csv')]

    # 准备参数列表
    args = [(file, gasf_output_folder, mtf_output_folder, gadf_output_folder) for file in files]

    # 使用Pool进行并行处理
    with Pool(processes=10) as pool:
        pool.map(process_file, args)

    print("转换完成！")


if __name__ == '__main__':
    csv_folder = r'D:\flow_idea2\flowdata_cleaned_fft259'
    gasf_output_folder = r'D:\flow_idea2\picture\gasf259'
    mtf_output_folder = r'D:\flow_idea2\picture\mtf259'
    gadf_output_folder = r'D:\flow_idea2\picture\gadf259'

    csv_to_images(csv_folder, gasf_output_folder, mtf_output_folder, gadf_output_folder)
