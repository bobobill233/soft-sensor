import os
import pandas as pd
from pyts.image import GramianAngularField, MarkovTransitionField
import matplotlib.pyplot as plt
from multiprocessing import Pool


def process_file(args):
    file_path, gasf_output_folder, mtf_output_folder, gadf_output_folder = args

    # 读取CSV文件
    df = pd.read_csv(file_path, header=None)

    # 提取时间序列数据
    time_series = df.iloc[0].dropna().astype(float).values

    # 初始化Gasf和Mtf对象
    gasf = GramianAngularField(image_size=len(time_series), method='summation')
    mtf = MarkovTransitionField(image_size=len(time_series))
    gadf = GramianAngularField(image_size=len(time_series), method='difference')

    # 转换成Gasf、Mtf、Gadf图像
    gasf_image = gasf.fit_transform([time_series])
    mtf_image = mtf.fit_transform([time_series])
    gadf_image = gadf.fit_transform([time_series])

    # 构建输出文件路径
    base_file_name = os.path.splitext(os.path.basename(file_path))[0]
    gasf_file_path = os.path.join(gasf_output_folder, base_file_name + '.png')
    mtf_file_path = os.path.join(mtf_output_folder, base_file_name + '.png')
    gadf_file_path = os.path.join(gadf_output_folder, base_file_name + '.png')

    # 保存图像
    plt.imsave(gasf_file_path, gasf_image[0])
    plt.imsave(mtf_file_path, mtf_image[0])
    plt.imsave(gadf_file_path, gadf_image[0])


def csv_to_images(csv_folder, gasf_output_folder, mtf_output_folder, gadf_output_folder):
    # 如果输出文件夹不存在，则创建
    os.makedirs(gasf_output_folder, exist_ok=True)
    os.makedirs(mtf_output_folder, exist_ok=True)
    os.makedirs(gadf_output_folder, exist_ok=True)

    # 获取CSV文件夹中的文件列表
    files = [os.path.join(csv_folder, file_name) for file_name in os.listdir(csv_folder) if file_name.endswith('.csv')]

    # 准备参数列表
    args = [(file, gasf_output_folder, mtf_output_folder, gadf_output_folder) for file in files]

    # 使用Pool进行并行处理
    with Pool(processes=10) as pool:
        pool.map(process_file, args)

    print("转换完成！")

if __name__ == '__main__':
    csv_folder = r'D:\flow_idea2\flowdata_cleaned_fftnew'
    gasf_output_folder = r'D:\flow_idea2\picture\gasfnew'
    mtf_output_folder = r'D:\flow_idea2\picture\mtfnew'
    gadf_output_folder = r'D:\flow_idea2\picture\gadfnew'

    csv_to_images(csv_folder, gasf_output_folder, mtf_output_folder, gadf_output_folder)