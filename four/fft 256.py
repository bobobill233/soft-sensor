import os
import numpy as np
import pandas as pd
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt


# 步骤1: 加载数据
def load_data(file_path):
    data = pd.read_csv(file_path, header=None)
    return data


# 步骤2: 应用傅里叶变换进行降采样
def downsample_data(data, new_length):
    fft_values = fft(data)
    middle_index = new_length // 2
    reduced_fft_values = np.zeros(len(data), dtype=complex)
    reduced_fft_values[:middle_index] = fft_values[:middle_index]
    reduced_fft_values[-middle_index:] = fft_values[-middle_index:]
    ifft_values = ifft(reduced_fft_values)
    indices = np.linspace(0, len(data) - 1, new_length, dtype=int)
    downsampled_data = ifft_values[indices].real
    return downsampled_data


# 新增: 指数平滑滤波函数
def exponential_smoothing(data, alpha):
    """
    对数据进行指数平滑滤波。

    参数:
    data (numpy array): 输入的降采样后的数据
    alpha (float): 平滑系数，范围在0到1之间

    返回:
    smoothed_data (numpy array): 平滑后的数据
    """
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # 初始化第一项为原始数据的第一项

    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]

    return smoothed_data


# 修改: 绘制三个单独的折线图
def plot_separate_line_plots(data, filename="line_plots_image.png"):
    """
    绘制三个折线图，分别为红、绿、蓝三种颜色，并保存图像。

    参数:
    data (numpy array): 输入数据
    filename (str): 输出图像文件名
    """
    plt.figure(figsize=(5, 5))

    # 第一个折线图 - 红色
    plt.subplot(3, 1, 1)
    plt.plot(data, color='red')
    # plt.title('Red Line')
    plt.xticks([])  # 隐藏横轴刻度
    plt.yticks([])  # 隐藏纵轴刻度

    # 第二个折线图 - 绿色
    plt.subplot(3, 1, 2)
    plt.plot(data, color='green')
    # plt.title('Green Line')
    plt.xticks([])  # 隐藏横轴刻度
    plt.yticks([])  # 隐藏纵轴刻度

    # 第三个折线图 - 蓝色
    plt.subplot(3, 1, 3)
    plt.plot(data, color='blue')
    # plt.title('Blue Line')
    plt.xticks([])  # 隐藏横轴刻度
    plt.yticks([])  # 隐藏纵轴刻度

    plt.tight_layout()

    # 保存图像到指定路径
    output_file = os.path.join(r'D:\flow_idea2', filename)
    plt.savefig(output_file)
    plt.close()
    print(f"Line plots image saved to {output_file}")


# 步骤3: 保存数据到CSV文件
def save_data(data, output_file):
    pd.DataFrame(data).T.to_csv(output_file, index=False, header=False)


# 遍历目录下的所有CSV文件
def process_files(input_dir, output_dir):
    i = 0
    alpha = 0.3  # 指数平滑的系数
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                df = load_data(file_path)

                # 提取第二行数据并进行FFT降采样
                second_row = df.iloc[1]
                downsampled_data = downsample_data(second_row, 2048)

                # 应用指数平滑滤波
                smoothed_data = exponential_smoothing(downsampled_data, alpha)

                # 提取第四行的最大值
                df.iloc[3] = pd.to_numeric(df.iloc[3], errors='coerce')
                max_value = df.iloc[3].dropna().max()

                # 生成新的文件名
                new_filename = f"{i}-{float(max_value)}.csv"
                output_file = os.path.join(output_dir, new_filename)
                output_file1 = os.path.join(output_dir1, new_filename)

                # 保存降采样并平滑后的数据和新文件名
                save_data(downsampled_data, output_file)
                save_data(smoothed_data, output_file1)

                # 在 i == 642 时生成包含三个折线图的图像
                if i == 1:
                    plot_separate_line_plots(downsampled_data, filename=f"lines_plots_image_{i}.png")
                    plot_separate_line_plots(smoothed_data, filename=f"line_plots_image_{i}.png")

                i += 1

    print("All files processed. Results saved to", output_dir)


# 主程序流程
if __name__ == "__main__":
    # 定义输入和输出目录
    input_dir = r'D:\flow_idea2\sci_picture'
    output_dir = r'D:\flow_idea2\sci_picture\fft'
    output_dir1 = r'D:\flow_idea2\sci_picture\exp'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_dir1):
        os.makedirs(output_dir1)
    # 处理文件
    process_files(input_dir, output_dir)
