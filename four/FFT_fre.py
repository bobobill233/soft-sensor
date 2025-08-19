import os
import numpy as np
import pandas as pd
from scipy.fft import fft
import matplotlib.pyplot as plt


# 绘制频域图
def plot_frequency_spectrum(signal, sampling_rate, filename="frequency_spectrum.png"):
    """
    绘制信号的频域图。

    参数:
    signal (numpy array): 输入时域信号。
    sampling_rate (int): 信号的采样率（Hz）。
    filename (str): 输出图像的文件名。
    """
    # 计算 FFT
    fft_values = fft(signal)
    N = len(signal)
    freqs = np.fft.fftfreq(N, 1 / sampling_rate)  # 计算频率
    fft_magnitude = np.abs(fft_values)  # 计算 FFT 幅值

    # 只绘制正频率部分
    half_n = N // 2
    freqs = freqs[:half_n]
    fft_magnitude = fft_magnitude[:half_n]

    # 绘制频域图
    plt.figure(figsize=(8, 5))
    plt.plot(freqs, fft_magnitude, color='blue')
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()

    # 保存图像
    output_file = os.path.join(r'D:\flow_idea2', filename)
    plt.savefig(output_file)
    plt.close()
    print(f"Frequency spectrum image saved to {output_file}")


# 数据加载函数
def load_data(file_path):
    """
    从 CSV 文件加载数据。

    参数:
    file_path (str): 输入文件路径。
    返回:
    pandas.DataFrame: 加载的表格数据。
    """
    data = pd.read_csv(file_path, header=None)
    return data


# 主函数处理流程
def process_and_plot(input_file, sampling_rate, output_image):
    """
    加载数据、计算 FFT 并绘制频域图。

    参数:
    input_file (str): 输入 CSV 文件路径。
    sampling_rate (int): 信号的采样率（Hz）。
    output_image (str): 输出频域图的文件名。
    """
    # 加载数据
    df = load_data(input_file)

    # 提取第二行作为信号
    signal = df.iloc[1].dropna().to_numpy()

    # 绘制频域图
    plot_frequency_spectrum(signal, sampling_rate, filename=output_image)


# 主程序入口
if __name__ == "__main__":
    # 输入文件路径和采样率
    input_file = r"D:\flow_idea2\flowdata_cleanednew\2_720_2024_09_21_21_40_33.csv"  # 替换为你的文件路径
    sampling_rate = 1000  # 假设采样率为 1000 Hz
    output_image = r"D:\flow_idea2\flowdata_cleanednew\FFTfd.png"

    # 执行处理和绘图
    process_and_plot(input_file, sampling_rate, output_image)
