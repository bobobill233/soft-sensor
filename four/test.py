import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_selector import get_model  # 从模型选择模块导入
from test_utils import test_model  # 从训练模块导入
import dataloader  # 从数据加载模块导入
import os
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for multiple models")
    parser.add_argument('--model', type=str, required=True,
                        choices=['CNN_LSTM_GAM', 'Swin_CNN_LSTM_GAM', 'Swin_CNN', 'Swin_CNN_LSTM', 'SCLA_CEN',
                                 'Swin_CNN_LSTM_SelfAttention'],
                        help="Choose which model to train")
    parser.add_argument('--batch_size_train', type=int, default=4, help="Batch size for training")
    parser.add_argument('--batch_size_valid', type=int, default=8, help="Batch size for validation")
    parser.add_argument('--target', type=float, default=7, help="Target to predict")
    parser.add_argument('--log', type=str, default='train_sc.txt', help="Path to save the log")
    parser.add_argument('--best_path', type=str, default='path', help="Path to save the best model")
    return parser.parse_args()


def generate_naive_forecast(data_loader):
    """
    生成 naive 预测，即假设下一时刻的预测值与上一时刻相同
    """
    all_labels = []
    for _, _, labels in data_loader:
        all_labels.extend(labels.numpy())  # 将标签添加到列表中

    # 使用前一个时间点的值作为 naive_forecast
    naive_forecast = np.roll(np.array(all_labels), 1)  # 向前滚动一位
    naive_forecast[0] = all_labels[0]  # 第一个预测值设置为第一个标签的值，避免不合法预测
    return naive_forecast


def main():
    args = parse_args()

    # Step 2: 动态选择模型
    model = get_model(args.model)  # 从模型选择模块获取模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Step 4: 开始训练
    source_folder = r'D:\flow_idea2\tensor_mixpre_all1'
    train_csv = 'train_data.csv'
    val_csv = 'valid_data.csv'
    csv_file = "txt/test_results.csv"
    batch_size_train = args.batch_size_train
    batch_size_valid = args.batch_size_valid

    data_loader_wrapper = dataloader.DataLoaderWrapper(source_folder, train_csv, val_csv)

    train_loader, valid_loader = data_loader_wrapper.get_loaders(train_batch_size=batch_size_train,
                                                                 val_batch_size=batch_size_valid)
    scaler = data_loader_wrapper.get_scaler()

    # 生成 naive_forecast
    naive_forecast = generate_naive_forecast(valid_loader)

    # 清空文件（如果之前存在）
    open(csv_file, 'w').close()

    # 打开日志文件，使用 'a' 模式（追加模式）
    with open(args.log, 'a') as log_file:
        # 循环运行 test_model 100 次，每次将日志写入同一个文件
        for I in range(0, 100):
            # log_file.write(f"\n=== Iteration {i}/100 ===\n")  # 写入当前迭代的标识
            # print(f"Running test_model iteration {i}, logging to {args.log}")

            # 将 log_file_path 传递给 test_model 以记录每次的结果
            test_model(model, test_loader=valid_loader, best_path=args.best_path, device=device,
                       scaler=scaler, target=args.target, log_file=args.log, I=I, naive_forecast=naive_forecast, csv_file=csv_file)


if __name__ == "__main__":
    main()

# 多日志
# import argparse
# import torch
# import torch.optim as optim
# import torch.nn as nn
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from model_selector import get_model  # 从模型选择模块导入
# from test_utils import test_model  # 从训练模块导入
# import dataloader  # 从数据加载模块导入
# import os
#
#
# def parse_args():
#     parser = argparse.ArgumentParser(description="Training script for multiple models")
#     parser.add_argument('--model', type=str, required=True,
#                         choices=['CNN_LSTM_GAM', 'shijie', 'Swin_CNN', 'Swin_CNN_LSTM', 'Swin_CNN_LSTM_GAM',
#                                  'Swin_CNN_LSTM_SelfAttention'],
#                         help="Choose which model to train")
#     parser.add_argument('--batch_size_train', type=int, default=4, help="Batch size for training")
#     parser.add_argument('--batch_size_valid', type=int, default=8, help="Batch size for validation")
#     parser.add_argument('--target', type=float, default=7, help="Target to predict")
#     parser.add_argument('--log', type=str, default='train_sc.txt', help="Base log file name")
#     parser.add_argument('--best_path', type=str, default='path', help="Path to save the best model")
#     return parser.parse_args()
#
#
# def main():
#     args = parse_args()
#
#     # Step 2: 动态选择模型
#     model = get_model(args.model)  # 从模型选择模块获取模型
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#
#     # Step 3: 定义损失函数和优化器 (可根据需要启用)
#     # criterion = nn.MSELoss()
#     # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
#     # scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
#
#     # Step 4: 开始训练
#     source_folder = r'D:\flow_idea2\tensor_mixpre_all1'
#     train_csv = 'train_data.csv'
#     val_csv = 'valid_data.csv'
#     batch_size_train = args.batch_size_train
#     batch_size_valid = args.batch_size_valid
#
#     data_loader_wrapper = dataloader.DataLoaderWrapper(source_folder, train_csv, val_csv)
#
#     train_loader, valid_loader = data_loader_wrapper.get_loaders(train_batch_size=batch_size_train,
#                                                                  val_batch_size=batch_size_valid)
#     scaler = data_loader_wrapper.get_scaler()
#
#     # 循环运行 test_model 100 次，每次保存不同的日志文件
#     for i in range(1, 101):
#         log_file_name = f"train_sc_{i}.txt"  # 动态生成日志文件名
#         log_file_path = os.path.join(os.path.dirname(args.log), log_file_name)  # 确保文件路径正确
#         print(f"Running test_model iteration {i}, logging to {log_file_path}")
#
#         test_model(model, test_loader=valid_loader, best_path=args.best_path, device=device,
#                    scaler=scaler, target=args.target, log_file=log_file_path)
#
#
# if __name__ == "__main__":
#     main()
