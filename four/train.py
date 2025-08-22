import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_selector import get_model  # 从模型选择模块导入
from train_utils import train_model  # 从训练模块导入
import dataloader  # 从数据加载模块导入


def parse_args():
    parser = argparse.ArgumentParser(description="Training script for multiple models")
    parser.add_argument('--model', type=str, required=True,
                        choices=['CNN_LSTM_GAM', 'Swin_CNN_LSTM_GAM', 'Swin_CNN', 'Swin_CNN_LSTM', 'SCLA_CEN',
                                 'Swin_CNN_LSTM_SelfAttention'],
                        help="Choose which model to train")
    parser.add_argument('--batch_size_train', type=int, default=4, help="Batch size for training")
    parser.add_argument('--batch_size_valid', type=int, default=8, help="Batch size for validation")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-5, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="Weight decay")
    parser.add_argument('--target', type=float, default=7, help="Target to predict")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/model.pth', help="Path to save the model")
    parser.add_argument('--log', type=str, default='train_sc.txt', help="Path to save the log")
    return parser.parse_args()

def main():
    args = parse_args()

    # Step 2: 动态选择模型
    model = get_model(args.model)  # 从模型选择模块获取模型

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Step 3: 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Step 4: 开始训练
    source_folder = r'D:\flow_idea2\tensor_mixpre_all1'
    train_csv = 'train_data.csv'
    val_csv = 'valid_data.csv'
    batch_size_train = args.batch_size_train
    batch_size_valid = args.batch_size_valid

    data_loader_wrapper = dataloader.DataLoaderWrapper(source_folder, train_csv, val_csv)

    train_loader, valid_loader = data_loader_wrapper.get_loaders(train_batch_size=batch_size_train,
                                                                 val_batch_size=batch_size_valid)
    scaler = data_loader_wrapper.get_scaler()
    train_model(model, train_loader, valid_loader, optimizer, criterion, num_epochs=args.epochs, device=device,
                scheduler=scheduler, save_path=args.checkpoint, scaler=scaler, target=args.target, log_file=args.log)


if __name__ == "__main__":
    main()