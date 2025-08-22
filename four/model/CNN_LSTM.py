import torch
import torch.nn as nn

class EDF(nn.Module):
    def __init__(self, in_channels):
        super(EDF, self).__init__()
        self.reduction = nn.Linear(4 * in_channels, 2 * in_channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, H // 2, 2, W // 2, 2)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous().view(B, H // 2, W // 2, 4 * C)
        x = self.reduction(x)
        return x.permute(0, 3, 1, 2).contiguous()

class CNNLSTM1(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=1):
        super(CNNLSTM1, self).__init__()

        # Preprocessing layer (CNN)
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=64, kernel_size=1),  # 初始卷积层
            nn.BatchNorm2d(64),  # 批归一化
            nn.ReLU(),
            EDF(64),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 最大池化层
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 最大池化层
        )

        # LSTM 层
        self.lstm = nn.LSTM(input_size=512 * 8 * 8, hidden_size=1024, num_layers=3, batch_first=True, dropout=0.25)

        # 回归层
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),  # 从 1024 降维到 512
            nn.ReLU(),  # 激活函数
            nn.Dropout(0.5),  # Dropout
            nn.Linear(512, 256),  # 从 512 降维到 256
            nn.ReLU(),  # 激活函数
            nn.Linear(256, num_classes)  # 最终输出层
        )

    def forward(self, gasf_tensor, mtf_tensor):
        # 重新调整输入的形状为 [batch_size, channels, height, width]

        # 特征融合 (将两个特征图拼接)
        fused_features = torch.cat(( gasf_tensor, mtf_tensor), dim=1)

        # 预处理层 (CNN)
        x = self.preprocess(fused_features)

        # 应用GAM (全局注意力机制)
        # x = self.gam1(x)

        # 将x的形状调整为适合LSTM输入
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 512 * 8 * 8)

        # LSTM层
        lstm_out, _ = self.lstm(x)

        # 第二个GAM层
        lstm_out = lstm_out.view(batch_size, 1, -1, 1024).permute(0, 3, 1, 2)
        # lstm_out = self.gam2(lstm_out)
        lstm_out = lstm_out.reshape(batch_size, -1, 1024)

        # 取LSTM最后一个时间步的输出
        final_output = lstm_out[:, -1, :]

        # 回归层 (预测输出)
        out = self.regressor(final_output)

        return out
