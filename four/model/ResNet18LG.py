import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# 定义 Global Attention Mechanism (GAM)
class GAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(GAM, self).__init__()
        # 通道注意力部分
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        # 空间注意力部分
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa = self.spatial_attention(x)
        x = x * sa
        return x


# 定义整体模型，使用 ResNet18 作为特征提取骨干网络
class ResNet18LSTMGAM(nn.Module):
    def __init__(self, num_classes=1):
        super(ResNet18LSTMGAM, self).__init__()
        # 使用 torchvision 提供的 ResNet18，并修改第一层卷积以接受 6 通道输入
        self.backbone = resnet18(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 去除全连接层和原始全局平均池化
        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.Identity()

        # 采用自适应池化将特征图尺寸调整为 (8,8)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # 第一个 GAM 层 (应用于 ResNet18 提取的特征)
        self.gam1 = GAM(in_channels=512)

        # LSTM 层：输入尺寸为 512*8*8，隐藏状态维度为 1024，3 层 LSTM，dropout=0.25
        self.lstm = nn.LSTM(input_size=512 * 8 * 8, hidden_size=1024, num_layers=3,
                            batch_first=True, dropout=0.25)

        # 第二个 GAM 层 (应用于 LSTM 输出)
        self.gam2 = GAM(in_channels=1024)

        # 回归层
        self.regressor = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, gasf_tensor, mtf_tensor):
        # 融合两个特征图 (假设 gasf_tensor 和 mtf_tensor 均为 [B, 3, H, W]，拼接后为 [B,6,H,W])
        fused_features = torch.cat((gasf_tensor, mtf_tensor), dim=1)

        # 使用 ResNet18 作为特征提取骨干
        x = self.backbone(fused_features)  # 输出形状约为 [B,512,H',W']
        x = self.adaptive_pool(x)  # 调整为 [B,512,8,8]

        # 应用第一个 GAM 层
        x = self.gam1(x)

        # 将 x 展平为适合 LSTM 输入的形状 [B, T, 512*8*8]
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 512 * 8 * 8)

        # LSTM 层
        lstm_out, _ = self.lstm(x)  # 输出形状 [B, T, 1024]

        # 将 LSTM 输出通过 GAM 层进行进一步特征增强
        # 首先调整形状为 [B,1,T,1024]，再进行 permute 得到 [B,1024,1,T]
        lstm_out = lstm_out.view(batch_size, 1, -1, 1024).permute(0, 3, 1, 2)
        lstm_out = self.gam2(lstm_out)
        lstm_out = lstm_out.reshape(batch_size, -1, 1024)

        # 取 LSTM 最后一个时间步的输出作为最终特征
        final_output = lstm_out[:, -1, :]

        # 回归层输出预测结果
        out = self.regressor(final_output)
        return out