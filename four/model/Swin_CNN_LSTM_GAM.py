import torch
import torch.nn as nn
import timm

class SwinExtractor(nn.Module):
    def __init__(self, model_name='swinv2_base_window8_256.ms_in1k', pretrained=True):
        super(SwinExtractor, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

    def forward(self, x):
        return self.model(x)


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


class GAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(GAM, self).__init__()

        # 通道注意力部分
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
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
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca  # 按通道权重加权输入特征图

        # 空间注意力
        sa = self.spatial_attention(x)
        x = x * sa  # 按空间权重加权输入特征图

        return x


class SwinCNNLSTMGAM(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=1):
        super(SwinCNNLSTMGAM, self).__init__()

        # 初始化 SwinExtractor
        self.swin_extractor = SwinExtractor()

        # Preprocessing layer
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),  # Add BN layer
            nn.ReLU(),
            EDF(128),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),  # Add BN layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Add GAM
        self.gam1 = GAM(in_channels=512)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=512 * 8 * 8, hidden_size=1024, num_layers=3, batch_first=True, dropout=0.25)

        self.gam2 = GAM(in_channels=1024)

        # Regressor layer
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),  # Add Dropout layer before the final regressor
            nn.Linear(1024, num_classes)  # Final linear layer reducing from 1024 to num_classes
        )

    def forward(self, gasf_tensor, mtf_tensor):
        # 使用 SwinExtractor 对输入进行特征提取
        gasf_features = self.swin_extractor(gasf_tensor)
        mtf_features = self.swin_extractor(mtf_tensor)

        # 重新调整特征维度
        batch_size, num_features = gasf_features.shape
        gasf_features = gasf_features.view(batch_size, num_features // (32 * 32), 32, 32)
        mtf_features = mtf_features.view(batch_size, num_features // (32 * 32), 32, 32)

        # 特征融合
        fused_features = torch.cat((gasf_features, mtf_features), dim=1)

        # 预处理
        x = self.preprocess(fused_features)

        # 应用 GAM
        x = self.gam1(x)

        # 调整为 LSTM 输入
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 512 * 8 * 8)

        # LSTM 层
        lstm_out, _ = self.lstm(x)

        # 应用第二个 GAM
        lstm_out = lstm_out.view(batch_size, 1, -1, 1024).permute(0, 3, 1, 2)
        lstm_out = self.gam2(lstm_out)
        lstm_out = lstm_out.view(batch_size, -1, 1024)

        # 取最后一个时间步的输出
        final_output = lstm_out[:, -1, :]

        # Regressor 层
        out = self.regressor(final_output)
        return out
