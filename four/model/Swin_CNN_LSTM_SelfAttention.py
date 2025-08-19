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
    def __init__(self, in_channels, reduction_ratio=8):
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


class SwinCNNLSTMSelfAttention(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=1):
        super(SwinCNNLSTMSelfAttention, self).__init__()
        # 初始化 SwinExtractor
        self.swin_extractor = SwinExtractor()
        # Preprocessing layer
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),  # Add BN layer
            nn.ReLU(),
            EDF(128),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),  # Add BN layer
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Add GAM
        # self.gam1 = GAM(in_channels=512)

        # LSTM layer
        self.lstm = nn.LSTM(input_size=512 * 8 * 8, hidden_size=1024, num_layers=3, batch_first=True, dropout=0.25)

        # self.gam2 = GAM(in_channels=1024)

        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=1024, num_heads=8, batch_first=True)

        # Regressor layer
        self.regressor = nn.Sequential(
            nn.Dropout(0.5),  # Add Dropout layer before the final regressor
            nn.Linear(1024, num_classes) # Final linear layer reducing from 128 to 1
        )

    def forward(self, gasf_tensor, mtf_tensor):
        # Reshape SwinV2 output to [batch_size, channels, height, width]

        gasf_features = self.swin_extractor(gasf_tensor)
        mtf_features = self.swin_extractor(mtf_tensor)

        batch_size, num_features = gasf_features.shape
        gasf_features = gasf_features.view(batch_size, num_features // (32 * 32), 32, 32)
        mtf_features = mtf_features.view(batch_size, num_features // (32 * 32), 32, 32)

        # print(f'GASF features shape after reshape: {gasf_features}')
        # print(f'MTF features shape after reshape: {mtf_features}')

        # Feature fusion
        fused_features = torch.cat((gasf_features, mtf_features), dim=1)
        # print(f'Fused features shape: {fused_features.shape}')

        # Preprocessing layer
        x = self.preprocess(fused_features)
        # print(f'Shape after preprocessing: {x}')

        # # CNN layer with Patch Merging
        # x = self.cnn(x)
        # # print(f'Shape after CNN: {x.shape}')

        # Apply GAM
        # x = self.gam1(x)
        # print(f'Shape after GAM: {x.shape}')

        # Reshape x to fit LSTM input
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 512 * 8 * 8)
        # print(f'Shape after reshaping for LSTM: {x.shape}')

        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # print(f'Shape after LSTM: {lstm_out}')

        # Apply second GAM
        lstm_out = lstm_out.view(batch_size, 1, -1, 1024).permute(0, 3, 1, 2)
        # lstm_out = self.gam2(lstm_out)
        lstm_out = lstm_out.view(batch_size, -1, 1024)

        # Take the output of the last time step
        final_output = lstm_out[:, -1, :]

        # Attention mechanism
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = attn_output[:, -1, :]  # Take the output of the last time step
        # print(f'Shape after Attention: {attn_output}')

        # Regressor layer
        out = self.regressor(final_output)
        # print(f'Shape after regressor: {out}')
        return out