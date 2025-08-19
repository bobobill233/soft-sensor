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


class SwinCNN(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=1):
        super(SwinCNN, self).__init__()

        # 初始化 SwinExtractor
        self.swin_extractor = SwinExtractor()

        # Preprocessing layer
        self.preprocess = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            EDF(128),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Update Linear layers based on expected input size
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),  # 假设经过处理后的特征维度为 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)  # 最终输出层
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

        # 处理特征
        processed_features = self.preprocess(fused_features)

        # 全局平均池化
        pooled_features = nn.AdaptiveAvgPool2d((1, 1))(processed_features)  # Shape: [B, 512, 1, 1]
        flattened_features = pooled_features.view(pooled_features.size(0), -1)  # Shape: [B, 512]

        # 通过回归层
        output = self.regressor(flattened_features)  # Shape: [B, num_classes]

        return output

