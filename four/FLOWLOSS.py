import torch
import torch.nn as nn

class CustomLoss(nn.Module):
    def __init__(self, L_init=1.0, alpha_init=1.0, beta_init=1.0, reduction="mean"):
        super(CustomLoss, self).__init__()

        # 定义可学习的参数
        self.L = nn.Parameter(torch.tensor(L_init, dtype=torch.float32))  # 可学习的系数 L
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))  # 权重系数 alpha
        self.beta = nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))  # 权重系数 beta

        # ✅ 使用 nn.MSELoss 计算 MSE
        self.mse_loss = nn.MSELoss(reduction=reduction)  # 默认是 "mean"

    def forward(self, M, rho, v, F, delta_h, y_true):
        """
        计算自定义损失
        参数:
        - M: 预测的质量流量
        - rho: 溶液密度
        - v: 速度
        - F: 冲击力特征（取第4行和第5行的均值）
        - delta_h: 液位高度
        - y_true: 真实质量流量
        """
        eps = 1  # 避免除以零

        min_beta = 1e-4
        max_beta = 1e-3
        self.beta.data = torch.clamp(self.beta.data, min=min_beta, max=max_beta)

        # ✅ 计算物理信息损失
        predicted_value = (self.L * F * delta_h) / (rho * v ** 2 + eps)
        physics_loss = torch.square(M - predicted_value)  # 计算平方误差

        # ✅ 使用 nn.MSELoss 计算 MSE
        mse_loss = self.mse_loss(M, y_true)

        # ✅ 加权后的总损失
        total_loss = self.alpha * mse_loss + self.beta * torch.mean(physics_loss)
        return total_loss
