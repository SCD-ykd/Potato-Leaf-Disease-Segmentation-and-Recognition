import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange
from timm.models.layers import trunc_normal_

# ------------------------------------------------------------
# 1. StarReLU 激活函数
# ------------------------------------------------------------
class StarReLU(nn.Module):
    """StarReLU: s * relu(x) ** 2 + b"""
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True):
        super().__init__()
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

# ------------------------------------------------------------
# 2. 轻量级 MLP（用于生成路由权重）
# ------------------------------------------------------------
class Mlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0., bias=False):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

# ------------------------------------------------------------
# 3. FDAM 核心模块（GroupDynamicScale）
# ------------------------------------------------------------
class GroupDynamicScale(nn.Module):
    """
    Frequency Dynamic Attention Modulation (FDAM)
    论文: Frequency-Dynamic Attention Modulation (ICCV 2025)
    作者: Linwei Chen
    代码来源: https://github.com/Linwei-Chen/FDAM
    """
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.125,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=True,
                 group=32, init_scale=1e-5, **kwargs):
        super().__init__()
        self.size = size
        self.filter_size = size // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.weight_resize = weight_resize

        # 路由网络：从空间特征中学习 band weights
        self.reweight = Mlp(dim, reweight_expansion_ratio, group * num_filters, bias=False)

        # 可学习的频域复数核（实数形式，分别存储实部和虚部？原代码存储为单一实数，后面与路由权重组合）
        # 原实现: complex_weights shape = [num_filters, dim//group, size, filter_size]
        self.complex_weights = nn.Parameter(
            torch.randn(num_filters, dim // group, self.size, self.filter_size, dtype=torch.float32) * init_scale
        )
        trunc_normal_(self.complex_weights, std=init_scale)

        self.act2 = act2_layer()  # 恒等映射

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征图
        Returns:
            out: (B, C, H, W) 频域动态调制后的特征图
        """
        B, C, H, W = x.shape
        # 1. 实数 FFT (RFFT)
        x_rfft = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')
        # x_rfft shape: (B, C, H, W//2+1)

        # 2. 将特征从 BCHW -> BHWC 用于路由权重计算（因为 reweight 期望 B, N, C）
        x_perm = x.permute(0, 2, 3, 1)           # (B, H, W, C)
        route_feat = x_perm.mean(dim=(1, 2))      # (B, C) 全局池化

        # 3. 生成路由权重 (B, group * num_filters) -> (B, group, num_filters)
        routeing = self.reweight(route_feat).view(B, -1, self.num_filters).tanh_()   # (B, group, num_filters)

        # 4. 获取复数权重，必要时调整大小以匹配频谱尺寸
        weight = self.complex_weights            # (num_filters, C_per_group, size, filter_size)
        if weight.shape[2:4] != x_rfft.shape[2:4]:
            weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)

        # 5. 组合路由权重和复数核：输出 (B, group, C_per_group, H, W//2+1) -> 合并为 (B, C, H, W//2+1)
        # Einstein summation: (B, g, f) 与 (f, cg, h, w)  -> (B, g, cg, h, w)
        weight = torch.einsum('bgf,fchw->bgchw', routeing, weight)   # (B, group, C_per_group, H, W')
        weight = weight.reshape(B, C, *x_rfft.shape[2:])             # (B, C, H, W')

        # 6. 将权重应用到频谱（实部与虚部分别相乘，因为 x_rfft 是复数）
        # 原代码：x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        x_rfft = torch.view_as_complex(
            torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1)
        )

        # 7. 逆傅里叶变换回空间域
        out = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')
        return out