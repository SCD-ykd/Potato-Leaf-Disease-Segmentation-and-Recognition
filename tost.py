import torch
from torch import nn

class TSSA(nn.Module):
    def __init__(self, dim, heads=8, dim_head=None):
        super().__init__()
        self.heads = heads
        if dim_head is None:
            # 保证 heads * dim_head 能被 dim 整除？不，我们直接投影回 dim
            dim_head = dim // heads if dim % heads == 0 else dim
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim_head * heads * 3)
        # 新增输出投影层，将 heads*dim_head 投影回 dim
        self.proj = nn.Linear(dim_head * heads, dim)

    def forward(self, x):
        b, n, _ = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(b, n, self.heads, -1).transpose(1, 2), qkv)
        # 统计量 (b, heads, n)
        stats = torch.einsum('bhid,bhjd->bhij', q, k).mean(dim=-1)
        gate = torch.sigmoid(stats)                          # (b, heads, n)
        # 加权聚合
        out = torch.einsum('bhi,bhjd->bhid', gate, v)        # (b, heads, n, dim_head)
        out = out.transpose(1, 2).reshape(b, n, -1)         # (b, n, heads*dim_head)
        out = self.proj(out)                                # (b, n, dim)
        return out

class ToSTBlock(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.attn = TSSA(dim, heads=heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ToST(nn.Module):
    def __init__(self, num_layers=12, dim=768, heads=12):
        super().__init__()
        self.layers = nn.ModuleList([
            ToSTBlock(dim, heads) for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x