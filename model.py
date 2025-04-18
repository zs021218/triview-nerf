import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import pos_encoding


class NeRF(nn.Module):
    def __init__(self, config):
        super(NeRF, self).__init__()
        self.config = config

        # 计算编码后的输入维度
        pos_enc_dims = 3 + 3 * 2 * config.pos_encoding_dims
        dir_enc_dims = 3 + 3 * 2 * config.dir_encoding_dims

        # 位置编码的MLP部分
        self.pts_linears = nn.ModuleList(
            [nn.Linear(pos_enc_dims, config.hidden_dims)] +
            [nn.Linear(config.hidden_dims, config.hidden_dims) for _ in range(config.num_layers - 1)]
        )

        # 密度预测 - 使用正确的初始化
        self.density_linear = nn.Linear(config.hidden_dims, 1)
        nn.init.xavier_uniform_(self.density_linear.weight)  # 更好的初始化

        # 视角相关的MLP部分
        self.feature_linear = nn.Linear(config.hidden_dims, config.hidden_dims // 2)
        self.views_linear = nn.Linear(dir_enc_dims + config.hidden_dims // 2, config.hidden_dims // 2)

        # RGB颜色预测 - 使用正确的初始化
        self.rgb_linear = nn.Linear(config.hidden_dims // 2, 3)
        nn.init.xavier_uniform_(self.rgb_linear.weight)  # 更好的初始化

    def forward(self, x, d):
        # 应用位置编码
        input_pts = pos_encoding(x, self.config.pos_encoding_dims)
        input_dirs = pos_encoding(d, self.config.dir_encoding_dims)

        # 处理位置信息
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        # 预测密度 - 确保输出为正值，使用softplus而不是relu
        density = F.softplus(self.density_linear(h))  # 使用softplus保证密度平滑正值

        # 处理特征和方向
        feat = self.feature_linear(h)
        h = torch.cat([feat, input_dirs], -1)

        h = self.views_linear(h)
        h = F.relu(h)

        # 预测RGB颜色
        rgb = torch.sigmoid(self.rgb_linear(h))  # 确保RGB值在[0,1]范围

        return torch.cat([rgb, density], -1)  # [B, 4] -> (R,G,B,density)


class TriViewNeRF(nn.Module):
    """简化版三视图NeRF模型"""

    def __init__(self, config):
        super(TriViewNeRF, self).__init__()
        self.config = config

        # 使用单个共享网络
        self.nerf_network = NeRF(config)

        # 简化的融合网络，使用恰当的初始化
        self.fusion_network = nn.Sequential(
            nn.Linear(4, config.hidden_dims // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dims // 2, 4)
        )

        # 初始化融合网络
        for m in self.fusion_network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, d):
        # 使用相同的网络处理
        features = self.nerf_network(x, d)

        # 应用简单的特征融合
        output = self.fusion_network(features)

        # 分离RGB和密度，并确保值域正确
        rgb = torch.sigmoid(output[..., :3])  # 确保RGB值在[0,1]
        density = F.softplus(output[..., 3:4])  # 使用softplus确保密度为正值

        return torch.cat([rgb, density], dim=-1)