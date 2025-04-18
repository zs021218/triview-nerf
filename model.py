import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import pos_encoding


class NeRF(nn.Module):
    def __init__(self, config):
        super(NeRF, self).__init__()
        self.config = config

        # 计算编码后的输入维度
        pos_enc_dims = 3 + 3 * 2 * config.pos_encoding_dims  # (x,y,z) + sin/cos编码
        dir_enc_dims = 3 + 3 * 2 * config.dir_encoding_dims  # (dx,dy,dz) + sin/cos编码

        # 位置编码的MLP部分
        self.pts_linears = nn.ModuleList(
            [nn.Linear(pos_enc_dims, config.hidden_dims)] +
            [nn.Linear(config.hidden_dims, config.hidden_dims) for _ in range(config.num_layers - 1)]
        )

        # 密度预测
        self.density_linear = nn.Linear(config.hidden_dims, 1)

        # 视角相关的MLP部分
        self.feature_linear = nn.Linear(config.hidden_dims, config.hidden_dims)
        self.views_linear = nn.Linear(dir_enc_dims + config.hidden_dims, config.hidden_dims // 2)

        # RGB颜色预测
        self.rgb_linear = nn.Linear(config.hidden_dims // 2, 3)

    def forward(self, x, d):
        # x: [B, 3] 位置坐标
        # d: [B, 3] 视角方向

        # 应用位置编码
        input_pts = pos_encoding(x, self.config.pos_encoding_dims)
        input_dirs = pos_encoding(d, self.config.dir_encoding_dims)

        # 处理位置信息
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

        # 预测密度
        density = self.density_linear(h)
        density = F.relu(density)

        # 处理特征和方向
        feat = self.feature_linear(h)
        h = torch.cat([feat, input_dirs], -1)

        h = self.views_linear(h)
        h = F.relu(h)

        # 预测RGB颜色
        rgb = self.rgb_linear(h)
        rgb = torch.sigmoid(rgb)  # 确保RGB值在[0,1]范围

        return torch.cat([rgb, density], -1)  # [B, 4] -> (R,G,B,density)


class TriViewNeRF(nn.Module):
    """三视图NeRF模型"""

    def __init__(self, config):
        super(TriViewNeRF, self).__init__()
        self.config = config

        # 为每个视角创建一个单独的NeRF编码器
        self.view_encoders = nn.ModuleDict({
            view: NeRF(config) for view in config.views
        })

        # 特征融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(len(config.views) * 4, config.hidden_dims),
            nn.ReLU(),
            nn.Linear(config.hidden_dims, config.hidden_dims),
            nn.ReLU(),
            nn.Linear(config.hidden_dims, 4)  # 输出RGB和密度
        )

    def forward(self, x, d):
        # 从每个视角获取特征
        view_features = []
        for view in self.config.views:
            features = self.view_encoders[view](x, d)
            view_features.append(features)

        # 拼接所有视角的特征
        combined_features = torch.cat(view_features, dim=-1)

        # 融合特征
        output = self.fusion_network(combined_features)

        # 分离RGB和密度
        rgb = torch.sigmoid(output[..., :3])
        density = F.relu(output[..., 3:4])

        return torch.cat([rgb, density], dim=-1)