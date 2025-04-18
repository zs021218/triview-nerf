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

        # 特征层
        self.feature_linear = nn.Linear(config.hidden_dims, config.hidden_dims)

        # 密度输出
        self.density_linear = nn.Linear(config.hidden_dims, 1)

        # 视角相关的MLP部分
        self.views_linear = nn.Linear(dir_enc_dims + config.hidden_dims, config.hidden_dims // 2)

        # RGB颜色输出
        self.rgb_linear = nn.Linear(config.hidden_dims // 2, 3)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, d):
        # 应用位置编码
        input_pts = pos_encoding(x, self.config.pos_encoding_dims)
        input_dirs = pos_encoding(d, self.config.dir_encoding_dims)

        # 通过空间MLP
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)

            if i == 4 and hasattr(self.config, 'skips') and self.config.skips:
                h = torch.cat([input_pts, h], -1)

        # 计算特征
        feature = self.feature_linear(h)
        feature = F.relu(feature)

        # 计算密度
        density = self.density_linear(h)

        # 计算颜色 - 结合视角信息
        h = torch.cat([feature, input_dirs], -1)
        h = self.views_linear(h)
        h = F.relu(h)

        # RGB输出
        rgb = self.rgb_linear(h)
        rgb = torch.sigmoid(rgb)

        outputs = torch.cat([rgb, density], -1)
        return outputs


class MicroalgaeNeRF(nn.Module):
    """针对微藻优化的NeRF模型"""

    def __init__(self, config):
        super(MicroalgaeNeRF, self).__init__()
        self.config = config

        # 使用基础NeRF网络
        self.nerf_network = NeRF(config)

        # 添加透明度网络分支
        self.transparency_network = nn.Sequential(
            nn.Linear(config.hidden_dims, config.hidden_dims // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dims // 2, 1),
            nn.Sigmoid()  # 输出透明度值(0-1)
        )

        # 微藻颜色增强网络
        if config.color_enhancement:
            self.color_enhancement = nn.Sequential(
                nn.Linear(3, 16),
                nn.ReLU(),
                nn.Linear(16, 3),
                nn.Sigmoid()
            )

        # 特征融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(4, config.hidden_dims),
            nn.ReLU(),
            nn.Linear(config.hidden_dims, config.hidden_dims // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dims // 2, 4)
        )

        # 边缘增强模块 - 帮助保持微藻的圆形边界
        self.edge_network = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, d):
        # 使用基本网络获取特征
        features = self.nerf_network(x, d)

        # 获取基本RGB和密度
        base_rgb = features[..., :3]
        base_density = features[..., 3:4]

        # 用于透明度估计的特征
        hidden_features = torch.cat([x, d], dim=-1)

        # 应用特征融合
        fused_features = self.fusion_network(features)

        # 分离颜色和密度
        rgb = torch.sigmoid(fused_features[..., :3])
        density = fused_features[..., 3:4]

        # 应用边缘增强
        edge_factor = self.edge_network(features)

        # 估计透明度
        transparency = self.transparency_network(
            torch.cat([x, torch.norm(x, dim=-1, keepdim=True)], dim=-1)
        )

        # 调整密度 - 考虑透明度和边缘因子
        adjusted_density = density * (1.0 - self.config.transparency_weight * transparency)
        adjusted_density = adjusted_density * (1.0 + edge_factor * self.config.edge_sampling_boost)

        # 增强颜色（如果启用）
        if hasattr(self.config, 'color_enhancement') and self.config.color_enhancement:
            enhanced_rgb = self.color_enhancement(rgb)
            # 混合原始RGB和增强RGB
            final_rgb = 0.7 * enhanced_rgb + 0.3 * rgb
        else:
            final_rgb = rgb

        # 确保颜色在有效范围内
        final_rgb = torch.clamp(final_rgb, 0.0, 1.0)

        return torch.cat([final_rgb, adjusted_density], dim=-1)