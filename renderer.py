import torch
import torch.nn.functional as F


class VolumeRenderer:
    def __init__(self, config):
        self.config = config

    def render_rays(self, model, rays_o, rays_d):
        """体渲染函数: 沿光线进行体积积分"""
        batch_size = rays_o.shape[0]
        device = rays_o.device

        # 生成采样点
        t_vals = torch.linspace(self.config.near, self.config.far, self.config.num_samples, device=device)
        z_vals = t_vals.expand(batch_size, self.config.num_samples)

        # 添加随机抖动以减轻分层伪影
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

        # 计算采样点的3D位置
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        # 展平点以并行评估网络
        pts_flat = pts.reshape(-1, 3)
        dirs_flat = rays_d.unsqueeze(1).expand(pts.shape).reshape(-1, 3)
        dirs_flat = F.normalize(dirs_flat, dim=-1)

        # 网络评估
        raw = model(pts_flat, dirs_flat)
        raw = raw.reshape(batch_size, self.config.num_samples, 4)

        # 解析网络输出
        rgb = raw[..., :3]  # [N_rays, N_samples, 3]
        sigma = raw[..., 3]  # [N_rays, N_samples]

        # 计算沿光线的距离
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-3], -1)

        # Alpha合成
        alpha = 1.0 - torch.exp(-sigma * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((batch_size, 1), device=device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]

        # 计算最终的RGB和深度
        rgb_map = torch.sum(weights[..., None] * rgb, -2)
        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        return {
            'rgb': rgb_map,  # [N_rays, 3]
            'depth': depth_map,  # [N_rays]
            'acc': acc_map,  # [N_rays]
            'weights': weights,  # [N_rays, N_samples]
            'z_vals': z_vals,  # [N_rays, N_samples]
        }

    def render_image(self, model, H, W, focal, c2w):
        """渲染完整图像"""
        from utils import generate_rays

        rays_o, rays_d = generate_rays(H, W, focal, c2w)
        rays_o = rays_o.to(self.config.device)
        rays_d = rays_d.to(self.config.device)

        # 将图像分成小批次进行渲染
        all_rgb = []
        all_depth = []

        for i in range(0, rays_o.shape[0], self.config.batch_size):
            end_i = min(i + self.config.batch_size, rays_o.shape[0])
            results = self.render_rays(
                model,
                rays_o[i:end_i].reshape(-1, 3),
                rays_d[i:end_i].reshape(-1, 3)
            )
            all_rgb.append(results['rgb'])
            all_depth.append(results['depth'])

        rgb = torch.cat(all_rgb, 0).reshape(H, W, 3)
        depth = torch.cat(all_depth, 0).reshape(H, W)

        return {
            'rgb': rgb,  # [H, W, 3]
            'depth': depth  # [H, W]
        }