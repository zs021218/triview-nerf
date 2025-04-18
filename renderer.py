import torch
import torch.nn.functional as F


class VolumeRenderer:
    def __init__(self, config):
        self.config = config

    def render_rays(self, model, rays_o, rays_d):
        """体渲染函数: 确保可见性"""
        batch_size = rays_o.shape[0]
        device = rays_o.device

        # 生成采样点，使用更大的扰动来确保好的采样
        t_vals = torch.linspace(self.config.near, self.config.far, self.config.num_samples, device=device)
        z_vals = t_vals.expand(batch_size, self.config.num_samples)

        # 添加随机抖动 - 使用更大的扰动范围
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

        # 使用分块处理来减少内存使用
        raw = []
        for i in range(0, pts_flat.shape[0], self.config.model_chunk_size):
            end_i = min(i + self.config.model_chunk_size, pts_flat.shape[0])
            pts_chunk = pts_flat[i:end_i]
            dirs_chunk = dirs_flat[i:end_i]
            raw_chunk = model(pts_chunk, dirs_chunk)
            raw.append(raw_chunk)

        raw = torch.cat(raw, 0)
        raw = raw.reshape(batch_size, self.config.num_samples, 4)

        # 解析网络输出
        rgb = raw[..., :3]  # [N_rays, N_samples, 3]
        sigma = raw[..., 3]  # [N_rays, N_samples]

        # 确保密度不为零 - 增加一个很小的值防止全黑
        sigma = sigma + 1e-5

        # 计算沿光线的距离
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e-3], -1)

        # Alpha合成 - 调整透明度算法
        alpha = 1.0 - torch.exp(-sigma * dists)

        # 调试: 打印一些值以检查渲染过程
        if torch.isnan(alpha).any():
            print("警告: 发现NaN值")
            alpha = torch.nan_to_num(alpha, 0.0)

        # 修改权重计算以避免数值问题
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((batch_size, 1), device=device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]

        # 计算最终的RGB和深度
        rgb_map = torch.sum(weights[..., None] * rgb, -2)

        # 确保RGB值不为零 - 如果全部为零，添加一个小偏移
        if torch.all(rgb_map < 1e-4):
            print("警告: RGB值非常小，可能导致全黑图像")
            # 为输出添加一个小的偏移，使其可见
            rgb_map = rgb_map + 0.01

        # 确保RGB值在有效范围内
        rgb_map = torch.clamp(rgb_map, 0.0, 1.0)

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)

        return {
            'rgb': rgb_map,  # [N_rays, 3]
            'depth': depth_map,  # [N_rays]
            'acc': acc_map,  # [N_rays]
            'weights': weights,  # [N_rays, N_samples]
            'z_vals': z_vals,  # [N_rays, N_samples]
            'sigma': sigma.mean().item(),  # 记录平均密度用于调试
            'alpha': alpha.mean().item(),  # 记录平均alpha用于调试
        }

    def render_image(self, model, H, W, focal, c2w):
        """渲染完整图像，使用分块处理"""
        from utils import generate_rays

        rays_o, rays_d = generate_rays(H, W, focal, c2w)
        rays_o = rays_o.to(self.config.device)
        rays_d = rays_d.to(self.config.device)

        # 将光线展平
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)

        # 使用分块处理来渲染图像
        all_rgb = []
        all_depth = []

        for i in range(0, rays_o.shape[0], self.config.chunk_size):
            # 清除先前的缓存
            torch.cuda.empty_cache()

            end_i = min(i + self.config.chunk_size, rays_o.shape[0])
            results = self.render_rays(
                model,
                rays_o[i:end_i],
                rays_d[i:end_i]
            )

            # 收集结果
            all_rgb.append(results['rgb'].detach())
            all_depth.append(results['depth'].detach())

            # 显式清理不再需要的张量
            del results
            torch.cuda.empty_cache()

        # 合并所有结果
        rgb = torch.cat(all_rgb, 0).reshape(H, W, 3)
        depth = torch.cat(all_depth, 0).reshape(H, W)

        return {
            'rgb': rgb,  # [H, W, 3]
            'depth': depth  # [H, W]
        }